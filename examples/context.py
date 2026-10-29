"""MMR diversity, parent expansion, neighboring windows, compression, and budgets."""

import argparse

from rag.core import Chunk, Hit, Tfidf, corpus, dot, load_documents, make_chunks, print_hits, retrieve, sentences, tokens


def mmr(query: str, hits: list[Hit], k: int = 3, relevance_weight: float = 0.7) -> list[Hit]:
    """Maximize lambda * relevance - (1-lambda) * redundancy, using cosine for both."""
    if not 0 <= relevance_weight <= 1:
        raise ValueError("relevance_weight must lie in [0, 1]")
    model = Tfidf([h.chunk.text for h in hits])
    q = model.encode(query)
    vectors = {h.chunk.id: model.encode(h.chunk.text) for h in hits}
    selected, remaining = [], list(hits)
    while remaining and len(selected) < k:
        def objective(h: Hit) -> float:
            v = vectors[h.chunk.id]
            redundancy = max((dot(v, vectors[s.chunk.id]) for s in selected), default=0)
            return relevance_weight * dot(q, v) - (1 - relevance_weight) * redundancy
        best = max(remaining, key=objective)
        selected.append(best)
        remaining.remove(best)
    return selected


def parent_expansion(hits: list[Hit], tenant: str = "alpha") -> list[Hit]:
    parents = {d.id: d for d in load_documents(tenant)}
    result, seen = [], set()
    for hit in hits:
        if hit.chunk.metadata.get("tenant") != tenant:
            raise ValueError("Cross-tenant child cannot be expanded")
        if hit.chunk.doc_id not in seen and hit.chunk.doc_id in parents:
            doc = parents[hit.chunk.doc_id]
            result.append(Hit(make_chunks(doc, [doc.text])[0], hit.score))
            seen.add(doc.id)
    return result


def neighbor_windows(hits: list[Hit], chunks: list[Chunk], radius: int = 1) -> list[Hit]:
    result, seen = [], set()
    for hit in hits:
        for c in chunks:
            if (c.doc_id == hit.chunk.doc_id and c.metadata.get("tenant") == hit.chunk.metadata.get("tenant")
                    and abs(c.position - hit.chunk.position) <= radius and c.id not in seen):
                seen.add(c.id)
                result.append(Hit(c, hit.score))
    return result


def compress(query: str, hits: list[Hit]) -> list[Hit]:
    """Extract query-matching sentences. Can lose negation context or exceptions!"""
    result = []
    terms = set(tokens(query))
    for hit in hits:
        kept = [s for s in sentences(hit.chunk.text) if terms & set(tokens(s))]
        if kept:
            c = hit.chunk
            result.append(Hit(Chunk(c.id, c.doc_id, " ".join(kept), c.position, c.metadata), hit.score))
    return result


def edge_reorder(hits: list[Hit]) -> list[Hit]:
    """Place high-ranked passages near both ends; evaluate before adopting."""
    return hits[::2] + list(reversed(hits[1::2]))


def pack_context(hits: list[Hit], budget_words: int = 300) -> tuple[str, dict[str, Chunk]]:
    """Budget includes citation labels and source headers; retain whole chunks."""
    if budget_words < 0:
        raise ValueError("budget_words must be nonnegative")
    blocks, evidence, seen = [], {}, set()
    used = 0
    for hit in hits:
        c = hit.chunk
        if c.id in seen:
            continue
        label = f"S{len(evidence) + 1}"
        block = f"[{label}] source={c.metadata.get('source', c.doc_id)}\n{c.text}"
        cost = len(block.split())
        if used + cost > budget_words:
            continue
        blocks.append(block)
        evidence[label] = c
        seen.add(c.id)
        used += cost
    return "\n\n".join(blocks), evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["mmr", "parent", "window", "compress", "reorder"], default="mmr")
    parser.add_argument("--query", default="What happens after cancellation and can I get a refund?")
    parser.add_argument("--budget", type=int, default=220)
    args = parser.parse_args()
    chunks = corpus()
    hits = retrieve(args.query, chunks, 8)
    methods = {"mmr": lambda: mmr(args.query, hits), "parent": lambda: parent_expansion(hits[:3]),
               "window": lambda: neighbor_windows(hits[:2], chunks), "compress": lambda: compress(args.query, hits[:3]),
               "reorder": lambda: edge_reorder(hits[:5])}
    result = methods[args.method]()
    print_hits(result)
    context, evidence = pack_context(result, args.budget)
    print(f"\nPacked {len(context.split())}/{args.budget} words, {len(evidence)} sources:\n{context}")


if __name__ == "__main__":
    main()
