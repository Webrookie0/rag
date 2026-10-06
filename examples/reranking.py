"""Rerank a candidate pool using a transparent heuristic or a real cross-encoder."""

import argparse

from rag.core import Hit, corpus, print_hits, retrieve, sentences, tokens


def lexical_rerank(query: str, hits: list[Hit], k: int = 3) -> list[Hit]:
    """Teaching baseline: strongest sentence coverage; NOT a neural reranker."""
    terms = set(tokens(query))
    def score(hit: Hit) -> float:
        coverage = max((len(terms & set(tokens(s))) / max(1, len(terms)) for s in sentences(hit.chunk.text)), default=0)
        # Small whole-passage tiebreaker; raw retrieval scores aren't comparable.
        return coverage + 0.05 * len(terms & set(tokens(hit.chunk.text))) / max(1, len(terms))
    return sorted((Hit(h.chunk, score(h)) for h in hits), key=lambda h: (-h.score, h.chunk.id))[:max(0, k)]


def neural_rerank(query: str, hits: list[Hit], k: int = 3) -> list[Hit]:
    if not hits or k <= 0:
        return []
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise SystemExit("Install rag/requirements-optional.txt for the neural reranker") from exc
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    scores = model.predict([(query, h.chunk.text) for h in hits])
    # These model scores are not calibrated probabilities.
    return sorted((Hit(h.chunk, float(s)) for h, s in zip(hits, scores)), key=lambda h: -h.score)[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", action="store_true")
    parser.add_argument("--query", default="Are monthly subscriptions refundable?")
    args = parser.parse_args()
    candidates = retrieve(args.query, corpus(), 8)
    print("Candidates:")
    print_hits(candidates)
    print("\nReranked:", "cross-encoder" if args.neural else "lexical heuristic")
    print_hits((neural_rerank if args.neural else lexical_rerank)(args.query, candidates))


if __name__ == "__main__":
    main()
