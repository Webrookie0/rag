"""Rewrite, multi-query/RAG fusion, decomposition, step-back, and HyDE.

Offline transformations below are fixtures for the DEFAULT question, not
general-purpose LLM reasoning. --ollama generates transformations for any query.
HyDE becomes neural document-space search only when --dense is also supplied.
"""

import argparse
import json

from rag.core import Hit, corpus, dot, print_hits, retrieve, rrf

DEFAULT_QUERY = "Can I get my money back and stop the next payment?"
FIXTURES = {
    "rewrite": ["What is the Atlas subscription refund and cancellation policy?"],
    "multi": ["Atlas subscription refund eligibility", "Cancel subscription and stop renewal", "Get money back after purchase"],
    "decompose": ["When is an Atlas subscription refundable?", "How do I cancel an Atlas subscription?"],
    "step-back": ["What are the Atlas billing lifecycle and subscription policies?"],
    "hyde": ["A subscription policy describes refund eligibility, purchase dates, payment methods, cancellation, and future renewals."],
}


def transform(query: str, mode: str, use_llm: bool = False) -> list[str]:
    if not use_llm:
        if query != DEFAULT_QUERY:
            raise ValueError("Offline fixtures only support the default question. Use --ollama for your own query.")
        return FIXTURES[mode]
    from rag.examples.llm import chat
    instructions = {
        "rewrite": "Rewrite the question as one precise standalone search query.",
        "multi": "Write three distinct search queries expressing the same information need.",
        "decompose": "Split the question into at most three independently searchable subquestions.",
        "step-back": "Write one broader conceptual question that helps answer the original.",
        "hyde": "Write a short hypothetical answer passage for retrieval only. It may be wrong and will never be cited.",
    }
    raw = chat(instructions[mode] + ' Return JSON with a nonempty array of strings named "queries".', query, True)
    result = json.loads(raw).get("queries")
    if not isinstance(result, list) or not 1 <= len(result) <= 5 or not all(isinstance(x, str) and x.strip() for x in result):
        raise ValueError("Model returned an invalid query list")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(FIXTURES), default="multi")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()
    transformed = transform(args.query, args.mode, args.ollama)
    print("Transform source:", "local LLM" if args.ollama else "curated fixture")
    print(json.dumps(transformed, indent=2))
    chunks = corpus()
    if args.dense:
        from rag.examples.embeddings import DenseEncoder
        encoder = DenseEncoder()
        vectors = encoder.documents([c.text for c in chunks])
        rankings = []
        for text in [args.query] + transformed:
            # HyDE embeds the invented passage in DOCUMENT space; it isn't evidence.
            q = encoder.documents([text])[0] if args.mode == "hyde" and text != args.query else encoder.query(text)
            rankings.append(sorted([Hit(c, dot(q, v)) for c, v in zip(chunks, vectors)], key=lambda h: -h.score)[:8])
    else:
        if args.mode == "hyde":
            print("Lexical HyDE analogue; use --ollama --dense for generated neural HyDE.")
        rankings = [retrieve(q, chunks, 8) for q in [args.query] + transformed]
    print_hits(rrf(rankings, 5))


if __name__ == "__main__":
    main()
