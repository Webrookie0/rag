"""Compare BM25, TF-IDF, hybrid fusion, and optional dense + BM25 fusion."""

import argparse

from rag.core import bm25, corpus, print_hits, retrieve, rrf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["bm25", "tfidf", "hybrid", "dense-hybrid"], default="hybrid")
    parser.add_argument("--query", default="What does AUTH_401 mean?")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    chunks = corpus()
    if args.method == "dense-hybrid":
        from rag.examples.embeddings import dense_search
        hits = rrf([bm25(args.query, chunks, 10), dense_search(args.query, chunks, 10)], args.k)
    else:
        hits = retrieve(args.query, chunks, args.k, args.method)
    print(f"Method: {args.method}; hybrid without 'dense' means BM25 + TF-IDF.")
    print_hits(hits)


if __name__ == "__main__":
    main()
