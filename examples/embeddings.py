"""Create normalized lexical or real neural embeddings; inspect cosine retrieval."""

import argparse

from rag.core import Chunk, Hit, Tfidf, corpus, dot, print_hits


class DenseEncoder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit("Install optional dependencies: python -m pip install -r rag/requirements-optional.txt") from exc
        self.model_id = model
        self.model = SentenceTransformer(model)

    def documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode_document(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def query(self, text: str) -> list[float]:
        return self.model.encode_query(text, normalize_embeddings=True, show_progress_bar=False).tolist()


def dense_search(query: str, chunks: list[Chunk], k: int = 5, encoder: DenseEncoder | None = None) -> list[Hit]:
    if not chunks or k <= 0:
        return []
    encoder = encoder or DenseEncoder()
    q = encoder.query(query)
    vectors = encoder.documents([c.text for c in chunks])
    hits = [Hit(c, dot(q, v)) for c, v in zip(chunks, vectors)]
    return sorted(hits, key=lambda h: (-h.score, h.chunk.id))[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["tfidf", "dense"], default="tfidf")
    parser.add_argument("--query", default="How can I get my annual subscription money back?")
    args = parser.parse_args()
    chunks = corpus()
    if args.backend == "dense":
        print_hits(dense_search(args.query, chunks))
    else:
        model = Tfidf([c.text for c in chunks])
        print(f"Lexical dimensions: {len(model.vocab)}; unknown query terms have zero weight.")
        from rag.core import vector_search
        print_hits(vector_search(args.query, chunks))


if __name__ == "__main__":
    main()
