"""Persist vectors with a model/corpus manifest; optionally compare FAISS HNSW.

SQLite + exhaustive cosine search teaches persistence, not ANN acceleration.
Default files are temporary. --path stores a reusable index at your chosen path.
"""

import argparse
import hashlib
import json
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path

from rag.core import Chunk, Hit, Tfidf, corpus, dot, print_hits


def build_index(path: Path, chunks: list[Chunk], encoder: Tfidf) -> dict:
    manifest = {"schema": 1, "model": "tfidf-v1", "vocab": encoder.vocab, "idf": encoder.idf,
                "normalized": True, "dimension": len(encoder.vocab),
                "fingerprint": hashlib.sha256(json.dumps([asdict(c) for c in chunks], sort_keys=True).encode()).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS rag_manifest (id INTEGER PRIMARY KEY, payload TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS rag_vectors (id TEXT PRIMARY KEY, chunk TEXT, vector TEXT)")
        existing = db.execute("SELECT payload FROM rag_manifest WHERE id=1").fetchone()
        if existing and json.loads(existing[0]) != manifest:
            raise ValueError("Index manifest changed. Choose a new --path for a new index generation.")
        db.execute("INSERT OR REPLACE INTO rag_manifest VALUES (1, ?)", (json.dumps(manifest),))
        db.executemany("INSERT OR REPLACE INTO rag_vectors VALUES (?, ?, ?)",
                       [(c.id, json.dumps(asdict(c)), json.dumps(encoder.encode(c.text))) for c in chunks])
    return manifest


def search_index(path: Path, query: str, tenant: str = "alpha", k: int = 3) -> list[Hit]:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        manifest = json.loads(db.execute("SELECT payload FROM rag_manifest WHERE id=1").fetchone()[0])
        if manifest["schema"] != 1 or manifest["model"] != "tfidf-v1":
            raise ValueError("Unsupported index manifest")
        encoder = Tfidf([])
        encoder.vocab, encoder.idf = manifest["vocab"], manifest["idf"]
        q = encoder.encode(query)
        hits = []
        for chunk_json, vector_json in db.execute("SELECT chunk, vector FROM rag_vectors"):
            chunk = Chunk(**json.loads(chunk_json))
            # Filter before ranking. Use indexed ACL predicates in a real database.
            if chunk.metadata["tenant"] != tenant:
                continue
            score = dot(q, json.loads(vector_json))
            if score > 0:
                hits.append(Hit(chunk, score))
    return sorted(hits, key=lambda h: (-h.score, h.chunk.id))[:max(0, k)]


def faiss_demo(chunks: list[Chunk], encoder: Tfidf, query: str) -> None:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Install rag/requirements-optional.txt for FAISS") from exc
    vectors = np.asarray([encoder.encode(c.text) for c in chunks], dtype="float32")
    q = np.asarray([encoder.encode(query)], dtype="float32")
    exact = faiss.IndexFlatIP(vectors.shape[1])
    ann = faiss.IndexHNSWFlat(vectors.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
    ann.hnsw.efConstruction = 80
    ann.hnsw.efSearch = 32
    exact.add(vectors)
    ann.add(vectors)
    k = min(5, len(chunks))
    _, exact_ids = exact.search(q, k)
    _, ann_ids = ann.search(q, k)
    print("HNSW overlap with exact top-k:", len(set(exact_ids[0]) & set(ann_ids[0])) / k)
    print("Tiny-corpus mechanics only; this is not an ANN performance benchmark.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--faiss", action="store_true")
    args = parser.parse_args()
    chunks = corpus()
    encoder = Tfidf([c.text for c in chunks])
    query = "How long are backups retained?"
    if args.faiss:
        faiss_demo(chunks, encoder, query)
    with tempfile.TemporaryDirectory(prefix="rag-index-") as tmp:
        path = args.path or Path(tmp) / "index.sqlite"
        manifest = build_index(path, chunks, encoder)
        print(f"Index dimension={manifest['dimension']} fingerprint={manifest['fingerprint'][:12]}")
        print_hits(search_index(path, query))


if __name__ == "__main__":
    main()
