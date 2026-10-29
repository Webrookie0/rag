"""Small transparent building blocks shared by the lessons.

TF-IDF is a lexical vector baseline, NOT a neural semantic embedding.
The retrievers here scan a tiny in-memory corpus; they are not production indexes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).parent / "data"
STOPWORDS = set("a an the is are was were to of for in on and or with how what when who i my do does can have it its be as by from should than once".split())


def tokens(text: str) -> list[str]:
    """Simple Unicode word tokenizer; underscores preserve identifiers like AUTH_401."""
    return [t for t in re.findall(r"\w+", text.casefold()) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    """Teaching splitter: abbreviations, code, and complex languages need a parser."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str
    tenant: str = "alpha"
    version: str = "1"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    position: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def load_documents(tenant: str = "alpha", path: Path | None = None) -> list[Document]:
    """Pre-filter at the trusted boundary; tenant must come from authenticated state."""
    rows = json.loads((path or DATA / "documents.json").read_text(encoding="utf-8"))
    return [Document(**row) for row in rows if row["tenant"] == tenant]


def make_chunks(doc: Document, parts: list[str]) -> list[Chunk]:
    chunks = []
    for i, part in enumerate(p for p in parts if p.strip()):
        digest = hashlib.sha256(f"{doc.tenant}|{doc.id}|{doc.version}|{i}|{part}".encode()).hexdigest()[:12]
        chunks.append(Chunk(f"{doc.id}:{digest}", doc.id, part.strip(), i,
                            {"title": doc.title, "source": doc.source,
                             "tenant": doc.tenant, "version": doc.version}))
    return chunks


def corpus(tenant: str = "alpha") -> list[Chunk]:
    return [c for d in load_documents(tenant) for c in make_chunks(d, d.text.split("\n\n"))]


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions differ; rebuild with the same model.")
    return sum(x * y for x, y in zip(a, b))


def normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


class Tfidf:
    """Fit vocabulary + IDF on documents only; reuse both for every query."""

    def __init__(self, texts: list[str]):
        counts = Counter(t for text in texts for t in set(tokens(text)))
        self.vocab = sorted(counts)
        self.idf = {t: math.log((len(texts) + 1) / (counts[t] + 1)) + 1 for t in counts}

    def encode(self, text: str) -> list[float]:
        counts = Counter(tokens(text))
        return normalize([(1 + math.log(counts[t])) * self.idf[t] if counts[t] else 0.0
                          for t in self.vocab])


def vector_search(query: str, chunks: list[Chunk], k: int = 5) -> list[Hit]:
    model = Tfidf([c.text for c in chunks])
    q = model.encode(query)
    hits = [Hit(c, dot(q, model.encode(c.text))) for c in chunks]
    return sorted((h for h in hits if h.score > 0), key=lambda h: (-h.score, h.chunk.id))[:max(0, k)]


def bm25(query: str, chunks: list[Chunk], k: int = 5,
         k1: float = 1.5, b: float = 0.75) -> list[Hit]:
    """Positive-IDF BM25 with term-frequency saturation and length normalization."""
    if not chunks or k <= 0:
        return []
    bags = [Counter(tokens(c.text)) for c in chunks]
    avg_len = sum(sum(bag.values()) for bag in bags) / len(bags) or 1
    df = Counter(t for bag in bags for t in bag)
    hits = []
    for chunk, bag in zip(chunks, bags):
        score = 0.0
        for term in set(tokens(query)):
            freq = bag[term]
            if not freq:
                continue
            idf = math.log(1 + (len(chunks) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * sum(bag.values()) / avg_len))
        if score > 0:
            hits.append(Hit(chunk, score))
    return sorted(hits, key=lambda h: (-h.score, h.chunk.id))[:k]


def rrf(rankings: list[list[Hit]], k: int = 5, constant: int = 60) -> list[Hit]:
    """Reciprocal rank fusion combines ranks, not incompatible raw scores."""
    if constant < 0:
        raise ValueError("RRF constant must be nonnegative")
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.chunk.id in seen:
                continue
            seen.add(hit.chunk.id)
            by_id[hit.chunk.id] = hit.chunk
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0) + 1 / (constant + rank)
    ids = sorted(scores, key=lambda cid: (-scores[cid], cid))[:max(0, k)]
    return [Hit(by_id[cid], scores[cid]) for cid in ids]


def retrieve(query: str, chunks: list[Chunk], k: int = 5, method: str = "hybrid") -> list[Hit]:
    if method == "bm25":
        return bm25(query, chunks, k)
    if method == "tfidf":
        return vector_search(query, chunks, k)
    if method != "hybrid":
        raise ValueError(f"Unknown retrieval method: {method}")
    return rrf([bm25(query, chunks, max(10, k)), vector_search(query, chunks, max(10, k))], k)


def print_hits(hits: list[Hit]) -> None:
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit.chunk.id} score={hit.score:.4f}\n   {hit.chunk.text[:180]}")
