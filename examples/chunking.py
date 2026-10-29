"""Fixed, sentence, recursive, Markdown-section, and semantic-boundary chunking.

Sizes are WORDS, not model tokens. --neural enables semantic embeddings for
the semantic strategy; the default is explicitly a lexical boundary proxy.
"""

import argparse
import re

from rag.core import Tfidf, dot, load_documents, make_chunks, sentences


def fixed_words(text: str, size: int = 50, overlap: int = 10) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    words = text.split()
    parts = []
    for start in range(0, len(words), size - overlap):
        parts.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return parts


def pack(units: list[str], size: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    result, current = [], ""
    for unit in units:
        # Oversized units fall back to words so the maximum is a hard limit.
        for piece in fixed_words(unit, size, 0):
            if current and len((current + " " + piece).split()) > size:
                result.append(current)
                current = ""
            current = (current + " " + piece).strip()
    return result + ([current] if current else [])


def recursive(text: str, size: int = 50, separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not text.strip():
        return []
    if len(text.split()) <= size:
        return [text.strip()]
    if not separators:
        return fixed_words(text, size, 0)
    # Preserve separators so punctuation and paragraph meaning aren't discarded.
    sep = separators[0]
    pieces = text.split(sep)
    units = []
    for i, piece in enumerate(pieces):
        preserved = piece + (sep if i < len(pieces) - 1 else "")
        units.extend(recursive(preserved, size, separators[1:]))
    return pack(units, size)


def markdown_sections(text: str, size: int = 50) -> list[str]:
    """Keep the nearest heading on each subchunk. A teaching Markdown parser."""
    result = []
    for section in re.split(r"(?m)(?=^#{1,6} )", text):
        if not section.strip():
            continue
        lines = section.strip().splitlines()
        heading = lines[0] if lines[0].startswith("#") else ""
        body = "\n".join(lines[1:]) if heading else section
        capacity = size - len(heading.split())
        if capacity <= 0:
            raise ValueError("Chunk size must leave room beyond the heading")
        if not body.strip():
            result.append(heading)
        for part in recursive(body, capacity):
            result.append(f"{heading}\n{part}".strip())
    return result


def semantic(text: str, size: int = 50, threshold: float = 0.25, neural: bool = False) -> list[str]:
    units = sentences(text)
    if not units:
        return []
    if neural:
        from rag.examples.embeddings import DenseEncoder
        vectors = DenseEncoder().documents(units)
    else:
        encoder = Tfidf(units)
        vectors = [encoder.encode(s) for s in units]
    groups, current = [], [units[0]]
    for i in range(1, len(units)):
        if dot(vectors[i - 1], vectors[i]) < threshold:
            groups.extend(pack(current, size))
            current = []
        current.append(units[i])
    return groups + pack(current, size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed", "sentence", "recursive", "markdown", "semantic"], default="markdown")
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--neural", action="store_true")
    args = parser.parse_args()
    doc = load_documents()[0]
    strategies = {
        "fixed": lambda: fixed_words(doc.text, args.size, args.overlap),
        "sentence": lambda: pack(sentences(doc.text), args.size),
        "recursive": lambda: recursive(doc.text, args.size),
        "markdown": lambda: markdown_sections(doc.text, args.size),
        "semantic": lambda: semantic(doc.text, args.size, args.threshold, args.neural),
    }
    if args.strategy == "semantic":
        print("Boundary model:", "neural embeddings" if args.neural else "TF-IDF lexical proxy")
    for chunk in make_chunks(doc, strategies[args.strategy]()):
        print(f"\n[{chunk.id}] {len(chunk.text.split())} words\n{chunk.text}")


if __name__ == "__main__":
    main()
