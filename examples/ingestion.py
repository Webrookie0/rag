"""Normalize and deduplicate text while preserving source metadata.

Run: python3 -m rag.examples.ingestion [--directory ./my_notes]
This importer handles UTF-8 .md/.txt, not PDFs, scans, or arbitrary HTML.
"""

import argparse
import hashlib
import re
import unicodedata
from dataclasses import replace
from pathlib import Path

from rag.core import Document, load_documents


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    # Keep paragraph boundaries, table rows, and code indentation intact.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def deduplicate(documents: list[Document]) -> list[Document]:
    seen = set()
    result = []
    for doc in documents:
        text = clean(doc.text)
        key = (doc.tenant, hashlib.sha256(text.encode()).hexdigest())
        if text and key not in seen:
            seen.add(key)
            result.append(replace(doc, text=text))
    return result


def read_directory(directory: Path, tenant: str = "alpha") -> list[Document]:
    docs = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            relative = path.relative_to(directory).as_posix()
            identity = hashlib.sha256(relative.encode()).hexdigest()[:12]
            docs.append(Document(identity, path.stem, path.read_text(encoding="utf-8"), relative, tenant))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.directory and not args.directory.is_dir():
        parser.error("--directory must point to an existing directory")
    docs = read_directory(args.directory) if args.directory else load_documents()
    augmented = docs + docs[:1]  # Deliberate duplicate to make the effect visible.
    result = deduplicate(augmented)
    print(f"Before: {len(augmented)}; after normalization/deduplication: {len(result)}")
    for doc in result:
        print(f"{doc.id}: {doc.source} ({len(doc.text)} characters)")


if __name__ == "__main__":
    main()
