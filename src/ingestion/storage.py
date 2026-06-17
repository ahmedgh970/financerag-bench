"""Persist and load chunks as JSONL.

JSONL (one ``Chunk`` per line) is the hand-off artifact between ingestion and the
downstream vectorization/indexing stage: it is streamable, greppable, diffable
and versionable, and round-trips trivially through the pydantic ``Chunk`` model.
Embeddings are intentionally absent here — they are computed later and stored in
the vector DB, so the same chunks file can serve several embedding models.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from src.ingestion.schema import Chunk


def write_chunks(chunks: Iterable[Chunk], path: str) -> int:
    """Write ``chunks`` to ``path`` as JSONL. Returns the number written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
            n += 1
    return n


def read_chunks(path: str) -> Iterator[Chunk]:
    """Yield ``Chunk`` objects from a JSONL file."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Chunk.model_validate_json(line)
