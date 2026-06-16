"""Unstructured-native chunkers. Consume list[Element], emit list[Chunk].

Unstructured ships exactly two chunking strategies (verified against
``unstructured.chunking``): ``by_title`` (``chunk_by_title``) and ``basic``
(``chunk_elements``). Per-page behaviour is a parameter of these strategies
(``multipage_sections``), not a separate chunker.
"""

from __future__ import annotations

from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title

from src.ingestion.schema import Chunk


def _to_chunks(elements, doc_id: str, name: str) -> list[Chunk]:
    """Map chunked Unstructured elements to the shared Chunk contract."""
    chunks: list[Chunk] = []
    for i, el in enumerate(elements):
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{i}",
                doc_id=doc_id,
                text=el.text,
                page=el.metadata.page_number or 1,
                metadata={"parser": "unstructured", "chunker": name, "category": el.category},
            )
        )
    return chunks


def by_title_chunk(elements, doc_id: str, **params) -> list[Chunk]:
    """Structure-aware chunking via Unstructured's ``chunk_by_title``.

    Starts a new chunk at each section title, keeping related content together.
    ``params`` are forwarded (e.g. ``max_characters``, ``multipage_sections``).
    """
    return _to_chunks(chunk_by_title(elements, **params), doc_id, "by_title")


def basic_chunk(elements, doc_id: str, **params) -> list[Chunk]:
    """Size-based chunking via Unstructured's ``basic`` strategy.

    Combines sequential elements up to a max size, ignoring section titles.
    ``params`` are forwarded (e.g. ``max_characters``, ``overlap``).
    """
    return _to_chunks(chunk_elements(elements, **params), doc_id, "basic")


CHUNKERS = {
    "by_title": by_title_chunk,
    "basic": basic_chunk,
}
