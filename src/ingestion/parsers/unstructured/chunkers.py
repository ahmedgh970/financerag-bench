"""Unstructured-native chunkers. Consume list[Element], emit list[Chunk].

Unstructured ships exactly two chunking strategies (verified against
``unstructured.chunking``): ``by_title`` (``chunk_by_title``) and ``basic``
(``chunk_elements``). Per-page behaviour is not a separate chunker — it is a
parameter of these strategies (``multipage_sections`` / ``CHUNK_MULTI_PAGE_DEFAULT``).
"""

from __future__ import annotations

from src.ingestion.schema import Chunk


def by_title_chunk(elements, doc_id: str, **params) -> list[Chunk]:
    """Structure-aware chunking via Unstructured's ``chunk_by_title``.

    Starts a new chunk at each section title, keeping related content together.
    """
    # TODO: from unstructured.chunking.title import chunk_by_title; run it over
    #       `elements`, map to Chunk (page from element.metadata.page_number).
    raise NotImplementedError


def basic_chunk(elements, doc_id: str, **params) -> list[Chunk]:
    """Size-based chunking via Unstructured's ``basic`` strategy.

    Combines sequential elements up to a max size, without using section titles.
    """
    # TODO: from unstructured.chunking.basic import chunk_elements; run it over
    #       `elements`, map to Chunk.
    raise NotImplementedError


CHUNKERS = {
    "by_title": by_title_chunk,
    "basic": basic_chunk,
}
