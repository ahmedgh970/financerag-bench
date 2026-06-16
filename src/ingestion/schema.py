"""Output contract for the ingestion subpackage.

Every (parser, chunker) combination must emit a ``list[Chunk]`` regardless of
the parser used internally. This is the single shared invariant of the package:
downstream stages (retrieval, evaluation) only ever see ``Chunk`` objects and
stay agnostic to how the document was parsed or chunked.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of text with its provenance.

    Attributes:
        chunk_id: Stable unique id (e.g. ``f"{doc_id}::{index}"``).
        doc_id: Source document id (usually the PDF stem).
        text: The chunk text, ready to embed / index.
        page: 1-based source page number. Crucial for evidence↔chunk matching
            during retrieval evaluation (Week 2). Use the first page the chunk
            spans when it crosses a page boundary.
        metadata: Free-form extras (parser name, chunker name, section title,
            element type, bounding box, token count...). Kept open on purpose so
            each parser/chunker can attach what it has without changing the schema.
    """

    chunk_id: str
    doc_id: str
    text: str
    page: int
    metadata: dict = Field(default_factory=dict)
