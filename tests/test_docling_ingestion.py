"""Integration tests for the Docling parser + chunkers.

Marked ``slow``: they run a real Docling conversion (loads ML models), so they
are excluded from the fast suite with ``-m "not slow"``.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ingestion.pipeline import ingest

PDF = Path(__file__).parents[1] / "data/pdfs/JOHNSON_JOHNSON_2023_8K_dated-2023-08-23.pdf"


@pytest.fixture(scope="module")
def pdf_path() -> str:
    if not PDF.exists():
        pytest.skip(f"test PDF not available: {PDF}")
    return str(PDF)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("chunker", "params"),
    [("hybrid", {}), ("hierarchical", {})],
)
def test_ingest_emits_valid_chunks(pdf_path, chunker, params):
    config = SimpleNamespace(parser="docling", chunker=chunker, chunker_params=params)
    chunks = ingest(pdf_path, config)

    assert chunks, "no chunks produced"
    assert all(c.text.strip() for c in chunks), "empty chunk text"
    assert all(c.page >= 1 for c in chunks), "invalid page number"
    assert len({c.chunk_id for c in chunks}) == len(chunks), "duplicate chunk_id"
    assert all(c.doc_id == PDF.stem for c in chunks), "wrong doc_id"
    assert all(c.metadata["chunker"] == chunker for c in chunks), "wrong chunker tag"
