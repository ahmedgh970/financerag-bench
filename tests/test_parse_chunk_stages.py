"""Integration tests for the two-stage parse -> save -> load -> chunk flow.

Marked ``slow``: they run a real parse. They verify the native parsed document
survives a disk round-trip and still chunks into a valid Chunk contract, across
every (parser, chunker) combination.
"""

from pathlib import Path

import pytest

# Skip this whole module when the optional ingestion stack is absent (e.g. the
# fast CI environment that installs only the core + dev deps).
pytest.importorskip("docling")
pytest.importorskip("unstructured")

from src.ingestion.parsers import docling, unstructured  # noqa: E402

PDF = Path(__file__).parents[1] / "data/pdfs/JOHNSON_JOHNSON_2023_8K_dated-2023-08-23.pdf"
DOC_ID = PDF.stem


@pytest.fixture(scope="module")
def pdf_path() -> str:
    if not PDF.exists():
        pytest.skip(f"test PDF not available: {PDF}")
    return str(PDF)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("mod", "parse_kwargs", "chunker"),
    [
        (docling, {"do_ocr": False, "do_table_structure": True}, "hybrid"),
        (docling, {"do_ocr": False, "do_table_structure": True}, "hierarchical"),
        (unstructured, {"strategy": "fast"}, "by_title"),
        (unstructured, {"strategy": "fast"}, "basic"),
    ],
)
def test_parse_save_load_chunk_roundtrip(pdf_path, tmp_path, mod, parse_kwargs, chunker):
    out = tmp_path / "doc.json"

    native = mod.parse(pdf_path, **parse_kwargs)
    mod.save(native, str(out))
    reloaded = mod.load(str(out))

    chunks = mod.CHUNKERS[chunker](reloaded, doc_id=DOC_ID)

    assert chunks, "no chunks produced after reload"
    assert all(c.text.strip() for c in chunks), "empty chunk text"
    assert all(c.page >= 1 for c in chunks), "invalid page number"
    assert len({c.chunk_id for c in chunks}) == len(chunks), "duplicate chunk_id"
    assert all(c.doc_id == DOC_ID for c in chunks), "wrong doc_id"
    assert all(c.metadata["chunker"] == chunker for c in chunks), "wrong chunker tag"
    assert all(isinstance(c.metadata["labels"], list) for c in chunks), "labels not a list"


@pytest.mark.slow
def test_docling_page_range_limits_parsed_pages(pdf_path):
    # page_range=(1, 1) must restrict parsing to the first page only.
    doc = docling.parse(pdf_path, page_range=(1, 1))
    assert len(doc.pages) == 1
