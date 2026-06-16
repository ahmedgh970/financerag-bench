"""Parse a PDF with Docling into a DoclingDocument."""

from __future__ import annotations

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter

# Module-level cache: DocumentConverter loads heavy ML models, so it is built
# once on first use and reused across calls.
_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def parse(pdf_path: str) -> DoclingDocument:
    """Parse ``pdf_path`` and return its DoclingDocument.

    The returned document carries the full structure (sections, tables,
    per-element provenance) consumed downstream by the Docling chunkers.

    Raises:
        RuntimeError: if Docling fails to convert the document.
    """
    result = _get_converter().convert(pdf_path)

    if result.status == ConversionStatus.FAILURE:
        raise RuntimeError(f"Docling failed to convert {pdf_path}: {result.errors}")
    return result.document
