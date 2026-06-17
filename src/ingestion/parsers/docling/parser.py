"""Parse a PDF with Docling into a DoclingDocument."""

from __future__ import annotations

from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

# Cache converters by the options that define them: a DocumentConverter loads
# heavy ML models, so reuse one per (do_ocr, do_table_structure, table_mode)
# combination instead of rebuilding it for every document.
_converters: dict[tuple, DocumentConverter] = {}


def _get_converter(do_ocr: bool, do_table_structure: bool, table_mode: str) -> DocumentConverter:
    key = (do_ocr, do_table_structure, table_mode)
    if key not in _converters:
        opts = PdfPipelineOptions()
        opts.do_ocr = do_ocr
        opts.do_table_structure = do_table_structure
        opts.table_structure_options.mode = TableFormerMode(table_mode)
        _converters[key] = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _converters[key]


def parse(
    pdf_path: str,
    do_ocr: bool = True,
    do_table_structure: bool = True,
    table_mode: str = "accurate",
    page_range: tuple[int, int] | None = None,
) -> DoclingDocument:
    """Parse ``pdf_path`` and return its DoclingDocument.

    Args:
        do_ocr: run OCR. Disable for native-text PDFs to save time.
        do_table_structure: reconstruct table structure (kept on for financial
            filings, where tables carry the answers).
        table_mode: table recognition quality, ``"accurate"`` or ``"fast"``.
        page_range: ``(first, last)`` 1-based page range to parse; ``None`` parses
            the whole document. Useful to probe a large filing quickly.

    The returned document carries the full structure (sections, tables,
    per-element provenance) consumed downstream by the Docling chunkers.

    Raises:
        RuntimeError: if Docling fails to convert the document.
    """
    converter = _get_converter(do_ocr, do_table_structure, table_mode)
    convert_kwargs = {"page_range": page_range} if page_range is not None else {}
    result = converter.convert(pdf_path, **convert_kwargs)
    if result.status == ConversionStatus.FAILURE:
        raise RuntimeError(f"Docling failed to convert {pdf_path}: {result.errors}")
    return result.document


def save(doc: DoclingDocument, path: str) -> None:
    """Serialize a DoclingDocument to JSON on disk (lossless)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc.model_dump_json(), encoding="utf-8")


def load(path: str) -> DoclingDocument:
    """Load a DoclingDocument previously written by ``save``."""
    return DoclingDocument.model_validate_json(Path(path).read_text(encoding="utf-8"))
