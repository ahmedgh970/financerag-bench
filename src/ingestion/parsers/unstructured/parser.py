"""Parse a PDF with Unstructured into a list of Elements."""

from __future__ import annotations

from pathlib import Path

from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_from_json, elements_to_json


def parse(
    pdf_path: str,
    strategy: str = "hi_res",
    infer_table_structure: bool = True,
    languages: list[str] | None = None,
    **kwargs,
) -> list[Element]:
    """Parse ``pdf_path`` with Unstructured into typed elements.

    Args:
        strategy: ``"hi_res"`` (layout model, reconstructs tables — the relevant
            default for financial filings) or ``"fast"`` (native text only, no
            tables, much quicker) or ``"ocr-only"``(extracts the text from the
        document using OCR and processes it).
        infer_table_structure: keep table structure as HTML (only used by
            ``hi_res``).
        languages: OCR/text languages; defaults to ``["eng"]``. Setting it
            explicitly avoids Unstructured's language auto-detection and silences
            its "defaulting to English" warning.
        kwargs: forwarded to ``partition_pdf`` (e.g. ``detect_language_per_element``).

    The Unstructured chunkers consume the returned ``list[Element]`` directly.
    """
    return partition_pdf(
        filename=pdf_path,
        strategy=strategy,
        infer_table_structure=infer_table_structure,
        languages=languages or ["eng"],
        **kwargs,
    )


def save(elements: list[Element], path: str) -> None:
    """Serialize a list of Unstructured elements to JSON on disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(elements_to_json(elements), encoding="utf-8")


def load(path: str) -> list[Element]:
    """Load Unstructured elements previously written by ``save``."""
    return elements_from_json(text=Path(path).read_text(encoding="utf-8"))
