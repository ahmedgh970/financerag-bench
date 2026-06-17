"""Parse a PDF with Unstructured into a list of Elements."""

from __future__ import annotations

from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf


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
            tables, much quicker).
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
