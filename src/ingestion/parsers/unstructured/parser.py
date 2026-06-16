"""Parse a PDF with Unstructured into a list of Elements."""

from __future__ import annotations


def parse(pdf_path: str):
    """Parse ``pdf_path`` with Unstructured.

    Returns the native ``list[Element]`` (typed blocks: Title, NarrativeText,
    Table, ...). The Unstructured chunkers in ``chunkers.py`` consume this list.

    Choose the partition strategy via params later if needed ("fast" vs
    "hi_res"); "hi_res" is slower but reconstructs tables (text_as_html).
    """
    # TODO: call partition_pdf(pdf_path, strategy=...) and return the elements.
    raise NotImplementedError
