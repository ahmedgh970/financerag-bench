"""Registry of available parsers.

Each parser package exposes ``parse``/``save``/``load`` and a ``CHUNKERS`` dict
of the chunkers compatible with its native output. The runners look parsers up
here by name.
"""

from __future__ import annotations

from src.ingestion.parsers import docling, unstructured

PARSERS = {
    "docling": docling,
    "unstructured": unstructured,
}
