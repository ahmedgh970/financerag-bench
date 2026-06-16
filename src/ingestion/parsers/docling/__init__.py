"""Docling parser + its native chunkers."""

from src.ingestion.parsers.docling.chunkers import CHUNKERS
from src.ingestion.parsers.docling.parser import parse

__all__ = ["CHUNKERS", "parse"]
