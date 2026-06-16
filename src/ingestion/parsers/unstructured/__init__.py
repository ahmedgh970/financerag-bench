"""Unstructured parser + its native chunkers."""

from src.ingestion.parsers.unstructured.chunkers import CHUNKERS
from src.ingestion.parsers.unstructured.parser import parse

__all__ = ["CHUNKERS", "parse"]
