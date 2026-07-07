"""Common contract shared by every retriever.

The evaluation harness depends only on this interface (``Retriever``) and output
type (``ScoredChunk``). New strategies — bm25, hybrid, reranker — implement the
same ``retrieve`` signature, so they plug in without changing the eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.ingestion.schema import Chunk


@dataclass
class ScoredChunk:
    """A retrieved chunk and its relevance score (higher = more relevant)."""

    chunk: Chunk
    score: float


class Retriever(Protocol):
    """Anything that returns the top-k chunks for a query.

    ``doc_id`` optionally restricts the search to a single document (FinanceBench
    questions target a known filing); ``None`` searches the whole collection.
    """

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None) -> list[ScoredChunk]: ...
