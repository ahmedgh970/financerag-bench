"""Normalized schema for the FinanceBench evaluation golden set."""

from __future__ import annotations

from pydantic import BaseModel


class Evidence(BaseModel):
    """One piece of gold evidence: the text and where it lives in the corpus."""

    text: str
    doc_name: str
    page: int


class QAItem(BaseModel):
    """A normalized FinanceBench question with its gold answer and evidence.

    ``doc_name`` matches a chunk's ``doc_id``; each ``Evidence.page`` matches a
    chunk's ``page`` — that pair is what the retrieval metrics score against.
    """

    id: str
    question: str
    answer: str
    company: str
    doc_name: str
    question_type: str
    evidence: list[Evidence]
