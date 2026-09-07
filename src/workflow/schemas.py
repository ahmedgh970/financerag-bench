"""Pydantic schemas for the workflow's structured LLM outputs.

Each is passed to Ollama as a JSON Schema, so decoding is constrained to emit a
valid object. That has a consequence worth keeping in mind: the model can no longer
abstain by refusing to answer, so a schema must make abstention *expressible* -- an
all-false list of booleans, say -- rather than leaving the model no way to say
"none of these".
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BatchGrades(BaseModel):
    """Relevance verdict for a batch of passages, one boolean per passage, in order.

    Deliberately flat -- a list of booleans rather than a list of objects -- so the
    JSON Schema carries no nested definitions, which small models follow far more
    reliably. "No passage is relevant" is simply an all-false list, which is exactly
    the verdict the correction loop keys on.
    """

    relevant: list[bool] = Field(description="One true/false per passage, in order")


class ChunkGrade(BaseModel):
    """Relevance verdict for a single passage.

    The per-chunk counterpart of :class:`BatchGrades`. Grading one passage per call
    removes the counting problem entirely -- one call can only produce one verdict --
    at the cost of k calls instead of one.
    """

    relevant: bool = Field(description="True if this passage helps answer the question")
