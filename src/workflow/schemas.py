"""Pydantic schemas for the workflow's structured LLM outputs.

Passed to Ollama as JSON Schema, so decoding is constrained to a valid object. The
consequence worth keeping in mind: the model can no longer abstain by refusing, so
abstention must be *expressible in the schema* -- here, the grade 0.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChunkGrade(BaseModel):
    """Graded relevance of a single passage, on the four-point UMBRELA scale.

    A graded scale rather than a boolean, for two reasons. It matches how relevance
    actually distributes -- a passage carrying one of the two line items a ratio needs
    is neither irrelevant nor an answer -- and it turns the keep/drop decision into a
    *threshold applied afterwards*, so one run yields the whole precision/recall curve
    instead of one irreversible verdict.

    ``Literal`` rather than a bounded int: it compiles to a JSON Schema enum, which
    constrained decoding actually enforces, where a numeric range may not be.
    """

    grade: Literal[0, 1, 2, 3] = Field(description="0 irrelevant, 1 related, 2 partial, 3 exact")
