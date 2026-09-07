"""State carried through the CRAG graph.

One flat TypedDict rather than nested objects: LangGraph merges the partial dict a
node returns into the state, so a flat shape keeps every node's contract obvious.
Fields are grouped by the phase that fills them; a disabled phase simply leaves its
fields unset, which is what lets one graph serve the whole ablation matrix.
"""

from __future__ import annotations

from typing import TypedDict

from src.ingestion.schema import Chunk


class CragState(TypedDict, total=False):
    """Everything the graph reads or writes for one question."""

    # Inputs
    question: str
    doc_id: str | None

    # Retrieval
    query: str  # the current query -- rewritten by the correction loop
    chunks: list[Chunk]
    scores: list[float]
    initial_chunks: list[Chunk]  # first round, kept as the give-up fallback
    tried_queries: list[str]

    # Grading / correction loop
    graded: list[Chunk]  # chunks kept by the grader; unset when grading is off
    kept_per_round: list[int]  # chunks kept at each grading round, to see if a rewrite paid off
    rewrites: int
    bound_hit: bool  # the rewrite budget ran out; we fall back instead of refusing
    grader_errors: int

    # Numeric path
    is_numeric: bool
    formula_name: str | None
    expression: str | None
    computed: str | None

    # Output
    answer: str
    sources: list[Chunk]

    # Instrumentation
    node_latencies: dict[str, float]
    llm_calls: int
