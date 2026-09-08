"""The graph's nodes. Each returns the slice of state it owns; the graph routes.

Every node here either fetches, judges or generates -- none of them chooses what runs
next. That split is what makes this a workflow rather than an agent.
"""

from __future__ import annotations

import time

from src.ingestion.schema import Chunk
from src.llm.client import StructuredOutputError, generate, generate_structured
from src.llm.prompts import build_prompt
from src.retrieval.base import Retriever
from src.workflow.config import WorkflowConfig
from src.workflow.prompts import build_grading_prompt
from src.workflow.schemas import ChunkGrade
from src.workflow.state import CragState


def _timed(state: CragState, node: str, started: float) -> dict[str, float]:
    """Accumulated per-node latency (the default reducer replaces, so merge here)."""
    return {**state.get("node_latencies", {}), node: time.perf_counter() - started}


def make_retrieve(retriever: Retriever, config: WorkflowConfig):
    """Fetch the top-k passages. Their order is the retriever's own ranking."""

    def retrieve(state: CragState) -> dict:
        started = time.perf_counter()
        results = retriever.retrieve(state["question"], k=config.k, doc_id=state.get("doc_id"))
        return {
            "chunks": [sc.chunk for sc in results],
            "scores": [sc.score for sc in results],
            "node_latencies": _timed(state, "retrieve", started),
        }

    return retrieve


def _apply_floor(kept: list[Chunk], ranked: list[Chunk], minimum: int) -> list[Chunk]:
    """Top a thin selection up to ``minimum`` with the best passages it does not hold.

    Walks the retriever's ranking and skips anything already kept, so a passage the
    grader selected is never added twice -- a duplicate would appear under two source
    numbers in the prompt and read as two independent pieces of evidence.
    """
    if len(kept) >= minimum:
        return kept
    seen = {c.chunk_id for c in kept}
    topped = list(kept)
    for chunk in ranked:
        if chunk.chunk_id in seen:
            continue
        topped.append(chunk)
        seen.add(chunk.chunk_id)
        if len(topped) >= minimum:
            break
    return topped


def make_grade(config: WorkflowConfig):
    """Grade every passage 0-3, keep those above the threshold, top up to the floor.

    One call per passage. Grading twenty passages in a single prompt was implemented
    first and measured unusable: asked for twenty verdicts at once the model returned
    one, and rejected everything. One passage per call makes the miscount structurally
    impossible, at the cost of k calls.
    """
    grading = config.grading

    def grade(state: CragState) -> dict:
        started = time.perf_counter()
        chunks = state.get("chunks", [])
        if not chunks:
            return {"graded": [], "grades": [], "node_latencies": _timed(state, "grade", started)}

        grades: list[int] = []
        for chunk in chunks:
            try:
                grades.append(
                    generate_structured(
                        build_grading_prompt(state["question"], chunk), config.llm, ChunkGrade
                    ).grade
                )
            except StructuredOutputError:
                # Fail open: a decoding failure must not silently drop a passage.
                grades.append(3)

        by_grade = [c for c, g in zip(chunks, grades, strict=True) if g >= grading.keep_threshold]
        context = _apply_floor(by_grade, chunks, grading.min_chunks)
        # Present the context in the retriever's ranking, whatever each passage's
        # provenance, so the strongest evidence leads regardless of who selected it.
        order = {c.chunk_id: i for i, c in enumerate(chunks)}
        context.sort(key=lambda c: order[c.chunk_id])

        return {
            "graded": context,
            "grades": grades,
            "n_kept_by_grade": len(by_grade),
            "n_kept_by_floor": len(context) - len(by_grade),
            "max_grade": max(grades),
            "low_confidence": max(grades) < grading.low_confidence_below,
            "llm_calls": state.get("llm_calls", 0) + len(chunks),
            "node_latencies": _timed(state, "grade", started),
        }

    return grade


def make_generate(config: WorkflowConfig):
    """Answer from the passages the graph selected."""

    def generate_node(state: CragState) -> dict:
        started = time.perf_counter()
        graded = state.get("graded")
        sources = graded if graded is not None else state["chunks"]
        text = generate(build_prompt(state["question"], sources), config.llm)
        return {
            "answer": text,
            "sources": sources,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "node_latencies": _timed(state, "generate", started),
        }

    return generate_node
