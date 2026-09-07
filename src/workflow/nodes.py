"""The graph's nodes. Each returns the slice of state it owns; the graph routes.

Every node here either fetches, judges or generates -- none of them chooses what
runs next. Routing lives in the conditional-edge functions at the bottom, which
read a judgement out of the state and return the name of the next node. That split
is what makes this a workflow rather than an agent.
"""

from __future__ import annotations

import time

from src.llm.client import (
    StructuredOutputError,
    estimated_context,
    generate,
    generate_structured,
)
from src.llm.prompts import build_prompt
from src.retrieval.base import Retriever
from src.workflow.config import WorkflowConfig
from src.workflow.prompts import (
    build_grading_prompt,
    build_rewrite_prompt,
    build_single_grading_prompt,
)
from src.workflow.schemas import BatchGrades, ChunkGrade
from src.workflow.state import CragState


def _timed(state: CragState, node: str, started: float) -> dict[str, float]:
    """Accumulated per-node latency (the default reducer replaces, so merge here)."""
    return {**state.get("node_latencies", {}), node: time.perf_counter() - started}


def make_retrieve(retriever: Retriever, config: WorkflowConfig):
    """Fetch the top-k passages for the current query."""

    def retrieve(state: CragState) -> dict:
        started = time.perf_counter()
        query = state.get("query") or state["question"]
        results = retriever.retrieve(query, k=config.k, doc_id=state.get("doc_id"))
        chunks = [sc.chunk for sc in results]
        out = {
            "chunks": chunks,
            "scores": [sc.score for sc in results],
            "tried_queries": [*state.get("tried_queries", []), query],
            "node_latencies": _timed(state, "retrieve", started),
        }
        # The first round is kept untouched: it is what the give-up path falls back
        # to, so that path degenerates exactly to the baseline and can never do worse.
        if not state.get("initial_chunks"):
            out["initial_chunks"] = chunks
        return out

    return retrieve


def make_grade(config: WorkflowConfig):
    """Judge each passage's relevance -- absolutely, unlike the reranker's ordering."""

    def grade(state: CragState) -> dict:
        started = time.perf_counter()
        chunks = state.get("chunks", [])
        rounds = state.get("kept_per_round", [])
        if not chunks:
            return {
                "graded": [],
                "kept_per_round": [*rounds, 0],
                "node_latencies": _timed(state, "grade", started),
            }

        errors = state.get("grader_errors", 0)
        if config.grading.mode == "per_chunk":
            flags, calls = [], len(chunks)
            for chunk in chunks:
                try:
                    flags.append(
                        generate_structured(
                            build_single_grading_prompt(state["question"], chunk),
                            config.llm,
                            ChunkGrade,
                        ).relevant
                    )
                except StructuredOutputError:
                    flags.append(True)  # fail open, per chunk
                    errors += 1
        else:
            calls = 1
            try:
                flags = generate_structured(
                    build_grading_prompt(state["question"], chunks), config.llm, BatchGrades
                ).relevant
            except StructuredOutputError:
                # Fail open: a grader failure must not manufacture a retrieval failure
                # and send the graph into rewrites it does not need.
                flags, errors = [True] * len(chunks), errors + 1
            if len(flags) != len(chunks):
                flags = (flags + [False] * len(chunks))[: len(chunks)]

        kept = [c for c, ok in zip(chunks, flags, strict=True) if ok]
        return {
            "graded": kept,
            "kept_per_round": [*rounds, len(kept)],
            "grader_errors": errors,
            "llm_calls": state.get("llm_calls", 0) + calls,
            "node_latencies": _timed(state, "grade", started),
        }

    return grade


def make_rewrite(config: WorkflowConfig):
    """Reformulate the query in the filing's own vocabulary and retry retrieval."""

    def rewrite_query(state: CragState) -> dict:
        started = time.perf_counter()
        new_query = (
            generate(
                build_rewrite_prompt(state["question"], state.get("tried_queries", [])),
                config.llm,
            )
            .strip()
            .strip('"')
        )
        return {
            "query": new_query or state["question"],
            "rewrites": state.get("rewrites", 0) + 1,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "node_latencies": _timed(state, "rewrite_query", started),
        }

    return rewrite_query


def make_fallback(config: WorkflowConfig):
    """Rewrite budget exhausted: fall back to the first round's passages.

    Deliberately NOT a canned refusal. Measured on this benchmark, questions whose
    retrieval is judged to have failed are still answered correctly 33% of the time
    and stay grounded 87% of the time, so refusing outright would trade those away.
    Falling back to the original query's passages makes this path degenerate to the
    baseline: the correction loop can then only ever help, never hurt.

    Those passages are the unfiltered top-k, so with a pinned ``num_ctx`` the prompt
    can overflow -- and Ollama truncates from the TOP, which would drop the grounding
    instructions first and the best-ranked passages next, keeping only the weakest.
    They are therefore trimmed from the END here instead: deliberate, deterministic,
    and it keeps the instructions and the strongest passages. How many fit is derived
    from the configured context, never a fixed count.
    """

    def fallback(state: CragState) -> dict:
        chunks = state.get("initial_chunks", [])
        num_ctx = config.llm.num_ctx
        if num_ctx is not None:
            fitting: list = []
            for chunk in chunks:
                candidate = [*fitting, chunk]
                needed = estimated_context(
                    build_prompt(state["question"], candidate), config.llm.max_tokens
                )
                if needed > num_ctx:
                    break
                fitting = candidate
            chunks = fitting
        return {"graded": chunks, "bound_hit": True}

    return fallback


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


def make_route_after_grading(config: WorkflowConfig):
    """Conditional edge: the graph -- not the model -- decides what happens next."""

    def route(state: CragState) -> str:
        if state.get("graded"):
            return "generate"
        if state.get("rewrites", 0) < config.max_rewrites:
            return "rewrite_query"
        return "fallback"

    return route
