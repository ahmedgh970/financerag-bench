"""The CRAG graph: retrieve -> [grade -> correct] -> generate.

Only the nodes enabled by the config are wired in, so one graph serves every row of
the ablation matrix. With grading off it reduces to retrieve -> generate, which is
the advanced-RAG baseline running on this exact code path -- that is what makes a
difference between rows attributable to the node under test and not to the plumbing.

The correction loop is bounded by ``max_rewrites`` from the config; when the budget
runs out the graph falls back to the first round's passages rather than refusing
(see :func:`src.workflow.nodes.make_fallback` for why).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph

from src.ingestion.schema import Chunk
from src.retrieval.base import Retriever
from src.workflow.config import WorkflowConfig
from src.workflow.nodes import (
    make_fallback,
    make_generate,
    make_grade,
    make_retrieve,
    make_rewrite,
    make_route_after_grading,
)
from src.workflow.state import CragState


@dataclass
class WorkflowAnswer:
    """A workflow answer with its sources and the instrumentation the eval needs."""

    answer: str
    sources: list[Chunk]
    latency_s: float
    n_rewrites: int = 0
    bound_hit: bool = False
    n_retrieved: int = 0
    n_kept: int = 0
    grader_errors: int = 0
    kept_per_round: list[int] = field(default_factory=list)
    llm_calls: int = 0
    node_latencies: dict[str, float] = field(default_factory=dict)


def build_graph(retriever: Retriever, config: WorkflowConfig):
    """Compile the graph for ``config`` -- only the enabled nodes are wired in."""
    builder = StateGraph(CragState)
    builder.add_node("retrieve", make_retrieve(retriever, config))
    builder.add_node("generate", make_generate(config))
    builder.add_edge(START, "retrieve")

    if config.grading.enabled:
        builder.add_node("grade", make_grade(config))
        builder.add_node("rewrite_query", make_rewrite(config))
        builder.add_node("fallback", make_fallback(config))
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            make_route_after_grading(config),
            {"generate": "generate", "rewrite_query": "rewrite_query", "fallback": "fallback"},
        )
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("fallback", "generate")
    else:
        builder.add_edge("retrieve", "generate")

    builder.add_edge("generate", END)
    return builder.compile()


def answer_workflow(
    question: str,
    retriever: Retriever,
    config: WorkflowConfig,
    doc_id: str | None = None,
) -> WorkflowAnswer:
    """Answer ``question`` by running the configured CRAG graph once."""
    graph = build_graph(retriever, config)
    start = time.perf_counter()
    state: CragState = graph.invoke(
        {"question": question, "doc_id": doc_id, "query": question, "rewrites": 0}
    )
    return WorkflowAnswer(
        answer=state.get("answer", ""),
        sources=state.get("sources", []),
        latency_s=time.perf_counter() - start,
        n_rewrites=state.get("rewrites", 0),
        bound_hit=state.get("bound_hit", False),
        n_retrieved=len(state.get("chunks", [])),
        n_kept=len(state.get("sources", [])),
        grader_errors=state.get("grader_errors", 0),
        kept_per_round=state.get("kept_per_round", []),
        llm_calls=state.get("llm_calls", 0),
        node_latencies=state.get("node_latencies", {}),
    )
