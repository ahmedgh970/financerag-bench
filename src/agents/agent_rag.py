"""Agentic RAG: a LangGraph tool-using agent (retrieve + calculator) over a filing.

Same generator as the naive baseline (default granite4.1:8b) and the same
``reranked(dense)`` retriever -- the only difference vs ``src/rag/naive.py`` is
the agent LOOP: the LLM decides *when* and with *what* (reformulated) query to
retrieve (possibly several times) and *when* to compute, before answering. Built
for the naive-vs-agentic comparison (Semaine 9).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from src.agents.prompts import get_prompt
from src.agents.tools import build_retrieve_tool, calculator
from src.ingestion.schema import Chunk
from src.llm.client import _model_name
from src.llm.config import LLMConfig
from src.retrieval.base import Retriever


@dataclass
class AgentAnswer:
    """An agentic answer with its sources, latency and agent-loop counters."""

    answer: str
    sources: list[Chunk]
    latency_s: float
    n_llm_calls: int
    n_retrieve: int
    n_calculator: int


def answer_agentic(
    question: str,
    retriever: Retriever,
    llm_config: LLMConfig,
    doc_id: str | None = None,
    depths: tuple[int, ...] = (5, 10, 20),
    num_ctx: int = 32768,
    recursion_limit: int = 12,
    prompt_version: str = "v1",
) -> AgentAnswer:
    """Answer ``question`` with a depth-escalating tool-using LangGraph agent.

    ``retriever`` should be dense-only (no reranker): the agent itself filters the
    passages, and the escalating ``depths`` (5 → 10 → 20) let it start cheap and
    go deeper only when needed. ``num_ctx`` is large because the passages of every
    depth accumulate in the agent's context. ``prompt_version`` selects the system
    prompt from :mod:`src.agents.prompts`.
    """
    sink: list[Chunk] = []
    tools = [build_retrieve_tool(retriever, doc_id=doc_id, depths=depths, sink=sink), calculator]
    llm = ChatOllama(
        model=_model_name(llm_config.model),
        temperature=llm_config.temperature,
        num_ctx=num_ctx,
        num_predict=llm_config.max_tokens,
    )
    agent = create_react_agent(llm, tools, prompt=get_prompt(prompt_version))

    start = time.perf_counter()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": recursion_limit},
        )
        msgs = result["messages"]
        answer_text = msgs[-1].content if msgs else ""
    except GraphRecursionError:
        msgs = []
        answer_text = "[agent stopped: reached the step limit without a final answer]"
    latency = time.perf_counter() - start

    n_llm_calls = sum(1 for m in msgs if m.type == "ai")
    tool_names = [tc["name"] for m in msgs if getattr(m, "tool_calls", None) for tc in m.tool_calls]
    n_retrieve = tool_names.count("retrieve")
    n_calculator = tool_names.count("calculator")

    # De-dupe the retrieved chunks by id, preserving first-seen order.
    seen: set[str] = set()
    sources: list[Chunk] = []
    for c in sink:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            sources.append(c)

    return AgentAnswer(answer_text, sources, latency, n_llm_calls, n_retrieve, n_calculator)
