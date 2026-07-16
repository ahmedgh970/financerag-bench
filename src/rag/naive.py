"""Naive RAG pipeline: retrieve once, generate once, no loop.

Wires a retriever (dense/bm25/hybrid/reranked, from src.retrieval) to the LLM
client (src.llm) — the simplest possible pipeline, and the baseline every
later improvement (agentic patterns) is measured against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.ingestion.schema import Chunk
from src.llm.client import generate
from src.llm.config import LLMConfig
from src.llm.prompts import build_prompt
from src.retrieval.base import Retriever


@dataclass
class RagAnswer:
    """A generated answer with its source chunks and end-to-end latency."""

    answer: str
    sources: list[Chunk]
    latency_s: float


def answer(
    question: str,
    retriever: Retriever,
    llm_config: LLMConfig,
    k: int = 10,
    doc_id: str | None = None,
) -> RagAnswer:
    """Answer ``question`` by retrieving once and generating once."""
    start = time.perf_counter()
    results = retriever.retrieve(question, k=k, doc_id=doc_id)
    sources = [sc.chunk for sc in results]
    prompt = build_prompt(question, sources)
    text = generate(prompt, llm_config)
    return RagAnswer(answer=text, sources=sources, latency_s=time.perf_counter() - start)
