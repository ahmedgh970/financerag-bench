"""FastAPI serving app: one grounded RAG answer per request.

Builds the retriever (reranker + BGE-M3) once at startup and reuses it across
requests -- loading it per request would add seconds of latency each time. The
pipeline is entirely local (Ollama); no external provider. The served config is
chosen by the ``RAG_CONFIG`` env var (default: granite4.1:8b at k10, the best
judged generator from ADR 0002).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.config import load_rag_config
from src.rag.naive import answer
from src.retrieval.registry import build_retriever

RAG_CONFIG = os.getenv("RAG_CONFIG", "configs/rag/serve_reranked_dense_1024_k10_granite.yaml")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the config and build the retriever ONCE (the reranker cross-encoder
    # and BGE-M3 embedder are heavy); every request reuses this instance.
    cfg = load_rag_config(RAG_CONFIG)
    _state["cfg"] = cfg
    _state["retriever"] = build_retriever(cfg.retriever, cfg)
    yield
    _state.clear()


app = FastAPI(title="financerag-bench", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None  # scope retrieval to one filing; None = search all


class Source(BaseModel):
    doc_id: str
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_s: float


@app.get("/health")
def health() -> dict:
    """Liveness + which config is being served."""
    cfg = _state.get("cfg")
    return {
        "status": "ok" if cfg else "loading",
        "retriever": cfg.retriever if cfg else None,
        "model": cfg.llm.model if cfg else None,
        "k": cfg.k if cfg else None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Retrieve + rerank + generate one grounded answer, with cited sources."""
    cfg = _state["cfg"]
    result = answer(req.question, _state["retriever"], cfg.llm, k=cfg.k, doc_id=req.doc_id)
    sources = [Source(doc_id=c.doc_id, page=c.page, text=c.text[:500]) for c in result.sources]
    return AskResponse(answer=result.answer, sources=sources, latency_s=result.latency_s)
