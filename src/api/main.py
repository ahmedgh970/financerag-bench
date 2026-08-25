"""FastAPI serving app: one grounded RAG answer per request.

The heavy models (BGE-M3 embedder, cross-encoder reranker) are loaded once and
cached by name, so a retriever is built per Qdrant collection lazily and reused
-- switching collection is cheap. The LLM, retrieval depth ``k`` and collection
can be overridden per request; defaults come from the served config
(``RAG_CONFIG``, default granite4.1:8b at k10, ADR 0002). Everything is local
(Ollama); no external provider.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI
from pydantic import BaseModel

from src.llm.client import _model_name
from src.rag.config import load_rag_config
from src.rag.naive import answer
from src.retrieval.registry import build_retriever

RAG_CONFIG = os.getenv("RAG_CONFIG", "configs/rag/serve_reranked_dense_1024_k10_granite.yaml")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Ollama models that aren't chat generators (embedders, judges, benchmark variants).
_NON_GENERATORS = ("bge", "prometheus", "ragas-critic", "genbench", "embed", "nomic")

_state: dict = {}
_retrievers: dict[str, object] = {}  # collection name -> built retriever


def _get_retriever(collection: str):
    """Get (or lazily build + cache) the retriever for a Qdrant collection."""
    if collection not in _retrievers:
        cfg = _state["cfg"].model_copy(update={"collection_name": collection})
        _retrievers[collection] = build_retriever(cfg.retriever, cfg)
    return _retrievers[collection]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_rag_config(RAG_CONFIG)
    _state["cfg"] = cfg
    _get_retriever(cfg.collection_name)  # warm the default (loads embedder + reranker once)
    yield
    _state.clear()
    _retrievers.clear()


app = FastAPI(title="financerag-bench", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None  # scope retrieval to one filing; None = search all
    model: str | None = None  # Ollama model name; None = served default
    k: int | None = None  # retrieval depth; None = served default
    collection: str | None = None  # Qdrant collection; None = served default


class Source(BaseModel):
    doc_id: str
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_s: float
    model: str
    k: int
    collection: str


def _ollama_models() -> list[str]:
    """Locally available chat generators (non-generators filtered out)."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return []
    out = []
    for m in resp.json().get("models", []):
        name = m["name"].removesuffix(":latest")
        if "/" in name or any(bad in name.lower() for bad in _NON_GENERATORS):
            continue
        out.append(name)
    return sorted(out)


def _qdrant_collections() -> list[str]:
    """Collections available in Qdrant."""
    try:
        resp = requests.get(f"{_state['cfg'].qdrant_location}/collections", timeout=10)
        resp.raise_for_status()
        return sorted(c["name"] for c in resp.json()["result"]["collections"])
    except (requests.exceptions.RequestException, KeyError):
        return []


@app.get("/health")
def health() -> dict:
    """Liveness + which config is being served."""
    cfg = _state.get("cfg")
    return {"status": "ok" if cfg else "loading", "config": RAG_CONFIG}


@app.get("/options")
def options() -> dict:
    """Choices for the UI: available models, collections, and the defaults."""
    cfg = _state["cfg"]
    return {
        "models": _ollama_models(),
        "collections": _qdrant_collections(),
        "defaults": {
            "model": _model_name(cfg.llm.model),
            "k": cfg.k,
            "collection": cfg.collection_name,
        },
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Retrieve + rerank + generate one grounded answer, with cited sources."""
    cfg = _state["cfg"]
    collection = req.collection or cfg.collection_name
    k = req.k or cfg.k
    llm = cfg.llm if req.model is None else cfg.llm.model_copy(update={"model": req.model})

    result = answer(req.question, _get_retriever(collection), llm, k=k, doc_id=req.doc_id)
    sources = [Source(doc_id=c.doc_id, page=c.page, text=c.text[:500]) for c in result.sources]
    return AskResponse(
        answer=result.answer,
        sources=sources,
        latency_s=result.latency_s,
        model=_model_name(llm.model),
        k=k,
        collection=collection,
    )
