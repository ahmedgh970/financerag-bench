"""Registry of retrieval strategies, selectable by name.

Each builder constructs a retriever from a config object. New strategies (bm25,
hybrid, reranker) register a builder here; the eval harness builds retrievers via
``build_retriever`` and never has to change. Builders import their retriever
lazily so unused strategies don't pull heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Callable

from src.retrieval.base import Retriever


def _build_dense(cfg) -> Retriever:
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.qdrant_store import get_client

    return DenseRetriever(
        client=get_client(cfg.qdrant_location),
        collection_name=cfg.collection_name,
        model_name=cfg.embedding_model,
    )


_BUILDERS: dict[str, Callable[[object], Retriever]] = {
    "dense": _build_dense,
}


def build_retriever(name: str, cfg) -> Retriever:
    """Build the retriever registered under ``name`` from ``cfg``."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown retriever '{name}'. Available: {list(_BUILDERS)}")
    return _BUILDERS[name](cfg)
