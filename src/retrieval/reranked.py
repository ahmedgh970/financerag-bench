"""Retrieve-then-rerank: wrap any retriever with a cross-encoder reranking pass.

The base retriever (dense, bm25 or hybrid) narrows the whole collection down to
``prefetch`` candidates cheaply; the reranker then re-scores just that shortlist
with a cross-encoder and the top-k of the reranked order is returned.
"""

from __future__ import annotations

from src.retrieval.base import Retriever, ScoredChunk
from src.retrieval.reranker import Reranker


class RerankedRetriever:
    """Rerank a base retriever's candidates with a cross-encoder."""

    def __init__(self, base: Retriever, reranker: Reranker, prefetch: int = 50):
        self.base = base
        self.reranker = reranker
        self.prefetch = prefetch

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None) -> list[ScoredChunk]:
        candidates = self.base.retrieve(query, k=max(self.prefetch, k), doc_id=doc_id)
        if not candidates:
            return candidates
        scores = self.reranker.score(query, [sc.chunk.text for sc in candidates])
        reranked = sorted(
            (
                ScoredChunk(chunk=sc.chunk, score=score)
                for sc, score in zip(candidates, scores, strict=True)
            ),
            key=lambda sc: sc.score,
            reverse=True,
        )
        return reranked[:k]
