"""Hybrid retrieval: fuse dense and sparse rankings with Reciprocal Rank Fusion.

Dense (semantic) and BM25 (lexical) rank chunks differently; RRF combines them by
rank alone — ``score = sum over lists of 1 / (rrf_k + rank)`` — which is robust
because it ignores the two retrievers' incomparable score scales.
"""

from __future__ import annotations

from collections import defaultdict

from src.retrieval.base import Retriever, ScoredChunk


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]],
    k: int,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists into one top-k list by Reciprocal Rank Fusion.

    ``rrf_k`` (conventionally 60) dampens the weight of top ranks so no single
    list dominates. ``weights`` scales each list's contribution (default equal) —
    useful when one retriever is much stronger. Chunks are keyed by ``chunk_id``.
    """
    weights = weights if weights is not None else [1.0] * len(rankings)
    fused: dict[str, float] = defaultdict(float)
    chunk_by_id: dict[str, ScoredChunk] = {}
    for weight, ranking in zip(weights, rankings, strict=True):
        for rank, sc in enumerate(ranking):
            fused[sc.chunk.chunk_id] += weight / (rrf_k + rank + 1)
            chunk_by_id[sc.chunk.chunk_id] = sc
    top = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]
    return [ScoredChunk(chunk=chunk_by_id[cid].chunk, score=fused[cid]) for cid in top]


class HybridRetriever:
    """Combine a dense and a sparse retriever via weighted RRF.

    Each sub-retriever returns ``prefetch`` candidates (deeper than the final k so
    fusion has material to work with); the fused list is truncated to k. Defaults
    favour the dense retriever (``sparse_weight`` < 1) and keep the candidate pool
    shallow: on FinanceBench a deep, equally weighted sparse list drags recall
    down, since BM25 is much weaker than BGE-M3 there.
    """

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        prefetch: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.3,
    ):
        self.dense = dense
        self.sparse = sparse
        self.prefetch = prefetch
        self.rrf_k = rrf_k
        self.weights = [dense_weight, sparse_weight]

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None) -> list[ScoredChunk]:
        n = max(self.prefetch, k)
        dense_hits = self.dense.retrieve(query, k=n, doc_id=doc_id)
        sparse_hits = self.sparse.retrieve(query, k=n, doc_id=doc_id)
        return reciprocal_rank_fusion(
            [dense_hits, sparse_hits], k=k, rrf_k=self.rrf_k, weights=self.weights
        )
