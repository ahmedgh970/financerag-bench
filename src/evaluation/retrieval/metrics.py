"""Standard ranked-retrieval metrics.

All functions take ``relevances``: a ranked list of booleans, one per retrieved
item in retrieval order (``relevances[0]`` is the top hit). ``num_relevant`` is
the total number of relevant targets for the query (the recall/nDCG denominator).
Pure functions — the matching that produces ``relevances`` lives elsewhere.
"""

from __future__ import annotations

import math


def recall_at_k(relevances: list[bool], k: int, num_relevant: int) -> float:
    """Fraction of relevant targets retrieved within the top-k."""
    if num_relevant == 0:
        return 0.0
    return sum(relevances[:k]) / num_relevant


def precision_at_k(relevances: list[bool], k: int) -> float:
    """Fraction of the top-k that is relevant."""
    if k == 0:
        return 0.0
    return sum(relevances[:k]) / k


def reciprocal_rank(relevances: list[bool]) -> float:
    """1 / rank of the first relevant item (0 if none)."""
    for rank, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevances: list[bool], k: int, num_relevant: int) -> float:
    """Normalized discounted cumulative gain over the top-k (binary relevance)."""
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], start=1))
    ideal_hits = min(num_relevant, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
