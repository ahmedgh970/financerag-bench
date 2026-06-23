"""Fast unit tests for the retrieval metrics on synthetic relevance lists."""

import math

import pytest

from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k():
    rel = [False, True, False, True]  # 2 relevant, at ranks 2 and 4
    assert recall_at_k(rel, k=4, num_relevant=2) == 1.0
    assert recall_at_k(rel, k=2, num_relevant=2) == 0.5  # only rank-2 hit in top-2
    assert recall_at_k(rel, k=4, num_relevant=0) == 0.0


def test_precision_at_k():
    rel = [False, True, False, True]
    assert precision_at_k(rel, k=4) == 0.5
    assert precision_at_k(rel, k=2) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank([False, True, False]) == 0.5  # first hit at rank 2
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_at_k():
    # Perfect ranking: relevant first -> nDCG == 1.
    assert ndcg_at_k([True, False], k=2, num_relevant=1) == 1.0
    # Relevant at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    assert ndcg_at_k([False, True], k=2, num_relevant=1) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([False, False], k=2, num_relevant=1) == 0.0
