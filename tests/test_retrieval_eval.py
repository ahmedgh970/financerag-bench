"""Unit tests for the retrieval evaluation (no Qdrant, no embedder)."""

from pathlib import Path

from src.evaluation.retrieval.config import load_retrieval_eval_config
from src.evaluation.retrieval.evaluate import aggregate, dedup_relevances, score_qa
from src.evaluation.run_retrieval import report_path
from src.ingestion.schema import Chunk
from src.retrieval.base import ScoredChunk


def _hit(page, doc="DOC", i=0):
    return ScoredChunk(Chunk(chunk_id=f"{doc}::{i}", doc_id=doc, text="...", page=page), 1.0)


def test_only_the_first_chunk_reaching_a_gold_page_counts():
    # Two chunks of gold page 46: a second hit on the same page is not a new relevant item.
    results = [_hit(3, i=0), _hit(46, i=1), _hit(46, i=2), _hit(58, i=3)]
    assert dedup_relevances(results, "DOC", {46, 58}) == [False, True, False, True]


def test_hits_from_another_document_are_never_relevant():
    assert dedup_relevances([_hit(46, doc="OTHER")], "DOC", {46}) == [False]


def test_score_qa_reports_every_metric_at_every_k():
    scores = score_qa([False, True], num_gold=1, k_values=[1, 2])
    assert scores["mrr"] == 0.5
    assert scores["recall@1"] == 0.0 and scores["recall@2"] == 1.0
    assert set(scores) == {
        "mrr",
        "recall@1",
        "precision@1",
        "ndcg@1",
        "recall@2",
        "precision@2",
        "ndcg@2",
    }


def test_aggregate_averages_over_the_evaluated_qas():
    assert aggregate([{"mrr": 1.0}, {"mrr": 0.0}]) == {"mrr": 0.5}
    assert aggregate([]) == {}


def test_report_is_named_after_the_config():
    path = report_path("configs/evaluation/retrieval/chunks512_dense.yaml", "20260911-120000")
    assert path == Path("docs/benchmarks/retrieval_chunks512_dense_20260911-120000.json")


def test_every_shipped_retrieval_config_loads():
    configs = sorted(Path("configs/evaluation/retrieval").glob("*.yaml"))
    assert len(configs) == 6
    for path in configs:
        cfg = load_retrieval_eval_config(str(path))
        assert cfg.doc_scoped
        assert (cfg.retriever == "reranked") == ("reranked" in path.stem)
