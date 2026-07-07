"""CLI runner: evaluate a retriever against the FinanceBench golden set.

For each QA it resolves the gold evidence to physical pages, retrieves the top-k
chunks, marks which are relevant (on a gold page, deduplicated per page) and
computes recall@k / precision@k / MRR / nDCG@k. Results are averaged over the QAs
whose document is present in the indexed corpus and saved to ``benchmarks/``.

Usage:
    python -m src.evaluation.runner --config configs/eval_hybrid512_dense.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from src.evaluation.config import EvalConfig, load_eval_config
from src.evaluation.golden_set import load_golden_set
from src.evaluation.matching import is_relevant, resolve_gold_pages, words
from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.ingestion.storage import read_chunks
from src.retrieval.base import ScoredChunk
from src.retrieval.registry import build_retriever


def _build_page_index(chunks_path: str, doc_names: set[str]) -> dict[str, dict[int, set[str]]]:
    """Build {doc: {page: word set}} from the corpus, for the needed docs only."""
    index: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for chunk in read_chunks(chunks_path):
        if chunk.doc_id in doc_names:
            index[chunk.doc_id][chunk.page] |= words(chunk.text)
    return index


def _dedup_relevances(
    results: list[ScoredChunk], doc_name: str, gold_pages: set[int]
) -> list[bool]:
    """Ranked relevance, one True per gold page (first chunk that reaches it)."""
    seen: set[int] = set()
    flags: list[bool] = []
    for sc in results:
        rel = is_relevant(sc.chunk, doc_name, gold_pages) and sc.chunk.page not in seen
        flags.append(rel)
        if rel:
            seen.add(sc.chunk.page)
    return flags


def _score_qa(relevances: list[bool], num_gold: int, k_values: list[int]) -> dict[str, float]:
    scores: dict[str, float] = {"mrr": reciprocal_rank(relevances)}
    for k in k_values:
        scores[f"recall@{k}"] = recall_at_k(relevances, k, num_gold)
        scores[f"precision@{k}"] = precision_at_k(relevances, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(relevances, k, num_gold)
    return scores


def run(config: EvalConfig) -> dict:
    """Run the retrieval evaluation and write a JSON report. Returns the report."""
    qas = load_golden_set(config.golden_set_path)
    page_index = _build_page_index(config.chunks_path, {q.doc_name for q in qas})
    retriever = build_retriever(config.retriever, config)
    top_k = max(config.k_values)

    per_qa: list[dict[str, float]] = []
    skipped = 0
    for qa in tqdm(qas, desc=f"eval [{config.collection_name}]"):
        gold_pages = resolve_gold_pages(qa, page_index.get(qa.doc_name, {}))
        if not gold_pages:  # document not in the indexed corpus
            skipped += 1
            continue
        results = retriever.retrieve(
            qa.question, k=top_k, doc_id=qa.doc_name if config.doc_scoped else None
        )
        relevances = _dedup_relevances(results, qa.doc_name, gold_pages)
        per_qa.append(_score_qa(relevances, len(gold_pages), config.k_values))

    metrics = _aggregate(per_qa)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "collection": config.collection_name,
        "embedding_model": config.embedding_model,
        "retriever": config.retriever,
        "doc_scoped": config.doc_scoped,
        "n_evaluated": len(per_qa),
        "n_skipped_doc_absent": skipped,
        "metrics": metrics,
    }
    _save_report(report, config.collection_name)
    return report


def _aggregate(per_qa: list[dict[str, float]]) -> dict[str, float]:
    if not per_qa:
        return {}
    keys = per_qa[0].keys()
    return {k: round(sum(q[k] for q in per_qa) / len(per_qa), 4) for k in keys}


def _save_report(report: dict, collection: str) -> None:
    out_dir = Path("benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"eval_{collection}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nEvaluated {report['n_evaluated']} QA (skipped {report['n_skipped_doc_absent']})")
    for name, value in report["metrics"].items():
        print(f"  {name:14}: {value}")
    print(f"-> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a retriever on FinanceBench.")
    parser.add_argument("--config", required=True, help="Path to an eval YAML config.")
    run(load_eval_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
