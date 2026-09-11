"""CLI runner: score every answer in an existing answers JSONL with Ragas.

Reads a ``data/processed/answers/*.jsonl`` file and scores each record on
faithfulness, answer relevancy, context precision, and context recall,
writing results to a JSONL. Resumable: ids already scored are skipped and new
scores are appended.

Usage:
    python -m src.evaluation.ragas_runner --config configs/ragas/ollama8b_k5.yaml
    python -m src.evaluation.ragas_runner --config configs/ragas/ollama8b_k5.yaml --id financebench_id_03029
    python -m src.evaluation.ragas_runner --config configs/ragas/ollama8b_k5.yaml --limit 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.evaluation.common.io import append_jsonl, derived_path, done_ids, read_jsonl, select
from src.evaluation.config import RagasConfig, load_ragas_config
from src.evaluation.ragas_eval import build_metrics, score_record


def _output_path(config: RagasConfig) -> Path:
    """One file per (answers file, Ragas LLM) combination."""
    return derived_path(config.answers_path, "data/processed/ragas", f"ragas_by_{config.llm.model}")


def run(config: RagasConfig, qa_id: str | None = None, limit: int | None = None) -> str:
    """Score every not-yet-scored answer (or just ``qa_id``/``limit``) and append to a JSONL."""
    records = select(read_jsonl(config.answers_path), qa_id, limit)

    out_path = _output_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = done_ids(out_path)
    remaining = [r for r in records if r["id"] not in done]
    if not remaining:
        print(f"All {len(records)} answers already scored -> {out_path}")
        return str(out_path)

    metrics = build_metrics(config.llm, config.embedding_model, config.metrics)

    totals: dict[str, float] = {}
    for r in tqdm(remaining, desc="ragas"):
        scores = score_record(r, metrics)
        append_jsonl(out_path, {"id": r["id"], "question": r["question"], **scores})
        for k, v in scores.items():
            totals[k] = totals.get(k, 0.0) + v

    n = len(remaining)
    summary = " | ".join(f"{k}={v / n:.3f}" for k, v in totals.items())
    print(f"Scored {n} new answers (skipped {len(done)} already scored) -> {out_path}\n  {summary}")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an answers JSONL with Ragas.")
    parser.add_argument("--config", required=True, help="Path to a Ragas YAML config.")
    parser.add_argument("--id", help="Score only this QA id, skipping the rest.")
    parser.add_argument("--limit", type=int, help="Score only the first N answers.")
    args = parser.parse_args()
    run(load_ragas_config(args.config), qa_id=args.id, limit=args.limit)


if __name__ == "__main__":
    main()
