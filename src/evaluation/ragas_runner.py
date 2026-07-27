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
import json
from pathlib import Path

from tqdm import tqdm

from src.evaluation.config import RagasConfig, load_ragas_config
from src.evaluation.ragas_eval import build_metrics, score_record


def _scored_ids(path: Path) -> set[str]:
    """QA ids already scored in ``path`` (empty set if it doesn't exist yet)."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _load_records(path: str, qa_id: str | None, limit: int | None) -> list[dict]:
    """Answer records from ``path`` -- just ``qa_id``, or the first ``limit``."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if qa_id is not None:
        selected = [r for r in records if r["id"] == qa_id]
        if not selected:
            raise SystemExit(f"No answer with id {qa_id!r} in {path!r}.")
        return selected
    return records[:limit] if limit is not None else records


def _output_path(config: RagasConfig) -> Path:
    """One file per (answers file, Ragas LLM) combination."""
    model = config.llm.model.replace("/", "_")
    answers_name = Path(config.answers_path).stem
    return Path("data/processed/ragas") / f"{answers_name}_ragas_by_{model}.jsonl"


def run(config: RagasConfig, qa_id: str | None = None, limit: int | None = None) -> str:
    """Score every not-yet-scored answer (or just ``qa_id``/``limit``) and append to a JSONL."""
    records = _load_records(config.answers_path, qa_id, limit)

    out_path = _output_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _scored_ids(out_path)
    remaining = [r for r in records if r["id"] not in done]
    if not remaining:
        print(f"All {len(records)} answers already scored -> {out_path}")
        return str(out_path)

    metrics = build_metrics(config.llm, config.embedding_model, config.embedding_device)

    totals = {
        k: 0.0 for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    }
    with out_path.open("a", encoding="utf-8") as f:
        for r in tqdm(remaining, desc="ragas"):
            scores = score_record(r, metrics)
            record = {"id": r["id"], "question": r["question"], **scores}
            f.write(json.dumps(record) + "\n")
            f.flush()
            for k, v in scores.items():
                totals[k] += v

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
