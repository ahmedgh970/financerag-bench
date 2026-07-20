"""CLI runner: judge every answer in an existing answers JSONL against gold.

Reads a ``data/processed/answers/*.jsonl`` file (produced by ``src.rag.runner``)
and scores each record with the LLM judge, writing verdicts to a JSONL.
Resumable: ids already judged are skipped and new verdicts are appended, so
hitting a provider's quota mid-run doesn't lose progress.

Usage:
    python -m src.evaluation.judge_runner --config configs/judge/llama70b.yaml
    python -m src.evaluation.judge_runner --config configs/judge/llama70b.yaml --id financebench_id_03029
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.evaluation.config import JudgeConfig, load_judge_config
from src.evaluation.judge import judge


def _judged_ids(path: Path) -> set[str]:
    """QA ids already judged in ``path`` (empty set if it doesn't exist yet)."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _load_records(path: str, qa_id: str | None) -> list[dict]:
    """Answer records from ``path``, or just the one matching ``qa_id``."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if qa_id is None:
        return records
    selected = [r for r in records if r["id"] == qa_id]
    if not selected:
        raise SystemExit(f"No answer with id {qa_id!r} in {path!r}.")
    return selected


def _output_path(config: JudgeConfig) -> Path:
    """One file per (answers file, judge model) combination."""
    model = config.llm.model.replace("/", "_")
    answers_name = Path(config.answers_path).stem
    return Path("data/processed/judged") / f"{answers_name}_judged_by_{model}.jsonl"


def run(config: JudgeConfig, qa_id: str | None = None) -> str:
    """Judge every not-yet-judged answer (or just ``qa_id``) and append verdicts to a JSONL."""
    records = _load_records(config.answers_path, qa_id)

    out_path = _output_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _judged_ids(out_path)
    remaining = [r for r in records if r["id"] not in done]
    if not remaining:
        print(f"All {len(records)} answers already judged -> {out_path}")
        return str(out_path)

    n_equivalent = 0
    n_correct_not_grounded = 0
    with out_path.open("a", encoding="utf-8") as f:
        for r in tqdm(remaining, desc="judge"):
            verdict = judge(r["question"], r["gold_answer"], r["generated_answer"], config.llm)
            record = {**r, **verdict, "judge_model": config.llm.model}
            f.write(json.dumps(record) + "\n")
            f.flush()

            n_equivalent += verdict["equivalent"]
            n_correct_not_grounded += verdict["correct"] and not verdict["grounded"]

    print(
        f"Judged {len(remaining)} new answers (skipped {len(done)} already judged) -> {out_path}\n"
        f"  equivalent: {n_equivalent}/{len(remaining)}"
        f" | correct but not grounded: {n_correct_not_grounded}/{len(remaining)}"
    )
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge an answers JSONL against gold.")
    parser.add_argument("--config", required=True, help="Path to a judge YAML config.")
    parser.add_argument("--id", help="Judge only this QA id, skipping the rest.")
    args = parser.parse_args()
    run(load_judge_config(args.config), qa_id=args.id)


if __name__ == "__main__":
    main()
