"""CLI runner: judge every answer in an existing answers JSONL against gold.

Reads a ``data/processed/answers/*.jsonl`` file (produced by ``src.rag.runner``)
and scores each record with the LLM judge, writing verdicts to a JSONL.
Resumable: ids already judged are skipped and new verdicts are appended, so
hitting a provider's quota mid-run doesn't lose progress.

Usage:
    python -m src.evaluation.judge_runner --config configs/judge/llm_judge.yaml
    python -m src.evaluation.judge_runner --config configs/judge/llm_judge.yaml \
        --answers data/processed/answers/<run>.jsonl --id financebench_id_03029
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.evaluation.common.io import append_jsonl, derived_path, done_ids, read_jsonl, select
from src.evaluation.config import JudgeConfig, load_judge_config
from src.evaluation.judge import judge


def _output_path(config: JudgeConfig) -> Path:
    """One file per (answers file, judge model) combination."""
    return derived_path(
        config.answers_path, "data/processed/judged", f"judged_by_{config.llm.model}"
    )


def run(config: JudgeConfig, qa_id: str | None = None) -> str:
    """Judge every not-yet-judged answer (or just ``qa_id``) and append verdicts to a JSONL."""
    records = select(read_jsonl(config.answers_path), qa_id)

    out_path = _output_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = done_ids(out_path)
    remaining = [r for r in records if r["id"] not in done]
    if not remaining:
        print(f"All {len(records)} answers already judged -> {out_path}")
        return str(out_path)

    n_equivalent = 0
    n_correct_not_grounded = 0
    for r in tqdm(remaining, desc="judge"):
        verdict = judge(r["question"], r["gold_answer"], r["generated_answer"], config.llm)
        append_jsonl(out_path, {**r, **verdict, "judge_model": config.llm.model})
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
    parser.add_argument("--answers", help="Answers JSONL to judge, overriding the config's.")
    args = parser.parse_args()
    config = load_judge_config(args.config)
    if args.answers:
        config = config.model_copy(update={"answers_path": args.answers})
    run(config, qa_id=args.id)


if __name__ == "__main__":
    main()
