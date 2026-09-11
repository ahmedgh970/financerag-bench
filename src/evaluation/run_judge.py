"""CLI: judge generated answers against gold, then place them in the outcome grid.

Two commands:

- ``score``: an LLM grades every answer of one or more answers files with a judging
  protocol (``correct_grounded`` or ``prometheus``), writing
  ``data/processed/judged/{answers stem}_judged_by_{judge}.jsonl``. Resumable: ids
  already judged are skipped, so an interrupted run continues where it stopped.
- ``grid``: combines an external judge's verdicts with whether the gold evidence
  reached the prompt, writing ``..._grid_by_{judge}.jsonl`` (ADR 0004). No LLM call.

Usage:
    python -m src.evaluation.run_judge score --config configs/evaluation/judge/correct_grounded.yaml
    python -m src.evaluation.run_judge score --config configs/evaluation/judge/correct_grounded.yaml \
        --answers data/processed/answers/naif_rag/*_k20.jsonl --model ollama_chat/qwen3.5:9b
    python -m src.evaluation.run_judge score --config configs/evaluation/judge/prometheus.yaml --id financebench_id_03029
    python -m src.evaluation.run_judge grid --config configs/evaluation/judge/evidence_grid.yaml \
        --answers data/processed/answers/<run>.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.evaluation.common.io import append_jsonl, derived_path, done_ids, read_jsonl, select
from src.evaluation.judge.config import (
    GridConfig,
    JudgeConfig,
    load_grid_config,
    load_judge_config,
)
from src.evaluation.judge.grid import Outcome, build_grid, grid_path
from src.evaluation.judge.protocols import get_protocol

JUDGED_DIR = Path("data/processed/judged")


def judged_path(config: JudgeConfig) -> Path:
    """``data/processed/judged/{answers stem}_judged_by_{judge}.jsonl``."""
    return derived_path(config.answers_path, JUDGED_DIR, f"judged_by_{config.judge_name}")


def with_model(config: JudgeConfig, model: str) -> JudgeConfig:
    """``config`` graded by ``model`` instead, re-validated, under its own output tag."""
    data = config.model_dump()
    data["llm"]["model"] = model
    data["name"] = None
    return JudgeConfig(**data)


def score(config: JudgeConfig, qa_id: str | None = None, limit: int | None = None) -> Path:
    """Judge the not-yet-judged answers of ``config.answers_path``; returns the output file."""
    protocol = get_protocol(config.protocol)
    records = select(read_jsonl(config.answers_path), qa_id, limit)
    out_path = judged_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = done_ids(out_path)
    remaining = [r for r in records if r["id"] not in done]
    for r in tqdm(remaining, desc=f"{config.protocol} [{config.judge_name}]"):
        verdict = protocol.judge(r, config.llm)
        append_jsonl(
            out_path,
            {
                "id": r["id"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "generated_answer": r["generated_answer"],
                **verdict,
                "judge_model": config.llm.model,
            },
        )

    judged = [v for v in read_jsonl(out_path) if v["id"] in {r["id"] for r in records}]
    print(
        f"{Path(config.answers_path).name}: judged {len(remaining)} new, "
        f"skipped {len(records) - len(remaining)} -> {out_path}\n  {protocol.summarize(judged)}"
    )
    return out_path


def grid(config: GridConfig) -> Path:
    """Write the outcome grid of one judged answers file and print its counts."""
    records = build_grid(config)
    out_path = grid_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    counts = Counter(r["outcome"] for r in records)
    n = len(records)
    print(f"{n} answers -> {out_path}")
    for o in Outcome:
        print(f"  {o.value:<14} {counts[o.value]:>4}  {100 * counts[o.value] / n:5.1f}%")
    missed = sum(r["outcome"] == Outcome.DONT_KNOW and r["evidence_retrieved"] for r in records)
    print(f"  dont_know with the evidence in context: {missed}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge answers against gold.")
    commands = parser.add_subparsers(dest="command", required=True)

    p_score = commands.add_parser("score", help="LLM-judge one or more answers files.")
    p_score.add_argument("--config", required=True, help="Judge YAML config.")
    p_score.add_argument("--answers", nargs="+", help="Answers JSONL(s), overriding the config's.")
    p_score.add_argument("--model", help="Judge model, overriding the config's.")
    p_score.add_argument("--id", help="Judge only this QA id.")
    p_score.add_argument("--limit", type=int, help="Judge only the first N answers per file.")

    p_grid = commands.add_parser("grid", help="Outcome grid from existing verdicts.")
    p_grid.add_argument("--config", required=True, help="Grid YAML config.")
    p_grid.add_argument("--answers", nargs="+", help="Answers JSONL(s), overriding the config's.")

    args = parser.parse_args()
    if args.command == "score":
        config = load_judge_config(args.config)
        if args.model:
            config = with_model(config, args.model)
        for answers in args.answers or [config.answers_path]:
            score(config.model_copy(update={"answers_path": answers}), args.id, args.limit)
    else:
        config = load_grid_config(args.config)
        for answers in args.answers or [config.answers_path]:
            grid(config.model_copy(update={"answers_path": answers}))


if __name__ == "__main__":
    main()
