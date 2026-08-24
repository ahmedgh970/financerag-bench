"""Batch LLM-judge over every answers file in data/processed/answers/.

Loops the judge across all generated answer files (one per model x k), reusing
judge_runner.run (resumable per file and per question), then prints an accuracy
summary: correct / grounded / equivalent per (model, k).

Judge defaults to the local Prometheus-2 model. Resumable per file and per
question, so an interrupted run continues where it left off.

    uv run python scripts/eval_all.py
    uv run python scripts/eval_all.py --max-files 1                 # gauge time on one file
    uv run python scripts/eval_all.py --model ollama_chat/mistral-nemo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.evaluation import judge_runner
from src.evaluation.config import JudgeConfig
from src.llm.config import LLMConfig

ANSWERS_DIR = Path("data/processed/answers")
DEFAULT_JUDGE = "ollama_chat/ggozad/prometheus2:latest"
# ollama model link for prompt template https://ollama.com/ggozad/prometheus2


def _label(path: Path) -> str:
    """reranked_..._ollama_chat_{model}_k{k}.jsonl -> {model}_k{k} (the table row)."""
    stem = path.stem
    return stem.split("ollama_chat_", 1)[-1] if "ollama_chat_" in stem else stem


def _accuracy(judged_path: Path) -> dict:
    """Correct / grounded / equivalent rates over a judged file (all verdicts)."""
    recs = [json.loads(line) for line in judged_path.read_text().splitlines() if line.strip()]
    n = len(recs)
    if not n:
        return {"n": 0, "correct": 0.0, "grounded": 0.0, "equivalent": 0.0}
    return {
        "n": n,
        "correct": sum(r["correct"] for r in recs) / n,
        "grounded": sum(r["grounded"] for r in recs) / n,
        "equivalent": sum(r["equivalent"] for r in recs) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch LLM-judge over all answer files.")
    ap.add_argument("--model", default=DEFAULT_JUDGE, help="Judge LLM (ollama_chat/<name>).")
    ap.add_argument("--max-files", type=int, help="Only the first N answer files (gauge run).")
    args = ap.parse_args()

    files = sorted(ANSWERS_DIR.glob("*.jsonl"))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No answer files in {ANSWERS_DIR}")

    print(f"Judging {len(files)} file(s) with {args.model}\n")
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        cfg = JudgeConfig(answers_path=str(f), llm=LLMConfig(model=args.model))
        print(f"[{i}/{len(files)}] {_label(f)}")
        t = time.perf_counter()
        try:
            judge_runner.run(cfg)
        except Exception as e:  # noqa: BLE001 - report and continue the matrix
            print(f"  ERROR on {_label(f)}: {e} -- skipping to next file")
            continue
        acc = _accuracy(judge_runner._output_path(cfg))
        acc["label"] = _label(f)
        acc["time_min"] = (time.perf_counter() - t) / 60
        rows.append(acc)

    total_min = (time.perf_counter() - t0) / 60
    print("\n" + "=" * 74)
    print(f"{'model_k':<26} {'n':>4} {'correct':>8} {'grounded':>9} {'equiv':>7} {'min':>6}")
    print("-" * 74)
    for r in sorted(rows, key=lambda x: x["label"]):
        print(
            f"{r['label']:<26} {r['n']:>4} {r['correct'] * 100:>7.1f}% "
            f"{r['grounded'] * 100:>8.1f}% {r['equivalent'] * 100:>6.1f}% {r['time_min']:>6.1f}"
        )
    print("-" * 74)
    print(f"files judged: {len(rows)}/{len(files)}  |  total {total_min:.1f} min")


if __name__ == "__main__":
    main()
