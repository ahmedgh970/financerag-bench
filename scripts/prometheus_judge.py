"""Prometheus-2 judge over selected generation answer files.

Scores each answer with the Prometheus-2 Absolute Grading judge (1-5, see
``src.evaluation.prometheus_judge``) and writes verdicts alongside the Claude
verdicts, one file per answers file:
``data/processed/judged/{stem}_judged_by_prometheus.jsonl``
(schema ``id/question/gold_answer/generated_answer/score/feedback/judge_model``).

Resumable per file: ids already scored are skipped and new verdicts appended.
Default selection is the four models judged at k10/k20 for the judge-ranking
comparison; override with ``--models`` / ``--ks`` / ``--files``.

    uv run python scripts/prometheus_judge.py
    uv run python scripts/prometheus_judge.py --models qwen3.5:4b --ks 10 --limit 5
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.evaluation.prometheus_judge import PROMETHEUS_MODEL, judge_prometheus

ANSWERS_DIR = Path("data/processed/answers")
JUDGED_DIR = Path("data/processed/judged")
COLLECTION = "docling_hybrid_1024_bge-m3"
RETRIEVER = "reranked"

# Subset chosen for the Claude-vs-Prometheus ranking comparison.
DEFAULT_MODELS = ["qwen3.5:4b", "qwen3.5:9b", "granite4.1:3b", "granite4.1:8b"]
DEFAULT_KS = [10, 20]


def _answers_path(model: str, k: int) -> Path:
    model_id = f"ollama_chat/{model}".replace("/", "_")
    return ANSWERS_DIR / f"{RETRIEVER}_{COLLECTION}_{model_id}_k{k}.jsonl"


def _output_path(answers_path: Path) -> Path:
    return JUDGED_DIR / f"{answers_path.stem}_judged_by_prometheus.jsonl"


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _judge_file(answers_path: Path, limit: int | None) -> None:
    if not answers_path.exists():
        print(f"  missing {answers_path.name} -- skipped")
        return
    records = [json.loads(line) for line in answers_path.open(encoding="utf-8") if line.strip()]
    if limit is not None:
        records = records[:limit]

    out_path = _output_path(answers_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    remaining = [r for r in records if r["id"] not in done]
    if not remaining:
        print(f"  {answers_path.name}: all {len(records)} already judged -> {out_path.name}")
        return

    scores: list[int] = []
    unparsed = 0
    with out_path.open("a", encoding="utf-8") as f:
        for r in tqdm(remaining, desc=answers_path.stem.split("ollama_chat_", 1)[-1]):
            v = judge_prometheus(r["question"], r["gold_answer"], r["generated_answer"])
            if v["score"] is None:
                unparsed += 1
                tqdm.write(f"  UNPARSED [RESULT] for {r['id']}: {v['raw'][-80:]!r}")
            else:
                scores.append(v["score"])
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "question": r["question"],
                        "gold_answer": r["gold_answer"],
                        "generated_answer": r["generated_answer"],
                        "score": v["score"],
                        "feedback": v["feedback"],
                        "judge_model": PROMETHEUS_MODEL,
                    }
                )
                + "\n"
            )
            f.flush()

    dist = dict(sorted(Counter(scores).items()))
    mean = st.mean(scores) if scores else float("nan")
    flag = f" -- {unparsed} UNPARSED" if unparsed else ""
    print(
        f"  {answers_path.name}: judged {len(remaining)} new (skipped {len(done)}) | "
        f"mean {mean:.2f} | dist {dist}{flag} -> {out_path.name}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prometheus-2 judge over selected answer files.")
    ap.add_argument("--models", help=f"Comma-separated (default: {','.join(DEFAULT_MODELS)}).")
    ap.add_argument("--ks", help=f"Comma-separated k (default: {','.join(map(str, DEFAULT_KS))}).")
    ap.add_argument(
        "--files", help="Comma-separated explicit answer file paths (overrides model/k)."
    )
    ap.add_argument("--limit", type=int, help="Only the first N questions per file (smoke test).")
    args = ap.parse_args()

    if args.files:
        paths = [Path(p) for p in args.files.split(",")]
    else:
        models = args.models.split(",") if args.models else DEFAULT_MODELS
        ks = [int(k) for k in args.ks.split(",")] if args.ks else DEFAULT_KS
        paths = [_answers_path(m, k) for m in models for k in ks]

    print(f"Prometheus judge: {PROMETHEUS_MODEL}  |  {len(paths)} file(s)\n")
    for p in paths:
        _judge_file(p, args.limit)


if __name__ == "__main__":
    main()
