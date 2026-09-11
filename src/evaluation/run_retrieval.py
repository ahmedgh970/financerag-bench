"""CLI: evaluate a retriever against the FinanceBench golden set.

Scores the retriever described by a config (see :mod:`src.evaluation.retrieval.evaluate`)
and writes the report to ``docs/benchmarks/retrieval_{config name}_{timestamp}.json``,
so every report says which retriever setup produced it.

Usage:
    python -m src.evaluation.run_retrieval --config configs/evaluation/retrieval/chunks512_dense.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.evaluation.retrieval.config import load_retrieval_eval_config
from src.evaluation.retrieval.evaluate import evaluate

REPORTS_DIR = Path("docs/benchmarks")


def report_path(config_path: str, stamp: str) -> Path:
    """``docs/benchmarks/retrieval_{config stem}_{stamp}.json``."""
    return REPORTS_DIR / f"retrieval_{Path(config_path).stem}_{stamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a retriever on FinanceBench.")
    parser.add_argument("--config", required=True, help="Path to a retrieval-eval YAML config.")
    args = parser.parse_args()

    report = {"config": args.config, **evaluate(load_retrieval_eval_config(args.config))}

    out = report_path(args.config, datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nEvaluated {report['n_evaluated']} QA (skipped {report['n_skipped_doc_absent']})")
    for name, value in report["metrics"].items():
        print(f"  {name:14}: {value}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
