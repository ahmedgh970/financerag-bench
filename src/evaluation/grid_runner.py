"""CLI runner: place every judged answer in the evidence-grounded outcome grid.

Joins three inputs per QA id -- the answers JSONL (whose ``sources`` are the
passages the generator saw), the judge's verdicts, and the golden set's evidence
-- resolves each evidence to its physical page in the corpus, checks whether a
chunk of that page reached the prompt, and writes one outcome per answer (see
:mod:`src.evaluation.grounding`).

Usage:
    python -m src.evaluation.grid_runner --config configs/judge/evidence_grid.yaml
    python -m src.evaluation.grid_runner --config configs/judge/evidence_grid.yaml \
        --answers data/processed/answers/<run>.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.evaluation.common.golden_set import load_golden_set
from src.evaluation.common.io import read_by_id
from src.evaluation.common.matching import build_page_index, resolve_evidence_page
from src.evaluation.common.schema import QAItem
from src.evaluation.config import GridConfig, load_grid_config
from src.evaluation.grounding import Outcome, evidence_retrieved, outcome


def _verdicts_path(config: GridConfig) -> Path:
    """The configured verdicts file, or ``verdicts/{answers stem}.{judge}.jsonl``."""
    if config.verdicts_path:
        return Path(config.verdicts_path)
    return Path("data/processed/judged/verdicts") / (
        f"{Path(config.answers_path).stem}.{config.judge_model}.jsonl"
    )


def _output_path(config: GridConfig) -> Path:
    return Path("data/processed/judged") / (
        f"{Path(config.answers_path).stem}_grid_by_{config.judge_model}.jsonl"
    )


def grade(answer: dict, verdict: dict, qa: QAItem, evidence_pages: list[int | None]) -> dict:
    """One grid record: the verdict, where the evidence sits, and the resulting outcome."""
    retrieved = evidence_retrieved(answer["sources"], qa.doc_name, evidence_pages)
    alt_supported = verdict.get("alt_supported", False)
    return {
        "id": answer["id"],
        "outcome": outcome(
            correct=verdict["correct"],
            refused=verdict["refused"],
            retrieved=retrieved,
            alt_supported=alt_supported,
        ).value,
        "correct": verdict["correct"],
        "refused": verdict["refused"],
        "alt_supported": alt_supported,
        "evidence_retrieved": retrieved,
        "evidence_pages": evidence_pages,
        "n_sources": len(answer["sources"]),
        "justification": verdict.get("justification", ""),
    }


def run(config: GridConfig) -> str:
    """Grade every judged answer and write the grid records to a JSONL."""
    answers = read_by_id(config.answers_path)
    verdicts = read_by_id(_verdicts_path(config))
    qas = {qa.id: qa for qa in load_golden_set(config.golden_set_path)}

    missing = sorted(answers.keys() - verdicts.keys())
    if missing:
        raise SystemExit(f"{len(missing)} answers have no verdict yet, e.g. {missing[:3]}")

    page_index = build_page_index(config.chunks_path, {qas[i].doc_name for i in answers})
    records = []
    for i in answers:
        qa = qas[i]
        pages = [
            resolve_evidence_page(ev.text, page_index.get(qa.doc_name, {})) for ev in qa.evidence
        ]
        records.append(
            {**grade(answers[i], verdicts[i], qa, pages), "judge_model": config.judge_model}
        )

    out_path = _output_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    counts = Counter(r["outcome"] for r in records)
    n = len(records)
    print(f"{n} answers -> {out_path}")
    for o in Outcome:
        print(f"  {o.value:<14} {counts[o.value]:>4}  {100 * counts[o.value] / n:5.1f}%")
    missed = sum(r["outcome"] == Outcome.DONT_KNOW and r["evidence_retrieved"] for r in records)
    print(f"  dont_know with the evidence in context: {missed}")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-grounded outcome grid.")
    parser.add_argument("--config", required=True, help="Path to a grid YAML config.")
    parser.add_argument("--answers", help="Answers JSONL to grade, overriding the config's.")
    args = parser.parse_args()
    config = load_grid_config(args.config)
    if args.answers:
        config = config.model_copy(update={"answers_path": args.answers})
    run(config)


if __name__ == "__main__":
    main()
