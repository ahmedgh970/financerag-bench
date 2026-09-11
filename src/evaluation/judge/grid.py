"""Evidence-grounded outcome grid for generated answers.

A generated answer is only trustworthy if it is right *and* justified by the
passages the generator actually saw. Grounding is therefore measured against
the retrieved context, not against the model's reasoning, and with the same
relevance definition as the retrieval metrics: each gold evidence is resolved
to its physical page (:mod:`src.evaluation.common.matching`), and it counts as
retrieved when a chunk of that page reached the prompt.

Combined with a judge's reading of the answer, every record falls in exactly
one outcome, which also attributes the failure to retrieval or generation:

- ``dont_know``: the model declined to answer (checked first).
- ``good_job``: correct, and the evidence was retrieved -- or the figures it
  used were verified in another retrieved passage (``alt_supported``).
- ``need_help``: wrong although the evidence was retrieved (generation failure).
- ``hallucinating``: answered without the evidence in its context.

The judge's verdicts (``correct`` / ``refused`` / ``alt_supported``) come from an
external judge and are read from a verdicts file; whether the evidence reached
the prompt is computed here, never taken from the judge.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from src.evaluation.common.golden_set import load_golden_set
from src.evaluation.common.io import read_by_id
from src.evaluation.common.matching import build_page_index, resolve_evidence_page
from src.evaluation.common.schema import QAItem
from src.evaluation.judge.config import GridConfig


class Outcome(StrEnum):
    GOOD_JOB = "good_job"
    HALLUCINATING = "hallucinating"
    NEED_HELP = "need_help"
    DONT_KNOW = "dont_know"


def evidence_retrieved(
    sources: list[dict], doc_name: str, evidence_pages: list[int | None]
) -> bool:
    """Whether every gold evidence has a chunk of its page among ``sources``.

    ``sources`` are the prompt's passages (``doc_id``, ``page``, ``text``);
    ``evidence_pages`` holds one resolved page per evidence (None if it could not
    be resolved). All evidences, not any: a ratio built from two statements is not
    grounded when only one of them reached the prompt.
    """
    seen = {s["page"] for s in sources if s["doc_id"] == doc_name}
    return bool(evidence_pages) and all(p is not None and p in seen for p in evidence_pages)


def outcome(*, correct: bool, refused: bool, retrieved: bool, alt_supported: bool) -> Outcome:
    """Place one judged answer in the grid.

    ``correct`` and ``refused`` come from the judge's reading of the answer;
    ``alt_supported`` is the judge's check that a correct answer's figures sit in
    a retrieved passage other than the gold one. ``retrieved`` is computed.
    """
    if refused:
        return Outcome.DONT_KNOW
    if correct and (retrieved or alt_supported):
        return Outcome.GOOD_JOB
    if retrieved:
        return Outcome.NEED_HELP
    return Outcome.HALLUCINATING


def verdicts_path(config: GridConfig) -> Path:
    """The configured verdicts file, or ``verdicts/{answers stem}.{judge}.jsonl``."""
    if config.verdicts_path:
        return Path(config.verdicts_path)
    return Path("data/processed/judged/verdicts") / (
        f"{Path(config.answers_path).stem}.{config.judge_model}.jsonl"
    )


def grid_path(config: GridConfig) -> Path:
    """``data/processed/judged/{answers stem}_grid_by_{judge}.jsonl``."""
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


def build_grid(config: GridConfig) -> list[dict]:
    """One grid record per answer; exits if any answer has no verdict yet."""
    answers = read_by_id(config.answers_path)
    verdicts = read_by_id(verdicts_path(config))
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
    return records
