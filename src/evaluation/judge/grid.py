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
- ``good_job``: correct, and the gold evidence was in the prompt.
- ``unverified``: correct, but unverified in the context -- the gold evidence
  never reached the prompt, so the answer may rest on another passage (an MD&A
  table repeating the statement, say) or on luck; the grid does not tell which.
- ``need_help``: wrong although the evidence was in the prompt (generation failure).
- ``hallucinating``: wrong, and the evidence never reached the prompt.

The judge only reads the answer (``correct`` / ``refused``, written to a verdicts
file by the ``grid`` judging protocol, or by any external judge in the same format);
whether the evidence reached the prompt is computed here, never taken from the judge.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from src.evaluation.common.golden_set import load_golden_set
from src.evaluation.common.io import derived_path, read_by_id
from src.evaluation.common.matching import build_page_index, resolve_evidence_page
from src.evaluation.common.schema import QAItem
from src.evaluation.judge.config import JudgeConfig

JUDGED_DIR = Path("data/processed/judged")


class Outcome(StrEnum):
    GOOD_JOB = "good_job"
    UNVERIFIED = "unverified"  # correct but unverified in the context
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


def outcome(*, correct: bool, refused: bool, retrieved: bool) -> Outcome:
    """Place one judged answer in the grid.

    ``correct`` and ``refused`` come from the judge's reading of the answer;
    ``retrieved`` (the gold evidence reached the prompt) is computed.
    """
    if refused:
        return Outcome.DONT_KNOW
    if correct:
        return Outcome.GOOD_JOB if retrieved else Outcome.UNVERIFIED
    return Outcome.NEED_HELP if retrieved else Outcome.HALLUCINATING


def verdicts_path(config: JudgeConfig) -> Path:
    """``data/processed/judged/verdicts/{answers stem}.{judge}.jsonl``."""
    judge = config.judge_name.replace("/", "_")
    return JUDGED_DIR / "verdicts" / f"{Path(config.answers_path).stem}.{judge}.jsonl"


def grid_path(config: JudgeConfig) -> Path:
    """``data/processed/judged/{answers stem}_grid_by_{judge}.jsonl``."""
    return derived_path(config.answers_path, JUDGED_DIR, f"grid_by_{config.judge_name}")


def with_gold_justification(records: list[dict], golden_set_path: str) -> list[dict]:
    """``records`` with FinanceBench's gold justification, which answers files do not carry."""
    qas = {qa.id: qa for qa in load_golden_set(golden_set_path)}
    return [{**r, "gold_justification": qas[r["id"]].justification} for r in records]


def grade(answer: dict, verdict: dict, qa: QAItem, evidence_pages: list[int | None]) -> dict:
    """One grid record: the verdict, where the evidence sits, and the resulting outcome."""
    retrieved = evidence_retrieved(answer["sources"], qa.doc_name, evidence_pages)
    return {
        "id": answer["id"],
        "outcome": outcome(
            correct=verdict["correct"], refused=verdict["refused"], retrieved=retrieved
        ).value,
        "correct": verdict["correct"],
        "refused": verdict["refused"],
        "evidence_retrieved": retrieved,
        "evidence_pages": evidence_pages,
        "n_sources": len(answer["sources"]),
        "justification": verdict.get("justification", ""),
    }


def build_grid(config: JudgeConfig) -> list[dict]:
    """One grid record per answer; exits if any answer has no verdict yet."""
    answers = read_by_id(config.answers_path)
    path = verdicts_path(config)
    verdicts = read_by_id(path) if path.exists() else {}
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
            {**grade(answers[i], verdicts[i], qa, pages), "judge_model": config.judge_name}
        )
    return records
