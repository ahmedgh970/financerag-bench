"""Evidence-grounded outcome grid for generated answers.

A generated answer is only trustworthy if it is right *and* justified by the
passages the generator actually saw. Grounding is therefore measured against
the retrieved context, not against the model's reasoning, and with the same
relevance definition as the retrieval metrics: each gold evidence is resolved
to its physical page (:mod:`src.evaluation.matching`), and it counts as
retrieved when a chunk of that page reached the prompt.

Combined with a judge's reading of the answer, every record falls in exactly
one outcome, which also attributes the failure to retrieval or generation:

- ``dont_know``: the model declined to answer (checked first).
- ``good_job``: correct, and the evidence was retrieved -- or the figures it
  used were verified in another retrieved passage (``alt_supported``).
- ``need_help``: wrong although the evidence was retrieved (generation failure).
- ``hallucinating``: answered without the evidence in its context.
"""

from __future__ import annotations

from enum import StrEnum


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
