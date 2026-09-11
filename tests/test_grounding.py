"""Fast unit tests for the evidence-grounded outcome grid (no LLM, tiny corpus)."""

import json

import pytest

from src.evaluation.common.matching import resolve_evidence_page
from src.evaluation.common.schema import Evidence, QAItem
from src.evaluation.config import GridConfig
from src.evaluation.grid_runner import _verdicts_path, grade, run
from src.evaluation.grounding import Outcome, evidence_retrieved, outcome

EVIDENCE = "Purchases of property, plant and equipment (1,577) (1,373)"


def _src(page, text="...", doc="DOC"):
    return {"doc_id": doc, "page": page, "text": text}


def _qa(*texts):
    return QAItem(
        id="q1",
        question="capex?",
        answer="1577",
        company="c",
        doc_name="DOC",
        question_type="t",
        evidence=[Evidence(text=t, doc_name="DOC", page=0) for t in texts],
    )


def test_retrieved_when_a_chunk_of_the_gold_page_is_in_the_prompt():
    assert evidence_retrieved([_src(3), _src(46)], "DOC", [46]) is True


def test_not_retrieved_when_the_gold_page_is_missing():
    # Words spread over other pages must not count: only the gold page does.
    assert evidence_retrieved([_src(3), _src(12)], "DOC", [46]) is False


def test_every_evidence_page_must_be_in_the_prompt():
    assert evidence_retrieved([_src(46), _src(58)], "DOC", [46, 58]) is True
    assert evidence_retrieved([_src(46)], "DOC", [46, 58]) is False  # one statement missing


def test_not_retrieved_for_another_document_or_an_unresolved_page():
    assert evidence_retrieved([_src(46, doc="OTHER")], "DOC", [46]) is False
    assert evidence_retrieved([_src(46)], "DOC", [None]) is False
    assert evidence_retrieved([_src(46)], "DOC", []) is False


def test_resolve_evidence_page_picks_best_overlap():
    pages = {
        3: {"risk", "factors"},
        46: {"purchases", "property", "plant", "equipment", "1", "577"},
    }
    assert resolve_evidence_page(EVIDENCE, pages) == 46
    assert resolve_evidence_page(EVIDENCE, {}) is None


@pytest.mark.parametrize(
    ("correct", "refused", "retrieved", "alt", "expected"),
    [
        (True, False, True, False, Outcome.GOOD_JOB),
        (True, False, False, True, Outcome.GOOD_JOB),  # figures verified in another passage
        (True, False, False, False, Outcome.HALLUCINATING),  # right without support
        (False, False, False, False, Outcome.HALLUCINATING),
        (False, False, True, False, Outcome.NEED_HELP),  # evidence there, still wrong
        (False, True, True, False, Outcome.DONT_KNOW),  # refusal wins over everything
        (False, True, False, False, Outcome.DONT_KNOW),
    ],
)
def test_outcome(correct, refused, retrieved, alt, expected):
    got = outcome(correct=correct, refused=refused, retrieved=retrieved, alt_supported=alt)
    assert got is expected


def test_grade_combines_the_verdict_with_the_computed_evidence():
    answer = {"id": "q1", "sources": [_src(46, EVIDENCE), _src(3, "unrelated risk factors")]}
    verdict = {"correct": False, "refused": False, "justification": "wrong line"}
    record = grade(answer, verdict, _qa(EVIDENCE), [46])
    assert record["evidence_retrieved"] is True
    assert record["outcome"] == "need_help"
    assert record["evidence_pages"] == [46]
    assert record["alt_supported"] is False  # absent from the verdict -> not supported


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    golden = tmp_path / "gold.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "financebench_id": "q1",
                "question": "capex?",
                "answer": "1577",
                "company": "3M",
                "doc_name": "DOC",
                "question_type": "metrics",
                "evidence": [
                    {"evidence_text": EVIDENCE, "doc_name": "DOC", "evidence_page_num": 0}
                ],
            }
        ],
    )
    corpus = tmp_path / "chunks.jsonl"
    _write_jsonl(
        corpus,
        [
            {"chunk_id": "DOC::0", "doc_id": "DOC", "page": 3, "text": "unrelated risk factors"},
            {"chunk_id": "DOC::1", "doc_id": "DOC", "page": 46, "text": EVIDENCE},
        ],
    )
    answers = tmp_path / "answers.jsonl"
    _write_jsonl(answers, [{"id": "q1", "sources": [_src(3, "unrelated risk factors")]}])
    verdicts = tmp_path / "verdicts.jsonl"
    _write_jsonl(
        verdicts, [{"id": "q1", "correct": True, "refused": False, "alt_supported": False}]
    )

    config = GridConfig(
        answers_path=str(answers),
        verdicts_path=str(verdicts),
        chunks_path=str(corpus),
        judge_model="claude",
        golden_set_path=str(golden),
    )
    out = json.loads((tmp_path / run(config)).read_text().splitlines()[0])
    assert out["evidence_pages"] == [46]
    assert out["outcome"] == "hallucinating"  # right answer, gold page never reached the prompt


def test_verdicts_path_defaults_to_the_answers_stem():
    config = GridConfig(
        answers_path="data/processed/answers/run_a.jsonl",
        chunks_path="unused.jsonl",
        judge_model="claude",
    )
    assert str(_verdicts_path(config)) == "data/processed/judged/verdicts/run_a.claude.jsonl"
    explicit = config.model_copy(update={"verdicts_path": "elsewhere.jsonl"})
    assert str(_verdicts_path(explicit)) == "elsewhere.jsonl"


def test_run_refuses_to_grade_with_missing_verdicts(tmp_path):
    answers = tmp_path / "answers.jsonl"
    _write_jsonl(answers, [{"id": "q1", "sources": []}])
    verdicts = tmp_path / "verdicts.jsonl"
    _write_jsonl(verdicts, [])
    config = GridConfig(
        answers_path=str(answers),
        verdicts_path=str(verdicts),
        chunks_path="unused.jsonl",
        judge_model="claude",
    )
    with pytest.raises(SystemExit):
        run(config)
