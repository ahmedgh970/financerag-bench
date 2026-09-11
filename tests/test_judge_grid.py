"""Fast unit tests for the evidence-grounded outcome grid and its judging protocol
(fake LLM, tiny corpus)."""

import json

import pytest

from src.evaluation.common.matching import resolve_evidence_page
from src.evaluation.common.schema import Evidence, QAItem
from src.evaluation.judge import grid as grid_module
from src.evaluation.judge.config import JudgeConfig
from src.evaluation.judge.grid import (
    Outcome,
    build_grid,
    evidence_retrieved,
    grade,
    grid_path,
    outcome,
    verdicts_path,
)
from src.evaluation.judge.protocols import grid as grid_protocol
from src.evaluation.run_judge import judge, judged_path, with_model
from src.llm.config import LLMConfig

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
    ("correct", "refused", "retrieved", "expected"),
    [
        (True, False, True, Outcome.GOOD_JOB),
        (True, False, False, Outcome.UNVERIFIED),  # right, but the gold evidence never showed
        (False, False, True, Outcome.NEED_HELP),  # evidence there, still wrong
        (False, False, False, Outcome.HALLUCINATING),
        (False, True, True, Outcome.DONT_KNOW),  # refusal wins over everything
        (False, True, False, Outcome.DONT_KNOW),
    ],
)
def test_outcome(correct, refused, retrieved, expected):
    assert outcome(correct=correct, refused=refused, retrieved=retrieved) is expected


def test_grade_combines_the_verdict_with_the_computed_evidence():
    answer = {"id": "q1", "sources": [_src(46, EVIDENCE), _src(3, "unrelated risk factors")]}
    verdict = {"correct": False, "refused": False, "justification": "wrong line"}
    record = grade(answer, verdict, _qa(EVIDENCE), [46])
    assert record["evidence_retrieved"] is True
    assert record["outcome"] == "need_help"
    assert record["evidence_pages"] == [46]
    assert "alt_supported" not in record  # the grid no longer asks the judge for it


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _grid_config(tmp_path, **kwargs) -> JudgeConfig:
    return JudgeConfig(
        answers_path=str(tmp_path / "answers.jsonl"),
        protocol="grid",
        chunks_path=str(tmp_path / "chunks.jsonl"),
        golden_set_path=str(tmp_path / "gold.jsonl"),
        **kwargs,
    )


def _write_run(tmp_path):
    """A one-question golden set, corpus and answers file whose gold page never reached the prompt."""
    _write_jsonl(
        tmp_path / "gold.jsonl",
        [
            {
                "financebench_id": "q1",
                "question": "capex?",
                "answer": "1577",
                "justification": "Read from the cash flow statement.",
                "company": "3M",
                "doc_name": "DOC",
                "question_type": "metrics",
                "evidence": [
                    {"evidence_text": EVIDENCE, "doc_name": "DOC", "evidence_page_num": 0}
                ],
            }
        ],
    )
    _write_jsonl(
        tmp_path / "chunks.jsonl",
        [
            {"chunk_id": "DOC::0", "doc_id": "DOC", "page": 3, "text": "unrelated risk factors"},
            {"chunk_id": "DOC::1", "doc_id": "DOC", "page": 46, "text": EVIDENCE},
        ],
    )
    _write_jsonl(
        tmp_path / "answers.jsonl",
        [
            {
                "id": "q1",
                "question": "capex?",
                "gold_answer": "1577",
                "generated_answer": "1,577 million.",
                "sources": [_src(3, "unrelated risk factors")],
            }
        ],
    )


def test_an_external_judges_complete_verdicts_go_straight_to_the_grid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path)
    config = _grid_config(tmp_path, name="claude")
    verdicts = verdicts_path(config)
    verdicts.parent.mkdir(parents=True)
    _write_jsonl(verdicts, [{"id": "q1", "correct": True, "refused": False}])
    monkeypatch.setattr(grid_protocol, "generate_structured", pytest.fail)  # no LLM call

    out = json.loads(judge(config).read_text().splitlines()[0])
    assert out["evidence_pages"] == [46]
    assert out["outcome"] == "unverified"  # right answer, gold page never reached the prompt
    assert out["judge_model"] == "claude"


def test_the_grid_protocol_judges_with_the_gold_justification_then_builds_the_grid(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path)
    prompts = []

    def fake_generate_structured(prompt, config, model):
        prompts.append(prompt)
        return model(justification="1,577 = gold.", refused=False, correct=True)

    monkeypatch.setattr(grid_protocol, "generate_structured", fake_generate_structured)
    config = _grid_config(tmp_path, llm=LLMConfig(model="ollama_chat/mistral-nemo"))

    out = judge(config)
    assert out == grid_path(config)
    assert out.name == "answers_grid_by_ollama_chat_mistral-nemo.jsonl"
    assert "Read from the cash flow statement." in prompts[0]
    verdict = json.loads(verdicts_path(config).read_text())
    assert (verdict["correct"], verdict["refused"]) == (True, False)

    judge(config)  # every answer already judged: the grid is rebuilt without the LLM
    assert len(prompts) == 1


def test_a_spot_check_writes_verdicts_but_no_grid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path)
    monkeypatch.setattr(
        grid_protocol,
        "generate_structured",
        lambda prompt, config, model: model(justification="-", refused=True, correct=False),
    )
    config = _grid_config(tmp_path, name="judge")
    assert judge(config, qa_id="q1") == verdicts_path(config)
    assert not grid_path(config).exists()


def test_grid_paths_follow_the_answers_stem_and_the_judge():
    config = JudgeConfig(
        answers_path="data/processed/answers/run_a.jsonl",
        protocol="grid",
        chunks_path="unused.jsonl",
        name="claude",
    )
    assert str(judged_path(config)) == "data/processed/judged/verdicts/run_a.claude.jsonl"
    assert str(grid_path(config)) == "data/processed/judged/run_a_grid_by_claude.jsonl"
    other = with_model(config, "ollama_chat/mistral-nemo")
    assert verdicts_path(other).name == "run_a.ollama_chat_mistral-nemo.jsonl"


def test_run_refuses_to_grade_with_missing_verdicts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path)
    config = _grid_config(tmp_path, name="claude")
    with pytest.raises(SystemExit):
        build_grid(config)


def test_gold_justification_is_attached_from_the_golden_set(tmp_path):
    _write_run(tmp_path)
    records = grid_module.with_gold_justification([{"id": "q1"}], str(tmp_path / "gold.jsonl"))
    assert records == [{"id": "q1", "gold_justification": "Read from the cash flow statement."}]


# --- grid protocol ------------------------------------------------------------------


def test_the_prompt_shows_the_gold_justification_only_when_there_is_one():
    with_it = grid_protocol.build_prompt("q?", "1577", "1,577", "Read from the 10-K.")
    assert "How the gold was obtained: Read from the 10-K." in with_it
    assert "How the gold was obtained" not in grid_protocol.build_prompt("q?", "1577", "1,577")


def test_a_refusal_is_never_correct(monkeypatch):
    monkeypatch.setattr(
        grid_protocol,
        "generate_structured",
        lambda prompt, config, model: model(justification="-", refused=True, correct=True),
    )
    record = {"id": "q1", "question": "q?", "gold_answer": "g", "generated_answer": "a"}
    verdict = grid_protocol.judge(record, LLMConfig())
    assert verdict == {"correct": False, "refused": True, "justification": "-"}


def test_a_prompt_larger_than_the_pinned_context_is_refused_not_truncated(monkeypatch):
    monkeypatch.setattr(grid_protocol, "generate_structured", pytest.fail)
    record = {"id": "q1", "question": "q?", "gold_answer": "g", "generated_answer": "x" * 20000}
    with pytest.raises(ValueError, match="num_ctx"):
        grid_protocol.judge(record, LLMConfig(num_ctx=4096))


def test_grid_protocol_summary():
    verdicts = [
        {"correct": True, "refused": False},
        {"correct": False, "refused": True},
        {"correct": False, "refused": False},
    ]
    assert grid_protocol.summarize(verdicts) == "correct 1/3 | refused 1/3"


def test_grid_config_needs_the_corpus_and_refuses_a_prometheus_model():
    with pytest.raises(ValueError, match="chunks_path"):
        JudgeConfig(answers_path="a.jsonl", protocol="grid")
    with pytest.raises(ValueError, match="Prometheus"):
        JudgeConfig(
            answers_path="a.jsonl",
            protocol="grid",
            chunks_path="c.jsonl",
            llm=LLMConfig(model="ollama_chat/ggozad/prometheus2"),
        )
