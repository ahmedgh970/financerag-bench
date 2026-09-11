"""Tests for the judging protocols and the ``score`` command.

Offline tests cover parsing, summaries, config validation and a resumable run with a
fake LLM. The ``eval``-marked tests call a real local model and are skipped when
Ollama is not running.
"""

import json

import pytest
import requests

from src.evaluation.judge.config import JudgeConfig
from src.evaluation.judge.protocols import correct_grounded, get_protocol, prometheus
from src.evaluation.run_judge import judged_path, score, with_model
from src.llm.config import LLMConfig


def _ollama_up() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def _record(generated: str, question: str = "q?", gold: str = "g", qa_id: str = "q1") -> dict:
    return {"id": qa_id, "question": question, "gold_answer": gold, "generated_answer": generated}


# --- correct / grounded -------------------------------------------------------------


def test_correct_grounded_parses_both_verdicts_and_combines_them():
    verdict = correct_grounded.parse("CORRECT: yes\nGROUNDED: no\nJUSTIFICATION: made up figure")
    assert verdict == {
        "correct": True,
        "grounded": False,
        "equivalent": False,
        "justification": "made up figure",
    }


def test_correct_grounded_rejects_a_reply_without_its_verdict_lines():
    with pytest.raises(ValueError):
        correct_grounded.parse("The answer looks fine to me.")


def test_correct_grounded_summary():
    verdicts = [
        {"correct": True, "grounded": True, "equivalent": True},
        {"correct": True, "grounded": False, "equivalent": False},
    ]
    assert correct_grounded.summarize(verdicts) == ("equivalent 1/2 | correct but not grounded 1/2")


# --- prometheus ---------------------------------------------------------------------


def test_prometheus_reads_the_result_tag_and_falls_back_to_a_score_mention():
    assert prometheus.parse_score("Feedback: solid. [RESULT] 4") == 4
    assert prometheus.parse_score("Feedback: solid. Overall score: 3") == 3
    assert prometheus.parse_score("no score at all") is None


def test_prometheus_sends_its_system_prompt_and_splits_feedback(monkeypatch):
    calls = {}

    def fake_generate(prompt, config, system=None):
        calls["prompt"], calls["system"] = prompt, system
        return "Feedback: matches the reference. [RESULT] 5"

    monkeypatch.setattr(prometheus, "generate", fake_generate)
    verdict = prometheus.judge(_record("1,577"), LLMConfig(model="ollama_chat/p"))
    assert verdict == {"score": 5, "feedback": "matches the reference."}
    assert calls["system"].startswith("You are a fair judge assistant")
    assert "###Reference Answer (Score 5):\ng" in calls["prompt"]


def test_prometheus_summary_counts_unparsed_outputs():
    verdicts = [{"score": 5}, {"score": 3}, {"score": None}]
    assert prometheus.summarize(verdicts) == "mean 4.00 | distribution {3: 1, 5: 1} | 1 UNPARSED"


# --- config and runner --------------------------------------------------------------


def test_unknown_protocol_is_rejected():
    with pytest.raises(ValueError):
        JudgeConfig(answers_path="a.jsonl", protocol="nope")
    with pytest.raises(ValueError):
        get_protocol("nope")


def test_a_prometheus_model_cannot_run_the_correct_grounded_prompt():
    with pytest.raises(ValueError):
        JudgeConfig(answers_path="a.jsonl", llm=LLMConfig(model="ollama_chat/ggozad/prometheus2"))
    base = JudgeConfig(answers_path="a.jsonl", llm=LLMConfig(model="ollama_chat/llama3.1:8b"))
    with pytest.raises(ValueError):
        with_model(base, "ollama_chat/ggozad/prometheus2")  # the override is re-validated


def test_output_is_tagged_by_name_else_by_model_and_a_new_model_gets_its_own_file():
    named = JudgeConfig(
        answers_path="x/run.jsonl",
        protocol="prometheus",
        name="prometheus",
        llm=LLMConfig(model="ollama_chat/ggozad/prometheus2:latest"),
    )
    assert judged_path(named).name == "run_judged_by_prometheus.jsonl"
    other = with_model(named, "ollama_chat/other")
    assert judged_path(other).name == "run_judged_by_ollama_chat_other.jsonl"


def test_score_writes_one_verdict_per_answer_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = tmp_path / "run.jsonl"
    answers.write_text("".join(json.dumps(_record("a", qa_id=i)) + "\n" for i in ("q1", "q2")))
    replies = iter(["CORRECT: yes\nGROUNDED: yes", "CORRECT: no\nGROUNDED: yes"])
    monkeypatch.setattr(correct_grounded, "generate", lambda prompt, config: next(replies))

    config = JudgeConfig(answers_path=str(answers))
    out = score(config, limit=1)
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["q1"]

    score(config)  # q1 already judged: only q2 is sent to the model
    verdicts = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(v["id"], v["equivalent"]) for v in verdicts] == [("q1", True), ("q2", False)]
    assert set(verdicts[0]) >= {"question", "gold_answer", "generated_answer", "judge_model"}


# --- real model ---------------------------------------------------------------------

live = pytest.mark.skipif(not _ollama_up(), reason="no local Ollama running")


def _judge(question: str, gold: str, generated: str) -> dict:
    return correct_grounded.judge(_record(generated, question, gold), LLMConfig())


@pytest.mark.eval
@live
def test_judge_flags_refusal_as_not_equivalent():
    result = _judge(
        "What was the net PP&E?", "$8.70 billion", "The context does not contain the answer."
    )
    assert result["correct"] is False
    assert result["equivalent"] is False
    assert isinstance(result["justification"], str)


@pytest.mark.eval
@live
def test_judge_flags_matching_grounded_answer_as_equivalent():
    result = _judge(
        "What is the FY2018 capital expenditure amount for 3M?",
        "$1577.00",
        "According to the cash flow statement, purchases of property, plant "
        "and equipment in 2018 were $1,577 million.",
    )
    assert result["correct"] is True
    assert result["grounded"] is True
    assert result["equivalent"] is True


@pytest.mark.eval
@live
def test_judge_flags_correct_but_hallucinated_answer_as_not_equivalent():
    # Right final number, reached by inventing a starting balance out of thin
    # air rather than by extracting it from the filing -- not grounded, so
    # equivalent must be False even though the value happens to land close.
    result = _judge(
        "What is the year-end FY2018 net PP&E for 3M, in USD billions?",
        "$8.70",
        "The context doesn't give the beginning PP&E balance, so let's assume "
        "it was around $10 billion based on 3M's size. Subtracting our "
        "estimated $1.3 billion net change gives approximately $8.70 billion.",
    )
    assert result["correct"] is True
    assert result["grounded"] is False
    assert result["equivalent"] is False


@pytest.mark.eval
@live
def test_judge_handles_textual_composite_gold():
    result = _judge(
        "Is the company capital-intensive?",
        "Yes, CAPEX/Revenue is 5.1% and Fixed assets/Total assets is 20%.",
        "Yes, the company appears capital-intensive given its asset base.",
    )
    assert isinstance(result["equivalent"], bool)
