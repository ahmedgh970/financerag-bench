"""Tests for Ragas scoring.

Offline tests cover the critic variant, the output naming, config validation and a
resumable run with fake metrics. The ``eval``-marked test scores a real record and
needs Ollama running locally with the critic and bge-m3.
"""

import json

import pytest

from src.evaluation import run_ragas
from src.evaluation.ragas import critic
from src.evaluation.ragas.config import RagasConfig, load_ragas_config
from src.evaluation.ragas.critic import critic_config, ensure_critic, variant_name
from src.evaluation.ragas.metrics import build_metrics, score_record
from src.llm.config import LLMConfig


def test_critic_runs_as_its_pinned_context_variant_only_when_num_ctx_is_set():
    assert variant_name("ollama_chat/mistral-nemo", 16384) == "ragas-critic-mistral-nemo-16384"
    assert variant_name("ollama_chat/llama3.1:8b", 8192) == "ragas-critic-llama3.1-8b-8192"
    plain = LLMConfig(model="ollama_chat/mistral-nemo")
    assert critic_config(plain) is plain
    pinned = critic_config(LLMConfig(model="ollama_chat/mistral-nemo", num_ctx=16384))
    assert pinned.model == "ollama_chat/ragas-critic-mistral-nemo-16384"


def test_shipped_config_writes_to_the_existing_ragas_file_name():
    # Same name as the scores produced before, so an interrupted run resumes them.
    config = load_ragas_config("configs/evaluation/ragas/ragas.yaml")
    assert run_ragas.ragas_path(config).name == (
        "reranked_docling_hybrid_1024_bge-m3_ollama_chat_granite4.1:8b_k10"
        "_ragas_by_ollama_chat_ragas-critic-mistral-nemo-16384.jsonl"
    )


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError):
        RagasConfig(answers_path="a.jsonl", metrics=["faithfulness", "bleu"])


class _Resp:
    def __init__(self, ok):
        self.ok = ok

    def raise_for_status(self):
        pass


def test_ensure_critic_creates_the_variant_only_when_missing(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None, present=False):
        calls.append((url.rsplit("/", 1)[-1], json))
        return _Resp(ok=present or url.endswith("/create"))

    llm = LLMConfig(model="ollama_chat/mistral-nemo", num_ctx=16384)
    monkeypatch.setattr(critic.requests, "post", lambda *a, **k: fake_post(*a, **k, present=True))
    ensure_critic(llm)
    assert [c[0] for c in calls] == ["show"]

    calls.clear()
    monkeypatch.setattr(critic.requests, "post", fake_post)
    ensure_critic(llm)
    assert [c[0] for c in calls] == ["show", "create"]
    assert calls[1][1] == {
        "model": "ragas-critic-mistral-nemo-16384",
        "from": "mistral-nemo",
        "parameters": {"num_ctx": 16384},
        "stream": False,
    }


def test_score_appends_one_record_per_answer_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = tmp_path / "run.jsonl"
    answers.write_text(
        "".join(json.dumps({"id": i, "question": "q?"}) + "\n" for i in ("q1", "q2"))
    )
    seen = []
    monkeypatch.setattr(run_ragas, "ensure_critic", lambda llm: None)
    monkeypatch.setattr(run_ragas, "build_metrics", lambda *a: {"faithfulness": None})
    monkeypatch.setattr(
        run_ragas, "score_record", lambda r, m: seen.append(r["id"]) or {"faithfulness": 1.0}
    )

    config = RagasConfig(answers_path=str(answers), metrics=["faithfulness"])
    out = run_ragas.score(config, limit=1)
    run_ragas.score(config)  # q1 already scored: only q2 reaches the critic
    assert seen == ["q1", "q2"]
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["q1", "q2"]


@pytest.mark.eval
def test_score_record_returns_all_four_metrics_in_range():
    metrics = build_metrics(LLMConfig(model="ollama_chat/mistral-nemo"))
    record = {
        "id": "smoke_test",
        "question": "What was the FY2018 capital expenditure for 3M?",
        "gold_answer": "$1577.00",
        "generated_answer": "According to the cash flow statement, purchases of "
        "property, plant and equipment in 2018 were $1,577 million.",
        "sources": [
            {
                "doc_id": "3M_2018_10K",
                "page": 46,
                "text": "Purchases of property, plant and equipment (PP&E), 2018 = $1,577 million.",
            }
        ],
    }

    scores = score_record(record, metrics)

    assert set(scores) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }
    for name, value in scores.items():
        assert 0.0 <= value <= 1.0, f"{name}={value} out of [0, 1]"
