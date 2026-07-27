"""Tests for Ragas scoring (needs Ollama running locally + downloads BGE-M3 -- marked eval)."""

import pytest

from src.evaluation.ragas_eval import build_metrics, score_record
from src.llm.config import LLMConfig


@pytest.mark.eval
def test_score_record_returns_all_four_metrics_in_range():
    metrics = build_metrics(LLMConfig(model="ollama_chat/mistral-nemo"), embedding_device="cpu")
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
