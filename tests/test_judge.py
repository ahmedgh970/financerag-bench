"""Tests for the LLM judge (needs an LLM -- marked eval, skipped without a key)."""

import os

import pytest

from src.evaluation.judge import judge


@pytest.mark.eval
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY set")
def test_judge_flags_refusal_as_not_equivalent():
    result = judge(
        question="What was the net PP&E?",
        gold_answer="$8.70 billion",
        generated_answer="The context does not contain the answer.",
    )
    assert result["correct"] is False
    assert result["equivalent"] is False
    assert isinstance(result["justification"], str)


@pytest.mark.eval
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY set")
def test_judge_flags_matching_grounded_answer_as_equivalent():
    result = judge(
        question="What is the FY2018 capital expenditure amount for 3M?",
        gold_answer="$1577.00",
        generated_answer=(
            "According to the cash flow statement, purchases of property, plant "
            "and equipment in 2018 were $1,577 million."
        ),
    )
    assert result["correct"] is True
    assert result["grounded"] is True
    assert result["equivalent"] is True


@pytest.mark.eval
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY set")
def test_judge_flags_correct_but_hallucinated_answer_as_not_equivalent():
    # Right final number, reached by inventing a starting balance out of thin
    # air rather than by extracting it from the filing -- not grounded, so
    # equivalent must be False even though the value happens to land close.
    result = judge(
        question="What is the year-end FY2018 net PP&E for 3M, in USD billions?",
        gold_answer="$8.70",
        generated_answer=(
            "The context doesn't give the beginning PP&E balance, so let's assume "
            "it was around $10 billion based on 3M's size. Subtracting our "
            "estimated $1.3 billion net change gives approximately $8.70 billion."
        ),
    )
    assert result["correct"] is True
    assert result["grounded"] is False
    assert result["equivalent"] is False


@pytest.mark.eval
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY set")
def test_judge_handles_textual_composite_gold():
    result = judge(
        question="Is the company capital-intensive?",
        gold_answer="Yes, CAPEX/Revenue is 5.1% and Fixed assets/Total assets is 20%.",
        generated_answer="Yes, the company appears capital-intensive given its asset base.",
    )
    assert isinstance(result["equivalent"], bool)
