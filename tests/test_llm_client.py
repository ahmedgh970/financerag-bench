"""Integration test for the LLM client: a real call, needs GROQ_API_KEY."""

import os

import pytest

from src.llm.client import generate
from src.llm.config import LLMConfig


@pytest.mark.eval
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY set")
def test_generate_returns_nonempty_string():
    answer = generate("Reply with exactly: OK", LLMConfig())
    assert isinstance(answer, str)
    assert answer.strip() != ""
