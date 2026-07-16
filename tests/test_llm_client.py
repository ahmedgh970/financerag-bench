"""Tests for the LLM client: a real call (needs GROQ_API_KEY) plus the
fast, offline request-size guard."""

import os

import pytest

from src.llm.client import RequestTooLargeError, generate
from src.llm.config import LLMConfig


@pytest.mark.eval
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY set")
def test_generate_returns_nonempty_string():
    answer = generate("Reply with exactly: OK", LLMConfig())
    assert isinstance(answer, str)
    assert answer.strip() != ""


def test_generate_fails_fast_on_oversized_prompt():
    # ~30k tokens of padding, well over groq/llama-3.1-8b-instant's known 6k TPM
    # cap — must fail immediately (no retry, no network call), not hang/retry.
    huge_prompt = "word " * 30_000
    config = LLMConfig(model="groq/llama-3.1-8b-instant")

    with pytest.raises(RequestTooLargeError):
        generate(huge_prompt, config)


def test_generate_allows_small_prompt_through_size_check():
    from src.llm.client import _check_request_size

    config = LLMConfig(model="groq/llama-3.1-8b-instant")
    _check_request_size("a short prompt", config)  # should not raise


def test_check_request_size_skips_unknown_models():
    from src.llm.client import _check_request_size

    config = LLMConfig(model="groq/some-future-model")
    _check_request_size("word " * 30_000, config)  # no known limit -> no raise
