"""Tests for the Ollama client: a real call (needs a running Ollama) plus an
offline check that the request payload is shaped right."""

import pytest
import requests

from src.llm.client import generate
from src.llm.config import LLMConfig


def _ollama_up() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


@pytest.mark.eval
@pytest.mark.skipif(not _ollama_up(), reason="no local Ollama running")
def test_generate_returns_string():
    answer = generate("Reply with exactly: OK", LLMConfig())
    assert isinstance(answer, str)


class _FakeResp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "ok"}}


def _capture_post(monkeypatch) -> dict:
    """Patch requests.post to record the JSON body and return a canned reply."""
    from src.llm import client

    captured: dict = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured.update(json or {})
        return _FakeResp()

    monkeypatch.setattr(client.requests, "post", fake_post)
    return captured


def test_generate_strips_prefix_and_disables_thinking(monkeypatch):
    # The ollama_chat/ prefix (repo naming convention) is stripped before the
    # call, and thinking is disabled so a model's reasoning can't exhaust
    # num_predict and return empty content.
    captured = _capture_post(monkeypatch)
    generate("hi", LLMConfig(model="ollama_chat/qwen3.5:4b"))
    assert captured["model"] == "qwen3.5:4b"
    assert captured["think"] is False
    assert captured["options"]["temperature"] == 0.0
    assert captured["url"].endswith("/api/chat")
