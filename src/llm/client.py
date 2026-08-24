"""Thin Ollama client: one function to generate a completion locally.

The whole pipeline runs on local Ollama models (generation, judging, serving),
so this talks to Ollama's native ``/api/chat`` endpoint directly -- no provider
abstraction, no per-minute token guard (there is no quota locally). Transient
transport failures (connection dropped, timeout, 5xx) are retried with backoff;
a bad request fails immediately since retrying can't help.
"""

from __future__ import annotations

import os

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.config import LLMConfig

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Transport-level failures worth retrying; a 4xx (bad request) is not among them.
_TRANSIENT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _model_name(model: str) -> str:
    """Strip a ``provider/`` prefix, keeping the Ollama model name.

    Configs still name models ``ollama_chat/llama3.1:8b`` (the prefix drives the
    output-file naming across the repo); Ollama itself wants just ``llama3.1:8b``.
    """
    return model.split("/", 1)[1] if "/" in model else model


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
)
def generate(prompt: str, config: LLMConfig) -> str:
    """Generate a completion for ``prompt`` on a local Ollama model.

    ``think: False`` disables the reasoning channel on thinking-capable models
    (qwen3.5, gemma4): left on, their chain-of-thought is emitted first and can
    exhaust ``num_predict`` before any answer token, returning empty content.
    Plain-instruct models ignore the flag, so it is always safe to send.
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": _model_name(config.model),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": config.temperature, "num_predict": config.max_tokens},
        },
        timeout=1800,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")
