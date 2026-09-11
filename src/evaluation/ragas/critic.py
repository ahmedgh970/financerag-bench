"""Serve the Ragas critic LLM with the context window its prompts need.

Ragas reaches Ollama through the OpenAI-compatible endpoint, which has no way of
setting the context size per request (Ollama docs, "Setting the context size"): the
model would load at Ollama's default window and silently truncate a long
faithfulness prompt. The documented fix is a model derived from the critic with
``num_ctx`` pinned, so when the config sets ``llm.num_ctx`` the critic runs as that
variant, created on first use and reused afterwards.
"""

from __future__ import annotations

import requests

from src.llm.client import OLLAMA_URL
from src.llm.config import LLMConfig


def _ollama_name(model: str) -> str:
    """``ollama_chat/mistral-nemo`` -> ``mistral-nemo`` (the repo keeps the prefix)."""
    return model.split("/", 1)[-1]


def variant_name(model: str, num_ctx: int) -> str:
    """Name of the critic variant pinned at ``num_ctx``, e.g. ``ragas-critic-mistral-nemo-16384``."""
    return f"ragas-critic-{_ollama_name(model).replace(':', '-')}-{num_ctx}"


def critic_config(llm: LLMConfig) -> LLMConfig:
    """The LLM config Ragas should call: the pinned-context variant, else ``llm`` itself.

    Pure: it only names the model, so output paths can be derived without Ollama.
    """
    if llm.num_ctx is None:
        return llm
    return llm.model_copy(update={"model": f"ollama_chat/{variant_name(llm.model, llm.num_ctx)}"})


def ensure_critic(llm: LLMConfig) -> None:
    """Create the pinned-context variant of the critic in Ollama if it does not exist yet."""
    if llm.num_ctx is None:
        return
    name = variant_name(llm.model, llm.num_ctx)
    if requests.post(f"{OLLAMA_URL}/api/show", json={"model": name}, timeout=30).ok:
        return
    response = requests.post(
        f"{OLLAMA_URL}/api/create",
        json={
            "model": name,
            "from": _ollama_name(llm.model),
            "parameters": {"num_ctx": llm.num_ctx},
            "stream": False,
        },
        timeout=600,
    )
    response.raise_for_status()
    print(f"created critic variant {name} (num_ctx {llm.num_ctx})")
