"""Configuration schema for an LLM client (1 YAML = 1 experiment)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Parameters of a single local Ollama model.

    ``model`` keeps an ``ollama_chat/`` prefix -- it drives the output-file
    naming across the repo; the client strips it before calling Ollama.
    """

    model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "ollama_chat/llama3.1:8b")
    )
    temperature: float = 0.0
    max_tokens: int = 1024


def load_llm_config(path: str) -> LLMConfig:
    """Load and validate an LLM config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return LLMConfig(**data)
