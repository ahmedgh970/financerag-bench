"""Configuration schema for an LLM client (1 YAML = 1 experiment)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Parameters of a single LLM, in LiteLLM's ``provider/model`` naming."""

    model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "groq/llama-3.3-70b-versatile")
    )
    temperature: float = 0.0
    max_tokens: int = 1024


def load_llm_config(path: str) -> LLMConfig:
    """Load and validate an LLM config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return LLMConfig(**data)
