"""Configuration schema for the Ragas runs (1 YAML = 1 experiment)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.llm.config import LLMConfig


class RagasConfig(BaseModel):
    """Parameters of a Ragas run: score an existing answers JSONL on faithfulness,
    answer relevancy, context precision, and context recall."""

    answers_path: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    # Ollama's own bge-m3 (served through its OpenAI-compatible endpoint), not
    # a separate sentence-transformers load -- Ollama picks its own GPU/CPU
    # dispatch for it, same as the LLM.
    embedding_model: str = "bge-m3"
    # Subset of metrics to score (None -> all four). context_precision/recall
    # are retriever-only (identical across generators at fixed k), so a
    # model-comparison run can keep just faithfulness/answer_relevancy.
    metrics: list[str] | None = None


def load_ragas_config(path: str) -> RagasConfig:
    """Load and validate a Ragas config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return RagasConfig(**data)
