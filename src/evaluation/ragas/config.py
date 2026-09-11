"""Configuration schema for a Ragas run (1 YAML = 1 experiment)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from src.llm.config import LLMConfig

ALL_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class RagasConfig(BaseModel):
    """Parameters of a Ragas run over an answers JSONL.

    ``llm`` is the critic that scores the answers. Set ``llm.num_ctx`` to the window
    its longest prompt needs: Ragas cannot pass it per request, so the critic then
    runs as a pinned-context variant (:mod:`src.evaluation.ragas.critic`).
    """

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

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, value: list[str] | None) -> list[str] | None:
        unknown = [m for m in value or [] if m not in ALL_METRICS]
        if unknown:
            raise ValueError(f"unknown Ragas metric(s) {unknown}; known: {ALL_METRICS}")
        return value


def load_ragas_config(path: str) -> RagasConfig:
    """Load and validate a Ragas config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return RagasConfig(**data)
