"""Configuration schemas for the judge and Ragas runs (1 YAML = 1 experiment)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.llm.config import LLMConfig


class JudgeConfig(BaseModel):
    """Parameters of a judge run: score an existing answers JSONL against gold."""

    answers_path: str
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_judge_config(path: str) -> JudgeConfig:
    """Load and validate a judge config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return JudgeConfig(**data)


class GridConfig(BaseModel):
    """Parameters of an evidence-grounded grid run over judged answers.

    ``verdicts_path`` holds the judge's reading of each answer (``correct``,
    ``refused``, ``alt_supported``, ``justification``); left unset, it is derived
    from the answers file name, so one config grades any judged run. ``chunks_path``
    is the indexed corpus, used to resolve each gold evidence to its physical page --
    the same resolution as the retrieval metrics.
    """

    answers_path: str
    verdicts_path: str | None = None
    chunks_path: str
    judge_model: str
    golden_set_path: str = "data/jsons/financebench_open_source.jsonl"


def load_grid_config(path: str) -> GridConfig:
    """Load and validate a grid config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return GridConfig(**data)


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
