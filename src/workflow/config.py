"""Configuration schema for a CRAG workflow run (1 YAML = 1 reproducible experiment).

Extends the RAG config with the switches that select a row of the ablation matrix.
Grading and the calculator act on different failure modes -- noisy retrieval versus
numeric questions -- so their effects should be attributable independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from src.rag.config import RagConfig


class GradingConfig(BaseModel):
    """Relevance grading of the retrieved chunks, and the correction loop it drives."""

    enabled: bool = False
    mode: Literal["batch", "per_chunk"] = "per_chunk"


class CalculatorConfig(BaseModel):
    """Deterministic arithmetic path for numeric questions."""

    enabled: bool = False
    routing: Literal["heuristic", "llm"] = "heuristic"


class WorkflowConfig(RagConfig):
    """Parameters of a CRAG workflow run."""

    grading: GradingConfig = Field(default_factory=GradingConfig)
    calculator: CalculatorConfig = Field(default_factory=CalculatorConfig)
    max_rewrites: int = 2


def load_workflow_config(path: str) -> WorkflowConfig:
    """Load and validate a workflow config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return WorkflowConfig(**data)


def variant_name(config: WorkflowConfig) -> str:
    """Short name of the ablation cell this config selects.

    Used in the output filename so each cell of the matrix is a separate, resumable
    experiment rather than an overwrite of the previous one.
    """
    return {
        (False, False): "advanced",
        (True, False): "grading",
        (False, True): "calc",
        (True, True): "crag_full",
    }[(config.grading.enabled, config.calculator.enabled)]
