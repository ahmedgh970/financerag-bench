"""Configuration schema for a CRAG workflow run (1 YAML = 1 reproducible experiment).

Extends the RAG config with the switches that select a row of the ablation matrix.
Grading and the calculator act on different failure modes -- noisy retrieval versus
numeric questions -- so their effects should be attributable independently.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.rag.config import RagConfig


class GradingConfig(BaseModel):
    """Graded relevance filtering of the retrieved passages.

    Every passage is graded 0-3; ``keep_threshold`` turns those grades into a context.
    Because the grades are recorded per passage, the threshold can be re-examined
    offline from a single run rather than requiring one run per value.

    ``min_chunks`` is the floor: measured, the grader keeps 1-2 passages on 45% of the
    questions and those collapse to 45% accuracy against 73% when 3-5 survive, so a
    selection thinner than the floor is topped up with the best-ranked passages it did
    not already keep. The value doubles as the choice of policy -- 0 disables the floor,
    1 only rescues an empty selection, 3 always guarantees three passages.

    Whether that ranking comes from a cross-encoder is not a switch here: it follows
    from ``retriever`` (dense or reranked) on the parent config.
    """

    enabled: bool = False
    keep_threshold: int = Field(default=2, ge=0, le=3)
    min_chunks: int = Field(default=3, ge=0)
    # Instrumentation only: no action is taken, the flag is recorded for later analysis.
    low_confidence_below: int = Field(default=2, ge=0, le=3)


class CalculatorConfig(BaseModel):
    """Deterministic arithmetic path for numeric questions."""

    enabled: bool = False
    routing: str = "heuristic"


class WorkflowConfig(RagConfig):
    """Parameters of a CRAG workflow run."""

    grading: GradingConfig = Field(default_factory=GradingConfig)
    calculator: CalculatorConfig = Field(default_factory=CalculatorConfig)


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
