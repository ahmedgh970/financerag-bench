"""Configuration schemas for the judge family: LLM judging and the outcome grid."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.evaluation.judge.protocols import PROTOCOLS
from src.llm.config import LLMConfig


class JudgeConfig(BaseModel):
    """Parameters of an LLM judging run over an answers JSONL.

    ``protocol`` picks how answers are graded (see :mod:`src.evaluation.judge.protocols`)
    and ``llm.model`` which model grades them, so switching judge is a config or
    ``--model`` change. ``name`` tags the output file (``..._judged_by_{name}``);
    left unset, the model name is used, so two models never share a file.
    """

    answers_path: str
    protocol: str = "correct_grounded"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    name: str | None = None

    @field_validator("protocol")
    @classmethod
    def _known_protocol(cls, value: str) -> str:
        if value not in PROTOCOLS:
            raise ValueError(f"unknown judge protocol {value!r}; known: {sorted(PROTOCOLS)}")
        return value

    @model_validator(mode="after")
    def _prometheus_needs_its_own_protocol(self) -> JudgeConfig:
        # Prometheus is fine-tuned on one verbatim prompt; the correct/grounded prompt
        # would silently degrade it (ADR 0003).
        if self.protocol == "correct_grounded" and "prometheus" in self.llm.model.lower():
            raise ValueError("a Prometheus model must be run with protocol: prometheus")
        return self

    @property
    def judge_name(self) -> str:
        """Tag of the output file: ``name``, else the model."""
        return self.name or self.llm.model


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
