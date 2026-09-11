"""Configuration schema for the judge family: one config per judging protocol."""

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
    ``--model`` change. ``name`` tags the output files (``..._judged_by_{name}``);
    left unset, the model name is used, so two models never share a file.

    The ``grid`` protocol also places every answer in the outcome grid, which needs
    ``chunks_path`` (the indexed corpus, to resolve each gold evidence to its
    physical page, as the retrieval metrics do) and the golden set.
    """

    answers_path: str
    protocol: str = "correct_grounded"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    name: str | None = None
    chunks_path: str | None = None
    golden_set_path: str = "data/jsons/financebench_open_source.jsonl"

    @field_validator("protocol")
    @classmethod
    def _known_protocol(cls, value: str) -> str:
        if value not in PROTOCOLS:
            raise ValueError(f"unknown judge protocol {value!r}; known: {sorted(PROTOCOLS)}")
        return value

    @model_validator(mode="after")
    def _prometheus_needs_its_own_protocol(self) -> JudgeConfig:
        # Prometheus is fine-tuned on one verbatim prompt; any other prompt would
        # silently degrade it (ADR 0003).
        if self.protocol != "prometheus" and "prometheus" in self.llm.model.lower():
            raise ValueError("a Prometheus model must be run with protocol: prometheus")
        return self

    @model_validator(mode="after")
    def _grid_needs_the_corpus(self) -> JudgeConfig:
        if self.protocol == "grid" and not self.chunks_path:
            raise ValueError("protocol: grid needs chunks_path to locate the gold evidence")
        return self

    @property
    def judge_name(self) -> str:
        """Tag of the output file: ``name``, else the model."""
        return self.name or self.llm.model


def load_judge_config(path: str) -> JudgeConfig:
    """Load and validate a judge config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return JudgeConfig(**data)
