"""Configuration schema for a retrieval-evaluation run (1 YAML = 1 experiment)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    """Parameters of a retrieval evaluation.

    Carries ``qdrant_location`` / ``collection_name`` / ``embedding_model`` so it
    plugs straight into ``build_retriever``. ``chunks_path`` is the same corpus
    that was indexed — it is used to resolve each evidence to its physical page.
    """

    golden_set_path: str = "data/jsons/financebench_open_source.jsonl"
    chunks_path: str
    collection_name: str
    embedding_model: str = "BAAI/bge-m3"
    retriever: str = "dense"
    qdrant_location: str = Field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    k_values: list[int] = [1, 3, 5, 10]


def load_eval_config(path: str) -> EvalConfig:
    """Load and validate an evaluation config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return EvalConfig(**data)
