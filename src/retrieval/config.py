"""Configuration schema for an indexing run (1 YAML = 1 experiment)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.retrieval.embeddings import DEFAULT_MODEL


class IndexConfig(BaseModel):
    """Parameters of a single indexing run: chunks -> embeddings -> Qdrant.

    ``qdrant_location`` defaults to the local Docker server, overridable by the
    ``QDRANT_URL`` env var, or set to a filesystem path for the embedded store.
    """

    chunks_path: str
    collection_name: str
    embedding_model: str = DEFAULT_MODEL
    qdrant_location: str = Field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    batch_size: int = 128
    recreate: bool = False


def load_index_config(path: str) -> IndexConfig:
    """Load and validate an indexing config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return IndexConfig(**data)
