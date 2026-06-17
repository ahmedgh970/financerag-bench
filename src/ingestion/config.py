"""Configuration schema for parse / chunk runs (1 YAML = 1 experiment)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class IngestConfig(BaseModel):
    """Parameters of a parse and/or chunk run.

    A parse config needs ``parser`` (+ ``parser_params``, ``pdf_dir``); a chunk
    config additionally needs ``chunker`` (+ ``chunker_params``). The same schema
    drives both stages and the one-shot ``ingest`` pipeline.
    """

    parser: str
    parser_params: dict = Field(default_factory=dict)
    chunker: str | None = None
    chunker_params: dict = Field(default_factory=dict)
    pdf_dir: str = "data/pdfs"
    processed_dir: str = "data/processed"
    output_path: str | None = None

    @property
    def parsed_dir(self) -> str:
        """Directory holding one serialized parsed document per PDF."""
        return f"{self.processed_dir}/{self.parser}/parsed"

    @property
    def resolved_output_path(self) -> str:
        """Where chunks are written; derived from parser+chunker if unset."""
        return self.output_path or f"{self.processed_dir}/{self.parser}_{self.chunker}/chunks.jsonl"


def load_config(path: str) -> IngestConfig:
    """Load and validate a config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return IngestConfig(**data)
