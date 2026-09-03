"""Configuration schema for an agentic RAG run (1 YAML = 1 experiment).

Extends the naive :class:`RagConfig` with the agent-loop knobs. The retriever
should be dense-only: in the agent loop the model itself filters the passages,
and the escalating ``depths`` (5 -> 10 -> 20) let it start cheap and go deeper
only when needed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.rag.config import RagConfig


class AgentConfig(RagConfig):
    """Parameters of an agentic RAG run (see :func:`src.agents.agent_rag.answer_agentic`)."""

    retriever: str = "dense"
    depths: tuple[int, ...] = (5, 10, 20)
    num_ctx: int = 32768
    recursion_limit: int = 12


def load_agent_config(path: str) -> AgentConfig:
    """Load and validate an agent config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return AgentConfig(**data)
