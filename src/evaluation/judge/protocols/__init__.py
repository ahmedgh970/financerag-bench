"""Judging protocols, looked up by name from a judge config.

Each protocol module exposes the same two functions, so the runner never needs to
know which one it drives:

- ``judge(record, llm) -> dict``: the verdict fields for one answer record;
- ``summarize(verdicts) -> str``: a one-line summary of a batch of verdicts.
"""

from __future__ import annotations

from types import ModuleType

from src.evaluation.judge.protocols import correct_grounded, prometheus

PROTOCOLS: dict[str, ModuleType] = {
    "correct_grounded": correct_grounded,
    "prometheus": prometheus,
}


def get_protocol(name: str) -> ModuleType:
    """The protocol module registered under ``name``."""
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ValueError(f"Unknown judge protocol {name!r}; known: {sorted(PROTOCOLS)}") from None
