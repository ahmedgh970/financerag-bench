"""Fast unit tests for the parser/chunker registry (no parsing involved)."""

import pytest

from src.ingestion.registry import PARSERS, resolve


def test_resolve_returns_callables():
    parse_fn, chunk_fn = resolve("docling", "hybrid")
    assert callable(parse_fn)
    assert callable(chunk_fn)


def test_resolve_unknown_parser_raises():
    with pytest.raises(ValueError, match="Unknown parser"):
        resolve("does-not-exist", "hybrid")


def test_resolve_unknown_chunker_raises():
    with pytest.raises(ValueError, match="has no chunker"):
        resolve("docling", "does-not-exist")


def test_every_parser_exposes_chunkers():
    # Each registered parser must expose at least one chunker, all callable.
    for name, mod in PARSERS.items():
        assert mod.CHUNKERS, f"parser '{name}' exposes no chunkers"
        assert all(callable(fn) for fn in mod.CHUNKERS.values())
