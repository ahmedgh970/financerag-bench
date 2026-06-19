"""Fast unit tests for the parser registry."""

import pytest

# The registry imports the parser packages, which import docling/unstructured at
# module load. Skip this module when the optional ingestion stack is absent
# (e.g. the light CI that installs only core + dev).
pytest.importorskip("docling")
pytest.importorskip("unstructured")

from src.ingestion.registry import PARSERS  # noqa: E402


def test_every_parser_exposes_parse_save_load_and_chunkers():
    # Each registered parser must expose the parse/save/load API and at least
    # one callable chunker.
    for name, mod in PARSERS.items():
        assert callable(mod.parse), f"parser '{name}' has no parse()"
        assert callable(mod.save), f"parser '{name}' has no save()"
        assert callable(mod.load), f"parser '{name}' has no load()"
        assert mod.CHUNKERS, f"parser '{name}' exposes no chunkers"
        assert all(callable(fn) for fn in mod.CHUNKERS.values())
