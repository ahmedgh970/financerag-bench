"""Resolve a (parser, chunker) pair from config and validate the pairing.

Each chunker is owned by its parser (it consumes that parser's native output),
so not every (parser, chunker) combination is valid. ``resolve`` enforces this
with a clear error instead of letting an incompatible pair fail deep inside.
"""

from __future__ import annotations

from src.ingestion.parsers import docling, unstructured

PARSERS = {
    "docling": docling,
    "unstructured": unstructured,
}


def resolve(parser_name: str, chunker_name: str):
    """Return ``(parse_fn, chunk_fn)`` for the given names.

    Raises:
        ValueError: if the parser is unknown, or the chunker is not one of that
            parser's available chunkers.
    """
    if parser_name not in PARSERS:
        raise ValueError(f"Unknown parser '{parser_name}'. Available: {list(PARSERS)}")
    mod = PARSERS[parser_name]
    if chunker_name not in mod.CHUNKERS:
        raise ValueError(
            f"Parser '{parser_name}' has no chunker '{chunker_name}'. "
            f"Available: {list(mod.CHUNKERS)}"
        )
    return mod.parse, mod.CHUNKERS[chunker_name]
