"""Top-level ingestion entry point: PDF + config -> list[Chunk]."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.registry import resolve
from src.ingestion.schema import Chunk


def ingest(pdf_path: str, config) -> list[Chunk]:
    """Parse and chunk ``pdf_path`` according to ``config``.

    ``config`` is expected to carry at least:
        - ``parser``: parser name (e.g. "docling")
        - ``chunker``: chunker name valid for that parser (e.g. "hybrid")
        - ``chunker_params``: dict of kwargs forwarded to the chunker

    The chunker receives the parser's native output and returns ``list[Chunk]``.
    """
    parse, chunk = resolve(config.parser, config.chunker)
    native = parse(pdf_path)
    doc_id = Path(pdf_path).stem
    return chunk(native, doc_id=doc_id, **getattr(config, "chunker_params", {}))
