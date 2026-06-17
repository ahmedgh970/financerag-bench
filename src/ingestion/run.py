"""Single CLI runner for the ingestion stages: parse and chunk.

Parsing is the expensive stage, done once per parser and persisted to
``{processed_dir}/{parser}/parsed/{doc_id}.json``. Chunking reloads those without
re-parsing, so several chunkers can be compared from a single parse.

Usage:
    python -m src.ingestion.run parse --config configs/parse_docling.yaml
    python -m src.ingestion.run chunk --config configs/chunk_docling_hybrid.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from tqdm import tqdm

from src.ingestion.config import IngestConfig, load_config
from src.ingestion.registry import PARSERS
from src.ingestion.schema import Chunk
from src.ingestion.storage import write_chunks


def _process(items: Sequence[Path], fn: Callable[[Path], None], desc: str) -> list[tuple[str, str]]:
    """Run ``fn`` over ``items`` with a progress bar; skip and report failures."""
    failures: list[tuple[str, str]] = []
    for item in tqdm(items, desc=desc):
        try:
            fn(item)
        except Exception as exc:  # noqa: BLE001 - resilience: skip and report
            failures.append((item.name, str(exc)))
    return failures


def _report(failures: list[tuple[str, str]], summary: str) -> None:
    print(f"\n{summary}")
    for name, err in failures:
        print(f"  FAILED {name}: {err}")


def _parser_module(name: str):
    if name not in PARSERS:
        raise SystemExit(f"Unknown parser {name!r}. Available: {list(PARSERS)}")
    return PARSERS[name]


def run_parse(config: IngestConfig) -> str:
    """Parse every PDF in ``config.pdf_dir`` once and persist each document.

    Incremental: already-parsed documents are skipped, so the run is resumable.
    """
    mod = _parser_module(config.parser)
    pdfs = sorted(Path(config.pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.pdf_dir!r}")
    parsed_dir = Path(config.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    counts = {"done": 0, "skipped": 0}

    def _parse_one(pdf: Path) -> None:
        out = parsed_dir / f"{pdf.stem}.json"
        if out.exists():
            counts["skipped"] += 1
            return
        mod.save(mod.parse(str(pdf), **config.parser_params), str(out))
        counts["done"] += 1

    failures = _process(pdfs, _parse_one, f"parse [{config.parser}]")
    _report(
        failures,
        f"Parsed {counts['done']} new, skipped {counts['skipped']} existing, "
        f"{len(failures)} failed -> {parsed_dir}",
    )
    return str(parsed_dir)


def run_chunk(config: IngestConfig) -> str:
    """Chunk every parsed document under ``config.parsed_dir`` to JSONL.

    Requires the parse stage to have run for ``config.parser``.
    """
    mod = _parser_module(config.parser)
    if config.chunker not in mod.CHUNKERS:
        raise SystemExit(
            f"Parser {config.parser!r} has no chunker {config.chunker!r}. "
            f"Available: {list(mod.CHUNKERS)}"
        )
    chunk_fn = mod.CHUNKERS[config.chunker]
    parsed_files = sorted(Path(config.parsed_dir).glob("*.json"))
    if not parsed_files:
        raise SystemExit(
            f"No parsed documents in {config.parsed_dir!r}. Run the parse stage first."
        )

    all_chunks: list[Chunk] = []

    def _chunk_one(pf: Path) -> None:
        native = mod.load(str(pf))
        all_chunks.extend(chunk_fn(native, doc_id=pf.stem, **config.chunker_params))

    failures = _process(parsed_files, _chunk_one, f"chunk [{config.parser}/{config.chunker}]")
    out_path = config.resolved_output_path
    n = write_chunks(all_chunks, out_path)
    _report(
        failures,
        f"Wrote {n} chunks from {len(parsed_files) - len(failures)}/{len(parsed_files)} "
        f"parsed docs -> {out_path}",
    )
    return out_path


_MODES: dict[str, Callable[[IngestConfig], str]] = {"parse": run_parse, "chunk": run_chunk}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion stages: parse then chunk.")
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in _MODES:
        p = sub.add_parser(mode, help=f"run the {mode} stage")
        p.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()
    _MODES[args.mode](load_config(args.config))


if __name__ == "__main__":
    main()
