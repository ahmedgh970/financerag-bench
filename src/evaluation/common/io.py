"""JSONL plumbing shared by the evaluation runners.

Every runner reads an answers file keyed by QA ``id``, scores what is left, and
appends one record per answer to an output file named after the answers file.
Appending record by record, then skipping the ids already written on the next
run, is what makes a long judge or Ragas run resumable after an interruption.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    """All records of a JSONL file, blank lines skipped."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_by_id(path: str | Path) -> dict[str, dict]:
    """Records of a JSONL file keyed by their ``id``."""
    return {r["id"]: r for r in read_jsonl(path)}


def done_ids(path: str | Path) -> set[str]:
    """Ids already written to ``path``; empty when the file does not exist yet."""
    return set(read_by_id(path)) if Path(path).exists() else set()


def select(records: list[dict], qa_id: str | None = None, limit: int | None = None) -> list[dict]:
    """Just the record with ``qa_id``, else the first ``limit`` records, else all.

    An unknown ``qa_id`` exits with an error rather than silently scoring nothing.
    """
    if qa_id is not None:
        selected = [r for r in records if r["id"] == qa_id]
        if not selected:
            raise SystemExit(f"No record with id {qa_id!r}.")
        return selected
    return records if limit is None else records[:limit]


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record, flushed at once so an interrupted run loses nothing."""
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def derived_path(answers_path: str | Path, out_dir: str | Path, tag: str) -> Path:
    """``out_dir/{answers stem}_{tag}.jsonl``: one output per answers file and scorer.

    ``tag`` names the scorer (``judged_by_<model>``, ``ragas_by_<model>``...); a
    ``/`` in a model name is replaced so the tag stays a single path segment.
    """
    return Path(out_dir) / f"{Path(answers_path).stem}_{tag.replace('/', '_')}.jsonl"
