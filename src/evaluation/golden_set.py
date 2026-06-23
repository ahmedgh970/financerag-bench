"""Load the FinanceBench QA golden set into normalized ``QAItem`` objects."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.schema import Evidence, QAItem


def load_golden_set(path: str) -> list[QAItem]:
    """Read the FinanceBench JSONL and return the normalized QA items."""
    items: list[QAItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        items.append(
            QAItem(
                id=rec["financebench_id"],
                question=rec["question"],
                answer=rec["answer"],
                company=rec["company"],
                doc_name=rec["doc_name"],
                question_type=rec["question_type"],
                evidence=[
                    Evidence(
                        text=e["evidence_text"],
                        doc_name=e["doc_name"],
                        page=e["evidence_page_num"],
                    )
                    for e in rec["evidence"]
                ],
            )
        )
    return items
