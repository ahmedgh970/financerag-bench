"""Match retrieved chunks to gold evidence by resolving the physical page.

FinanceBench's ``evidence_page_num`` is offset from the parser's physical page
(observed +1, i.e. FinanceBench pages are 0-indexed). Rather than assume a fixed
offset, each evidence is resolved to the physical page whose text best overlaps
it; a retrieved chunk is then relevant if it sits on a resolved gold page of the
gold document.
"""

from __future__ import annotations

import re

from src.evaluation.schema import QAItem
from src.ingestion.schema import Chunk

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> set[str]:
    """Lowercase alphanumeric token set, for overlap scoring."""
    return set(_WORD.findall(text.lower()))


def resolve_gold_pages(qa: QAItem, page_words: dict[int, set[str]]) -> set[int]:
    """Resolve the QA's evidence to physical page numbers in the corpus.

    ``page_words`` maps each page of ``qa.doc_name`` to the set of words found on
    it (union of that page's chunks). For each evidence, the page with the most
    word overlap is taken as a gold page.
    """
    gold: set[int] = set()
    if not page_words:
        return gold
    for ev in qa.evidence:
        ev_words = words(ev.text)
        if ev_words:
            gold.add(max(page_words, key=lambda p: len(ev_words & page_words[p])))
    return gold


def is_relevant(chunk: Chunk, doc_name: str, gold_pages: set[int]) -> bool:
    """A chunk is relevant if it sits on a gold page of the gold document."""
    return chunk.doc_id == doc_name and chunk.page in gold_pages
