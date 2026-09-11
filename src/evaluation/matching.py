"""Match retrieved chunks to gold evidence by resolving the physical page.

FinanceBench's ``evidence_page_num`` is offset from the parser's physical page
(observed +1, i.e. FinanceBench pages are 0-indexed). Rather than assume a fixed
offset, each evidence is resolved to the physical page whose text best overlaps
it; a retrieved chunk is then relevant if it sits on a resolved gold page of the
gold document.
"""

from __future__ import annotations

import re
from collections import defaultdict

from src.evaluation.schema import QAItem
from src.ingestion.schema import Chunk
from src.ingestion.storage import read_chunks

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> set[str]:
    """Lowercase alphanumeric token set, for overlap scoring."""
    return set(_WORD.findall(text.lower()))


def build_page_index(chunks_path: str, doc_names: set[str]) -> dict[str, dict[int, set[str]]]:
    """Build {doc: {page: word set}} from the corpus, for the needed docs only."""
    index: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for chunk in read_chunks(chunks_path):
        if chunk.doc_id in doc_names:
            index[chunk.doc_id][chunk.page] |= words(chunk.text)
    return index


def resolve_evidence_page(evidence_text: str, page_words: dict[int, set[str]]) -> int | None:
    """The physical page whose words overlap ``evidence_text`` the most, if any."""
    ev_words = words(evidence_text)
    if not page_words or not ev_words:
        return None
    return max(page_words, key=lambda p: len(ev_words & page_words[p]))


def resolve_gold_pages(qa: QAItem, page_words: dict[int, set[str]]) -> set[int]:
    """Resolve the QA's evidence to physical page numbers in the corpus.

    ``page_words`` maps each page of ``qa.doc_name`` to the set of words found on
    it (union of that page's chunks). For each evidence, the page with the most
    word overlap is taken as a gold page.
    """
    pages = (resolve_evidence_page(ev.text, page_words) for ev in qa.evidence)
    return {p for p in pages if p is not None}


def is_relevant(chunk: Chunk, doc_name: str, gold_pages: set[int]) -> bool:
    """A chunk is relevant if it sits on a gold page of the gold document."""
    return chunk.doc_id == doc_name and chunk.page in gold_pages
