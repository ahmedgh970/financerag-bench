"""Sparse retrieval: rank chunks by BM25 lexical overlap with the query.

Complements dense retrieval by catching exact terms and figures (financial
questions hinge on them) that embeddings blur. The BM25 index is built once over
the corpus; ``doc_id`` restricts scoring to a single document at query time.
"""

from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from src.ingestion.schema import Chunk
from src.retrieval.base import ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (a list, so term frequencies are kept)."""
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Retrieve the top-k chunks by BM25 score over a fixed chunk corpus.

    IDF is computed over the whole corpus, so per-document scoping (``doc_id``)
    still benefits from corpus-wide term statistics; only the candidate set is
    restricted to that document.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        self._positions_by_doc: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            self._positions_by_doc[c.doc_id].append(i)

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None) -> list[ScoredChunk]:
        scores = self.bm25.get_scores(_tokenize(query))
        candidates = self._positions_by_doc[doc_id] if doc_id else range(len(self.chunks))
        top = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredChunk(chunk=self.chunks[i], score=float(scores[i])) for i in top]
