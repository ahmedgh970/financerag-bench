"""Fast unit test for RerankedRetriever (fake base + fake reranker, no model)."""

from src.ingestion.schema import Chunk
from src.retrieval.base import ScoredChunk
from src.retrieval.reranked import RerankedRetriever


class _FakeBase:
    """Returns a fixed, deliberately mis-ordered candidate list."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=c, score=0.0) for c in self.chunks[:k]]


class _FakeReranker:
    """Scores chunks by a hand-picked mapping, ignoring the base's order/score."""

    def __init__(self, scores_by_id: dict[str, float]):
        self.scores_by_id = scores_by_id

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [self.scores_by_id[t] for t in texts]


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", text=text, page=1)


def test_reranked_reorders_by_reranker_score():
    chunks = [_chunk("a", "low"), _chunk("b", "high"), _chunk("c", "mid")]
    base = _FakeBase(chunks)
    reranker = _FakeReranker({"low": 0.1, "high": 0.9, "mid": 0.5})

    retriever = RerankedRetriever(base=base, reranker=reranker, prefetch=3)
    results = retriever.retrieve("q", k=2)

    assert [sc.chunk.chunk_id for sc in results] == ["b", "c"]
    assert results[0].score == 0.9


def test_reranked_empty_base_returns_empty():
    retriever = RerankedRetriever(base=_FakeBase([]), reranker=_FakeReranker({}), prefetch=5)
    assert retriever.retrieve("q", k=3) == []
