"""Fast unit test for Reciprocal Rank Fusion (no retriever/corpus needed)."""

from src.ingestion.schema import Chunk
from src.retrieval.base import ScoredChunk
from src.retrieval.hybrid import reciprocal_rank_fusion


def _sc(cid: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id="d", text="", page=1), score=0.0)


def test_rrf_rewards_agreement_across_lists():
    dense = [_sc("c1"), _sc("c2"), _sc("c3")]
    sparse = [_sc("c2"), _sc("c3"), _sc("c4")]

    fused = reciprocal_rank_fusion([dense, sparse], k=4, rrf_k=60)
    order = [sc.chunk.chunk_id for sc in fused]

    # c2 and c3 appear in both lists -> ranked above c1/c4 that appear once.
    assert order[:2] == ["c2", "c3"]
    assert set(order) == {"c1", "c2", "c3", "c4"}


def test_rrf_truncates_to_k():
    dense = [_sc("a"), _sc("b"), _sc("c")]
    assert len(reciprocal_rank_fusion([dense], k=2)) == 2
