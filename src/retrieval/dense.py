"""Dense retrieval: embed a query and fetch the nearest chunks from Qdrant.

This is the mirror of indexing: a query is embedded with the same model, Qdrant
returns the closest vectors, and each hit's payload is rebuilt into a ``Chunk``
with its similarity score.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from src.ingestion.schema import Chunk
from src.retrieval.embeddings import DEFAULT_MODEL, embed_texts


@dataclass
class ScoredChunk:
    """A retrieved chunk and its similarity score (higher = closer)."""

    chunk: Chunk
    score: float


def _chunk_from_payload(payload: dict) -> Chunk:
    """Rebuild a Chunk from the payload stored at index time."""
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        text=payload["text"],
        page=payload["page"],
        metadata=payload.get("metadata", {}),
    )


class DenseRetriever:
    """Retrieve the top-k chunks closest to a query from one Qdrant collection.

    The ``model_name`` must match the one used to index the collection, otherwise
    the query and stored vectors live in different spaces.
    """

    def __init__(self, client: QdrantClient, collection_name: str, model_name: str = DEFAULT_MODEL):
        self.client = client
        self.collection_name = collection_name
        self.model_name = model_name

    def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]:
        vector = embed_texts([query], model_name=self.model_name)[0].tolist()
        hits = self.client.query_points(self.collection_name, query=vector, limit=k).points
        return [ScoredChunk(chunk=_chunk_from_payload(h.payload), score=h.score) for h in hits]
