"""Dense retrieval: embed a query and fetch the nearest chunks from Qdrant.

This is the mirror of indexing: a query is embedded with the same model, Qdrant
returns the closest vectors, and each hit's payload is rebuilt into a ``Chunk``
with its similarity score.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.retrieval.base import ScoredChunk
from src.vectorstore.embeddings import embed_texts
from src.vectorstore.qdrant_store import chunk_from_payload


class DenseRetriever:
    """Retrieve the top-k chunks closest to a query from one Qdrant collection.

    The ``model_name`` must match the one used to index the collection, otherwise
    the query and stored vectors live in different spaces.
    """

    def __init__(self, client: QdrantClient, collection_name: str, model_name: str):
        self.client = client
        self.collection_name = collection_name
        self.model_name = model_name

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None) -> list[ScoredChunk]:
        vector = embed_texts([query], model_name=self.model_name)[0].tolist()
        query_filter = (
            Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            if doc_id
            else None
        )
        hits = self.client.query_points(
            self.collection_name, query=vector, limit=k, query_filter=query_filter
        ).points
        return [ScoredChunk(chunk=chunk_from_payload(h.payload), score=h.score) for h in hits]
