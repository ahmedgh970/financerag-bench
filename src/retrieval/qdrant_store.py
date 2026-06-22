"""Connection to Qdrant — a local on-disk store or a remote server.

The rest of the retrieval code talks to Qdrant only through the client returned
here, so it never needs to know where Qdrant actually runs. Switching from the
local dev store to a Docker/cloud server is a one-line config change.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from src.ingestion.schema import Chunk
from src.retrieval.embeddings import DEFAULT_MODEL, embed_texts

# Default local store: an on-disk folder (persisted across runs, gitignored).
DEFAULT_LOCATION = "data/vectorstores/qdrant"


def get_client(location: str = DEFAULT_LOCATION) -> QdrantClient:
    """Return a configured ``QdrantClient``.

    ``location`` selects where Qdrant lives:
        - an ``http(s)`` URL -> connect to a Qdrant server (Docker / cloud)
        - ``":memory:"`` -> ephemeral in-process store (used by fast tests)
        - any other path -> on-disk local store, persisted across runs
    """
    if location.startswith("http"):
        return QdrantClient(url=location)
    if location == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=location)


def create_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> None:
    """Create a collection of ``dim``-dimensional vectors using cosine distance.

    Cosine matches our L2-normalized embeddings. With ``recreate=True`` an
    existing collection is dropped first (start from scratch); otherwise an
    already-existing collection is left untouched.
    """
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID from a chunk_id.

    Qdrant point ids must be ints or UUIDs, not arbitrary strings like
    ``"doc::3"``. Using a deterministic UUID means re-upserting the same chunk
    updates its point instead of creating a duplicate.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upsert_chunks(
    client: QdrantClient,
    name: str,
    chunks: Iterable[Chunk],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 128,
) -> int:
    """Embed ``chunks`` and upsert them as points into collection ``name``.

    Each chunk becomes one point: vector = embedding of ``chunk.text``, payload =
    the chunk fields needed to rebuild it on retrieval. Processed in batches to
    bound memory. Returns the number of points written.
    """
    chunks = list(chunks)
    desc = f"index [{name}]"
    for start in tqdm(range(0, len(chunks), batch_size), desc=desc, unit="batch"):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts([c.text for c in batch], model_name=model_name)
        points = [
            PointStruct(
                id=_point_id(c.chunk_id),
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "page": c.page,
                    "metadata": c.metadata,
                },
            )
            for i, c in enumerate(batch)
        ]
        client.upsert(collection_name=name, points=points)
    return len(chunks)


def chunk_from_payload(payload: dict) -> Chunk:
    """Rebuild a Chunk from a point payload (inverse of the payload built above)."""
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        text=payload["text"],
        page=payload["page"],
        metadata=payload.get("metadata", {}),
    )
