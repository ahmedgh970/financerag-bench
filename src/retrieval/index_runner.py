"""CLI runner: index a chunks JSONL file into a Qdrant collection.

Reads the chunks produced by the ingestion stage, embeds them and upserts them
into a Qdrant collection sized for the chosen embedding model.

Usage:
    python -m src.retrieval.index_runner --config configs/index_docling_hybrid.yaml
"""

from __future__ import annotations

import argparse

from src.ingestion.storage import read_chunks
from src.retrieval.config import IndexConfig, load_index_config
from src.retrieval.embeddings import embedding_dim
from src.retrieval.qdrant_store import create_collection, get_client, upsert_chunks


def run(config: IndexConfig) -> str:
    """Index ``config.chunks_path`` into the Qdrant collection. Returns its name."""
    chunks = list(read_chunks(config.chunks_path))
    if not chunks:
        raise SystemExit(f"No chunks found in {config.chunks_path!r}")

    client = get_client(config.qdrant_location)
    dim = embedding_dim(config.embedding_model)
    create_collection(client, config.collection_name, dim, recreate=config.recreate)
    n = upsert_chunks(
        client,
        config.collection_name,
        chunks,
        model_name=config.embedding_model,
        batch_size=config.batch_size,
    )

    print(
        f"Indexed {n} chunks -> collection '{config.collection_name}' "
        f"({dim}-d, {config.embedding_model}) @ {config.qdrant_location}"
    )
    return config.collection_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a chunks JSONL file into Qdrant.")
    parser.add_argument("--config", required=True, help="Path to an indexing YAML config.")
    run(load_index_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
