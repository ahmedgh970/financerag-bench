"""Docling-native chunkers. Each consumes a DoclingDocument, emits list[Chunk].

Docling ships exactly two chunkers (verified against ``docling.chunking``):
``HybridChunker`` and ``HierarchicalChunker``.
"""

from __future__ import annotations

from docling.chunking import BaseChunker, HierarchicalChunker, HybridChunker

from src.ingestion.schema import Chunk

# Cache HybridChunker instances by their params: building one loads a tokenizer,
# so reuse it across documents sharing the same config. Keyed on params so a
# benchmark that varies (e.g.) max_tokens gets the right instance each time.
_hybrid_cache: dict[tuple, HybridChunker] = {}


def _get_hybrid_chunker(**params) -> HybridChunker:
    key = tuple(sorted(params.items()))
    if key not in _hybrid_cache:
        _hybrid_cache[key] = HybridChunker(**params)
    return _hybrid_cache[key]


def _to_chunks(chunker: BaseChunker, doc, doc_id: str, name: str) -> list[Chunk]:
    """Run ``chunker`` over ``doc`` and map each Docling chunk to a Chunk.

    Text is contextualized (section headings prepended) so each chunk is
    self-contained for retrieval; the page is the first page the chunk spans,
    read from element provenance.
    """
    chunks: list[Chunk] = []
    for i, dc in enumerate(chunker.chunk(doc)):
        pages = [prov.page_no for item in dc.meta.doc_items for prov in item.prov]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{i}",
                doc_id=doc_id,
                text=chunker.contextualize(dc),
                page=min(pages) if pages else 1,
                metadata={"parser": "docling", "chunker": name, "headings": dc.meta.headings},
            )
        )
    return chunks


def hybrid_chunk(doc, doc_id: str, **params) -> list[Chunk]:
    """Token-aware, structure-aware chunking via Docling's ``HybridChunker``.

    Splits along the document structure, then merges/splits to respect a token
    budget. ``params`` are forwarded to ``HybridChunker``.

    To set a custom token budget, pass a ``tokenizer`` built from the embedding
    model (``HuggingFaceTokenizer(tokenizer=..., max_tokens=...)``); the token
    budget only makes sense when aligned with the model that will embed the
    chunks. Passing ``max_tokens`` directly is deprecated by Docling. With no
    params, Docling's default tokenizer is used.
    """
    return _to_chunks(_get_hybrid_chunker(**params), doc, doc_id, "hybrid")


def hierarchical_chunk(doc, doc_id: str, **params) -> list[Chunk]:
    """Structure-only chunking via Docling's ``HierarchicalChunker``.

    Chunks follow the document hierarchy (sections / subsections) without a token
    budget, so they are finer-grained than ``hybrid_chunk``. ``params`` are
    forwarded to ``HierarchicalChunker``. Unlike ``HybridChunker`` it loads no
    tokenizer, so it is cheap to construct and not cached.
    """
    return _to_chunks(HierarchicalChunker(**params), doc, doc_id, "hierarchical")


CHUNKERS = {
    "hybrid": hybrid_chunk,
    "hierarchical": hierarchical_chunk,
}
