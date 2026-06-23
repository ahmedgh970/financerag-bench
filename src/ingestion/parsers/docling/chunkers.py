"""Docling-native chunkers. Each consumes a DoclingDocument, emits list[Chunk].

Docling ships exactly two chunkers (verified against ``docling.chunking``):
``HybridChunker`` and ``HierarchicalChunker``.
"""

from __future__ import annotations

from docling.chunking import BaseChunker, HierarchicalChunker, HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from src.ingestion.schema import Chunk

DEFAULT_TOKENIZER_MODEL = "BAAI/bge-m3"

# Cache HybridChunker instances by their params: building one loads a tokenizer,
# so reuse it across documents sharing the same config. Keyed on params so a
# benchmark that varies (e.g.) max_tokens gets the right instance each time.
_hybrid_cache: dict[tuple, HybridChunker] = {}


def _get_hybrid_chunker(params: dict) -> HybridChunker:
    """Build (and cache) a HybridChunker for ``params``.

    ``max_tokens`` (and optional ``tokenizer_model``, default BGE-M3) are turned
    into a ``HuggingFaceTokenizer`` so the token budget is aligned with the model
    that will embed the chunks — Docling deprecated passing ``max_tokens`` direct.
    Without ``max_tokens`` nor ``tokenizer_model``, Docling's default tokenizer
    (all-MiniLM-L6-v2, 256-token budget) is used. Remaining params pass through.
    """
    key = tuple(sorted(params.items()))
    if key not in _hybrid_cache:
        kwargs = dict(params)
        max_tokens = kwargs.pop("max_tokens", None)
        tokenizer_model = kwargs.pop("tokenizer_model", None)
        if max_tokens is not None or tokenizer_model is not None:
            kwargs["tokenizer"] = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_model or DEFAULT_TOKENIZER_MODEL),
                max_tokens=max_tokens,
            )
        _hybrid_cache[key] = HybridChunker(**kwargs)
    return _hybrid_cache[key]


def _to_chunks(chunker: BaseChunker, doc, doc_id: str, name: str) -> list[Chunk]:
    """Run ``chunker`` over ``doc`` and map each Docling chunk to a Chunk.

    Text is contextualized (section headings prepended) so each chunk is
    self-contained for retrieval; the page is the first page the chunk spans,
    read from element provenance.

    Metadata attached per chunk:
        - ``headings`` | Optional[list[str]]: section headings the chunk belongs to, else ``None``.
        - ``labels``: element types composing the chunk (e.g. "text", "table",
            "section_header") — lets downstream spot table-bearing chunks.
        - ``captions`` | Optional[list[str]]: table/figure captions when present, else ``None``.
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
                metadata={
                    "parser": "docling",
                    "chunker": name,
                    "headings": dc.meta.headings,
                    "labels": [str(it.label) for it in dc.meta.doc_items],
                    "captions": dc.meta.captions,
                },
            )
        )
    return chunks


def hybrid_chunk(doc, doc_id: str, **params) -> list[Chunk]:
    """Token-aware, structure-aware chunking via Docling's ``HybridChunker``.

    Splits along the document structure, then merges/splits to respect a token
    budget. Useful ``params`` (from ``chunker_params`` in the YAML):
        - ``max_tokens``: token budget per chunk (e.g. 512). The budget only makes
            sense in the tokenizer of the model that will embed the chunks.
        - ``tokenizer_model``: HF model whose tokenizer counts the budget
            (default ``BAAI/bge-m3``); only applied when a budget is set.
    With neither, Docling's default tokenizer (all-MiniLM-L6-v2, 256) is used.
    """
    return _to_chunks(_get_hybrid_chunker(params), doc, doc_id, "hybrid")


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
