"""Text embeddings via a sentence-transformers model.

The default model is BGE-M3 (1024-dim, multilingual), the project's reference
embedder. Models are lazy-loaded and cached per name, and use the GPU when
available. Embeddings are L2-normalized so cosine similarity == dot product.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_models: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _models:
        # Force safetensors weights: torch 2.5.1 (pinned for the cu121 GPU stack)
        # is below the 2.6 that transformers now requires to load legacy .bin
        # checkpoints (CVE-2025-32434). safetensors loading is exempt.
        # fp16 weights ~halve VRAM and speed up the GPU forward; negligible
        # quality loss for inference (the GPU upcasts internally as needed).
        _models[model_name] = SentenceTransformer(
            model_name,
            model_kwargs={"use_safetensors": True, "torch_dtype": "float16"},
        )
    return _models[model_name]


def embedding_dim(model_name: str) -> int:
    """Output dimensionality of ``model_name`` (e.g. 1024 for BGE-M3)."""
    return _get_model(model_name).get_embedding_dimension()


def embed_texts(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Embed ``texts`` into an L2-normalized ``(len(texts), dim)`` array."""
    model = _get_model(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
