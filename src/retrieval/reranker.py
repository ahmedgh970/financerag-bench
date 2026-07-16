"""Cross-encoder reranking: score a query against a small candidate set.

Unlike a bi-encoder (query and chunk embedded independently, then compared by
cosine), a cross-encoder scores the pair jointly with full cross-attention
between the two texts. That is far more precise but too slow to run over an
entire collection, so it is only used to re-score the shortlist a first-stage
retriever already narrowed down.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

_models: dict[str, CrossEncoder] = {}


def _get_model(model_name: str) -> CrossEncoder:
    if model_name not in _models:
        _models[model_name] = CrossEncoder(model_name)
    return _models[model_name]


class Reranker:
    """Score (query, text) pairs with a cross-encoder."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, in the same order as ``texts``."""
        model = _get_model(self.model_name)
        pairs = [(query, text) for text in texts]
        return model.predict(pairs).tolist()
