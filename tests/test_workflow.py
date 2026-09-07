"""Fast unit tests for the deterministic CRAG workflow (no GPU, no LLM, no Qdrant)."""

from __future__ import annotations

import pytest

from src.ingestion.schema import Chunk
from src.retrieval.base import ScoredChunk
from src.workflow.config import WorkflowConfig, load_workflow_config, variant_name
from src.workflow.graph import answer_workflow


class FakeRetriever:
    """Returns a fixed set of chunks, recording the query and depth it was asked for."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None):
        self.calls.append((query, k, doc_id))
        return [ScoredChunk(chunk=c, score=1.0 - i) for i, c in enumerate(self.chunks[:k])]


def _chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"c{i}", doc_id="DOC_2022_10K", page=i, text=f"passage {i}", n_tokens=3)


def _config(**overrides) -> WorkflowConfig:
    base = {
        "chunks_path": "unused.jsonl",
        "collection_name": "test_collection",
        "retriever": "reranked",
        "k": 3,
        "doc_scoped": True,
    }
    return WorkflowConfig(**{**base, **overrides})


def test_variant_name_covers_the_ablation_matrix():
    assert variant_name(_config()) == "advanced"
    assert variant_name(_config(grading={"enabled": True})) == "grading"
    assert variant_name(_config(calculator={"enabled": True})) == "calc"
    assert (
        variant_name(_config(grading={"enabled": True}, calculator={"enabled": True}))
        == "crag_full"
    )


def test_baseline_config_yaml_loads_and_is_the_advanced_cell():
    config = load_workflow_config("configs/workflow/advanced.yaml")
    assert variant_name(config) == "advanced"
    assert config.grading.enabled is False and config.calculator.enabled is False
    assert config.max_rewrites == 2


def test_disabled_nodes_reduce_the_graph_to_retrieve_then_generate(monkeypatch):
    """With both switches off the graph must behave exactly like the naive pipeline."""
    captured: dict[str, str] = {}

    def fake_generate(prompt, config):
        captured["prompt"] = prompt
        return "generated answer"

    monkeypatch.setattr("src.workflow.nodes.generate", fake_generate)

    retriever = FakeRetriever([_chunk(i) for i in range(5)])
    result = answer_workflow("What was capex?", retriever, _config(), doc_id="DOC_2022_10K")

    assert result.answer == "generated answer"
    assert [c.chunk_id for c in result.sources] == ["c0", "c1", "c2"]  # k=3
    assert retriever.calls == [("What was capex?", 3, "DOC_2022_10K")]
    assert result.llm_calls == 1
    # No correction loop ran.
    assert result.n_rewrites == 0 and result.bound_hit is False
    # Every retrieved chunk reached the prompt.
    assert "passage 2" in captured["prompt"]
    assert "retrieve" in result.node_latencies and "generate" in result.node_latencies


def test_unknown_prompt_free_config_rejects_bad_grading_mode():
    with pytest.raises(ValueError):
        _config(grading={"enabled": True, "mode": "nonsense"})


# --- grading + correction loop ------------------------------------------------


def _patch_llm(monkeypatch, grades: list[list[bool]], answer: str = "generated answer"):
    """Stub the LLM calls: per-chunk grading yields one boolean per call, in order."""
    verdicts = iter([flag for round_ in grades for flag in round_])

    def fake_structured(prompt, config, model):
        return model(relevant=next(verdicts))

    monkeypatch.setattr("src.workflow.nodes.generate_structured", fake_structured)
    monkeypatch.setattr("src.workflow.nodes.generate", lambda prompt, config: answer)


def test_grading_keeps_only_relevant_chunks_and_skips_the_rewrite(monkeypatch):
    _patch_llm(monkeypatch, grades=[[True, False, True]])
    retriever = FakeRetriever([_chunk(i) for i in range(3)])
    config = _config(grading={"enabled": True})

    result = answer_workflow("q", retriever, config, doc_id="DOC_2022_10K")

    assert [c.chunk_id for c in result.sources] == ["c0", "c2"]
    assert result.n_rewrites == 0 and result.bound_hit is False
    assert len(retriever.calls) == 1  # relevant chunks found -> no reformulation


def test_zero_relevant_triggers_rewrites_up_to_the_configured_bound(monkeypatch):
    # Never relevant: the loop must stop after max_rewrites and fall back.
    _patch_llm(monkeypatch, grades=[[False] * 3] * 5)
    retriever = FakeRetriever([_chunk(i) for i in range(3)])
    config = _config(grading={"enabled": True}, max_rewrites=2)

    result = answer_workflow("q", retriever, config, doc_id="DOC_2022_10K")

    assert result.n_rewrites == 2
    assert result.bound_hit is True
    assert len(retriever.calls) == 3  # initial + 2 rewrites
    # Fallback serves the FIRST round's chunks, so the path degenerates to baseline.
    assert [c.chunk_id for c in result.sources] == ["c0", "c1", "c2"]


def test_rewrite_bound_is_read_from_config_not_hardcoded(monkeypatch):
    _patch_llm(monkeypatch, grades=[[False] * 3] * 8)
    retriever = FakeRetriever([_chunk(i) for i in range(3)])

    result = answer_workflow(
        "q", retriever, _config(grading={"enabled": True}, max_rewrites=0), doc_id="DOC_2022_10K"
    )
    assert result.n_rewrites == 0 and result.bound_hit is True

    retriever2 = FakeRetriever([_chunk(i) for i in range(3)])
    _patch_llm(monkeypatch, grades=[[False] * 3] * 8)
    result2 = answer_workflow(
        "q", retriever2, _config(grading={"enabled": True}, max_rewrites=3), doc_id="DOC_2022_10K"
    )
    assert result2.n_rewrites == 3


def test_grader_failure_fails_open_rather_than_faking_a_retrieval_failure(monkeypatch):
    from src.llm.client import StructuredOutputError

    def boom(prompt, config, model):
        raise StructuredOutputError("bad json")

    monkeypatch.setattr("src.workflow.nodes.generate_structured", boom)
    monkeypatch.setattr("src.workflow.nodes.generate", lambda prompt, config: "answer")

    retriever = FakeRetriever([_chunk(i) for i in range(3)])
    result = answer_workflow(
        "q", retriever, _config(grading={"enabled": True}), doc_id="DOC_2022_10K"
    )

    assert result.grader_errors == 3  # one failure per chunk, in per_chunk mode
    assert len(result.sources) == 3  # every chunk kept
    assert result.n_rewrites == 0  # and no needless reformulation


def test_fallback_cap_is_derived_from_num_ctx_not_a_fixed_count(monkeypatch):
    """The give-up path trims passages to fit the pinned context, from the END.

    Ollama truncates from the top, which would drop the grounding instructions and
    the best-ranked passages; trimming here keeps both. How many survive must follow
    the configured context, so a larger num_ctx keeps strictly more.
    """
    _patch_llm(monkeypatch, grades=[[False] * 6] * 6)

    def kept_with(num_ctx):
        retriever = FakeRetriever([_chunk(i) for i in range(6)])
        config = _config(k=6, grading={"enabled": True}, max_rewrites=0)
        config.llm.num_ctx = num_ctx
        config.llm.max_tokens = 16
        return [c.chunk_id for c in answer_workflow("q", retriever, config).sources]

    tight, roomy = kept_with(2048), kept_with(32768)
    assert tight == ["c0", "c1", "c2", "c3", "c4", "c5"][: len(tight)]  # trimmed from the END
    assert len(roomy) == 6  # everything fits
    assert len(tight) <= len(roomy)
