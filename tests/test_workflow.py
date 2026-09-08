"""Fast unit tests for the deterministic CRAG workflow (no GPU, no LLM, no Qdrant)."""

from __future__ import annotations

import json
from pathlib import Path

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
        "k": 5,
        "doc_scoped": True,
    }
    return WorkflowConfig(**{**base, **overrides})


def _patch_llm(monkeypatch, grades: list[int], answer: str = "generated answer"):
    """Stub the LLM: one graded verdict per passage, then the generation call."""
    from src.workflow.schemas import ChunkGrade

    verdicts = iter(grades)
    monkeypatch.setattr(
        "src.workflow.nodes.generate_structured",
        lambda prompt, config, model: ChunkGrade(grade=next(verdicts)),
    )
    monkeypatch.setattr("src.workflow.nodes.generate", lambda prompt, config: answer)


def test_variant_name_covers_the_ablation_matrix():
    assert variant_name(_config()) == "advanced"
    assert variant_name(_config(grading={"enabled": True})) == "grading"
    assert variant_name(_config(calculator={"enabled": True})) == "calc"
    assert (
        variant_name(_config(grading={"enabled": True}, calculator={"enabled": True}))
        == "crag_full"
    )


def test_shipped_configs_load_and_select_distinct_cells():
    advanced = load_workflow_config("configs/workflow/advanced.yaml")
    grading = load_workflow_config("configs/workflow/grading.yaml")
    control = load_workflow_config("configs/workflow/control_top3.yaml")
    assert variant_name(advanced) == "advanced" and advanced.retriever == "dense"
    assert variant_name(grading) == "grading" and grading.grading.min_chunks == 3
    # The control row is the floor without the grader: top-3, no grading.
    assert variant_name(control) == "advanced" and control.k == 3


def test_disabled_grading_reduces_the_graph_to_retrieve_then_generate(monkeypatch):
    captured: dict[str, str] = {}

    def fake_generate(prompt, config):
        captured["prompt"] = prompt
        return "generated answer"

    monkeypatch.setattr("src.workflow.nodes.generate", fake_generate)
    monkeypatch.setattr(
        "src.workflow.nodes.generate_structured",
        lambda *a, **k: pytest.fail("grading is off, the grader must not be called"),
    )

    retriever = FakeRetriever([_chunk(i) for i in range(5)])
    result = answer_workflow("What was capex?", retriever, _config(k=3), doc_id="DOC_2022_10K")

    assert [c.chunk_id for c in result.sources] == ["c0", "c1", "c2"]
    assert result.llm_calls == 1 and result.n_kept_by_grade == 0
    assert "passage 2" in captured["prompt"]


def test_grading_keeps_passages_at_or_above_the_threshold(monkeypatch):
    _patch_llm(monkeypatch, grades=[3, 0, 2, 1, 0])
    retriever = FakeRetriever([_chunk(i) for i in range(5)])
    config = _config(k=5, grading={"enabled": True, "keep_threshold": 2, "min_chunks": 0})

    result = answer_workflow("q", retriever, config, doc_id="DOC_2022_10K")

    assert [c.chunk_id for c in result.sources] == ["c0", "c2"]  # grades 3 and 2
    assert result.n_kept_by_grade == 2 and result.n_kept_by_floor == 0
    assert result.grades == [3, 0, 2, 1, 0]  # recorded for offline calibration
    assert result.max_grade == 3 and result.low_confidence is False
    assert result.llm_calls == 6  # five gradings plus the generation


def test_floor_tops_up_without_ever_duplicating_a_kept_passage(monkeypatch):
    """The kept passage is also top-ranked: the floor must skip it, not re-add it."""
    _patch_llm(monkeypatch, grades=[3, 0, 0, 0, 0])  # only c0 kept, and c0 ranks first
    retriever = FakeRetriever([_chunk(i) for i in range(5)])
    config = _config(k=5, grading={"enabled": True, "min_chunks": 3})

    result = answer_workflow("q", retriever, config, doc_id="DOC_2022_10K")

    ids = [c.chunk_id for c in result.sources]
    assert ids == ["c0", "c1", "c2"]  # c0 once, then the next best two
    assert len(ids) == len(set(ids)) == 3
    assert result.n_kept_by_grade == 1 and result.n_kept_by_floor == 2


def test_min_chunks_expresses_the_floor_policy(monkeypatch):
    """0 disables the floor, 1 only rescues an empty selection, 3 always guarantees three."""

    def kept_with(minimum, grades):
        _patch_llm(monkeypatch, grades=grades)
        retriever = FakeRetriever([_chunk(i) for i in range(5)])
        config = _config(k=5, grading={"enabled": True, "min_chunks": minimum})
        return [c.chunk_id for c in answer_workflow("q", retriever, config).sources]

    assert kept_with(0, [0, 0, 0, 0, 0]) == []  # no floor: an empty context is allowed
    assert kept_with(1, [0, 0, 0, 0, 0]) == ["c0"]  # rescue only
    assert kept_with(1, [3, 0, 0, 0, 0]) == ["c0"]  # already non-empty, floor idle
    assert kept_with(3, [0, 0, 0, 0, 0]) == ["c0", "c1", "c2"]


def test_context_follows_the_retriever_ranking_whatever_the_provenance(monkeypatch):
    """A late passage kept by the grader must not jump ahead of earlier floor passages."""
    _patch_llm(monkeypatch, grades=[0, 0, 0, 0, 3])  # only the last-ranked passage kept
    retriever = FakeRetriever([_chunk(i) for i in range(5)])
    config = _config(k=5, grading={"enabled": True, "min_chunks": 3})

    ids = [c.chunk_id for c in answer_workflow("q", retriever, config).sources]

    assert ids == ["c0", "c1", "c4"]  # ranking order, not selection order


def test_runner_writes_every_field_the_answer_exposes(monkeypatch, tmp_path):
    """End-to-end through the runner, which unit tests on answer_workflow never touch.

    Regression guard: the runner once referenced fields WorkflowAnswer no longer had,
    and neither ruff nor the graph tests caught it -- the failure only surfaced after a
    retriever and a model had been loaded, minutes into a 150-question run.
    """
    from src.evaluation.schema import QAItem
    from src.workflow import runner as runner_mod

    qa = QAItem(
        id="financebench_id_00001",
        question="q",
        answer="gold",
        company="C",
        doc_name="DOC_2022_10K",
        question_type="metrics-generated",
        evidence=[],
    )
    monkeypatch.setattr(runner_mod, "load_golden_set", lambda path: [qa])
    monkeypatch.setattr(
        runner_mod,
        "build_retriever",
        lambda name, config: FakeRetriever([_chunk(i) for i in range(5)]),
    )
    monkeypatch.setattr(runner_mod, "_output_path", lambda config: tmp_path / "out.jsonl")
    _patch_llm(monkeypatch, grades=[3, 2, 0, 0, 0])

    out = runner_mod.run(_config(k=5, grading={"enabled": True}))

    record = json.loads(Path(out).read_text().splitlines()[0])
    assert record["id"] == qa.id and record["generated_answer"] == "generated answer"
    assert record["grades"] == [3, 2, 0, 0, 0]
    assert record["n_kept_by_grade"] == 2 and record["n_kept_by_floor"] == 1  # floor tops to 3
    assert record["max_grade"] == 3 and record["low_confidence"] is False
    assert record["llm_calls"] == 6 and len(record["sources"]) == 3


def test_context_is_trimmed_from_the_bottom_to_fit_the_pinned_num_ctx(monkeypatch):
    """Trimming must follow num_ctx, and drop the weakest passages, not the strongest.

    Ollama truncates an oversized prompt from the top, which would eat the grounding
    instructions first and the best-ranked passages next. Trimming here inverts that.
    """
    _patch_llm(monkeypatch, grades=[3] * 10)  # two runs of five passages

    def sources_with(num_ctx):
        retriever = FakeRetriever([_chunk(i) for i in range(5)])
        config = _config(k=5, grading={"enabled": True, "min_chunks": 0})
        config.llm.num_ctx = num_ctx
        config.llm.max_tokens = 16
        return answer_workflow("q", retriever, config)

    roomy = sources_with(32768)
    assert [c.chunk_id for c in roomy.sources] == ["c0", "c1", "c2", "c3", "c4"]
    assert roomy.n_dropped_to_fit == 0

    tight = sources_with(2048)
    kept = [c.chunk_id for c in tight.sources]
    assert kept == ["c0", "c1", "c2", "c3", "c4"][: len(kept)]  # a prefix: weakest dropped
    assert len(kept) <= 5 and tight.n_dropped_to_fit == 5 - len(kept)


def test_a_single_oversized_passage_still_reaches_the_prompt(monkeypatch):
    """An empty context would be worse than one passage the window cannot hold."""
    _patch_llm(monkeypatch, grades=[3])
    huge = Chunk(chunk_id="big", doc_id="D", page=1, text="x" * 40_000, n_tokens=13_000)
    config = _config(k=1, grading={"enabled": True, "min_chunks": 0})
    config.llm.num_ctx = 2048

    result = answer_workflow("q", FakeRetriever([huge]), config)

    assert [c.chunk_id for c in result.sources] == ["big"]
