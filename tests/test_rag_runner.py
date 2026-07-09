"""Fast unit test for the RAG runner's resume logic (no retriever/LLM calls)."""

import json

from src.rag.config import RagConfig
from src.rag.runner import _answered_ids, _output_path


def test_answered_ids_empty_when_file_missing(tmp_path):
    assert _answered_ids(tmp_path / "missing.jsonl") == set()


def test_answered_ids_reads_existing_records(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n", encoding="utf-8"
    )

    assert _answered_ids(path) == {"a", "b"}


def test_output_path_encodes_retriever_model_and_k():
    cfg = RagConfig(
        chunks_path="x.jsonl",
        collection_name="c",
        retriever="reranked",
        k=5,
        llm={"model": "groq/llama-3.1-8b-instant"},
    )

    path = _output_path(cfg)

    assert path.name == "reranked_groq_llama-3.1-8b-instant_k5.jsonl"
