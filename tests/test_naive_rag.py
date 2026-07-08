"""Fast unit test for the naive RAG pipeline (mocked retriever + LLM, no real calls)."""

from unittest.mock import patch

from src.ingestion.schema import Chunk
from src.llm.config import LLMConfig
from src.rag.naive import answer
from src.retrieval.base import ScoredChunk


class _FakeRetriever:
    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None) -> list[ScoredChunk]:
        chunk = Chunk(chunk_id="d::0", doc_id="d", text="Capex was $20M.", page=1)
        return [ScoredChunk(chunk=chunk, score=0.9)]


def test_answer_wires_retriever_prompt_and_llm():
    with patch(
        "src.rag.naive.generate", return_value="Capex was $20M. (Source 1)"
    ) as mock_generate:
        result = answer("What was the capex?", _FakeRetriever(), LLMConfig())

    assert result.answer == "Capex was $20M. (Source 1)"
    assert [c.text for c in result.sources] == ["Capex was $20M."]
    assert result.latency_s >= 0

    mock_generate.assert_called_once()
    prompt_arg = mock_generate.call_args[0][0]
    assert "What was the capex?" in prompt_arg
    assert "Capex was $20M." in prompt_arg
