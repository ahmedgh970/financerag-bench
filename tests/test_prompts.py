"""Fast unit test for prompt assembly (pure string formatting, no LLM call)."""

from src.ingestion.schema import Chunk
from src.llm.prompts import build_prompt


def test_build_prompt_includes_question_and_numbered_sources():
    chunks = [
        Chunk(chunk_id="d::0", doc_id="d", text="Revenue was $100M.", page=1),
        Chunk(chunk_id="d::1", doc_id="d", text="Capex was $20M.", page=2),
    ]

    prompt = build_prompt("What was the capex?", chunks)

    assert "What was the capex?" in prompt
    assert "[Source 1] Revenue was $100M." in prompt
    assert "[Source 2] Capex was $20M." in prompt
