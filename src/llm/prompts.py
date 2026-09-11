"""Prompt template assembling a question and retrieved chunks into an LLM prompt.

Financial questions often need a precise number, so the prompt asks
explicitly for the exact figure and its unit, grounds the answer in the
given context only, and asks which source it came from.
"""

from __future__ import annotations

from src.ingestion.schema import Chunk

_TEMPLATE = """You are a financial analyst assistant. Answer the question using ONLY the context below — do not use outside knowledge.

If the answer is a number, state the exact figure with its unit (e.g. "$1,577 million", "12.4%"). Cite which source (e.g. "Source 2") the answer comes from. If the context does not contain the answer, say so explicitly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def _format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[Source {i + 1}] {c.text}" for i, c in enumerate(chunks))


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Assemble a grounded QA prompt from the question and its retrieved chunks."""
    return _TEMPLATE.format(context=_format_context(chunks), question=question)
