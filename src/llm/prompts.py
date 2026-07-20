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


_JUDGE_TEMPLATE = """You are grading a generated financial answer against a gold (reference) answer.

Question: {question}

Gold answer: {gold_answer}
Generated answer: {generated_answer}

Evaluate two things:

1. CORRECT: Does the gold answer's value/conclusion appear in the generated answer? Exact wording doesn't need to match, but the numeric value or yes/no direction must agree. A refusal or "context lacks the information" is NOT correct when gold contains a value.

2. GROUNDED: Is the reasoning that leads to the generated answer legitimate -- based on real figures and valid logic -- or is it hallucinated (invented numbers, unjustified assumptions, a non-sequitur)? A correct final value reached through fabricated reasoning is NOT grounded.

Respond in exactly this format:
CORRECT: yes or no
GROUNDED: yes or no
JUSTIFICATION: one or two sentences explaining both verdicts"""


def build_judge_prompt(question: str, gold_answer: str, generated_answer: str) -> str:
    """Assemble the prompt asking the judge LLM to compare a generated answer to gold."""
    return _JUDGE_TEMPLATE.format(
        question=question, gold_answer=gold_answer, generated_answer=generated_answer
    )
