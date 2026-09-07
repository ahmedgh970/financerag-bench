"""Prompts for the workflow's judgement nodes.

These ask the model to *judge*, never to control the flow: the graph reads the
judgement and decides which edge to follow. Kept apart from the generation prompt
(:mod:`src.llm.prompts`), which the baseline shares.
"""

from __future__ import annotations

from src.ingestion.schema import Chunk


def build_grading_prompt(question: str, chunks: list[Chunk]) -> str:
    """Ask for a relevance verdict on each passage, in order.

    The criterion is deliberately "could help answer", not "contains the answer": a
    ratio is often computed from two line items, neither of which answers the
    question on its own.
    """
    passages = "\n\n".join(f"[{i + 1}] (page {c.page}) {c.text}" for i, c in enumerate(chunks))
    return (
        "You are grading retrieved passages from an SEC filing for relevance.\n\n"
        f"QUESTION: {question}\n\n"
        f"PASSAGES:\n{passages}\n\n"
        "For each passage, decide whether it could help answer the question -- either "
        "because it states the answer, or because it holds a figure the answer is "
        "computed from. A passage about the right company but the wrong topic, period "
        "or statement is NOT relevant.\n"
        "Return one true/false per passage, in the same order. If none of them helps, "
        "return false for all of them."
    )


def build_single_grading_prompt(question: str, chunk: Chunk) -> str:
    """Ask for a relevance verdict on ONE passage.

    Same criterion as the batch prompt, but the model judges a single short passage,
    which is a far easier task than tracking twenty of them in one 15k-token prompt.
    """
    return (
        "You are grading one passage from an SEC filing for relevance.\n\n"
        f"QUESTION: {question}\n\n"
        f"PASSAGE (page {chunk.page}):\n{chunk.text}\n\n"
        "Does this passage help answer the question -- either because it states the "
        "answer, or because it holds a figure the answer is computed from? A passage "
        "about the right company but the wrong topic, period or statement is NOT "
        "relevant. Answer true or false."
    )


def build_rewrite_prompt(question: str, previous_queries: list[str]) -> str:
    """Ask for a reformulated search query after a retrieval attempt found nothing."""
    tried = "\n".join(f"- {q}" for q in previous_queries)
    return (
        "A vector search over a single SEC filing returned no relevant passage.\n\n"
        f"ORIGINAL QUESTION: {question}\n\n"
        f"QUERIES ALREADY TRIED (all failed):\n{tried}\n\n"
        "Write ONE new search query that is more likely to match the wording of the "
        "filing itself. Prefer the exact line-item and statement vocabulary a 10-K "
        "uses (for example 'purchases of property, plant and equipment' rather than "
        "'capex'). Do not repeat a query already tried.\n"
        "Answer with the query text only, nothing else."
    )
