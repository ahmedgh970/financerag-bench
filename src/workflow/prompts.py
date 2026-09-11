"""Prompts for the workflow's judgement nodes.

These ask the model to *judge*, never to control the flow: the graph reads the
judgement and decides what happens. Kept apart from the generation prompt
(:mod:`src.llm.prompts`), which the baseline shares.
"""

from __future__ import annotations

from src.ingestion.schema import Chunk

_GRADE_SCALE = """3 = the passage is dedicated to the question and states the answer.
2 = the passage holds part of what the answer is built from, even if it is buried in a
    table or stated indirectly. A passage carrying ONE of the line items a computed
    metric needs -- a revenue, a cost, an asset balance -- is at least a 2.
1 = the passage is on a related topic but carries nothing the answer is built from.
0 = the passage is unrelated: wrong topic, wrong period, or wrong statement."""


def build_grading_prompt(question: str, chunk: Chunk) -> str:
    """Ask for a graded relevance judgement on ONE passage.

    The four points come from UMBRELA, the relevance assessor adopted by the TREC RAG
    track, which reports Kendall tau above 0.87 against human assessors. The one
    addition is the financial clause inside grade 2: a ratio is computed from two line
    items, and neither answers the question on its own -- judged by a yes/no criterion
    both get rejected, which is precisely how a grader loses the evidence it needs.
    """
    return (
        "You are assessing how relevant a passage from an SEC filing is to a financial "
        "question, on a four-point scale.\n\n"
        f"QUESTION: {question}\n\n"
        f"PASSAGE (page {chunk.page}):\n{chunk.text}\n\n"
        f"SCALE:\n{_GRADE_SCALE}\n\n"
        "Answer with the grade only."
    )
