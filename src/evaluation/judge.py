"""LLM judge: strict verdict on a generated answer against gold.

Checks two things independently -- CORRECT (does the final value/conclusion
match gold) and GROUNDED (is the reasoning that produced it legitimate, not
hallucinated) -- combined in code as ``equivalent = correct and grounded``.
A right answer reached through fabricated reasoning is not equivalent: the
LLM's own yes/no verdicts are read as-is, never re-judged here.
"""

from __future__ import annotations

import re

from src.llm.client import generate
from src.llm.config import LLMConfig
from src.llm.prompts import build_judge_prompt

_CORRECT_RE = re.compile(r"CORRECT:\s*(yes|no)", re.IGNORECASE)
_GROUNDED_RE = re.compile(r"GROUNDED:\s*(yes|no)", re.IGNORECASE)
_JUSTIFICATION_RE = re.compile(r"JUSTIFICATION:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_judge_response(text: str) -> dict:
    correct_m = _CORRECT_RE.search(text)
    grounded_m = _GROUNDED_RE.search(text)
    if not correct_m or not grounded_m:
        raise ValueError(f"Judge response missing a CORRECT: or GROUNDED: line: {text!r}")
    justification_m = _JUSTIFICATION_RE.search(text)

    correct = correct_m.group(1).lower() == "yes"
    grounded = grounded_m.group(1).lower() == "yes"
    return {
        "correct": correct,
        "grounded": grounded,
        "equivalent": correct and grounded,
        "justification": justification_m.group(1).strip() if justification_m else "",
    }


def judge(
    question: str, gold_answer: str, generated_answer: str, llm_config: LLMConfig | None = None
) -> dict:
    """LLM verdict on whether ``generated_answer`` is equivalent to ``gold_answer``."""
    config = llm_config or LLMConfig()
    prompt = build_judge_prompt(question, gold_answer, generated_answer)
    response = generate(prompt, config)
    return _parse_judge_response(response)
