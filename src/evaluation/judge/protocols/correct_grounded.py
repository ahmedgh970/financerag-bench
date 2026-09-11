"""Correct / grounded protocol: a strict two-axis verdict on an answer against gold.

Checks two things independently -- CORRECT (does the final value/conclusion match
gold) and GROUNDED (is the reasoning that produced it legitimate, not
hallucinated) -- combined in code as ``equivalent = correct and grounded``. A
right answer reached through fabricated reasoning is not equivalent: the LLM's
own yes/no verdicts are read as-is, never re-judged here.

Works with any instruction-following Ollama model; pick it with the config's
``llm.model`` or ``--model``.
"""

from __future__ import annotations

import re

from src.llm.client import generate
from src.llm.config import LLMConfig

_TEMPLATE = """You are grading a generated financial answer against a gold (reference) answer.

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

_CORRECT_RE = re.compile(r"CORRECT:\s*(yes|no)", re.IGNORECASE)
_GROUNDED_RE = re.compile(r"GROUNDED:\s*(yes|no)", re.IGNORECASE)
_JUSTIFICATION_RE = re.compile(r"JUSTIFICATION:\s*(.+)", re.IGNORECASE | re.DOTALL)


def build_prompt(question: str, gold_answer: str, generated_answer: str) -> str:
    """The grading prompt for one answer."""
    return _TEMPLATE.format(
        question=question, gold_answer=gold_answer, generated_answer=generated_answer
    )


def parse(text: str) -> dict:
    """Read the CORRECT / GROUNDED / JUSTIFICATION lines of the judge's reply."""
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


def judge(record: dict, llm: LLMConfig) -> dict:
    """Verdict on one answer record (``question``, ``gold_answer``, ``generated_answer``)."""
    prompt = build_prompt(record["question"], record["gold_answer"], record["generated_answer"])
    return parse(generate(prompt, llm))


def summarize(verdicts: list[dict]) -> str:
    """One line: how many answers are equivalent, and how many are right but ungrounded."""
    n = len(verdicts)
    equivalent = sum(v["equivalent"] for v in verdicts)
    ungrounded = sum(v["correct"] and not v["grounded"] for v in verdicts)
    return f"equivalent {equivalent}/{n} | correct but not grounded {ungrounded}/{n}"
