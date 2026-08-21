"""Prometheus-2 judge: native Absolute Grading of a generated answer vs gold.

Unlike the correct/grounded judge in ``judge.py``, this scores each answer on
Prometheus-2's own 1-5 rubric scale (a single integer plus written feedback),
using the exact Absolute Grading protocol the model was trained on
(prometheus-eval): a task description, the instruction, the response, a
reference answer worth a score of 5, and a score rubric. The model is served
locally through Ollama; its Modelfile already carries the required system
prompt and Mistral ``[INST]`` template, so we send the assembled prompt as the
user turn and read back ``Feedback: ... [RESULT] N``.

The 1-5 score is a different axis from the correct/grounded judge, not a
remapping of it: the two judges are compared by the model *ranking* they induce
(Spearman/Kendall), not question-by-question.
"""

from __future__ import annotations

import os
import re

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PROMETHEUS_MODEL = "ggozad/prometheus2:latest"

# Baked into the model's Modelfile too; sent explicitly so the call is
# self-contained and does not rely on the server default.
_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective "
    "feedback based on specific criteria, ensuring each assessment reflects the "
    "absolute standards set for performance."
)

# prometheus-eval ABSOLUTE_PROMPT, verbatim (deviating from it degrades the
# fine-tuned model). Filled with our rubric below.
_ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """

# Single financial-QA correctness rubric (correctness and groundedness fused
# into one 1-5 score, since Prometheus emits one score per call and we compare
# judges by ranking, not by mapping to the correct/grounded grille).
_RUBRIC = """[Does the response correctly and faithfully answer the financial question, matching the reference answer's value or conclusion with valid, non-fabricated reasoning?]
Score 1: The response is wrong or refuses to answer; its value or conclusion contradicts or fails to provide the reference answer's.
Score 2: The response is largely incorrect; it gets the direction right but the wrong magnitude, unit, or sign, or its answer is not supported by real figures.
Score 3: The response is partially correct; it reaches the reference value through weak or partly fabricated reasoning, or it reasons correctly but reports an imprecise value.
Score 4: The response is correct; its value or conclusion matches the reference, grounded in real figures, with only minor imprecision.
Score 5: The response is fully correct; it matches the reference answer's value or conclusion precisely, with sound reasoning grounded in real figures."""

_RESULT_RE = re.compile(r"\[RESULT\]\s*([1-5])")
_FALLBACK_RE = re.compile(r"(?:score|result)\D{0,10}?([1-5])\b", re.IGNORECASE)


def build_absolute_prompt(question: str, gold_answer: str, generated_answer: str) -> str:
    """Assemble the Prometheus Absolute Grading user prompt for one answer."""
    return _ABSOLUTE_PROMPT.format(
        instruction=question,
        response=generated_answer,
        reference_answer=gold_answer,
        rubric=_RUBRIC,
    )


def parse_score(text: str) -> int | None:
    """Extract the 1-5 score from a ``Feedback: ... [RESULT] N`` completion.

    Returns the integer score, or ``None`` if no score can be located (caller
    decides how to handle a malformed judge output).
    """
    m = _RESULT_RE.search(text)
    if m:
        return int(m.group(1))
    # Prometheus occasionally drops the [RESULT] tag; fall back to a trailing
    # "score ... N" mention before giving up.
    matches = _FALLBACK_RE.findall(text)
    return int(matches[-1]) if matches else None


def judge_prometheus(
    question: str,
    gold_answer: str,
    generated_answer: str,
    model: str = PROMETHEUS_MODEL,
    ollama_url: str = OLLAMA_URL,
    timeout: float = 600.0,
) -> dict:
    """Grade one generated answer against gold on Prometheus-2's 1-5 rubric.

    Returns ``{"score": int | None, "feedback": str, "raw": str}``. ``score`` is
    ``None`` when the model's output can't be parsed.
    """
    prompt = build_absolute_prompt(question, gold_answer, generated_answer)
    r = requests.post(
        f"{ollama_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0},
        },
        timeout=timeout,
    ).json()
    raw = r.get("message", {}).get("content", "")
    score = parse_score(raw)
    feedback = raw.split("[RESULT]", 1)[0].removeprefix("Feedback:").strip()
    return {"score": score, "feedback": feedback, "raw": raw}
