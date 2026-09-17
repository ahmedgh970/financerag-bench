"""Grid protocol: the judge's half of the evidence-grounded outcome grid (ADR 0004).

The judge reads one answer against the gold and returns two verdicts, ``refused``
and ``correct``, plus a short justification. It sees the question, the gold answer,
FinanceBench's gold justification (how the gold was obtained) and the full generated
answer -- never the retrieved passages: whether the gold evidence reached the prompt
is computed by :mod:`src.evaluation.judge.grid`, not judged. Keeping the passages out
keeps the prompt small enough for a local model on a pinned, short context window.

The reply is constrained to a JSON object filled in order: the answer's final value
or conclusion first, then what it rests on, then whether that is a refusal, then the
comparison with the gold and the verdict. Extracting the final answer and its basis
before anything else keeps the model from reading "the answer correctly says the
data is missing" as a correct answer, or a "No, the sources do not mention it" as a
finding.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.llm.client import estimated_context, generate_structured
from src.llm.config import LLMConfig

_TEMPLATE = """You are grading an answer to a financial question against the gold (reference) answer. You do not see the documents the answer was based on: judge only what the answer says.

Question: {question}

Gold answer: {gold_answer}
{gold_justification}
Generated answer:
<<<
{generated_answer}
>>>

Fill in five fields, in this order.

1. "final_answer": the final value or conclusion the generated answer gives to the question, in a few words ("$1,577 million", "No, it decreased", "Americas, EMEA, APAC"). Write "none" if it gives none.

2. "basis": what the generated answer rests its final answer on, in a few words: the figures or facts it cites ("net sales 34,229 and capex 1,749"), or the absence of information in its sources ("the sources do not mention X").

3. "refused": true if the generated answer does not give the value or conclusion the question asks for. That is the case when:
- it says the information is not available, not provided, or not in the sources or context;
- it says the value cannot be determined, calculated or assessed;
- its basis is the absence of information, whatever its final answer: "No, the context does not mention any legal battle" is refused.
Such an answer is refused even if what it says about the sources is true, and even if the gold answer is also negative. Saying that data is missing is never a correct answer. An answer that gives a value or conclusion while adding caveats is not refused.
When the question itself offers an alternative ("if X is not a useful metric for this company, state that and explain why"), an answer that takes it -- says the metric does not suit this company's business and explains why -- gives a conclusion and is not refused, even if it also says the data is missing.

4. "justification": one or two sentences comparing the final answer with the gold answer.

5. "correct": always false when "refused" is true. Otherwise true if the final answer agrees with the gold answer:
- Units and formats may differ: $1,577 million matches $1577.00, and 0.0143 matches 1.43%.
- A more precise value that rounds to the gold's precision matches ($302.58M vs $303M, 36.46% vs 36%). A different value at the same precision does not (7.8% vs 7.9%, 17.99 vs 17.98).
- A magnitude error is wrong (thousands read as millions, 0.15 vs 15%).
- For a yes/no or direction question, the direction must agree; small rounding in the supporting figures is tolerated.
- The right label reached through reasoning that is false, self-contradictory, a different metric than the one asked, or invented assumptions ("assume D&A is typically...") is wrong.
- For a list, every item the gold lists must be there; one wrong or missing item makes it wrong.
- For an explanation, the key conclusion or drivers of the gold must be there; wording does not matter."""


class Verdict(BaseModel):
    final_answer: str
    basis: str
    refused: bool
    justification: str
    correct: bool


def build_prompt(
    question: str, gold_answer: str, generated_answer: str, gold_justification: str = ""
) -> str:
    """The grading prompt for one answer; the justification line only when the gold has one."""
    return _TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        gold_justification=f"How the gold was obtained: {gold_justification}\n"
        if gold_justification
        else "",
        generated_answer=generated_answer,
    )


def judge(record: dict, llm: LLMConfig) -> dict:
    """Verdict on one answer record (``question``, ``gold_answer``, ``generated_answer``,
    and ``gold_justification`` when available).

    A pinned ``llm.num_ctx`` too small for the prompt is refused rather than sent:
    Ollama would silently drop the start of the prompt, instructions included.
    """
    prompt = build_prompt(
        record["question"],
        record["gold_answer"],
        record["generated_answer"],
        record.get("gold_justification", ""),
    )
    needed = estimated_context(prompt, llm.max_tokens)
    if llm.num_ctx is not None and needed > llm.num_ctx:
        raise ValueError(
            f"{record['id']}: the prompt needs about {needed} tokens of context, "
            f"above llm.num_ctx={llm.num_ctx}; raise it in the judge config"
        )
    verdict = generate_structured(prompt, llm, Verdict)
    return {
        "correct": verdict.correct and not verdict.refused,
        "refused": verdict.refused,
        "final_answer": verdict.final_answer,
        "basis": verdict.basis,
        "justification": verdict.justification,
    }


def summarize(verdicts: list[dict]) -> str:
    """One line: how many answers are correct, and how many are refusals."""
    n = len(verdicts)
    correct = sum(v["correct"] for v in verdicts)
    refused = sum(v["refused"] for v in verdicts)
    return f"correct {correct}/{n} | refused {refused}/{n}"
