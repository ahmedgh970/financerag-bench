"""Ragas evaluation: faithfulness, answer relevancy, context precision/recall.

Scores are computed directly off existing answer records (question, generated
answer, source chunks, gold answer) -- no new retrieval call. Covers both the
retriever (context precision/recall, reference-based) and the generator
(faithfulness, answer relevancy, reference-free).

Ragas expects an OpenAI-shaped client. Groq and Ollama both speak the
OpenAI-compatible protocol natively, so a plain ``openai.AsyncOpenAI`` client
is enough here -- Ragas's own documented path (``llm_factory``), simpler than
wrapping LiteLLM through ``instructor`` for no benefit when only one provider
is needed per run.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)
from ragas.metrics.result import MetricResult
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.config import LLMConfig

# Same transient-error/backoff policy as src/llm/client.py: Groq's TPM limit
# resets on a ~60s rolling window, and Ragas scores a question with up to 4 LLM
# calls (one per metric) -- several times our own judge's 1-call-per-question
# budget -- so it hits this limit more often. instructor's own built-in
# max_retries doesn't cover raw API errors like 429s (only response-validation
# retries), hence wrapping each metric call here instead.
_TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=60),
)
def _score(metric, **kwargs) -> MetricResult:
    return metric.score(**kwargs)


_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "ollama_chat": "http://localhost:11434/v1",
}

load_dotenv()


def _build_client(provider: str) -> AsyncOpenAI:
    # Async client: Ragas's sync .score() still drives everything through
    # asyncio.run(self.ascore(...)) internally, and refuses a sync client.
    base_url = _BASE_URLS.get(provider)
    if base_url is None:
        raise ValueError(
            f"Unsupported Ragas LLM provider: {provider!r} (known: {sorted(_BASE_URLS)})"
        )
    api_key = os.environ["GROQ_API_KEY"] if provider == "groq" else "ollama"
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _load_embeddings(model_name: str) -> OpenAIEmbeddings:
    """BGE-M3 embeddings served by Ollama, not sentence-transformers/HF.

    Ollama serves bge-m3 through its own OpenAI-compatible endpoint
    (`/v1/embeddings`), so this reuses the same AsyncOpenAI client pattern as
    the LLM instead of loading a separate sentence-transformers model in this
    process. Ollama picks its own GPU/CPU dispatch for it, same as the LLM --
    no device to choose here, and no torch/safetensors pinning concern either
    (that workaround only applied to the sentence-transformers path).
    """
    client = _build_client("ollama_chat")
    return OpenAIEmbeddings(client=client, model=model_name)


ALL_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def build_metrics(
    llm_config: LLMConfig,
    embedding_model: str = "bge-m3",
    metrics: list[str] | None = None,
) -> dict:
    """Instantiate the requested metrics from an ``LLMConfig`` (litellm-style ``provider/model``).

    ``metrics`` selects a subset (default: all four). context_precision/recall
    depend only on the retriever, so they are identical across generators at a
    fixed k -- a run comparing models can drop them and keep only the
    generation metrics (faithfulness, answer_relevancy).
    """
    provider, model = llm_config.model.split("/", 1)
    client = _build_client(provider)
    # llm_factory's ``provider`` selects Ragas's instructor-patching strategy, not
    # the backend model provider -- "groq" maps to the native groq SDK's
    # client.messages.create shape (Anthropic-like), which our plain
    # AsyncOpenAI client (pointed at Groq's/Ollama's OpenAI-compatible
    # endpoint) doesn't have. "openai" is the correct strategy for any
    # OpenAI-shaped client, regardless of which backend it actually talks to.
    #
    # max_tokens: Ragas defaults to 1024, too low for structured output --
    # faithfulness emits one JSON verdict per statement, and a long generated
    # answer yields many statements, so the JSON gets truncated mid-object and
    # instructor raises IncompleteOutputException. Ragas's own docs recommend
    # 4096+ here; drive it from the config so it's tunable per run.
    llm = llm_factory(model, provider="openai", client=client, max_tokens=llm_config.max_tokens)

    selected = list(metrics) if metrics else list(ALL_METRICS)
    unknown = [m for m in selected if m not in ALL_METRICS]
    if unknown:
        raise ValueError(f"Unknown Ragas metric(s): {unknown} (known: {ALL_METRICS})")

    # Embeddings are only needed by answer_relevancy -- skip the load otherwise.
    embeddings = _load_embeddings(embedding_model) if "answer_relevancy" in selected else None
    builders = {
        "faithfulness": lambda: Faithfulness(llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm, embeddings),
        "context_precision": lambda: ContextPrecisionWithReference(llm),
        "context_recall": lambda: ContextRecall(llm),
    }
    return {name: builders[name]() for name in selected}


def score_record(record: dict, metrics: dict) -> dict:
    """Score one answers-JSONL record on whichever metrics ``metrics`` contains."""
    user_input = record["question"]
    response = record["generated_answer"]
    reference = record["gold_answer"]
    retrieved_contexts = [s["text"] for s in record["sources"]]

    callers = {
        "faithfulness": lambda m: _score(
            m, user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
        ),
        "answer_relevancy": lambda m: _score(m, user_input=user_input, response=response),
        "context_precision": lambda m: _score(
            m, user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
        ),
        "context_recall": lambda m: _score(
            m, user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference
        ),
    }
    return {name: callers[name](metric).value for name, metric in metrics.items()}
