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

import logging
import os

import torch
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
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


def _load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """BGE-M3 embeddings, GPU if there's room, CPU otherwise.

    Mirrors Ollama's own graceful GPU/CPU dispatch: try CUDA first (fast), but
    unlike Ollama's server, sentence-transformers has no built-in fallback --
    a CUDA OOM here is a hard crash, not a silent CPU retry. A concurrent
    retriever/reranker process can already be using most of an 8GB GPU, so
    catch that one failure mode and retry on CPU instead of taking the
    process down.

    Force safetensors weights either way: torch 2.5.1 (pinned for the cu121
    GPU stack, see src/vectorstore/embeddings.py) is below the 2.6 that
    transformers now requires to load legacy .bin checkpoints (CVE-2025-32434).
    """
    try:
        return HuggingFaceEmbeddings(model=model_name, model_kwargs={"use_safetensors": True})
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        logging.warning("GPU out of memory loading %s embeddings, falling back to CPU", model_name)
        torch.cuda.empty_cache()
        return HuggingFaceEmbeddings(
            model=model_name, device="cpu", model_kwargs={"use_safetensors": True}
        )


def build_metrics(llm_config: LLMConfig, embedding_model: str = "BAAI/bge-m3") -> dict:
    """Instantiate the 4 metrics from an ``LLMConfig`` (litellm-style ``provider/model``)."""
    provider, model = llm_config.model.split("/", 1)
    client = _build_client(provider)
    # llm_factory's ``provider`` selects Ragas's instructor-patching strategy, not
    # the backend model provider -- "groq" maps to the native groq SDK's
    # client.messages.create shape (Anthropic-like), which our plain
    # AsyncOpenAI client (pointed at Groq's/Ollama's OpenAI-compatible
    # endpoint) doesn't have. "openai" is the correct strategy for any
    # OpenAI-shaped client, regardless of which backend it actually talks to.
    llm = llm_factory(model, provider="openai", client=client)
    embeddings = _load_embeddings(embedding_model)

    return {
        "faithfulness": Faithfulness(llm),
        "answer_relevancy": AnswerRelevancy(llm, embeddings),
        "context_precision": ContextPrecisionWithReference(llm),
        "context_recall": ContextRecall(llm),
    }


def score_record(record: dict, metrics: dict) -> dict:
    """Score one answers-JSONL record on all 4 metrics."""
    user_input = record["question"]
    response = record["generated_answer"]
    reference = record["gold_answer"]
    retrieved_contexts = [s["text"] for s in record["sources"]]

    faithfulness = _score(
        metrics["faithfulness"],
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    answer_relevancy = _score(metrics["answer_relevancy"], user_input=user_input, response=response)
    context_precision = _score(
        metrics["context_precision"],
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    context_recall = _score(
        metrics["context_recall"],
        user_input=user_input,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )

    return {
        "faithfulness": faithfulness.value,
        "answer_relevancy": answer_relevancy.value,
        "context_precision": context_precision.value,
        "context_recall": context_recall.value,
    }
