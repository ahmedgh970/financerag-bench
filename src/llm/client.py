"""Thin LiteLLM wrapper: one function to call any provider/model uniformly.

Retries transient failures (rate limits, timeouts, transport/server errors)
with backoff; permanent failures (bad auth, bad request, content policy) fail
immediately since retrying them can't help.
"""

from __future__ import annotations

import litellm
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.config import LLMConfig

_TRANSIENT_ERRORS = (
    RateLimitError,
    Timeout,
    APIConnectionError,
    ServiceUnavailableError,
    InternalServerError,
    BadGatewayError,
)


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    # Free-tier rate limits (e.g. Groq's tokens-per-minute cap) reset on a
    # ~60s rolling window, not in a couple seconds — a short backoff just
    # burns through attempts and still fails. Wait long enough for that
    # window to clear, and allow enough attempts to actually get through it.
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=60),
)
def generate(prompt: str, config: LLMConfig) -> str:
    """Generate a completion for ``prompt`` using ``config``."""
    response = litellm.completion(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return response.choices[0].message.content
