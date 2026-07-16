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

# Free-tier tokens-per-minute ceilings, confirmed from real 429/413 responses
# (not guessed): a single request over this can never succeed no matter how
# long we retry, since it exceeds even a fully-reset budget. Only models we
# have actually hit a limit for are listed; unknown models skip the check.
_KNOWN_TPM_LIMITS = {
    "groq/llama-3.1-8b-instant": 6_000,
    "groq/llama-3.3-70b-versatile": 12_000,
}


class RequestTooLargeError(Exception):
    """A prompt is bigger than the model's own per-minute token budget."""


def _check_request_size(prompt: str, config: LLMConfig) -> None:
    limit = _KNOWN_TPM_LIMITS.get(config.model)
    if limit is None:
        return
    estimated = (
        litellm.token_counter(model=config.model, messages=[{"role": "user", "content": prompt}])
        + config.max_tokens
    )
    if estimated > limit:
        raise RequestTooLargeError(
            f"Prompt (~{estimated} tokens incl. completion budget) exceeds "
            f"{config.model}'s known {limit} TPM limit — no retry can fix this. "
            "Lower k (fewer/smaller chunks) or switch to a model with a higher limit."
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
    """Generate a completion for ``prompt`` using ``config``.

    Fails immediately (no retry) if the prompt structurally exceeds the
    model's known per-minute token budget, rather than exhausting retries on
    a request that can never succeed.
    """
    _check_request_size(prompt, config)
    response = litellm.completion(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return response.choices[0].message.content
