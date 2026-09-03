"""Optional local tracing of the agent via Arize Phoenix (OpenTelemetry).

Off by default. Call :func:`setup_tracing` once per process (e.g. from the agent
runner when tracing is requested) to export every LangGraph / LangChain span --
the agent loop, each LLM call, each ``retrieve`` / ``calculator`` tool call -- to
a local Phoenix collector (the docker-compose ``phoenix`` service, UI at
http://localhost:6006). No account, no keys: everything stays on the machine.

Instrumentation is global (it patches LangChain for the whole process), so this is
a one-shot setup, not a per-call wrapper -- ``answer_agentic`` needs no change.
"""

from __future__ import annotations

_started = False


def setup_tracing(
    project_name: str = "financerag-agent",
    endpoint: str | None = None,
    batch: bool = False,
) -> None:
    """Wire OpenInference tracing to a local Phoenix collector (idempotent).

    ``endpoint`` defaults to the ``PHOENIX_COLLECTOR_ENDPOINT`` environment
    variable, else Phoenix's local default (http://localhost:6006). ``batch=False``
    exports each span synchronously, which is the safe choice for short-lived
    scripts; pass ``batch=True`` for a long run to buffer and flush asynchronously.
    """
    global _started
    if _started:
        return
    from phoenix.otel import register

    register(
        project_name=project_name,
        endpoint=endpoint,
        batch=batch,
        auto_instrument=True,  # patches the installed OpenInference libs (LangChain)
        verbose=False,
    )
    _started = True
