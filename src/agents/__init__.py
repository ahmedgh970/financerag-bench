"""Tool-using ReAct agent: the LLM decides which tool to call, and when.

Deferred tier. This package is kept intact as the FINAL comparison point of the
progression, evaluated last, after the deterministic workflow in :mod:`src.workflow`
has established what each capability contributes on its own. Going straight from
advanced RAG to a ReAct agent changes flow control, retrieval count, tooling and
query reformulation at once, so a win or a loss there cannot be attributed to any
one of them -- hence the intermediate deterministic tier.

Import rule: :mod:`src.workflow` must not import from this package, with the single
exception of the safe arithmetic evaluator, which is shared deliberately.
"""
