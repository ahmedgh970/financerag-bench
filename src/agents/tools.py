"""Tools the agentic RAG can call.

Two tools, exposed as LangChain tools so a LangGraph agent can bind them:

- ``calculator`` — a safe arithmetic evaluator (no ``eval``); financial questions
  are heavy on multi-step arithmetic, which plain-instruct LLMs fumble.
- ``retrieve`` — a *depth-escalating* search over the filing: each successive call
  returns a deeper slice (top 5 → 10 → 20), so the agent starts cheap and only
  goes deeper when the passages so far are insufficient. Dense-only (no reranker):
  in an agentic loop the model itself plays the reranker's role, and dropping the
  cross-encoder keeps every call fast (~1-2 s) instead of seconds-to-minutes.
"""

from __future__ import annotations

import ast
import operator

from langchain_core.tools import StructuredTool, tool

from src.retrieval.base import Retriever

# --- calculator -----------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST, allowing only numbers and +-*/%**."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / % ** and parentheses are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Supports + - * / % ** and parentheses, e.g. "(1577 / 34229) * 100". Use it
    for any numeric computation instead of doing the arithmetic yourself.
    """
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:  # noqa: BLE001 - report the error back to the agent
        return f"Error: could not evaluate {expression!r} ({e})"


# --- retrieve -------------------------------------------------------------


def build_retrieve_tool(
    retriever: Retriever,
    doc_id: str | None = None,
    depths: tuple[int, ...] = (5, 10, 20),
    sink: list | None = None,
) -> StructuredTool:
    """Build a depth-escalating ``retrieve(query)`` tool bound to a filing.

    The tool is stateful across calls within one question: the *i*-th call
    retrieves the top ``depths[i]`` passages (5 → 10 → 20 by default) and returns
    only the ones not seen yet, so the agent progressively deepens the search and
    never re-reads a passage. After the last depth it stops yielding more. If
    ``sink`` is given, every new chunk is appended to it so the caller can recover
    the sources actually used. ``doc_id`` scopes the search to one filing.
    """
    state = {"call": 0, "n": 0}
    seen: set[str] = set()

    def retrieve(query: str) -> str:
        """Search the filing for relevant passages.

        Each call searches deeper than the last and returns the *new* passages
        found. Call it again only if what you have is insufficient to answer;
        otherwise answer. There is a maximum search depth.
        """
        i = state["call"]
        if i >= len(depths):
            return "Maximum search depth reached -- answer from the passages you already have."
        state["call"] = i + 1
        results = retriever.retrieve(query, k=depths[i], doc_id=doc_id)
        new = [sc.chunk for sc in results if sc.chunk.chunk_id not in seen]
        for c in new:
            seen.add(c.chunk_id)
        if sink is not None:
            sink.extend(new)
        if not new:
            return "No new passages found -- answer from the passages you already have."
        lines = []
        for c in new:
            state["n"] += 1
            lines.append(f"[Source {state['n']}] (p.{c.page}) {c.text}")
        return "\n\n".join(lines)

    return StructuredTool.from_function(
        retrieve,
        name="retrieve",
        description=(
            "Search the filing for passages relevant to a query. Each call searches "
            "deeper and returns the new passages found; call it again only if what you "
            "have is insufficient. Gather evidence before answering."
        ),
    )
