"""Gradio demo UI over the RAG API.

A thin front-end on top of the FastAPI endpoints: pick the LLM, the retrieval
depth ``k`` and the Qdrant collection (fetched from ``GET /options``), type a
question (optionally scoped to one filing), and get the grounded answer, its
cited sources and the latency via ``POST /ask``. Start the API first
(``make serve``), then run this (``make demo``). Both are local -- no external
provider.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
GOLDEN_SET = "data/jsons/financebench_open_source.jsonl"

_FALLBACK = {
    "models": [],
    "collections": [],
    "defaults": {"model": "granite4.1:8b", "k": 10, "collection": "docling_hybrid_1024_bge-m3"},
}


def _options() -> dict:
    """Models / collections / defaults from the API (fallback if it's not up)."""
    try:
        resp = requests.get(f"{API_URL}/options", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return _FALLBACK


def _doc_ids() -> list[str]:
    """Unique filing ids from the golden set, for the scope dropdown."""
    path = Path(GOLDEN_SET)
    if not path.exists():
        return []
    ids = {json.loads(line)["doc_name"] for line in path.read_text().splitlines() if line.strip()}
    return sorted(ids)


def ask(question: str, doc_id: str, model: str, k: int, collection: str) -> tuple[str, str]:
    """Call POST /ask with the chosen settings and format answer + sources."""
    if not question.strip():
        return "_Pose une question._", ""
    payload = {
        "question": question,
        "doc_id": doc_id or None,
        "model": model or None,
        "k": int(k),
        "collection": collection or None,
    }
    try:
        resp = requests.post(f"{API_URL}/ask", json=payload, timeout=600)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"**Erreur API** : {e}\n\n(l'API tourne-t-elle ? `make serve`)", ""

    data = resp.json()
    answer_md = (
        f"{data['answer']}\n\n"
        f"_{data['model']} · k={data['k']} · {data['collection']} · {data['latency_s']:.1f} s_"
    )
    sources_md = "\n\n---\n\n".join(
        f"**{s['doc_id']}** — page {s['page']}\n\n{s['text']}" for s in data["sources"]
    )
    return answer_md, sources_md or "_(aucune source)_"


opts = _options()
defaults = opts["defaults"]

with gr.Blocks(title="financerag-bench") as demo:
    gr.Markdown(
        "# financerag-bench — RAG financier\n"
        "Pose une question sur un filing SEC (FinanceBench). Réponse ancrée dans les "
        "sources récupérées, générée en local (Ollama)."
    )
    with gr.Row():
        model = gr.Dropdown(choices=opts["models"], value=defaults["model"], label="LLM", scale=1)
        k = gr.Slider(
            minimum=1, maximum=20, step=1, value=defaults["k"], label="k (chunks)", scale=1
        )
        collection = gr.Dropdown(
            choices=opts["collections"],
            value=defaults["collection"],
            label="Collection Qdrant",
            scale=1,
        )
    with gr.Row():
        question = gr.Textbox(
            label="Question",
            placeholder="What was 3M's FY2018 capital expenditure?",
            scale=3,
        )
        doc = gr.Dropdown(
            choices=["", *_doc_ids()], value="", label="Document (optionnel — vide = tous)", scale=1
        )
    btn = gr.Button("Répondre", variant="primary")

    gr.Markdown("### Réponse")
    answer_out = gr.Markdown()
    gr.Markdown("### Sources")
    sources_out = gr.Markdown()

    inputs = [question, doc, model, k, collection]
    btn.click(ask, inputs=inputs, outputs=[answer_out, sources_out])
    question.submit(ask, inputs=inputs, outputs=[answer_out, sources_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
