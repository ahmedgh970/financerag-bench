"""Contract tests for the FastAPI serving app.

The retriever build and the RAG ``answer`` call are mocked, so these run fast
and offline (no Qdrant, no GPU, no Ollama) -- they check the endpoint wiring
and response schema, not the model.
"""

from fastapi.testclient import TestClient

from src.api import main
from src.ingestion.schema import Chunk
from src.rag.naive import RagAnswer


def _client(monkeypatch) -> TestClient:
    # Skip the heavy retriever build in the lifespan; answer() is mocked anyway.
    monkeypatch.setattr(main, "build_retriever", lambda name, cfg: object())
    return TestClient(main.app)


def test_health_ok(monkeypatch):
    with _client(monkeypatch) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "granite" in body["config"]


def test_options_lists_models_and_collections(monkeypatch):
    monkeypatch.setattr(main, "_ollama_models", lambda: ["granite4.1:8b", "qwen3.5:4b"])
    monkeypatch.setattr(main, "_qdrant_collections", lambda: ["docling_hybrid_1024_bge-m3"])
    with _client(monkeypatch) as client:
        body = client.get("/options").json()
    assert body["models"] == ["granite4.1:8b", "qwen3.5:4b"]
    assert body["collections"] == ["docling_hybrid_1024_bge-m3"]
    assert body["defaults"] == {
        "model": "granite4.1:8b",
        "k": 10,
        "collection": "docling_hybrid_1024_bge-m3",
    }


def test_ask_returns_answer_sources_and_echoes_choices(monkeypatch):
    chunk = Chunk(
        chunk_id="ACME_2022_10K::5",
        doc_id="ACME_2022_10K",
        text="Total revenue was $5,000 million in FY2022.",
        page=12,
    )
    monkeypatch.setattr(
        main,
        "answer",
        lambda q, r, llm, k, doc_id=None: RagAnswer(
            answer="$5,000 million", sources=[chunk], latency_s=0.42
        ),
    )
    with _client(monkeypatch) as client:
        resp = client.post(
            "/ask", json={"question": "What was FY2022 revenue?", "k": 5, "model": "qwen3.5:4b"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "$5,000 million"
    assert body["latency_s"] == 0.42
    assert body["model"] == "qwen3.5:4b"  # request override echoed back
    assert body["k"] == 5
    assert body["collection"] == "docling_hybrid_1024_bge-m3"  # served default
    assert body["sources"] == [{"doc_id": "ACME_2022_10K", "page": 12, "text": chunk.text}]


def test_ask_requires_a_question(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/ask", json={})
    assert resp.status_code == 422  # pydantic validation: question is required
