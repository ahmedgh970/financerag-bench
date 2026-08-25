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


def test_health_reports_served_config(monkeypatch):
    with _client(monkeypatch) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ollama_chat/granite4.1:8b"
    assert body["k"] == 10


def test_ask_returns_answer_and_sources(monkeypatch):
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
        resp = client.post("/ask", json={"question": "What was FY2022 revenue?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "$5,000 million"
    assert body["latency_s"] == 0.42
    assert body["sources"] == [{"doc_id": "ACME_2022_10K", "page": 12, "text": chunk.text}]


def test_ask_requires_a_question(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/ask", json={})
    assert resp.status_code == 422  # pydantic validation: question is required
