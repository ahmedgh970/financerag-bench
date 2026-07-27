.PHONY: help install install-all lint format test test-fast parse chunk index eval answer judge ragas prompts generate chunk-dist serve docker-up docker-down clean

PYTHON := python
CONFIG ?= configs/eval/hybrid512_dense.yaml

help:
	@echo "financerag-bench — available commands:"
	@echo ""
	@echo "  make install        Install runtime (deployable) dependencies only"
	@echo "  make install-all    Install every extra (ingestion, dev, dashboard, demo) + pre-commit"
	@echo "  make lint           Run ruff lint"
	@echo "  make format         Run ruff format"
	@echo "  make test           Run all tests"
	@echo "  make test-fast      Run fast tests only (skip slow/eval)"
	@echo "  make parse          Parse corpus once -> data/processed/PARSER/parsed/ (CONFIG=...)"
	@echo "  make chunk          Chunk parsed docs -> chunks.jsonl (CONFIG=...)"
	@echo "  make index          Embed chunks.jsonl -> Qdrant collection (CONFIG=...)"
	@echo "  make eval           Run evaluation (CONFIG=configs/...yaml)"
	@echo "  make answer         Run the naive RAG pipeline on the 150 QA -> data/processed/answers/ (CONFIG=..., optional ID=<qa_id> for one question)"
	@echo "  make judge          Judge an answers file against gold -> data/processed/judged/ (CONFIG=..., optional ID=<qa_id> for one question)"
	@echo "  make ragas          Score an answers file with Ragas -> data/processed/ragas/ (CONFIG=..., optional ID=<qa_id> or LIMIT=<n>)"
	@echo "  make prompts        Pre-materialize generation prompts per question x k -> data/processed/prompts/ (optional LIMIT=<n>)"
	@echo "  make generate       Run the local Ollama lineup on materialized prompts -> data/processed/answers/ (optional MODELS=, KS=, LIMIT=)"
	@echo "  make chunk-dist     Plot real chunk-size distribution per budget -> docs/adr/assets/ (needs install-all)"
	@echo "  make serve          Start FastAPI server"
	@echo "  make docker-up      Start Docker services (Qdrant, Langfuse)"
	@echo "  make docker-down    Stop Docker services"
	@echo "  make clean          Remove generated artefacts"

install:
	uv sync

install-all:
	uv sync --extra ingestion --extra dev --extra dashboard --extra demo
	uv run pre-commit install

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test:
	uv run --extra ingestion --extra dev pytest tests/ -v

test-fast:
	uv run --extra dev pytest tests/ -v -m "not slow and not eval"

parse:
	uv run --extra ingestion python -m src.ingestion.runner parse --config $(CONFIG)

chunk:
	uv run --extra ingestion python -m src.ingestion.runner chunk --config $(CONFIG)

index:
	uv run python -m src.indexing.runner --config $(CONFIG)

eval:
	uv run python -m src.evaluation.runner --config $(CONFIG)

answer:
	uv run python -m src.rag.runner --config $(CONFIG) $(if $(ID),--id $(ID),)

judge:
	uv run python -m src.evaluation.judge_runner --config $(CONFIG) $(if $(ID),--id $(ID),)

ragas:
	uv run python -m src.evaluation.ragas_runner --config $(CONFIG) $(if $(ID),--id $(ID),) $(if $(LIMIT),--limit $(LIMIT),)

prompts:
	uv run python scripts/materialize_prompts.py $(if $(LIMIT),--limit $(LIMIT),)

generate:
	uv run python scripts/generation_benchmark.py $(if $(MODELS),--models $(MODELS),) $(if $(KS),--ks $(KS),) $(if $(LIMIT),--limit $(LIMIT),)

chunk-dist:
	uv run --no-sync python scripts/chunk_size_dist.py

serve:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
