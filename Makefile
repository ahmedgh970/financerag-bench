.PHONY: help install install-dev lint format test test-fast parse chunk eval benchmark serve docker-up docker-down clean

PYTHON := python
CONFIG ?= configs/baseline.yaml

help:
	@echo "financerag-bench — available commands:"
	@echo ""
	@echo "  make install        Install runtime dependencies (uv)"
	@echo "  make install-dev    Install all deps including dev + pre-commit"
	@echo "  make lint           Run ruff lint"
	@echo "  make format         Run ruff format"
	@echo "  make test           Run all tests"
	@echo "  make test-fast      Run fast tests only (skip slow/eval)"
	@echo "  make parse          Parse corpus once -> data/processed/PARSER/parsed/ (CONFIG=...)"
	@echo "  make chunk          Chunk parsed docs -> chunks.jsonl (CONFIG=...)"
	@echo "  make eval           Run evaluation (CONFIG=configs/...yaml)"
	@echo "  make benchmark      Run full benchmark suite"
	@echo "  make serve          Start FastAPI server"
	@echo "  make docker-up      Start Docker services (Qdrant, Langfuse)"
	@echo "  make docker-down    Stop Docker services"
	@echo "  make clean          Remove generated artefacts"

install:
	uv sync

install-dev:
	uv sync --extra dev
	uv run pre-commit install

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -m "not slow and not eval"

parse:
	uv run python -m src.ingestion.run parse --config $(CONFIG)

chunk:
	uv run python -m src.ingestion.run chunk --config $(CONFIG)

eval:
	uv run python -m src.evaluation.runner --config $(CONFIG)

benchmark:
	uv run python -m src.evaluation.runner --config configs/full_benchmark.yaml

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
