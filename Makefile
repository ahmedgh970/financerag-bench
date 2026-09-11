.PHONY: help install install-all lint format test test-fast parse chunk index answer eval-retrieval judge grid ragas prompts generate judge-all chunk-dist serve demo docker-up docker-down clean

PYTHON := python
CONFIG ?= configs/evaluation/retrieval/chunks512_dense.yaml
# CONFIG when given on the command line, else the stage's own default config.
stage_config = $(if $(filter command line,$(origin CONFIG)),$(CONFIG),$(1))

help:
	@echo "financerag-bench — available commands:"
	@echo ""
	@echo "  make install        Install runtime (deployable) dependencies only"
	@echo "  make install-all    Install every extra (ingestion, dev, dashboard, demo, agents) + pre-commit"
	@echo "  make lint           Run ruff lint"
	@echo "  make format         Run ruff format"
	@echo "  make test           Run all tests"
	@echo "  make test-fast      Run fast tests only (skip slow/eval)"
	@echo "  make parse          Parse corpus once -> data/processed/PARSER/parsed/ (CONFIG=...)"
	@echo "  make chunk          Chunk parsed docs -> chunks.jsonl (CONFIG=...)"
	@echo "  make index          Embed chunks.jsonl -> Qdrant collection (CONFIG=...)"
	@echo "  make eval-retrieval Score a retriever: recall@k / MRR / nDCG -> docs/benchmarks/ (CONFIG=configs/evaluation/retrieval/...yaml)"
	@echo "  make answer         Run the naive RAG pipeline on the 150 QA -> data/processed/answers/ (CONFIG=..., optional ID=<qa_id> for one question)"
	@echo "  make judge          LLM-judge an answers file against gold -> data/processed/judged/ (optional ANSWERS=<answers.jsonl>, ID=<qa_id>)"
	@echo "  make grid           Place judged answers in the evidence-grounded outcome grid -> data/processed/judged/ (optional ANSWERS=<answers.jsonl>)"
	@echo "  make ragas          Score an answers file with Ragas -> data/processed/ragas/ (CONFIG=..., optional ID=<qa_id> or LIMIT=<n>)"
	@echo "  make prompts        Pre-materialize generation prompts per question x k -> data/processed/prompts/ (optional LIMIT=<n>)"
	@echo "  make generate       Run the local Ollama lineup on materialized prompts -> data/processed/answers/ (optional MODELS=, KS=, LIMIT=)"
	@echo "  make judge-all      Judge every answers file -> data/processed/judged/ + accuracy table (optional MODEL=, MAX_FILES=)"
	@echo "  make chunk-dist     Plot real chunk-size distribution per budget -> docs/adr/assets/ (needs install-all)"
	@echo "  make serve          Start FastAPI server"
	@echo "  make demo           Start the Gradio demo UI (needs make serve running)"
	@echo "  make docker-up      Start Docker services (Qdrant, Phoenix)"
	@echo "  make docker-down    Stop Docker services"
	@echo "  make clean          Remove generated artefacts"

install:
	uv sync

install-all:
	uv sync --extra ingestion --extra dashboard --extra demo --extra agents
	uv run pre-commit install

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test:
	uv run --extra ingestion pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -m "not slow and not eval"

parse:
	uv run --extra ingestion python -m src.ingestion.runner parse --config $(CONFIG)

chunk:
	uv run --extra ingestion python -m src.ingestion.runner chunk --config $(CONFIG)

index:
	uv run python -m src.indexing.runner --config $(CONFIG)

eval-retrieval:
	uv run python -m src.evaluation.run_retrieval --config $(CONFIG)

answer:
	uv run python -m src.rag.runner --config $(CONFIG) $(if $(ID),--id $(ID),)

judge:
	uv run python -m src.evaluation.judge_runner --config $(call stage_config,configs/judge/llm_judge.yaml) $(if $(ANSWERS),--answers $(ANSWERS),) $(if $(ID),--id $(ID),)

grid:
	uv run python -m src.evaluation.grid_runner --config $(call stage_config,configs/judge/evidence_grid.yaml) $(if $(ANSWERS),--answers $(ANSWERS),)

ragas:
	uv run python -m src.evaluation.ragas_runner --config $(CONFIG) $(if $(ID),--id $(ID),) $(if $(LIMIT),--limit $(LIMIT),)

prompts:
	uv run python scripts/materialize_prompts.py $(if $(LIMIT),--limit $(LIMIT),)

generate:
	uv run python scripts/generation_benchmark.py $(if $(MODELS),--models $(MODELS),) $(if $(KS),--ks $(KS),) $(if $(LIMIT),--limit $(LIMIT),)

judge-all:
	uv run python scripts/eval_all.py $(if $(MODEL),--model $(MODEL),) $(if $(MAX_FILES),--max-files $(MAX_FILES),)

chunk-dist:
	uv run --no-sync python scripts/chunk_size_dist.py

serve:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

demo:
	uv run --extra demo python demo/app.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
