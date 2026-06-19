# financerag-bench

> End-to-end RAG & Agentic RAG benchmark on [FinanceBench](https://github.com/patronus-ai/financebench) — 150 financial QA pairs, 368 SEC filings (10-K/10-Q).
> From naive retrieval to multi-agent system, every improvement is justified by a number.

![CI](https://github.com/ahmedgh970/financerag-bench/actions/workflows/ci.yml/badge.svg)

---

## Overview

FinanceBench shows that state-of-the-art RAG systems fail on ~80% of financial questions (GPT-4-Turbo, 2023). This project builds a rigorous benchmark to measure and systematically improve that number using open-source LLMs and modern RAG techniques.

**What this repo demonstrates:**
- Reproducible evaluation pipeline (retrieval metrics + LLM judge + Ragas)
- Progression: naive RAG → hybrid search + reranking → agentic RAG → multi-agent
- Multi-LLM open-source benchmark (quality / latency / cost)
- Production patterns: observability (Langfuse), CI with eval regression, FastAPI serving

---

## Results

*To be filled — Week 3+*

| Pipeline | Recall@5 | Accuracy (LLM judge) | Latency p50 |
|---|---|---|---|
| Naive RAG (baseline) | - | - | - |
| Hybrid + Reranker | - | - | - |
| Agentic RAG | - | - | - |

---

## Quickstart

*To be filled — Week 4*

```bash
# 1. Start services
docker compose up -d

# 2. Install dependencies
make install-all

# 3. Parse + chunk documents
make parse CONFIG=configs/parse_docling.yaml
make chunk CONFIG=configs/chunk_docling_hybrid.yaml

# 4. Run evaluation
make eval CONFIG=configs/baseline.yaml
```

---

## Project Structure

```
financerag-bench/
├── README.md
├── docs/                          # ADRs, benchmark reports
├── configs/                       # 1 YAML = 1 reproducible experiment
├── data/
│   ├── pdfs/                      # 368 docs
│   ├── jsons/                     # 150 QA pairs (FinanceBench open-source)
│   └── processed/                 # chunks, embeddings
├── src/
│   ├── ingestion/                 # PDF parsing, chunking strategies
│   ├── retrieval/                 # dense, BM25, hybrid, reranker
│   ├── llm/                       # LiteLLM client + versioned prompts
│   ├── rag/                       # pipelines: naive → advanced → agentic
│   ├── agents/                    # LangGraph: router, grader, rewriter
│   ├── evaluation/                # retrieval metrics + Ragas + runner
│   └── api/                       # FastAPI
├── benchmarks/                    # results (JSON/CSV) versioned
├── dashboard/                     # Streamlit benchmark explorer
├── tests/                         # pytest (unit + integration + eval regression)
├── .github/workflows/             # CI: lint, tests, smoke eval
├── docker-compose.yml             # Qdrant + Langfuse
└── Makefile                       # make ingest / eval / benchmark / serve
```

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+, uv |
| LLM abstraction | LiteLLM |
| Agent orchestration | LangChain + LangGraph |
| Vector DB | Qdrant |
| Lexical search | BM25 |
| Reranking | cross-encoder (BAAI/bge-reranker) |
| PDF parsing | Docling + PyMuPDF |
| Embeddings | BGE-M3 |
| Evaluation | Ragas + custom retrieval metrics |
| Observability | Langfuse |
| API | FastAPI |
| CI/CD | GitHub Actions |
