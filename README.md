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

Retrieval quality on the 150 FinanceBench QA, doc-scoped (each question
restricted to its source filing — see [ADR 0001](docs/adr/0001-retrieval-strategy.md)
for the full comparison including BM25 and hybrid fusion):

| Configuration | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Dense (global, no doc filter) | 0.230 | 0.297 | 0.182 | 0.205 |
| Dense (doc-scoped) | 0.402 | 0.552 | 0.338 | 0.377 |
| **Dense + cross-encoder reranker** (default) | **0.549** | **0.649** | **0.433** | **0.473** |

End-to-end generation quality (LLM-judged answer accuracy, faithfulness) is
the next milestone — results to follow.

---

## Quickstart

```bash
# 1. Start services (Qdrant + Langfuse)
docker compose up -d

# 2. Install dependencies
make install-all

# 3. Build the corpus: parse -> chunk -> index into Qdrant
make parse CONFIG=configs/parse/docling.yaml
make chunk CONFIG=configs/chunk/docling_hybrid_512.yaml
make index CONFIG=configs/index/docling_hybrid_512.yaml

# 4. Evaluate retrieval quality
make eval CONFIG=configs/eval/hybrid512_reranked_dense.yaml

# 5. Generate answers with the RAG pipeline (needs an LLM: GROQ_API_KEY, or a
#    local Ollama via a configs/rag/*_ollama.yaml config)
make answer CONFIG=configs/rag/naive_reranked_dense_512_k10_llama70b.yaml                            # all 150 QA
make answer CONFIG=configs/rag/naive_reranked_dense_512_k10_llama70b.yaml ID=financebench_id_03029  # one QA
```

---

## Project Structure

```
financerag-bench/
├── README.md
├── docs/                          # ADRs, benchmark reports
├── configs/                       # 1 YAML = 1 reproducible experiment, grouped by stage
│   ├── parse/
│   ├── chunk/
│   ├── index/
│   ├── eval/
│   └── rag/                    # naive RAG: retriever × LLM × k
├── data/
│   ├── pdfs/                      # 368 docs
│   ├── jsons/                     # 150 QA pairs (FinanceBench open-source)
│   └── processed/                 # chunks, embeddings
├── src/
│   ├── ingestion/                 # PDF parsing (Docling, Unstructured), chunking strategies
│   ├── vectorstore/               # embeddings + Qdrant client, shared by indexing and retrieval
│   ├── indexing/                  # offline job: chunks -> embeddings -> Qdrant
│   ├── retrieval/                 # dense, BM25, hybrid, reranker
│   ├── llm/                       # LiteLLM client + versioned prompts
│   ├── rag/                       # pipelines: naive → advanced → agentic
│   ├── agents/                    # LangGraph: router, grader, rewriter
│   ├── evaluation/                # retrieval metrics + Ragas + runner
│   └── api/                       # FastAPI
├── benchmarks/                    # results (JSON/CSV) versioned
├── dashboard/                     # Streamlit benchmark explorer
├── tests/                         # pytest (unit + integration + eval regression)
├── .github/workflows/             # CI: lint, format check, fast tests
├── docker-compose.yml             # Qdrant + Langfuse
└── Makefile                       # make parse / chunk / index / eval / answer / serve
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
| PDF parsing | Docling + Unstructured |
| Embeddings | BGE-M3 |
| Evaluation | Ragas + custom retrieval metrics |
| Observability | Langfuse |
| API | FastAPI |
| CI/CD | GitHub Actions |
