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

End-to-end generation quality on the 150 QA (corpus `docling_hybrid_1024_bge-m3`,
`reranked(dense)`, doc-scoped), LLM-judged on `equivalent` (answer **correct AND
grounded** in the retrieved context). Best point per model — full table, depth
ablation and analysis in [ADR 0002](docs/adr/0002-generation-model.md):

| Model | Params | equivalent (best k) |
|---|---|---|
| **granite4.1:8b** | 8.8B | **65.3** (k20) |
| qwen3.5:4b | 4.7B | 60.0 (k20) |
| qwen3.5:9b | 9.7B | 58.0 (k20) |
| llama3.1:8b | 8.0B | 50.7 (k20) |
| mistral-nemo | 12.2B | 47.3 (k5) |
| mistral:7b | 7.2B | 41.3 (k20) |
| granite4.1:3b | 3.4B | 40.0 (k5) |
| command-r7b | 8.0B | 38.7 (k5) |
| llama3.2:3b | 3.2B | 28.7 (k10) |

Key finding: **useful retrieval depth scales with model capability** — the
k10→k20 step only helps the strongest models (flat for the 3B tier). See ADR 0002.

---

## Generation benchmark — local LLM lineup

The generation stage runs open-weight LLMs locally via Ollama on a single 8GB
GPU (with automatic GPU/CPU layer offload for models that don't fully fit). The
lineup is capped at **~14B params** — the largest that keeps the majority of
layers on GPU; beyond that, inference is CPU-bound and impractical for the
150 QA grid. Specs below were measured on the actual hardware:


| Model | Params | Context | Disk | VRAM @8K ctx | Angle |
|---|---|---|---|---|---|
| `llama3.2:3b` | 3.2B | 128K | 2.0 GB | 3.1 GB · 100% GPU | Meta — small, fast floor |
| `granite4.1:3b` | 3.4B | 128K | 2.1 GB | 2.9 GB · 100% GPU | IBM (newest) — dense 3B |
| `qwen3.5:4b` | 4.7B | 256K | 3.4 GB | 3.4 GB · 100% GPU | Alibaba (newest) — small |
| `mistral:7b` | 7.2B | 32K | 4.4 GB | 5.6 GB · 100% GPU | Mistral |
| `command-r7b` | 8.0B | **8K** | 5.1 GB | 5.6 GB · 100% GPU | Cohere — RAG-native |
| `llama3.1:8b` | 8.0B | 128K | 4.9 GB | 5.8 GB · 100% GPU | Meta — baseline |
| `granite4.1:8b` | 8.8B | 128K | 5.3 GB | 6.6 GB · 100% GPU | IBM (newest) — enterprise/finance |
| `qwen3.5:9b` | 9.7B | 256K | 6.6 GB | 5.8 GB · 100% GPU | Alibaba (newest) |
| `mistral-nemo` | 12.2B | 1M | 7.1 GB | 8.6 GB · 80% GPU | Mistral — mid tier |

*Context = model's max window. Disk = on-disk size. VRAM @8K = loaded footprint
at `num_ctx=8192`, total size and GPU share.*

**Retrieval depth (k) and cached context.** Each model is benchmarked at
`k = 5 / 10 / 20` retrieved chunks. Every k fixes the `num_ctx` (the KV-cache
context reserved per request) to the true max real-prompt token count across
all 150 questions at that depth — measured with `mistral:7b`, the lineup's most
expensive tokenizer — plus a 1024-token output budget:

| k | `num_ctx` (cached context) |
|---|---|
| 5 | 10240 |
| 10 | 18432 |
| 20 | 30720 |

Two exceptions to the full k sweep: **`command-r7b`** runs at `k=5` only — its
8K context can't hold the larger budgets (and even at k5 it runs at its own 8K
max rather than 10240). **`mistral-nemo`** is also `k=5` only — its context is
large enough, but its partial CPU offload (80% GPU) makes generation at k=10/20
too slow to be practical. All other models cover the full `k = 5 / 10 / 20`.

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

# 5. Generate answers with the RAG pipeline (local Ollama, no quota)
make answer CONFIG=configs/rag/naive_reranked_dense_1024_k10_ollama.yaml                            # all 150 QA
make answer CONFIG=configs/rag/naive_reranked_dense_1024_k10_ollama.yaml ID=financebench_id_03029  # one QA
```

---

## Run the demo locally

Everything is local — Qdrant in Docker, the API / UI / LLM on the host (GPU).
No external inference provider.

```bash
# vector DB (Docker) + the served generator
docker compose up -d qdrant
ollama pull granite4.1:8b

# index the served collection once, if not already: docling_hybrid_1024_bge-m3
make index CONFIG=configs/index/docling_hybrid_1024.yaml

# serve the API + the UI (two terminals)
make serve        # FastAPI on :8000  — GET /health, POST /ask, GET /options
make demo         # Gradio UI on :7860  -> open http://localhost:7860

# or query the API directly
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"What was 3M FY2018 capital expenditure?","doc_id":"3M_2018_10K"}'
```

The LLM, retrieval depth `k` and Qdrant collection are picked in the UI (or per
request in `/ask`); the default is `granite4.1:8b` at k10 (ADR 0002), set by the
`RAG_CONFIG` env var. Retrieval models run on the host GPU; on an 8 GB card the
generator falls back to CPU when both compete for VRAM.

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
│   └── rag/                       # naive RAG: retriever × LLM × k
├── data/
│   ├── pdfs/                      # 368 docs
│   ├── jsons/                     # 150 QA pairs (FinanceBench open-source)
│   └── processed/                 # chunks, embeddings
├── src/
│   ├── ingestion/                 # PDF parsing (Docling, Unstructured), chunking strategies
│   ├── vectorstore/               # embeddings + Qdrant client, shared by indexing and retrieval
│   ├── indexing/                  # offline job: chunks -> embeddings -> Qdrant
│   ├── retrieval/                 # dense, BM25, hybrid, reranker
│   ├── llm/                       # Ollama client + versioned prompts
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
| LLM serving | Ollama (local) |
| Agent orchestration | LangChain + LangGraph |
| Vector DB | Qdrant |
| Lexical search | BM25 |
| Reranking | cross-encoder (BAAI/bge-reranker) |
| PDF parsing | Docling |
| Embeddings | BGE-M3 |
| Evaluation | Ragas + custom retrieval metrics + LLM as a judge |
| Observability | Langfuse |
| API | FastAPI |
| CI/CD | GitHub Actions |
