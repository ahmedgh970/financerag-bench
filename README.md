# financerag-bench 

### Structure
```
financerag-bench/                  
├── README.md                      # vitrine : archi, résultats, démo GIF
├── docs/                          # decisions (ADR), rapports de benchmark
├── configs/                       # 1 YAML = 1 expérience reproductible
├── data/
│   ├── pdfs/                      # 10-K/10-Q FinanceBench (script de download)
│   ├── golden_set/                # 150 QA FinanceBench + extensions
│   └── processed/                 # chunks, index
├── src/
│   ├── ingestion/                 # parsing PDF, chunking (3 stratégies)
│   ├── retrieval/                 # dense, BM25, hybrid, reranker
│   ├── llm/                       # client LiteLLM + prompts versionnés
│   ├── rag/                       # pipelines : naive → advanced → agentic
│   ├── agents/                    # LangGraph : router, grader, rewriter, multi-agent
│   ├── evaluation/                # métriques retrieval + Ragas + runner
│   └── api/                       # FastAPI
├── benchmarks/                    # scripts + résultats (JSON/CSV) versionnés
├── dashboard/                     # Streamlit benchmark explorer
├── tests/                         # pytest (unit + integration + eval regression)
├── .github/workflows/             # CI : lint, tests, smoke eval
├── docker-compose.yml             # api + qdrant + langfuse
└── Makefile                       # make ingest / eval / benchmark / serve
```