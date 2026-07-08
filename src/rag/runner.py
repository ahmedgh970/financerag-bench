"""CLI runner: answer every FinanceBench QA with the naive RAG pipeline.

Builds the configured retriever once, answers each of the 150 QA, and writes
the results (answer + source chunks + latency) to a JSONL for later judging
(Ragas + LLM judge).

Usage:
    python -m src.rag.runner --config configs/rag/naive_reranked_dense.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.evaluation.golden_set import load_golden_set
from src.rag.config import RagConfig, load_rag_config
from src.rag.naive import answer
from src.retrieval.registry import build_retriever


def run(config: RagConfig) -> str:
    """Answer every QA in the golden set and write results to a JSONL. Returns its path."""
    qas = load_golden_set(config.golden_set_path)
    retriever = build_retriever(config.retriever, config)

    out_dir = Path("data/processed/answers")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config.retriever}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for qa in tqdm(qas, desc=f"answer [{config.retriever}]"):
            result = answer(
                qa.question,
                retriever,
                config.llm,
                k=config.k,
                doc_id=qa.doc_name if config.doc_scoped else None,
            )
            record = {
                "id": qa.id,
                "question": qa.question,
                "gold_answer": qa.answer,
                "generated_answer": result.answer,
                "sources": [
                    {"doc_id": c.doc_id, "page": c.page, "text": c.text} for c in result.sources
                ],
                "latency_s": result.latency_s,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(qas)} answers -> {out_path}")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer FinanceBench QA with the naive RAG pipeline."
    )
    parser.add_argument("--config", required=True, help="Path to a RAG YAML config.")
    run(load_rag_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
