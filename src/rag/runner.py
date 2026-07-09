"""CLI runner: answer every FinanceBench QA with the naive RAG pipeline.

Builds the configured retriever once, answers each of the 150 QA, and writes
the results (answer + source chunks + latency) to a JSONL for later judging
(Ragas + LLM judge). Resumable: QA ids already present in the output file are
skipped and new answers are appended, so hitting a provider's daily quota
mid-run doesn't lose progress — rerun the same command later to continue.

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


def _answered_ids(path: Path) -> set[str]:
    """QA ids already answered in ``path`` (empty set if it doesn't exist yet)."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _output_path(config: RagConfig) -> Path:
    """Where answers are written: one file per (retriever, LLM, k) combination.

    Encoding the model and k avoids silently mixing answers generated under
    different settings when comparing LLMs/retrievers (Semaine 6).
    """
    model = config.llm.model.replace("/", "_")
    return Path("data/processed/answers") / f"{config.retriever}_{model}_k{config.k}.jsonl"


def run(config: RagConfig) -> str:
    """Answer every not-yet-answered QA and append results to a JSONL. Returns its path."""
    qas = load_golden_set(config.golden_set_path)

    out_path = _output_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _answered_ids(out_path)
    remaining = [qa for qa in qas if qa.id not in done]
    if not remaining:
        print(f"All {len(qas)} QA already answered -> {out_path}")
        return str(out_path)

    retriever = build_retriever(config.retriever, config)

    with out_path.open("a", encoding="utf-8") as f:
        for qa in tqdm(remaining, desc=f"answer [{config.retriever}]"):
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
            f.flush()

    print(f"Answered {len(remaining)} new QA (skipped {len(done)} already present) -> {out_path}")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer FinanceBench QA with the naive RAG pipeline."
    )
    parser.add_argument("--config", required=True, help="Path to a RAG YAML config.")
    run(load_rag_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
