"""Pre-materialize the real generation prompts for the whole golden set × k.

Runs the retrieval half of `make answer` once for every question, at every k,
and freezes the exact prompt build_prompt() would send to the LLM -- so the
multi-model / multi-num_ctx benchmark can feed prompts straight to each LLM
without ever re-running the retriever + reranker (the slow, GPU-bound part).

Efficiency: reranked(dense) prefetches a fixed pool (50) and returns the top-k
of one reranked ordering, so top-5/10/20 are nested prefixes. We retrieve once
at max(k) per question and slice -- identical results, a third of the retrieval.

Output: one JSONL per k at data/processed/prompts/prompts_k{k}.jsonl, each line
carrying everything downstream needs (generation, judge, Ragas):
    {id, question, gold_answer, doc_name, k, n_chunks,
     sources: [{doc_id, page, text}], prompt}

Resumable: a (question, k) already written is skipped.

    uv run python scripts/materialize_prompts.py
    uv run python scripts/materialize_prompts.py --limit 5   # smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.evaluation.golden_set import load_golden_set
from src.llm.prompts import build_prompt
from src.rag.config import RagConfig
from src.retrieval.registry import build_retriever

KS = [5, 10, 20]
OUT_DIR = Path("data/processed/prompts")

# Retriever setup, hard-coded on purpose (no external YAML): the materialized
# prompts have one canonical setup -- the reference retriever reranked(dense)
# on the 1024-token corpus, doc-scoped, same as `make answer`. Only the
# retrieval fields matter here; nothing generates, so no LLM is involved.
COLLECTION = "docling_hybrid_1024_bge-m3"
CHUNKS_PATH = "data/processed/docling/chunked/hybrid/chunks_1024.jsonl"
RETRIEVER = "reranked"
BASE_RETRIEVER = "dense"
DOC_SCOPED = True
PREFETCH = 50  # reference value (ADR 0001); must match `make answer` -- a smaller
# pool would rerank fewer candidates and change which top-k chunks come out.


def _out_path(k: int) -> Path:
    """Self-documenting filename encoding the retrieval setup (like the answers files)."""
    scope = "docscoped" if DOC_SCOPED else "global"
    stem = f"prompts_{RETRIEVER}-{BASE_RETRIEVER}_{COLLECTION}_{scope}_pf{PREFETCH}_k{k}"
    return OUT_DIR / f"{stem}.jsonl"


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def run(limit: int | None = None) -> None:
    cfg = RagConfig(
        chunks_path=CHUNKS_PATH,
        collection_name=COLLECTION,
        retriever=RETRIEVER,
        base_retriever=BASE_RETRIEVER,
        doc_scoped=DOC_SCOPED,
        rerank_prefetch=PREFETCH,
    )
    qas = load_golden_set(cfg.golden_set_path)
    if limit is not None:
        qas = qas[:limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {k: _out_path(k).open("a", encoding="utf-8") for k in KS}
    done = {k: _done_ids(_out_path(k)) for k in KS}

    retriever = build_retriever(cfg.retriever, cfg)
    written = {k: 0 for k in KS}

    for qa in tqdm(qas, desc="materialize"):
        missing = [k for k in KS if qa.id not in done[k]]
        if not missing:
            continue
        # retrieve once at the largest k; the smaller k are prefixes of this order
        top = retriever.retrieve(
            qa.question, k=max(KS), doc_id=qa.doc_name if cfg.doc_scoped else None
        )
        for k in missing:
            chunks = [sc.chunk for sc in top[:k]]
            record = {
                "id": qa.id,
                "question": qa.question,
                "gold_answer": qa.answer,
                "doc_name": qa.doc_name,
                "k": k,
                "n_chunks": len(chunks),
                "sources": [{"doc_id": c.doc_id, "page": c.page, "text": c.text} for c in chunks],
                "prompt": build_prompt(qa.question, chunks),
            }
            files[k].write(json.dumps(record) + "\n")
            files[k].flush()
            written[k] += 1

    for f in files.values():
        f.close()
    summary = " | ".join(f"k{k}: +{written[k]} (skipped {len(done[k])})" for k in KS)
    print(f"materialized prompts -> {OUT_DIR}/\n  {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-materialize generation prompts per question × k."
    )
    parser.add_argument("--limit", type=int, help="Only the first N questions (smoke test).")
    args = parser.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
