"""Ragas over the selected generation answer files, scored by a local critic.

Runs Ragas (faithfulness, answer_relevancy, context_precision, context_recall)
on the same (model, k) cells judged by Claude and Prometheus, so the three
evaluation lenses line up. Critic LLM = Ollama Mistral Nemo 12B (local, no TPM
cap -- Groq's free tier can't fit the k20 contexts), embeddings = Ollama bge-m3.

Reuses the tested ``ragas_runner.run`` per file (resumable, one output file per
answers file: ``data/processed/ragas/{stem}_ragas_by_{critic}.jsonl``). Scores
the first ``--limit`` questions per file (default 50) since Ragas is ~4 LLM
calls per question and the local critic is slow on a power-capped GPU.

    uv run python scripts/ragas_benchmark.py
    uv run python scripts/ragas_benchmark.py --models qwen3.5:4b --ks 10 --limit 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from src.evaluation import ragas_runner
from src.evaluation.config import RagasConfig
from src.llm.config import LLMConfig

ANSWERS_DIR = Path("data/processed/answers")
COLLECTION = "docling_hybrid_1024_bge-m3"
RETRIEVER = "reranked"

# The three best models by equivalent at k20 (ADR 0002) plus granite4.1:3b as a
# low-capability anchor.
DEFAULT_MODELS = ["granite4.1:8b", "qwen3.5:4b", "qwen3.5:9b", "granite4.1:3b"]
# k10, not k20: a k20 faithfulness call is ~21k tokens -> critic num_ctx ~24k
# whose KV cache no longer co-resides with bge-m3 on an 8 GB GPU (swap or
# truncation). k10 (~14k/call) fits num_ctx 16384 alongside bge-m3 -- no swap,
# no truncation. Ragas measures generation quality, which is not k-sensitive.
DEFAULT_KS = [10]
# Critic served as a num_ctx variant: Ragas reaches Ollama through the OpenAI
# endpoint, which can't pass num_ctx per request, so the base model would load
# at Ollama's 4096 default and silently truncate the ~14k faithfulness prompt.
# The variant (created on the fly) pins num_ctx; llama3.1:8b + its 16k KV +
# bge-m3 stay co-resident (measured ~7.6 GB), so no swap.
CRITIC_BASE = "llama3.1:8b"
CRITIC_NUM_CTX = 16384
CRITIC_VARIANT = f"ragas-critic-{CRITIC_BASE.replace(':', '-')}-{CRITIC_NUM_CTX}"
CRITIC_MODEL = f"ollama_chat/{CRITIC_VARIANT}"
# Ragas default (1024) truncates faithfulness's per-statement JSON on long
# answers; its docs recommend 4096+.
CRITIC_MAX_TOKENS = 4096
EMBEDDING_MODEL = "bge-m3"
# Generation metrics only: context_precision/recall are retriever-only
# (identical across generators at fixed k) and already covered by ADR 0001.
METRICS = ["faithfulness", "answer_relevancy"]


def _ensure_critic_variant() -> None:
    """Create the num_ctx critic variant in Ollama if it isn't there yet."""
    listed = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if CRITIC_VARIANT in listed:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(f"FROM {CRITIC_BASE}\nPARAMETER num_ctx {CRITIC_NUM_CTX}\n")
        path = f.name
    subprocess.run(["ollama", "create", CRITIC_VARIANT, "-f", path], capture_output=True)
    os.unlink(path)
    print(f"created critic variant {CRITIC_VARIANT} (num_ctx {CRITIC_NUM_CTX})")


def _answers_path(model: str, k: int) -> Path:
    model_id = f"ollama_chat/{model}".replace("/", "_")
    return ANSWERS_DIR / f"{RETRIEVER}_{COLLECTION}_{model_id}_k{k}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Ragas over selected answer files (local critic).")
    ap.add_argument("--models", help=f"Comma-separated (default: {','.join(DEFAULT_MODELS)}).")
    ap.add_argument("--ks", help=f"Comma-separated k (default: {','.join(map(str, DEFAULT_KS))}).")
    ap.add_argument(
        "--files", help="Comma-separated explicit answer file paths (overrides model/k)."
    )
    ap.add_argument(
        "--limit", type=int, default=50, help="First N questions per file (default 50)."
    )
    ap.add_argument("--metrics", help=f"Comma-separated metrics (default: {','.join(METRICS)}).")
    args = ap.parse_args()

    if args.files:
        paths = [Path(p) for p in args.files.split(",")]
    else:
        models = args.models.split(",") if args.models else DEFAULT_MODELS
        ks = [int(k) for k in args.ks.split(",")] if args.ks else DEFAULT_KS
        paths = [_answers_path(m, k) for m in models for k in ks]

    metrics = args.metrics.split(",") if args.metrics else METRICS
    _ensure_critic_variant()
    llm = LLMConfig(model=CRITIC_MODEL, max_tokens=CRITIC_MAX_TOKENS)
    print(f"Ragas critic: {CRITIC_MODEL} | embeddings: {EMBEDDING_MODEL} | metrics: {metrics}")
    print(f"limit {args.limit}/file | {len(paths)} file(s)\n")
    for p in paths:
        if not p.exists():
            print(f"  missing {p.name} -- skipped")
            continue
        config = RagasConfig(
            answers_path=str(p), llm=llm, embedding_model=EMBEDDING_MODEL, metrics=metrics
        )
        ragas_runner.run(config, limit=args.limit)


if __name__ == "__main__":
    main()
