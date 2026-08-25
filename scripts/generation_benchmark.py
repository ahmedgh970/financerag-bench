"""Multi-model generation benchmark: run the local Ollama lineup on the
pre-materialized prompts (scripts/materialize_prompts.py) across k=5/10/20.

Each (model, k) combination is answered independently, at the num_ctx budget
fixed for that k (10240/18432/30720 -- see README's "Generation benchmark"
section for how these were measured), and written in the exact schema/naming
`make answer` uses, so `make judge`/`make ragas` can score the result without
any changes.

Skips a (model, k) pair outright when the model's architectural context can't
fit the budget -- except command-r7b at k=5, which runs at its own max (8192)
instead of the shared budget (10240), since it's still the lineup's fastest
option and k5 is its one usable depth.

Resumable per (model, k) output file: question ids already answered are
skipped, so a run interrupted mid-model/mid-k can continue where it left off.

    uv run python scripts/generation_benchmark.py
    uv run python scripts/generation_benchmark.py --models llama3.2:3b,qwen3.5:4b
    uv run python scripts/generation_benchmark.py --ks 5
    uv run python scripts/generation_benchmark.py --limit 5   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

OLLAMA = "http://localhost:11434"
KS = [5, 10, 20]
TEMPERATURE = 0.0
NUM_PREDICT = 1024  # matches LLMConfig.max_tokens default
BUDGET_HOURS = 3  # soft target per (model, k) across the 150 QA -- warned, not enforced

# Same corpus/retriever identity as materialize_prompts.py and `make answer`,
# used both to locate the prompt files and to name the answer files so
# make judge / make ragas can point straight at them.
COLLECTION = "docling_hybrid_1024_bge-m3"
RETRIEVER = "reranked"
PROMPTS_DIR = Path("data/processed/prompts")
ANSWERS_DIR = Path("data/processed/answers")

# model -> architectural max context length (from `ollama show`)
MODELS = {
    "llama3.2:3b": 131072,
    "granite4.1:3b": 131072,
    "qwen3.5:4b": 262144,
    "mistral:7b": 32768,
    "command-r7b": 8192,
    "llama3.1:8b": 131072,
    "granite4.1:8b": 131072,
    "qwen3.5:9b": 262144,
    "gemma4:12b": 262144,
    "mistral-nemo": 1024000,
}

# num_ctx per k: the true max real-prompt token count across all 150 questions
# (measured with mistral:7b, the most expensive tokenizer in the lineup) plus
# a 1024-token output budget, no extra margin (see README).
TARGET_NUM_CTX = {5: 10240, 10: 18432, 20: 30720}

# command-r7b's own context (8192) is below the shared k5 budget (10240) --
# run it at its own max instead of skipping it outright.
NUM_CTX_OVERRIDE = {("command-r7b", 5): 8192}


def _prompts_path(k: int) -> Path:
    return PROMPTS_DIR / f"prompts_reranked-dense_{COLLECTION}_docscoped_pf50_k{k}.jsonl"


def _answers_path(model: str, k: int) -> Path:
    model_id = f"ollama_chat/{model}".replace("/", "_")
    return ANSWERS_DIR / f"{RETRIEVER}_{COLLECTION}_{model_id}_k{k}.jsonl"


def _answered_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _feasible_ks(model: str, archmax: int, ks: list[int]) -> list[tuple[int, int]]:
    """(k, num_ctx) pairs this model's context can actually fit, among ``ks``."""
    out = []
    for k in ks:
        nctx = NUM_CTX_OVERRIDE.get((model, k), TARGET_NUM_CTX[k])
        if nctx <= archmax:
            out.append((k, nctx))
        else:
            print(f"  k={k}: num_ctx {nctx} > arch max {archmax} -> skipped")
    return out


def _unload_all() -> None:
    for m in requests.get(f"{OLLAMA}/api/ps", timeout=30).json().get("models", []):
        subprocess.run(["ollama", "stop", m["name"]], capture_output=True)
    time.sleep(1)


def _create_variant(model: str, num_ctx: int) -> str:
    name = "genbench-" + model.replace(":", "-") + f"-{num_ctx}"
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(f"FROM {model}\nPARAMETER num_ctx {num_ctx}\n")
        path = f.name
    subprocess.run(["ollama", "create", name, "-f", path], capture_output=True)
    os.unlink(path)
    return name


def _remove_variant(name: str) -> None:
    subprocess.run(["ollama", "stop", name], capture_output=True)
    subprocess.run(["ollama", "rm", name], capture_output=True)


def _chat(variant: str, prompt: str) -> tuple[str, float, str]:
    """Chat-complete ``prompt`` on the loaded ``variant``; matches the ollama_chat/*
    chat route `make answer` uses (single user turn, no system prompt).

    ``think: False`` disables the reasoning channel on thinking-capable models
    (qwen3.5, gemma4). Left enabled, their chain-of-thought is emitted first and
    can exhaust the ``num_predict`` budget before any answer token is produced,
    so Ollama returns an empty ``content`` (the reasoning lands in a separate
    ``thinking`` field that is scratch work, not the answer). Plain-instruct
    models ignore the flag, so it is safe to send unconditionally.
    """
    start = time.perf_counter()
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": variant,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
        },
        timeout=1800,
    ).json()
    latency = time.perf_counter() - start
    return r.get("message", {}).get("content", ""), latency, r.get("done_reason", "")


def _run_one(model: str, k: int, num_ctx: int, limit: int | None) -> None:
    prompts_path = _prompts_path(k)
    if not prompts_path.exists():
        print(f"  k={k}: missing {prompts_path} -- run `make prompts` first, skipping")
        return

    records = [json.loads(line) for line in prompts_path.open(encoding="utf-8") if line.strip()]
    if limit is not None:
        records = records[:limit]

    out_path = _answers_path(model, k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _answered_ids(out_path)
    remaining = [r for r in records if r["id"] not in done]
    if not remaining:
        print(f"  k={k}: all {len(records)} QA already answered -> {out_path}")
        return

    variant = _create_variant(model, num_ctx)
    _unload_all()
    elapsed = 0.0
    empties = 0
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for rec in tqdm(remaining, desc=f"{model} k={k} (ctx={num_ctx})"):
                text, latency, done_reason = _chat(variant, rec["prompt"])
                elapsed += latency
                if not text.strip():
                    empties += 1
                    tqdm.write(f"  EMPTY content for {rec['id']} (done_reason={done_reason!r})")
                out = {
                    "id": rec["id"],
                    "question": rec["question"],
                    "gold_answer": rec["gold_answer"],
                    "generated_answer": text,
                    "sources": rec["sources"],
                    "latency_s": latency,
                }
                f.write(json.dumps(out) + "\n")
                f.flush()
    finally:
        _remove_variant(variant)

    budget_s = BUDGET_HOURS * 3600
    flag = " -- OVER BUDGET" if elapsed > budget_s else ""
    empty_flag = f" -- {empties} EMPTY" if empties else ""
    print(
        f"  k={k}: answered {len(remaining)} new QA (skipped {len(done)}) in "
        f"{elapsed / 60:.1f} min (budget {BUDGET_HOURS}h){flag}{empty_flag} -> {out_path}"
    )


def run(models: list[str], ks: list[int], limit: int | None) -> None:
    for model in models:
        archmax = MODELS[model]
        print(f"\n=== {model} (arch max {archmax}) ===")
        for k, num_ctx in _feasible_ks(model, archmax, ks):
            _run_one(model, k, num_ctx, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model generation benchmark.")
    parser.add_argument("--models", help="Comma-separated subset of the lineup (default: all).")
    parser.add_argument("--ks", help="Comma-separated subset of k (default: 5,10,20).")
    parser.add_argument("--limit", type=int, help="Only the first N questions (smoke test).")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else list(MODELS)
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"Unknown model {m!r}. Known: {', '.join(MODELS)}")
    ks = [int(k) for k in args.ks.split(",")] if args.ks else KS

    run(models, ks, args.limit)


if __name__ == "__main__":
    main()
