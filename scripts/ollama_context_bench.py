"""Context-window benchmark: real RAG prompts (k=5/10/20) across Ollama models.

Answers a concrete planning question: for each (model, k), what num_ctx do we
need so the retrieved chunks aren't truncated, does the model even support it
(architectural max), and what does it cost (VRAM, GPU/CPU split, inference time)?

The prompts are the *real* ones the generation pipeline sends: a real golden-set
question, retrieved + reranked through the actual reranked(dense) retriever at
each k, wrapped by the same build_prompt() as `make answer`.

Two phases, GPU-separated:
  1. --retrieve: build the real prompts (uses the GPU for BGE-M3 + reranker),
     write them to JSON, then exit so the GPU is fully freed.
  2. default: read those prompts and benchmark each (model, num_ctx) with the
     GPU free for Ollama alone (an accurate VRAM/dispatch reading).

    uv run python scripts/ollama_context_bench.py
    uv run python scripts/ollama_context_bench.py --force   # re-retrieve prompts
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

OLLAMA = "http://localhost:11434"
PROMPTS_FILE = Path(
    "/tmp/claude-1000/-home-ahmed-VSCodeProjects-7-financerag-bench/"
    "4274992a-def3-4544-be1b-0da320d14a10/scratchpad/ctxbench_prompts.json"
)
RAG_CONFIG = "configs/rag/naive_reranked_dense_1024_k5_ollama.yaml"
QUESTION_ID = "financebench_id_03029"
KS = [5, 10, 20]

# num_ctx per k: measured directly (not estimated) on the real corpus -- the
# max real-token count across all 150 questions per k, with mistral:7b (the
# most expensive tokenizer in the lineup) + 1024 tokens output budget, rounded
# up to 2048. No extra margin: this is already the true worst case, not a proxy.
TARGET_NUM_CTX = {5: 10240, 10: 18432, 20: 30720}

# model name -> architectural max context length (from `ollama show`)
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

# command-r7b's own context (8192) is below the shared k5 budget (10240) --
# run it at its own max instead of skipping it outright, since it's still the
# lineup's fastest/smallest option and k5 is its one usable depth.
NUM_CTX_OVERRIDE = {("command-r7b", 5): 8192}


# --------------------------------------------------------------------------- #
# Phase 1: build the real prompts (GPU: retriever + reranker), then exit.
# --------------------------------------------------------------------------- #
def phase_retrieve(outfile: str) -> None:
    from src.evaluation.golden_set import load_golden_set
    from src.llm.prompts import build_prompt
    from src.rag.config import load_rag_config
    from src.retrieval.registry import build_retriever

    cfg = load_rag_config(RAG_CONFIG)
    cfg.rerank_prefetch = max(cfg.rerank_prefetch, max(KS))  # prefetch must cover max k
    qa = next(q for q in load_golden_set(cfg.golden_set_path) if q.id == QUESTION_ID)
    retriever = build_retriever(cfg.retriever, cfg)

    out = {"question_id": qa.id, "question": qa.question, "doc": qa.doc_name, "prompts": {}}
    for k in KS:
        results = retriever.retrieve(
            qa.question, k=k, doc_id=qa.doc_name if cfg.doc_scoped else None
        )
        prompt = build_prompt(qa.question, [sc.chunk for sc in results])
        out["prompts"][str(k)] = {"n_chunks": len(results), "chars": len(prompt), "prompt": prompt}
        print(f"[retrieve] k={k}: {len(results)} chunks, {len(prompt)} chars")

    Path(outfile).write_text(json.dumps(out))
    print(f"[retrieve] wrote {outfile}")


# --------------------------------------------------------------------------- #
# Phase 2 helpers: Ollama load / measure / cleanup.
# --------------------------------------------------------------------------- #
def _api_ps() -> list[dict]:
    return requests.get(f"{OLLAMA}/api/ps", timeout=30).json().get("models", [])


def _unload_all() -> None:
    for m in _api_ps():
        subprocess.run(["ollama", "stop", m["name"]], capture_output=True)
    time.sleep(1)


def _create_variant(model: str, num_ctx: int) -> str:
    name = "ctxbench-" + model.replace(":", "-").replace("/", "-") + f"-{num_ctx}"
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(f"FROM {model}\nPARAMETER num_ctx {num_ctx}\n")
        path = f.name
    subprocess.run(["ollama", "create", name, "-f", path], capture_output=True)
    os.unlink(path)
    return name


def _remove_variant(name: str) -> None:
    subprocess.run(["ollama", "stop", name], capture_output=True)
    subprocess.run(["ollama", "rm", name], capture_output=True)


def measure(model: str, num_ctx: int, prompt: str) -> dict:
    """Load ``model`` at ``num_ctx``, run the prompt, return timing + dispatch + VRAM."""
    variant = _create_variant(model, num_ctx)
    _unload_all()
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": variant,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "2m",
            "options": {"num_predict": 64},  # short answer: measure context cost, not generation
        },
        timeout=1800,
    ).json()

    ps = next((m for m in _api_ps() if m["name"].startswith(variant)), None)
    size = ps["size"] if ps else 0
    vram = ps.get("size_vram", 0) if ps else 0
    _remove_variant(variant)

    prompt_eval_s = r.get("prompt_eval_duration", 0) / 1e9
    prompt_tokens = r.get("prompt_eval_count")
    return {
        "prompt_tokens": prompt_tokens,
        "truncated": bool(prompt_tokens) and prompt_tokens >= num_ctx,
        "load_s": r.get("load_duration", 0) / 1e9,
        "prompt_eval_s": prompt_eval_s,
        "prompt_tok_s": (prompt_tokens / prompt_eval_s) if prompt_eval_s and prompt_tokens else 0,
        "eval_s": r.get("eval_duration", 0) / 1e9,
        "size_gb": size / 1e9,
        "vram_gb": vram / 1e9,
        "gpu_pct": (100 * vram / size) if size else 0,
    }


# --------------------------------------------------------------------------- #
# Phase 2: benchmark matrix.
# --------------------------------------------------------------------------- #
def phase_bench() -> None:
    data = json.loads(PROMPTS_FILE.read_text())
    prompts = data["prompts"]
    print(f"\nquestion: {data['question_id']} ({data['doc']})")
    print("real prompts (question + reranked(dense) chunks + generation template):")
    for k in KS:
        p = prompts[str(k)]
        print(f"  k={k:2d}: {p['n_chunks']} chunks, {p['chars']} chars")

    print("\nnum_ctx target per k:", TARGET_NUM_CTX)

    rows = []
    for model, archmax in MODELS.items():
        for k in KS:
            nctx = NUM_CTX_OVERRIDE.get((model, k), TARGET_NUM_CTX[k])
            if nctx > archmax:
                print(f"\n{model} | k={k} | num_ctx {nctx} > arch max {archmax} -> OUT OF CONTEXT")
                rows.append({"model": model, "k": k, "num_ctx": nctx, "status": "OUT_OF_CONTEXT"})
                continue
            print(f"\n{model} | k={k} | num_ctx={nctx} | measuring ...")
            try:
                m = measure(model, nctx, prompts[str(k)]["prompt"])
            except Exception as e:  # noqa: BLE001 - report and continue the matrix
                print(f"  ERROR: {e}")
                rows.append({"model": model, "k": k, "num_ctx": nctx, "status": f"ERROR: {e}"})
                continue
            status = "TRUNCATED" if m["truncated"] else "ok"
            print(
                f"  {status} | prompt {m['prompt_tokens']} tok | VRAM {m['vram_gb']:.1f}/"
                f"{m['size_gb']:.1f} GB ({m['gpu_pct']:.0f}% GPU) | load {m['load_s']:.1f}s | "
                f"ctx-ingest {m['prompt_eval_s']:.1f}s ({m['prompt_tok_s']:.0f} tok/s)"
            )
            rows.append({"model": model, "k": k, "num_ctx": nctx, "status": status, **m})

    _print_table(rows)
    out = PROMPTS_FILE.with_name("ctxbench_results.json")
    out.write_text(json.dumps({"question": data["question_id"], "rows": rows}, indent=2))
    print(f"\nresults -> {out}")


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    hdr = f"{'model':<22} {'k':>3} {'num_ctx':>8} {'status':>12} {'GPU%':>5} {'VRAM':>7} {'ctx-ingest':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["status"] == "OUT_OF_CONTEXT":
            print(f"{r['model']:<22} {r['k']:>3} {r['num_ctx']:>8} {'OUT OF CTX':>12}")
        elif r["status"].startswith("ERROR"):
            print(f"{r['model']:<22} {r['k']:>3} {r['num_ctx']:>8} {'ERROR':>12}")
        else:
            print(
                f"{r['model']:<22} {r['k']:>3} {r['num_ctx']:>8} {r['status']:>12} "
                f"{r['gpu_pct']:>4.0f}% {r['vram_gb']:>5.1f}GB {r['prompt_eval_s']:>9.1f}s"
            )


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Context-window benchmark across Ollama models.")
    parser.add_argument("--retrieve", metavar="OUTFILE", help="(internal) build prompts and exit")
    parser.add_argument("--force", action="store_true", help="re-retrieve prompts even if cached")
    args = parser.parse_args()

    if args.retrieve:
        phase_retrieve(args.retrieve)
        return

    # Build the real prompts in a subprocess so its GPU memory (retriever +
    # reranker) is fully released before we benchmark the LLMs.
    if args.force or not PROMPTS_FILE.exists():
        PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        print("[main] building real prompts via retriever (subprocess, frees GPU on exit) ...")
        subprocess.run([sys.executable, __file__, "--retrieve", str(PROMPTS_FILE)], check=True)
    else:
        print(f"[main] using cached prompts: {PROMPTS_FILE} (use --force to rebuild)")

    phase_bench()


if __name__ == "__main__":
    main()
