"""Report real specs for a set of Ollama models: arch context, params, VRAM.

For each model: `ollama show` gives the architectural context length and param
count; loading it at a target num_ctx (default 8192, the k5 context budget) and
reading /api/ps gives the actual VRAM footprint and the GPU/CPU split -- the two
numbers that decide which models fit and which k they can run on an 8GB card.

    uv run python scripts/ollama_model_specs.py
    uv run python scripts/ollama_model_specs.py --num-ctx 14336   # k10 budget
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
import time

import requests

OLLAMA = "http://localhost:11434"
MODELS = [
    "llama3.2:3b",
    "granite4.1:3b",
    "qwen3.5:4b",
    "mistral:7b",
    "command-r7b",
    "llama3.1:8b",
    "granite4.1:8b",
    "qwen3.5:9b",
    "gemma4:12b",
    "mistral-nemo",
    "phi4",
]


def _show(model: str) -> tuple[str, str]:
    out = subprocess.run(["ollama", "show", model], capture_output=True, text=True).stdout
    ctx = re.search(r"context length\s+(\S+)", out)
    params = re.search(r"parameters\s+(\S+)", out)
    return (params.group(1) if params else "?", ctx.group(1) if ctx else "?")


def _api_ps() -> list[dict]:
    return requests.get(f"{OLLAMA}/api/ps", timeout=30).json().get("models", [])


def _unload_all() -> None:
    for m in _api_ps():
        subprocess.run(["ollama", "stop", m["name"]], capture_output=True)
    time.sleep(1)


def _measure_vram(model: str, num_ctx: int) -> dict:
    name = "spec-" + model.replace(":", "-").replace("/", "-")
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(f"FROM {model}\nPARAMETER num_ctx {num_ctx}\n")
        path = f.name
    subprocess.run(["ollama", "create", name, "-f", path], capture_output=True)
    os.unlink(path)
    _unload_all()
    requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": name, "prompt": "hi", "stream": False, "options": {"num_predict": 1}},
        timeout=600,
    )
    ps = next((m for m in _api_ps() if m["name"].startswith(name)), None)
    subprocess.run(["ollama", "stop", name], capture_output=True)
    subprocess.run(["ollama", "rm", name], capture_output=True)
    size = ps["size"] if ps else 0
    vram = ps.get("size_vram", 0) if ps else 0
    return {
        "size_gb": size / 1e9,
        "vram_gb": vram / 1e9,
        "gpu_pct": (100 * vram / size) if size else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Ollama model specs (ctx, params, VRAM).")
    parser.add_argument(
        "--num-ctx", type=int, default=8192, help="num_ctx to load at (default 8192)."
    )
    args = parser.parse_args()

    print(f"{'model':<24} {'params':>8} {'arch ctx':>10} {'VRAM(GPU/total)':>18} {'GPU%':>6}")
    print("-" * 70)
    for model in MODELS:
        params, ctx = _show(model)
        try:
            m = _measure_vram(model, args.num_ctx)
            vram = f"{m['vram_gb']:.1f}/{m['size_gb']:.1f} GB"
            gpu = f"{m['gpu_pct']:.0f}%"
        except Exception as e:  # noqa: BLE001
            vram, gpu = f"ERROR: {e}", "-"
        print(f"{model:<24} {params:>8} {ctx:>10} {vram:>18} {gpu:>6}")


if __name__ == "__main__":
    main()
