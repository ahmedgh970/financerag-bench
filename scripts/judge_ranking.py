"""Compare the Claude and Prometheus judges by the model ranking they induce.

The two judges score on different axes -- Claude on equivalent % (correct AND
grounded), Prometheus on a 1-5 rubric mean -- so they are not compared verdict
by verdict. Instead, each ranks the judged (model, k) cells, and we report how
much the two rankings agree (Spearman rho, Kendall tau). A high rank correlation
means the cheap local reproducible judge orders the models the same way as the
frontier judge, even though the raw numbers differ.

Auto-discovers every cell that has BOTH a ``*_judged_by_claude.jsonl`` and a
``*_judged_by_prometheus.jsonl`` in data/processed/judged/.

    uv run python scripts/judge_ranking.py
"""

from __future__ import annotations

import json
from pathlib import Path

JUDGED_DIR = Path("data/processed/judged")


def _cell(stem: str) -> str:
    """reranked_..._ollama_chat_{model}_k{k}_judged_by_X -> {model}_k{k}."""
    core = stem.split("_judged_by_", 1)[0]
    return core.split("ollama_chat_", 1)[-1] if "ollama_chat_" in core else core


def _claude_equivalent(path: Path) -> float:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return 100.0 * sum(bool(r["equivalent"]) for r in recs) / len(recs)


def _prometheus_mean(path: Path) -> float:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    scores = [r["score"] for r in recs if r.get("score") is not None]
    return sum(scores) / len(scores)


def _ranks(values: list[float]) -> list[float]:
    """Average (tie-corrected) ranks, rank 1 = highest value (best)."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank over the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rho = Pearson on the ranks (tie-corrected)."""
    return _pearson(_ranks(a), _ranks(b))


def _kendall_tau(a: list[float], b: list[float]) -> float:
    """Kendall tau-b (concordant vs discordant pairs, tie-adjusted)."""
    n = len(a)
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            s = da * db
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
            else:
                ta += da == 0
                tb += db == 0
    n0 = conc + disc + ta + tb
    denom = ((n0 - ta) * (n0 - tb)) ** 0.5
    return (conc - disc) / denom if denom else 0.0


def main() -> None:
    claude = {_cell(p.stem): p for p in JUDGED_DIR.glob("*_judged_by_claude.jsonl")}
    prometheus = {_cell(p.stem): p for p in JUDGED_DIR.glob("*_judged_by_prometheus.jsonl")}
    cells = sorted(set(claude) & set(prometheus))
    if not cells:
        raise SystemExit("No cell judged by BOTH claude and prometheus yet.")

    equiv = {c: _claude_equivalent(claude[c]) for c in cells}
    pmean = {c: _prometheus_mean(prometheus[c]) for c in cells}
    equiv_rank = dict(zip(cells, _ranks([equiv[c] for c in cells]), strict=True))
    pmean_rank = dict(zip(cells, _ranks([pmean[c] for c in cells]), strict=True))

    print(f"{len(cells)} cell(s) judged by both\n")
    print(f"{'cell':<20} {'Claude equiv%':>13} {'rank':>5}   {'Prom mean':>9} {'rank':>5}")
    print("-" * 62)
    for c in sorted(cells, key=lambda c: equiv[c], reverse=True):
        print(
            f"{c:<20} {equiv[c]:>12.1f}% {equiv_rank[c]:>5.1f}   "
            f"{pmean[c]:>9.2f} {pmean_rank[c]:>5.1f}"
        )
    print("-" * 62)

    a = [equiv[c] for c in cells]
    b = [pmean[c] for c in cells]
    print(f"\nSpearman rho = {_spearman(a, b):+.3f}   Kendall tau = {_kendall_tau(a, b):+.3f}")
    print("(rank correlation between the two judges' model orderings; +1 = identical order)")


if __name__ == "__main__":
    main()
