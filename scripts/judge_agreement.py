"""Compare judge models on the same answers: agreement vs a reference judge.

Given one answers file judged by several judges (data/processed/judged/*), this
picks a reference judge (default: the Groq llama-70b one, already validated at
97.4% vs human) and reports, for every other judge, how often its verdicts match
the reference -- percent agreement plus Cohen's kappa (chance-corrected) on the
headline `equivalent` verdict. Lets us pick a local judge empirically instead of
arguing capability in the abstract.

    uv run python scripts/judge_agreement.py --answers command-r7b_k5
    uv run python scripts/judge_agreement.py --answers command-r7b_k5 --reference-substr groq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

JUDGED_DIR = Path("data/processed/judged")


def _load(path: Path) -> dict[str, dict]:
    """id -> verdict record."""
    return {
        r["id"]: r
        for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def _cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary label lists (chance-corrected agreement)."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)  # expected agreement by chance
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def _agreement(ref: dict[str, dict], cand: dict[str, dict]) -> dict:
    ids = sorted(set(ref) & set(cand))
    n = len(ids)
    out = {"n": n}
    for key in ("correct", "grounded", "equivalent"):
        r = [bool(ref[i][key]) for i in ids]
        c = [bool(cand[i][key]) for i in ids]
        out[key] = sum(x == y for x, y in zip(r, c, strict=True)) / n if n else 0.0
    out["kappa_equiv"] = _cohen_kappa(
        [bool(ref[i]["equivalent"]) for i in ids],
        [bool(cand[i]["equivalent"]) for i in ids],
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge-vs-judge agreement on one answers file.")
    ap.add_argument("--answers", required=True, help="Answers stem, e.g. command-r7b_k5.")
    ap.add_argument(
        "--reference-substr", default="groq", help="Substring identifying the reference judge file."
    )
    args = ap.parse_args()

    files = sorted(JUDGED_DIR.glob(f"*{args.answers}_judged_by_*.jsonl"))
    if not files:
        raise SystemExit(f"No judged files matching *{args.answers}_judged_by_* in {JUDGED_DIR}")

    ref_files = [f for f in files if args.reference_substr in f.name]
    if not ref_files:
        raise SystemExit(
            f"No reference file (substring {args.reference_substr!r}) among:\n  "
            + "\n  ".join(f.name for f in files)
        )
    ref_path = ref_files[0]
    ref = _load(ref_path)

    def _judge_name(f: Path) -> str:
        return f.name.split("_judged_by_", 1)[-1].removesuffix(".jsonl")

    print(f"reference judge: {_judge_name(ref_path)}  (n={len(ref)})\n")
    print(
        f"{'candidate judge':<40} {'n':>4} {'correct':>8} {'grounded':>9} {'equiv':>7} {'kappa':>7}"
    )
    print("-" * 80)
    for f in files:
        if f == ref_path:
            continue
        a = _agreement(ref, _load(f))
        print(
            f"{_judge_name(f):<40} {a['n']:>4} {a['correct'] * 100:>7.1f}% "
            f"{a['grounded'] * 100:>8.1f}% {a['equivalent'] * 100:>6.1f}% {a['kappa_equiv']:>7.2f}"
        )
    print("-" * 80)
    print("agreement = % of questions where the candidate's verdict matches the reference.")
    print(
        "kappa = chance-corrected agreement on `equivalent` (>0.6 substantial, >0.8 near-perfect)."
    )


if __name__ == "__main__":
    main()
