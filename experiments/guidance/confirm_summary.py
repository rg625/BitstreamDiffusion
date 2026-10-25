#experiments/guidance/confirm_summary.py
"""Headline table for the confirmation stage (Phase 16).

`analyse.py` pairs each guided run against *its own seed's* baseline, which is
the right unit for checking a single run. But the confirmation grids ran three
seeds, and the question the report has to answer is about the method, not about
one seed of it. So this script pools differently:

  * for every GSM8K problem, average the per-problem outcome over seeds, for the
    guided arm and for the baseline arm separately;
  * bootstrap over *problems* (the independent sampling unit -- seeds are three
    repeated measurements of the same 1319 problems, not 3957 fresh draws).

Treating the 3x1319 rows as independent would understate the interval by ~sqrt(3);
resampling problems and carrying the seed-average keeps the pairing intact and
the unit of inference honest.

    python -m experiments.guidance.confirm_summary results/guidance
"""
from __future__ import annotations

import argparse
import csv
import json
import collections
import random
from pathlib import Path
from typing import Dict, List, Optional

CONFIRM_GRIDS = ("cfg_confirm", "sg_confirm", "ag_confirm", "factorial_confirm")


def _f(r: Dict, k: str) -> Optional[float]:
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return None


def label(r: Dict) -> str:
    m = r.get("method", "")
    if m == "CFG":
        return f"CFG w={_f(r,'guidance_scale'):g}"
    if m == "AG":
        return f"AG w={_f(r,'ag_scale'):g} bad={r.get('bad_step','?')}"
    if m in ("SG-prev", "SG-exact"):
        return f"{m} w={_f(r,'sg_scale'):g}"
    return m


def outcomes(path: str) -> Dict[int, int]:
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        return {}
    pp = d.get("per_problem") or {}
    idx, cor = pp.get("idx"), pp.get("correct")
    if not idx or not cor:
        return {}
    return {int(i): int(c) for i, c in zip(idx, cor)}


def seed_mean(files: List[str], min_idx: Optional[int] = None) -> Dict[int, float]:
    """Per-problem outcome averaged over seeds, optionally restricted to a holdout."""
    acc = collections.defaultdict(list)
    for f in files:
        for i, c in outcomes(f).items():
            if min_idx is not None and i < min_idx:
                continue
            acc[i].append(c)
    return {i: sum(v) / len(v) for i, v in acc.items() if v}


def paired_bootstrap(a: Dict[int, float], b: Dict[int, float],
                     n_boot: int = 20000, seed: int = 0):
    """Bootstrap the paired difference b - a, resampling problems."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 20:
        return None
    diffs = [b[k] - a[k] for k in keys]
    n = len(diffs)
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    # Two-sided bootstrap p-value for H0: delta = 0.
    centred = [x - point for x in boots]
    p = sum(1 for x in centred if abs(x) >= abs(point)) / n_boot
    return {"delta": point, "ci95_low": lo, "ci95_high": hi,
            "p_value": p, "n_problems": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="directory containing all_cells.csv")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--holdout-from", type=int, default=None, metavar="IDX",
                    help="Keep only problems with index >= IDX. The exploration "
                         "sweeps ran on a 250-problem PREFIX and chose the "
                         "operating points; the confirmation set contains those "
                         "250, so --holdout-from 250 re-estimates every effect "
                         "on problems that played no part in the selection.")
    args = ap.parse_args()

    root = Path(args.results)
    rows = list(csv.DictReader(open(root / "all_cells.csv")))
    conf = [r for r in rows if r.get("grid") in CONFIRM_GRIDS]
    if not conf:
        raise SystemExit("no confirmation-grid rows in all_cells.csv")

    by = collections.defaultdict(list)
    for r in conf:
        by[label(r)].append(r)

    base_files = [r["result_file"] for r in by.get("baseline", [])]
    if not base_files:
        raise SystemExit("no baseline rows in the confirmation grids")
    base = seed_mean(base_files, args.holdout_from)
    base_acc = sum(base.values()) / len(base)

    out = []
    for name, rs in by.items():
        if name == "baseline":
            continue
        arm = seed_mean([r["result_file"] for r in rs], args.holdout_from)
        st = paired_bootstrap(base, arm, n_boot=args.n_boot)
        if st is None:
            continue
        nfe = [_f(r, "efficiency.nfe_per_sample") for r in rs]
        nfe = [x for x in nfe if x is not None]
        st.update({
            "config": name,
            "accuracy": sum(arm.values()) / len(arm),
            "baseline": base_acc,
            "n_seeds": len(rs),
            "nfe_per_sample": (sum(nfe) / len(nfe)) if nfe else None,
        })
        out.append(st)
    out.sort(key=lambda x: -x["delta"])

    hdr = (f"{'config':<26}{'acc':>8}{'base':>8}{'delta':>9}"
           f"{'95% CI':>20}{'p':>8}{'NFE':>7}{'seeds':>7}")
    scope = (f"holdout: problems >= {args.holdout_from}"
             if args.holdout_from else "all problems")
    print(f"\nConfirmation stage -- GSM8K test, {len(base)} problems ({scope}), "
          f"seed-averaged, paired bootstrap over problems "
          f"({args.n_boot} resamples)\n")
    print(hdr)
    print("-" * len(hdr))
    for r in out:
        ci = f"[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]"
        p = "<1e-4" if r["p_value"] == 0 else f"{r['p_value']:.4f}"
        nfe = f"{r['nfe_per_sample']:.0f}" if r["nfe_per_sample"] else "-"
        print(f"{r['config']:<26}{r['accuracy']:>8.4f}{r['baseline']:>8.4f}"
              f"{r['delta']:>+9.4f}{ci:>20}{p:>8}{nfe:>7}{r['n_seeds']:>7}")

    suffix = f"_holdout{args.holdout_from}" if args.holdout_from else ""
    dest = Path(args.out or (root / f"confirm_summary{suffix}.json"))
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n[confirm] wrote {dest}")


if __name__ == "__main__":
    main()
