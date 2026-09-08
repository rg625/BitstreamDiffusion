"""Paired analysis of the temporal-ordering arms.

Every arm is evaluated on the SAME problems with the SAME seed, so the correct
comparison is PAIRED: bootstrap over problems, resampling the (control,
intervention) outcome pair together. An unpaired comparison would throw away the
pairing and inflate the interval.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load(path):
    d = json.loads(Path(path).read_text())
    pp = d.get("per_problem") or {}
    idx = pp.get("idx")
    cor = pp.get("correct")
    if not idx or cor is None:
        return None
    return {
        "file": Path(path).name,
        "order_w": d.get("order_w"), "order_mode": d.get("order_mode"),
        "steps": d.get("steps"), "seed": d.get("seed"),
        "accuracy": d.get("accuracy"),
        "n": len(idx),
        "idx": np.asarray(idx), "correct": np.asarray(cor, dtype=float),
        "invalid_token_rate": d.get("invalid_token_rate"),
        "samples_per_sec": (d.get("efficiency") or {}).get("samples_per_sec"),
    }


def paired_delta(ctrl, arm, n_boot=10000, seed=0):
    """Paired bootstrap over problems. Returns (delta, lo, hi, p_two_sided)."""
    common, ci, ai = np.intersect1d(ctrl["idx"], arm["idx"], return_indices=True)
    c, a = ctrl["correct"][ci], arm["correct"][ai]
    d = float(a.mean() - c.mean())
    rng = np.random.default_rng(seed)
    n = len(common)
    bs = rng.integers(0, n, size=(n_boot, n))
    deltas = a[bs].mean(1) - c[bs].mean(1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided p by the sign test on the bootstrap distribution
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return d, float(lo), float(hi), float(min(p, 1.0)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ordering/screen")
    ap.add_argument("--out", default="results/ordering/summary.json")
    args = ap.parse_args()

    runs = [r for r in (load(p) for p in sorted(glob.glob(f"{args.dir}/*.json"))) if r]
    if not runs:
        print(f"[ord] no per-problem results in {args.dir}")
        return
    by_key = {}
    for r in runs:
        by_key.setdefault((r["steps"], r["seed"]), []).append(r)

    out = []
    print(f"{'arm':28s} {'steps':>6} {'seed':>5} {'n':>5} {'acc':>7} "
          f"{'delta':>8} {'95% CI':>18} {'p':>7} {'inval':>7}")
    for (steps, seed), group in sorted(by_key.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0)):
        ctrl = next((g for g in group if (g["order_w"] or 0.0) == 0.0), None)
        if ctrl is None:
            print(f"  [no control at steps={steps} seed={seed}]")
            continue
        print(f"{'CONTROL (w=0)':28s} {steps:>6} {seed:>5} {ctrl['n']:>5} "
              f"{ctrl['accuracy']:>7.4f} {'-':>8} {'-':>18} {'-':>7} "
              f"{(ctrl['invalid_token_rate'] or 0):>7.4f}")
        for g in sorted(group, key=lambda x: (x["order_mode"] or "", x["order_w"] or 0)):
            if g is ctrl:
                continue
            d, lo, hi, p, n = paired_delta(ctrl, g)
            name = f"{g['order_mode']} w={g['order_w']}"
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"{name:28s} {steps:>6} {seed:>5} {n:>5} {g['accuracy']:>7.4f} "
                  f"{d:>+8.4f} [{lo:>+7.4f},{hi:>+7.4f}] {p:>7.4f}{sig} "
                  f"{(g['invalid_token_rate'] or 0):>7.4f}")
            out.append({"arm": name, "steps": steps, "seed": seed, "n": n,
                        "accuracy": g["accuracy"], "control_accuracy": ctrl["accuracy"],
                        "delta": d, "ci95": [lo, hi], "p": p,
                        "significant": bool(lo > 0 or hi < 0)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[ord] wrote {args.out}   (* = 95% CI excludes zero)")


if __name__ == "__main__":
    main()
