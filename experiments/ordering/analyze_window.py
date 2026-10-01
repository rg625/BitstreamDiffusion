"""Window-ordering study: arm vs control with TRAINING-SEED variance, compute-matched.

Every comparison is at a matched (decoder, NFE, training step); cells are never
pooled across those. Per arm = (mode, k):

  acc_seed[s]   accuracy of training seed s, averaged over its sampling seeds
  mean, sd_tr   across training seeds -- sd_tr is the number the w=0.25 study
                never had, and the one that decides whether a delta is real
  sd_samp       sampling-seed SD within a trained model, for scale

Two intervals for delta = mean(arm) - mean(control), because they answer
different questions:

  ci_models     paired bootstrap over PROBLEMS for these specific trained
                models (sampling seeds averaged). Conditional on the weights;
                it cannot see training-seed luck.
  ci_method     Welch t over training seeds (df from the seed counts). The
                interval for the METHOD. With two seeds per arm it is wide on
                purpose; a result that is only significant in ci_models is a
                statement about two checkpoints, not about ordering.

    python experiments/ordering/analyze_window.py --dir results/ordering/window_eval_gh200
"""
import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN_RE = re.compile(r"runs/tasks/tinygsm/([^/]*?)_(none|l2r|r2l|random)_k(inf|\d+)(?:_lm(\d+))?_s(\d+)/checkpoints/")
TAG_RE = re.compile(r"__ck(\d+)__(own|standard)__n(\d+)__s(\d+)")


def load(path):
    d = json.loads(Path(path).read_text())
    m, t = RUN_RE.search(str(d.get("checkpoint", ""))), TAG_RE.search(Path(path).stem)
    pp = d.get("per_problem") or {}
    if not (m and t and pp.get("idx")):
        return None
    order = np.argsort(pp["idx"])
    # the supervision-matched control is its own arm, "lm<k>", compared to the
    # control like any window arm
    mode = m.group(2) if m.group(4) is None else f"lm{m.group(4)}"
    return {"tag": m.group(1), "mode": mode,
            "k": math.inf if m.group(3) == "inf" else int(m.group(3)),
            "train_seed": int(m.group(5)), "step": int(t.group(1)), "decoder": t.group(2),
            "nfe": int(t.group(3)), "samp_seed": int(t.group(4)),
            "idx": np.asarray(pp["idx"])[order],
            "correct": np.asarray(pp["correct"], dtype=float)[order],
            "T1": sum(1 for a in (pp.get("answer") or []) if a is None)}


def seed_matrix(cells):
    """{train_seed: [n_problems] mean correctness over sampling seeds}."""
    by = defaultdict(list)
    for c in cells:
        by[c["train_seed"]].append(c)
    out, samp_sd = {}, []
    for s, cs in by.items():
        out[s] = np.mean([c["correct"] for c in cs], axis=0)
        if len(cs) > 1:
            samp_sd.append(np.std([c["correct"].mean() for c in cs], ddof=1))
    return out, (float(np.mean(samp_sd)) if samp_sd else float("nan"))


def welch(a, c):
    a, c = np.asarray(a), np.asarray(c)
    d = a.mean() - c.mean()
    if len(a) < 2 or len(c) < 2:
        return d, float("nan"), float("nan"), float("nan")
    va, vc = a.var(ddof=1) / len(a), c.var(ddof=1) / len(c)
    se = math.sqrt(va + vc)
    if se == 0:
        return d, d, d, float("inf")
    df = (va + vc) ** 2 / ((va ** 2) / (len(a) - 1) + (vc ** 2) / (len(c) - 1))
    from scipy import stats
    q = stats.t.ppf(0.975, df)
    return d, d - q * se, d + q * se, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ordering/window_eval_gh200")
    ap.add_argument("--out", default="results/ordering/window_summary.json")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    cells = [c for c in (load(p) for p in sorted(glob.glob(f"{args.dir}/*.json"))) if c]
    if not cells:
        print(f"[win] no results in {args.dir}")
        return
    groups = defaultdict(list)
    for c in cells:
        groups[(c["mode"], c["k"], c["decoder"], c["nfe"], c["step"])].append(c)

    rng = np.random.default_rng(0)
    rows = []
    hdr = (f"{'arm':>12s} {'dec':>8s} {'nfe':>5s} {'step':>7s} {'seeds':>5s} {'acc':>7s} "
           f"{'sd_tr':>6s} {'sd_smp':>6s} {'delta':>8s} {'ci_models':>19s} {'ci_method':>19s}")
    print(hdr)
    for key in sorted(groups, key=lambda x: (x[3], x[2], x[4], -1 if x[1] == math.inf else -x[1], x[0])):
        mode, k, dec, nfe, step = key
        arm, samp_sd = seed_matrix(groups[key])
        ctrl_cells = groups.get(("none", math.inf, dec, nfe, step))
        per_seed = {s: float(v.mean()) for s, v in arm.items()}
        row = {"mode": mode, "k": ("inf" if k == math.inf else k), "decoder": dec, "nfe": nfe,
               "step": step, "acc_seed": per_seed, "acc": float(np.mean(list(per_seed.values()))),
               "sd_train": (float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else None),
               "sd_samp": samp_sd,
               "T1": float(np.mean([c["T1"] for c in groups[key]]))}
        dtxt = ci1 = ci2 = ""
        if ctrl_cells is not None and mode != "none":
            ctrl, _ = seed_matrix(ctrl_cells)
            A = np.mean(list(arm.values()), axis=0)
            C = np.mean(list(ctrl.values()), axis=0)
            n = len(A)
            bs = rng.integers(0, n, size=(args.n_boot, n))
            dd = A[bs].mean(1) - C[bs].mean(1)
            lo, hi = np.percentile(dd, [2.5, 97.5])
            d, wlo, whi, df = welch([v.mean() for v in arm.values()], [v.mean() for v in ctrl.values()])
            row.update(delta=float(A.mean() - C.mean()), ci_models=[float(lo), float(hi)],
                       ci_method=[wlo, whi], df_method=df,
                       sig_models=bool(lo > 0 or hi < 0),
                       sig_method=bool(np.isfinite(wlo) and (wlo > 0 or whi < 0)))
            dtxt = f"{row['delta']:+8.4f}"
            ci1 = f"[{lo:+.3f},{hi:+.3f}]{'*' if row['sig_models'] else ' '}"
            ci2 = (f"[{wlo:+.3f},{whi:+.3f}]{'*' if row['sig_method'] else ' '}"
                   if np.isfinite(wlo) else "n/a (1 seed)")
        armtxt = "control" if mode == "none" else f"{mode} k={k}"
        sdt = f"{row['sd_train']:.4f}" if row["sd_train"] is not None else "  -   "
        print(f"{armtxt:>12s} {dec:>8s} {nfe:>5d} {step//1000:>6d}k {len(per_seed):>5d} "
              f"{row['acc']:7.4f} {sdt:>6s} {samp_sd:6.4f} {dtxt:>8s} {ci1:>19s} {ci2:>19s}")
        rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n[win] wrote {args.out}   (* = interval excludes zero)")
    print("[win] ci_models conditions on the trained weights; ci_method is the claim about ordering.")


if __name__ == "__main__":
    main()
