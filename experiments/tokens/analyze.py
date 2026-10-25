"""Token-softmax vs bits: GSM8K learning curve and paired comparison.

Reads the result JSONs that scripts/hpc/arch/tokens_eval_gh200.sh writes into
ONE directory (one machine, one regime; never pooled with another) and reports,
per run and checkpoint step: accuracy per seed, the seed mean, and the
T1/T2/T3 failure breakdown. At every step both arms have, it reports the
paired delta token - bits:

  * per seed: paired bootstrap over problems (same problems, same seed index);
  * pooled:   each problem's correctness averaged over the seeds, then a paired
              bootstrap over problems. Resampling problems, not (problem, seed)
              rows, keeps the three seeds of one problem together, since they
              are not independent evidence.

--gate-only checks the two decode-path anchors and exits 0 (pass) or 1.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# The tag tokens_eval.slurm writes: <run>__ck<9-digit step>__s<seed>.
TAG_RE = re.compile(r"_ema\d_(?P<run>.+)__ck(?P<step>\d{9})__s(?P<seed>\d+)\.json$")

# Decode-path anchors: checkpoints whose accuracy is already known.
#   prod_anchor: production bits model; GH200 ordered-kind eval 0.127/0.135.
#   tok_fs500k : the batch-128 token run; CSD3 scored 0.022/0.026/0.029.
GATES = {
    "prod_anchor": (0.12, 0.20),
    "tok_fs500k_token_ce_s42": (0.010, 0.045),
}
SEEDS = (0, 1, 2)


def load(path):
    m = TAG_RE.search(Path(path).name)
    if not m:
        return None
    d = json.loads(Path(path).read_text())
    pp = d.get("per_problem") or {}
    if not pp.get("idx"):
        return None
    ans = pp.get("answer") or []
    cor = np.asarray(pp["correct"], dtype=float)
    recs = d.get("sample_records") or []
    eff = d.get("efficiency") or {}
    return {
        "run": m.group("run"), "step": int(m.group("step")), "seed": int(m.group("seed")),
        "n": int(d.get("num_examples") or len(cor)),
        "idx": np.asarray(pp["idx"]), "correct": cor,
        "accuracy": float(d["accuracy"]),
        # T1: program did not run / returned nothing. T2: ran, wrong number.
        "T1": sum(1 for a in ans if a is None),
        "T2": sum(1 for a, c in zip(ans, cor) if a is not None and not c),
        # T3: no function definition at all, i.e. degenerate text. Only the
        # first 100 responses are stored, so this is a rate over those.
        "T3_rate": (sum(1 for r in recs if "def " not in (r.get("response") or "")) / len(recs))
                   if recs else None,
        "invalid": d.get("invalid_token_rate"),
        "sps": eff.get("samples_per_sec"),
        "peak_gb": eff.get("peak_gpu_gb"),
        "batch": d.get("batch_size"),
    }


def paired(a_idx, a_cor, b_idx, b_cor, n_boot=10000, seed=0):
    """delta = mean(a) - mean(b) on the problems both evaluated."""
    common, ai, bi = np.intersect1d(a_idx, b_idx, return_indices=True)
    a, b = a_cor[ai], b_cor[bi]
    rng = np.random.default_rng(seed)
    bs = rng.integers(0, len(common), size=(n_boot, len(common)))
    dd = a[bs].mean(1) - b[bs].mean(1)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    p = min(1.0, 2.0 * min((dd <= 0).mean(), (dd >= 0).mean()))
    return {"delta": float(a.mean() - b.mean()), "ci95": [float(lo), float(hi)],
            "p": float(p), "n": int(len(common))}


def seed_mean_per_problem(cells):
    """Per-problem correctness averaged over seeds, on problems every seed has."""
    idx = cells[0]["idx"]
    for c in cells[1:]:
        idx = np.intersect1d(idx, c["idx"])
    rows = []
    for c in cells:
        pos = {int(i): k for k, i in enumerate(c["idx"])}
        rows.append(np.array([c["correct"][pos[int(i)]] for i in idx]))
    return idx, np.mean(rows, axis=0)


def gate(cells):
    ok = True
    for run, (lo, hi) in GATES.items():
        accs = {c["seed"]: c["accuracy"] for c in cells if c["run"] == run and c["n"] == 1319}
        passed = set(accs) >= set(SEEDS) and all(lo <= a <= hi for a in accs.values())
        print(f"  gate {run}: {dict(sorted(accs.items()))}  band {lo}-{hi}  "
              f"{'PASS' if passed else 'NOT PASSED'}")
        ok &= passed
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/tokens/eval_gh200")
    ap.add_argument("--token", default="tok_gh500k_token_ce_s42")
    ap.add_argument("--bits", default="ord_gh500k_none_s42")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()

    cells = [c for c in (load(f) for f in sorted(glob.glob(f"{args.dir}/*.json"))) if c]
    if args.gate_only:
        raise SystemExit(0 if gate(cells) else 1)

    gate(cells)
    by = defaultdict(dict)                       # (run, step) -> {seed: cell}
    for c in cells:
        by[(c["run"], c["step"])][c["seed"]] = c

    print(f"\n{'run':<28} {'step':>7} {'seeds':<17} {'mean':>6} "
          f"{'T1':>5} {'T2':>5} {'T3%':>4} {'inval':>8} {'samp/s':>7}")
    table = []
    for (run, step), seeds in sorted(by.items()):
        cs = [seeds[s] for s in sorted(seeds)]
        accs = [c["accuracy"] for c in cs]
        t3 = [c["T3_rate"] for c in cs if c["T3_rate"] is not None]
        inv = [c["invalid"] for c in cs if c["invalid"] is not None]
        sps = [c["sps"] for c in cs if c["sps"]]
        row = {"run": run, "step": step, "seeds": sorted(seeds), "acc": accs,
               "mean": float(np.mean(accs)),
               "T1": float(np.mean([c["T1"] for c in cs])),
               "T2": float(np.mean([c["T2"] for c in cs])),
               "T3_rate": float(np.mean(t3)) if t3 else None,
               "invalid_token_rate": float(np.mean(inv)) if inv else None,
               "samples_per_sec": float(np.mean(sps)) if sps else None}
        table.append(row)
        print(f"{run:<28} {step:>7} {' '.join(f'{a:.3f}' for a in accs):<17} {row['mean']:>6.3f} "
              f"{row['T1']:>5.0f} {row['T2']:>5.0f} "
              f"{(100 * row['T3_rate'] if row['T3_rate'] is not None else float('nan')):>4.0f} "
              f"{(row['invalid_token_rate'] or 0):>8.1e} {(row['samples_per_sec'] or 0):>7.2f}")

    print(f"\npaired delta, {args.token} - {args.bits}  (positive = tokens better)")
    comps = []
    steps = sorted({s for r, s in by if r == args.token} & {s for r, s in by if r == args.bits})
    for step in steps:
        tk, bt = by[(args.token, step)], by[(args.bits, step)]
        common = sorted(set(tk) & set(bt))
        per_seed = {}
        for s in common:
            per_seed[s] = paired(tk[s]["idx"], tk[s]["correct"], bt[s]["idx"], bt[s]["correct"], seed=s)
        pooled = None
        if common:
            ti, tm = seed_mean_per_problem([tk[s] for s in common])
            bi, bm = seed_mean_per_problem([bt[s] for s in common])
            pooled = paired(ti, tm, bi, bm)
        comps.append({"step": step, "seeds": common, "per_seed": per_seed, "pooled": pooled})
        for s, r in per_seed.items():
            print(f"  step {step:>7} seed {s}: {r['delta']:+.4f}  "
                  f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p={r['p']:.4f}")
        if pooled:
            print(f"  step {step:>7} POOLED ({len(common)} seeds): {pooled['delta']:+.4f}  "
                  f"[{pooled['ci95'][0]:+.4f}, {pooled['ci95'][1]:+.4f}]  p={pooled['p']:.4f}")
    if not steps:
        print("  (no step evaluated for both arms yet)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"dir": args.dir, "token": args.token, "bits": args.bits,
             "table": table, "paired": comps}, indent=2))
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
