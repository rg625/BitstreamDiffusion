"""Collect the gamma-dependent Euler study (configs/gamma_euler_eval.py) into one JSON (CPU).

    python scripts/dgc/gamma_euler_collect.py --out results/dgc/gamma_euler/summary.json

Reads runs/<experiment>/gamma_euler/<sampler>_<knob>/<arm>/seed<s>/results.jsonl (smoke runs skipped). Per cell
(data, sampler_knob, arm, T = actual model calls): seed-mean GenPPL and unigram entropy, and the paired seed
differences against the reference arm (corrected table) with a t interval over seeds.
"""
import argparse
import glob
import json
import math
import re
from collections import defaultdict

import numpy as np

EXPS = {"lm1b": "runs/paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting",
        "owt": "runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_1M"}
METS = {"gen_full_external_ppl": "genppl", "gen_full_token_unigram_entropy": "entropy"}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def load():
    rows = []
    for data, exp in EXPS.items():
        for f in glob.glob(f"{exp}/gamma_euler/*/*/seed*/results.jsonl"):
            parts = f.split("/")
            knob_dir, arm, seed_dir = parts[-4], parts[-3], parts[-2]
            if "smoke" in seed_dir:
                continue
            seed = int(seed_dir[4:])
            cell = defaultdict(dict)
            for line in open(f):
                r = json.loads(line)
                if r.get("metric") in METS and r.get("view", "full") == "full":
                    T = int(re.search(r"steps-(\d+)", r["tag"]).group(1))   # points = actual model calls
                    cell[T][METS[r["metric"]]] = float(r["value"])
            for T, v in cell.items():
                rows.append(dict(data=data, run=knob_dir, arm=arm, seed=seed, T=T, **v))
    return rows


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan")
    return float(x.mean()), float(T95.get(len(x) - 1, 1.96) * x.std(ddof=1) / math.sqrt(len(x)))


def summarise(rows, ref="table"):
    by = defaultdict(dict)
    for r in rows:
        by[(r["data"], r["run"], r["arm"], r["T"])][r["seed"]] = r
    out = []
    for (data, run, arm, T), seeds in sorted(by.items()):
        rec = dict(data=data, run=run, arm=arm, T=T, n_seeds=len(seeds), seeds=sorted(seeds))
        for m in ("genppl", "entropy"):
            vals = [s[m] for s in seeds.values() if m in s]
            rec[m], rec[f"{m}_ci"] = tci(vals)
            rec[f"{m}_per_seed"] = {k: s.get(m) for k, s in sorted(seeds.items())}
            refd = by.get((data, run, ref, T), {})
            common = [k for k in seeds if k in refd and m in seeds[k] and m in refd[k]]
            if arm != ref and common:
                rec[f"d_{m}"], rec[f"d_{m}_ci"] = tci([seeds[k][m] - refd[k][m] for k in common])
                rec[f"d_{m}_n"] = len(common)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dgc/gamma_euler/summary.json")
    a = ap.parse_args()
    rows = load()
    summ = summarise(rows)
    json.dump(dict(rows=rows, cells=summ), open(a.out, "w"), indent=1)
    for r in summ:
        d = (f"  dPPL {r['d_genppl']:+7.2f}±{r['d_genppl_ci']:.2f}  dH {r['d_entropy']:+.3f}±{r['d_entropy_ci']:.3f}"
             if "d_genppl" in r and "d_entropy" in r else "")
        print(f"{r['data']:5s} {r['run']:20s} {r['arm']:14s} T={r['T']:4d} n={r['n_seeds']}  "
              f"PPL {r.get('genppl', float('nan')):7.2f}±{r.get('genppl_ci', float('nan')):.2f}  "
              f"H {r.get('entropy', float('nan')):.3f}{d}")


if __name__ == "__main__":
    main()
