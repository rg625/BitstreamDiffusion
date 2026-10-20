"""Tables for the global score-temperature study (scripts/hpc/arch/scoretemp_gh200.sh).

    python experiments/scoretemp/report.py results/scoretemp/eval_gh200

One block per (arm, w, gamma, steps) group, rows tau: accuracy with its 95% CI,
executable rate, invalid-token rate, and the PAIRED delta to tau=1 in the same
group (bootstrap over problems; every cell sees identical problems and seed).
For the CFG arm, also the guided-direction size cfg_dir_rms per sigma band.
"""
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

LABEL = re.compile(r"__(?P<arm>[a-z0-9]+)_w(?P<w>[\d.]+)_t(?P<tau>[\d.]+)_g(?P<g>[\d.]+)_n(?P<n>\d+)")
SIGMA_BANDS = [(10.0, float("inf")), (1.0, 10.0), (0.1, 1.0), (0.0, 0.1)]


def paired_delta(a, b, n_boot=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def cfg_bands(r):
    curve = (r.get("guidance_diagnostics") or {}).get("per_step") or []
    out = []
    for lo, hi in SIGMA_BANDS:
        v = [row["cfg_dir_rms"] for row in curve
             if "cfg_dir_rms" in row and "sigma" in row and lo <= row["sigma"] < hi]
        out.append(f"{np.mean(v):.3f}" if v else "  -  ")
    return out


def main(out_dir):
    cells = {}
    for f in glob.glob(f"{out_dir}/gsm8k_results_*.json"):
        m = LABEL.search(f)
        if not m:
            continue
        r = json.load(open(f))
        key = (m["arm"], float(m["w"]), float(m["g"]), int(m["n"]))
        pp = r["per_problem"]
        r["_by_idx"] = dict(zip(pp["idx"], pp["correct"]))
        cells.setdefault(key, {})[float(m["tau"])] = r

    if not cells:
        print(f"no results under {out_dir}")
        return
    for key in sorted(cells):
        arm, w, g, n = key
        group = cells[key]
        base = group.get(1.0)
        print(f"\n== {arm}  w={w:g}  gamma={g:g}  steps={n}")
        hdr = f"{'tau':>6} {'acc%':>6} {'95% CI':>13} {'exec%':>6} {'inv_tok%':>8} {'d vs tau=1 (paired)':>24} {'n':>5}"
        if arm.startswith("cfg"):
            hdr += "   cfg_dir_rms by sigma: >10 | 1-10 | .1-1 | <.1"
        print(hdr)
        for tau in sorted(group, reverse=True):
            r = group[tau]
            acc = 100 * r["accuracy"]
            ci = f"[{100 * r['ci95_low']:.1f},{100 * r['ci95_high']:.1f}]"
            ex = r.get("executable_rate")
            ex = f"{100 * ex:.1f}" if ex is not None else "-"
            inv = f"{100 * r.get('invalid_token_rate', float('nan')):.2f}"
            if base is not None and tau != 1.0:
                common = sorted(set(base["_by_idx"]) & set(r["_by_idx"]))
                d, lo, hi = paired_delta([r["_by_idx"][i] for i in common],
                                         [base["_by_idx"][i] for i in common])
                dd = f"{100 * d:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}]"
            else:
                dd = "-"
            line = f"{tau:>6g} {acc:>6.1f} {ci:>13} {ex:>6} {inv:>8} {dd:>24} {r['num_examples']:>5}"
            if arm.startswith("cfg"):
                line += "   " + " | ".join(cfg_bands(r))
            print(line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/scoretemp/eval_gh200")
