"""TASK 6: guidance fix/break arithmetic and the break-even baseline accuracy.

The per-problem guidance records (CSD3 runs/guidance/*.json) are not on this
cluster, so neither is the n=1069 holdout subset. What is on disk:
results/guidance/paired_comparisons.json has, per (config, seed) on the full
n=1319 at 512 steps, gamma=0, the paired delta and the number of discordant
problems. That is enough to recover the counts exactly:

    fix - break = n * delta        fix + break = discordant
    fix_rate   = fix   / #baseline-wrong       break_rate = break / #baseline-correct

Net gain = fix_rate * (1 - a) - break_rate * a > 0  <=>  a < a* = fix_rate / (fix_rate + break_rate).
So guidance is positive BELOW a*, not above it.

Seeds: each paired entry is matched to its seed through results/guidance/all_cells.csv
(guided accuracy == the cell's accuracy, baseline accuracy == that seed's
baseline cell); duplicate entries of the same (config, seed) are dropped.

CIs: (i) a parametric bootstrap treating fix ~ Bin(#wrong, fix_rate) and
break ~ Bin(#correct, break_rate) independently per seed (within-seed sampling
noise only; it ignores that the three seeds share problems, so it is too
narrow), and (ii) the range over the three seeds.

Churn regime: no paired counts on disk; a* is recomputed from the fix/break
percentages tabulated in docs/tiny_gsm8k_error_analysis.md section 3 (doc-derived).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cpu_reanalysis.common import REPO, cpu_guard, write_json  # noqa: E402

cpu_guard()

CONFIGS = {  # name -> (method, cfg, ag, sg, bad_step)
    "CFG w=12": ("CFG", 12.0, 0.0, 0.0, ""),
    "CFG w=7": ("CFG", 7.0, 0.0, 0.0, ""),
    "CFG w=4": ("CFG", 4.0, 0.0, 0.0, ""),
    "CFG w=20": ("CFG", 20.0, 0.0, 0.0, ""),
    "SG-prev w=2": ("SG-prev", 0.0, 0.0, 2.0, ""),
    "SG-prev w=1": ("SG-prev", 0.0, 0.0, 1.0, ""),
    "AG w=15 bad=350k": ("AG", 0.0, 15.0, 0.0, "350000"),
    "AG w=6 bad=250k": ("AG", 0.0, 6.0, 0.0, "250000"),
}
DOC_CHURN = [  # docs/tiny_gsm8k_error_analysis.md section 3: regime, baseline, config, fix%, break%
    ("A gamma=0", 0.133, "CFG w=12", 12.6, 23.9), ("A gamma=0", 0.133, "CFG w=20", 13.4, 26.1),
    ("A gamma=0.3", 0.252, "CFG w=2", 10.7, 23.8), ("A gamma=0.3", 0.252, "CFG w=4", 11.3, 24.7),
    ("B gamma=0.41", 0.316, "CFG w=2", 8.2, 16.5), ("B gamma=0.41", 0.316, "CFG w=8", 9.4, 32.9),
]


def f(x):
    return float(x) if x not in ("", None) else 0.0


def main():
    cells = [r for r in csv.DictReader(open(REPO / "results/guidance/all_cells.csv"))
             if r["num_examples"] == "1319" and r["steps"] == "512" and f(r["gamma"]) == 0.0]
    base = {}
    for r in cells:
        if f(r["guidance_scale"]) == 0 and f(r["ag_scale"]) == 0 and f(r["sg_scale"]) == 0:
            base.setdefault(r["seed"], set()).add(round(f(r["accuracy"]), 9))
    pairs = [x for x in json.loads((REPO / "results/guidance/paired_comparisons.json").read_text())
             if x["n"] == "1319" and x["steps"] == "512" and x.get("paired_n_discordant") is not None]

    rng = np.random.default_rng(8)
    out = {"source": ["results/guidance/paired_comparisons.json", "results/guidance/all_cells.csv"],
           "n_problems": 1319, "steps": 512, "gamma": 0.0,
           "missing": "n=1069 holdout per-problem records (CSD3 runs/guidance/) not on this cluster",
           "configs": {}}
    for name, (meth, w, ag, sg, bad) in CONFIGS.items():
        seeds = {}
        for x in pairs:
            if (x["method"], f(x["cfg_scale"]), f(x["ag_scale"]), f(x["sg_scale"]), x["bad_step"]) != (meth, w, ag, sg, bad):
                continue
            match = [r for r in cells if abs(f(r["accuracy"]) - x["acc"]) < 1e-9
                     and f(r["guidance_scale"]) == w and f(r["ag_scale"]) == ag and f(r["sg_scale"]) == sg
                     and round(x["acc_baseline"], 9) in base.get(r["seed"], set())]
            if len({r["seed"] for r in match}) != 1:
                continue
            seed = match[0]["seed"]
            n, a0, disc, dlt = 1319, x["acc_baseline"], int(x["paired_n_discordant"]), x["paired_delta"]
            fix, brk = (disc + n * dlt) / 2, (disc - n * dlt) / 2
            assert abs(fix - round(fix)) < 1e-6 and abs(brk - round(brk)) < 1e-6, (name, seed, fix, brk)
            nw, nc = round(n * (1 - a0)), round(n * a0)
            seeds[seed] = {"seed": seed, "baseline_acc": a0, "guided_acc": x["acc"], "discordant": disc,
                           "fix": int(round(fix)), "break": int(round(brk)), "baseline_wrong": nw,
                           "baseline_correct": nc, "fix_rate": fix / nw, "break_rate": brk / nc,
                           "a_star": (fix / nw) / (fix / nw + brk / nc), "files": [r["result_file"] for r in match]}
        if not seeds:
            out["configs"][name] = {"missing": "no paired entry matched to a seed"}
            continue
        S = list(seeds.values())
        F, B = sum(s["fix"] for s in S), sum(s["break"] for s in S)
        W, C = sum(s["baseline_wrong"] for s in S), sum(s["baseline_correct"] for s in S)
        fr, br = F / W, B / C
        sims = []
        for _ in range(20000):
            fb = sum(rng.binomial(s["baseline_wrong"], s["fix_rate"]) for s in S)
            bb = sum(rng.binomial(s["baseline_correct"], s["break_rate"]) for s in S)
            sims.append(((fb / W) / (fb / W + bb / C), fb / W, bb / C))
        sims = np.asarray(sims)
        ci = lambda k: list(map(float, np.percentile(sims[:, k], [2.5, 97.5])))  # noqa: E731
        out["configs"][name] = {
            "n_seeds": len(S), "fix": F, "break": B, "fix_rate": fr, "break_rate": br,
            "fix_rate_ci95_binomial": ci(1), "break_rate_ci95_binomial": ci(2),
            "a_star": fr / (fr + br), "a_star_ci95_binomial": ci(0),
            "a_star_seed_range": [min(s["a_star"] for s in S), max(s["a_star"] for s in S)],
            "baseline_acc_mean": float(np.mean([s["baseline_acc"] for s in S])),
            "net_problems_per_seed": (F - B) / len(S), "seeds": S}
        r = out["configs"][name]
        print(f"{name:18s} seeds={len(S)} fix {fr:.3f} [{r['fix_rate_ci95_binomial'][0]:.3f},{r['fix_rate_ci95_binomial'][1]:.3f}] "
              f"break {br:.3f} [{r['break_rate_ci95_binomial'][0]:.3f},{r['break_rate_ci95_binomial'][1]:.3f}] "
              f"a*={r['a_star']:.3f} [{r['a_star_ci95_binomial'][0]:.3f},{r['a_star_ci95_binomial'][1]:.3f}] "
              f"seed range [{r['a_star_seed_range'][0]:.3f},{r['a_star_seed_range'][1]:.3f}] baseline {r['baseline_acc_mean']:.4f}")
    out["churn_doc_derived"] = [{"regime": g, "baseline_acc": a, "config": c, "fix_pct": fx, "break_pct": bk,
                                 "a_star": fx / (fx + bk), "baseline_below_a_star": a < fx / (fx + bk)}
                                for g, a, c, fx, bk in DOC_CHURN]
    for r in out["churn_doc_derived"]:
        print(f"[doc] {r['regime']:12s} {r['config']:9s} a*={r['a_star']:.3f} baseline {r['baseline_acc']:.3f} "
              f"-> {'positive' if r['baseline_below_a_star'] else 'negative'} predicted")
    print("->", write_json("t6_fix_break.json", out))


if __name__ == "__main__":
    main()
