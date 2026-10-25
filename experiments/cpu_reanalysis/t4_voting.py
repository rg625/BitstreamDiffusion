"""TASK 4: voting that skips programs that do not execute (non-FKC only).

Per (run, checkpoint), using the 3 sampling seeds as 3 samples per problem
(problem indices intersected explicitly across seeds):

  pass@1        mean correctness over the seeds;
  maj@3         plain vote over the 3 executed answers, a non-executing sample
                (answer None) being a candidate like any other;
  maj@3-exec    vote among executing samples only, falling back to the plain
                vote (i.e. None, wrong) when none executes;
  pass@3        any sample correct.

Ties (e.g. three distinct answers) get expected credit under a uniform random
tie-break, so no metric depends on seed order. Paired bootstrap over problems
for each metric minus pass@1.

k-curve: production 500k has 9 distinct samples per problem from three 3-seed
karras/256-step/gamma=0 sets (GH200 DDIM, GH200 ordered-uniform, CSD3
ordered-uniform). Duplicate cells (byte-identical answer vectors, e.g. the
window_eval copy of train_eval_gh200) are dropped. For k = 1..K, pass@k is the
unbiased estimator and maj@k is averaged over all k-subsets. The pooled sets
use near-equivalent, not identical, samplers: tagged [likely].
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cpu_reanalysis.common import (  # noqa: E402
    cpu_guard, gold_answers, load_dir, paired_bootstrap, parse_answer, per_problem, write_json,
)

cpu_guard()

from evaluation.tasks.sandbox_gsm8k import _numbers_equal  # noqa: E402

CELLS = "results/tokens/eval_gh200/gsm8k_results_*.json"
PROD_SETS = [
    "results/tokens/eval_gh200/gsm8k_results_*prod_anchor__ck000500000__s*.json",
    "results/ordering/train_eval_gh200/gsm8k_results_*prod_anchor__ck000500000__uniform__s*.json",
    "results/ordering/window_eval_gh200/gsm8k_results_*prod_anchor__ck000500000__*.json",
    "results/ordering/train_eval/gsm8k_results_*prod_anchor__ck000500000__uniform__s*.json",
]


def key(v):
    return None if v is None else repr(v)


def vote(answers, gold, exec_only):
    """Expected correctness of a majority vote with uniform tie-breaking."""
    vals = [parse_answer(a) for a in answers]
    pool = [v for v in vals if v is not None] if exec_only else vals
    if not pool:
        pool = vals                       # fall back to the plain vote
    cnt = Counter(key(v) for v in pool)
    top = max(cnt.values())
    winners = [k for k, c in cnt.items() if c == top]
    rep = {key(v): v for v in pool}
    return float(np.mean([bool(_numbers_equal(rep[w], gold)) for w in winners]))


def score(samples, gold):
    """samples: {idx: [answer_str, ...]} with the same k for every idx."""
    idx = sorted(samples)
    corr = np.array([[bool(_numbers_equal(parse_answer(a), gold[i])) for a in samples[i]] for i in idx], float)
    p1 = corr.mean(1)
    maj = np.array([vote(samples[i], gold[i], False) for i in idx])
    majx = np.array([vote(samples[i], gold[i], True) for i in idx])
    pk = corr.max(1)
    return idx, {"pass@1": p1, "maj": maj, "maj_exec": majx, "pass@k": pk}


def main():
    gold, _ = gold_answers()
    cells = load_dir(CELLS)
    groups = defaultdict(list)
    for d in cells:
        groups[(d["_run"], d["_step"])].append(d)

    rows = []
    for (run, step), ds in sorted(groups.items()):
        pps = [per_problem(d) for d in sorted(ds, key=lambda d: d["_seed"])]
        common = sorted(set.intersection(*[set(p) for p in pps]))
        samples = {i: [p[i][1] for p in pps] for i in common}
        idx, m = score(samples, gold)
        exec_rate = float(np.mean([[a is not None for a in samples[i]] for i in idx]))
        row = {"run": run, "step": step, "seeds": [d["_seed"] for d in ds], "n_problems": len(idx),
               "files": [d["_path"] for d in ds], "exec_rate": exec_rate}
        for name in m:
            row[name] = float(m[name].mean())
        for name in ("maj", "maj_exec", "pass@k"):
            row[f"{name}_minus_pass@1"] = paired_bootstrap(m[name], m["pass@1"], seed=2)
        row["maj_exec_minus_maj"] = paired_bootstrap(m["maj_exec"], m["maj"], seed=3)
        rows.append(row)
        print(f"{run:28s} {step:>7}: n={len(idx)} exec={exec_rate:.3f} pass@1={row['pass@1']:.4f} "
              f"maj@3={row['maj']:.4f} maj@3-exec={row['maj_exec']:.4f} pass@3={row['pass@k']:.4f} | "
              f"maj-exec - p1 {row['maj_exec_minus_pass@1']['delta']:+.4f} "
              f"[{row['maj_exec_minus_pass@1']['ci95'][0]:+.4f},{row['maj_exec_minus_pass@1']['ci95'][1]:+.4f}]")

    # ---- production k-curve over pooled, de-duplicated sample sets ----
    seen, pool = set(), []
    for pat in PROD_SETS:
        for d in load_dir(pat):
            pp = per_problem(d)
            sig = tuple((i, pp[i][1]) for i in sorted(pp))
            if sig in seen:
                continue
            seen.add(sig)
            pool.append((d["_path"], pp))
    common = sorted(set.intersection(*[set(p) for _, p in pool]))
    K = len(pool)
    ans = {i: [p[i][1] for _, p in pool] for i in common}
    corr = np.array([[bool(_numbers_equal(parse_answer(a), gold[i])) for a in ans[i]] for i in common], float)
    curve = []
    for k in range(1, K + 1):
        c = corr.sum(1)
        passk = np.array([1 - comb(K - int(ci), k) / comb(K, k) for ci in c])
        subs = list(combinations(range(K), k))
        maj = np.array([np.mean([vote([ans[i][j] for j in s], gold[i], False) for s in subs]) for i in common])
        majx = np.array([np.mean([vote([ans[i][j] for j in s], gold[i], True) for s in subs]) for i in common])
        curve.append({"k": k, "n_subsets": len(subs), "pass@k": float(passk.mean()),
                      "maj@k": float(maj.mean()), "maj@k_exec": float(majx.mean()),
                      "maj@k_exec_minus_pass@1": paired_bootstrap(majx, corr.mean(1), seed=4)})
        print(f"prod k={k}: pass@k={passk.mean():.4f} maj@k={maj.mean():.4f} maj@k-exec={majx.mean():.4f}", flush=True)
    out = {"scope": CELLS, "tie_break": "expected credit, uniform over tied answers",
           "rows": rows,
           "production_k_curve": {"tag": "[likely] pooled near-equivalent samplers", "K": K,
                                  "n_problems": len(common), "sets": [p for p, _ in pool], "curve": curve}}
    print("->", write_json("t4_voting.json", out))


if __name__ == "__main__":
    main()
