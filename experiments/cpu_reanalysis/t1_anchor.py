"""TASK 1: why the production 500k anchor moved from 0.164 (9 Sep) to 0.1314 (27 Sep).

Diffs every recorded setting between the 9 Sep evaluation and the 27 Sep
token-study production cells, then pairs them per problem on the indices both
contain (intersected explicitly). Seed-to-seed flips inside the 27 Sep cells,
on the same problems, give the reference for how much flipping seed noise alone
produces. The CSD3 karras 1319-problem cells (ordering train_eval) separate
"cluster" from "problem subset".
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cpu_reanalysis.common import (  # noqa: E402
    REPO, bootstrap_mean, cpu_guard, load_eval, per_problem, write_json,
)

cpu_guard()

SEP9 = "results/objective/eval/gsm8k_results_stochastic_g0.0_w0.0_s256_sd0.3998_ema1_production_500k.json"
SEP27 = [f"results/tokens/eval_gh200/gsm8k_results_stochastic_g0.0_w0.0_s256_sd0.3998_ema1_prod_anchor__ck000500000__s{s}.json"
         for s in range(3)]
CSD3_1319 = sorted(str(p.relative_to(REPO)) for p in (REPO / "results/ordering/train_eval").glob(
    "gsm8k_results_*prod_anchor__ck000500000__uniform__s*"))
GH_ORDERED = sorted(str(p.relative_to(REPO)) for p in (REPO / "results/ordering/train_eval_gh200").glob(
    "gsm8k_results_*prod_anchor__ck000500000__uniform__s*"))

SKIP = {"per_problem", "sample_records", "diversity", "efficiency", "guidance_diagnostics",
        "accuracy", "bootstrap_accuracy", "ci95_low", "ci95_high", "num_correct",
        "invalid_token_rate", "_path", "_run", "_step", "_seed"}


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if k in SKIP:
            continue
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def flips(a: dict, b: dict, idx):
    up = sum(1 for i in idx if a[i][0] == 0 and b[i][0] == 1)
    down = sum(1 for i in idx if a[i][0] == 1 and b[i][0] == 0)
    return {"n": len(idx), "acc_a": float(np.mean([a[i][0] for i in idx])),
            "acc_b": float(np.mean([b[i][0] for i in idx])),
            "wrong_to_right": up, "right_to_wrong": down, "mcnemar_p": mcnemar_exact(up, down)}


def main():
    d9 = load_eval(REPO / SEP9)
    d27 = [load_eval(REPO / p) for p in SEP27]
    csd = [load_eval(REPO / p) for p in CSD3_1319]
    gho = [load_eval(REPO / p) for p in GH_ORDERED]

    f9 = flat(d9)
    keys = sorted(set(f9) | set().union(*[flat(d) for d in d27]))
    diff = {}
    for k in keys:
        vals = [f9.get(k, "<absent>")] + [flat(d).get(k, "<absent>") for d in d27]
        if len({json.dumps(v, default=str) for v in vals}) > 1:
            diff[k] = {"sep9": vals[0], "sep27_s0": vals[1], "sep27_s1": vals[2], "sep27_s2": vals[3]}

    p9 = per_problem(d9)
    p27 = [per_problem(d) for d in d27]
    pcs = [per_problem(d) for d in csd]
    pgo = [per_problem(d) for d in gho]
    common = sorted(set(p9).intersection(*[set(p) for p in p27]))
    rest = sorted(set(p27[0]) - set(p9))

    def acc_on(p, idx):
        return float(np.mean([p[i][0] for i in idx]))

    subset = {
        "sep9_idx": [min(p9), max(p9), len(p9)],
        "sep9_acc": acc_on(p9, common),
        "sep27_on_sep9_problems": [acc_on(p, common) for p in p27],
        "sep27_on_other_problems": [acc_on(p, rest) for p in p27],
        "sep27_full": [acc_on(p, sorted(p)) for p in p27],
        "csd3_karras_1319_on_sep9_problems": [acc_on(p, common) for p in pcs],
        "csd3_karras_1319_on_other_problems": [acc_on(p, rest) for p in pcs],
        "csd3_karras_1319_full": [acc_on(p, sorted(p)) for p in pcs],
        "gh200_ordered_karras_on_sep9_problems": [acc_on(p, common) for p in pgo],
    }
    # Is the 250-problem prefix easier? Pool all karras 500k cells (6 seeds, 2 clusters).
    pooled = [p for p in p27 + pcs]
    prefix_vs_rest = {
        "prefix_mean": float(np.mean([acc_on(p, common) for p in pooled])),
        "rest_mean": float(np.mean([acc_on(p, rest) for p in pooled])),
    }
    # Per-problem mean correctness over the 6 seeds, bootstrap over problems.
    pm = {i: np.mean([p[i][0] for p in pooled]) for i in sorted(p27[0])}
    prefix_vs_rest["prefix_ci"] = bootstrap_mean([pm[i] for i in common])
    prefix_vs_rest["rest_ci"] = bootstrap_mean([pm[i] for i in rest])

    flip_9_vs_27 = [flips(p9, p, common) for p in p27]
    flip_9_vs_csd = [flips(p9, p, common) for p in pcs]
    seed_ref = [flips(p27[a], p27[b], common) for a, b in ((0, 1), (0, 2), (1, 2))]
    seed_ref_csd = [flips(pcs[a], pcs[b], common) for a, b in ((0, 1), (0, 2), (1, 2))] if len(pcs) == 3 else []
    # Problems the 9 Sep run got right that NO 27 Sep seed solved, and vice versa.
    sep9_only = [i for i in common if p9[i][0] == 1 and all(p[i][0] == 0 for p in p27)]
    sep27_only = [i for i in common if p9[i][0] == 0 and all(p[i][0] == 1 for p in p27)]
    # Same-seed answer agreement: seed 0 on both.
    same_ans = sum(1 for i in common if p9[i][1] == p27[0][i][1])

    out = {
        "files": {"sep9": SEP9, "sep27": SEP27, "csd3_karras_1319": CSD3_1319, "gh200_ordered": GH_ORDERED},
        "setting_diff": diff, "subset": subset, "prefix_vs_rest_pooled_6_seeds": prefix_vs_rest,
        "flips_sep9_vs_sep27_seed": flip_9_vs_27, "flips_sep9_vs_csd3_seed": flip_9_vs_csd,
        "flips_between_sep27_seeds": seed_ref, "flips_between_csd3_seeds": seed_ref_csd,
        "sep9_right_all_sep27_wrong": sep9_only, "sep9_wrong_all_sep27_right": sep27_only,
        "seed0_identical_answers_sep9_vs_sep27": f"{same_ans}/{len(common)}",
    }
    print(json.dumps({k: out[k] for k in ("subset", "prefix_vs_rest_pooled_6_seeds")}, indent=1))
    print("setting diffs:")
    for k, v in diff.items():
        print(f"  {k}: {v}")
    for name in ("flips_sep9_vs_sep27_seed", "flips_sep9_vs_csd3_seed", "flips_between_sep27_seeds",
                 "flips_between_csd3_seeds"):
        print(name, [(f["wrong_to_right"], f["right_to_wrong"], round(f["mcnemar_p"], 3)) for f in out[name]])
    print("sep9-only", len(sep9_only), "sep27-only", len(sep27_only), "seed0 same answer", out["seed0_identical_answers_sep9_vs_sep27"])
    print("->", write_json("t1_anchor.json", out))


if __name__ == "__main__":
    main()
