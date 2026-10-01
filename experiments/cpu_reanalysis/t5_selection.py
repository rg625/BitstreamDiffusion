"""TASK 5: is "token programs that run are more often correct" a selection effect?

200k cells of the token study (results/tokens/eval_gh200): tok_gh500k_token_ce_s42
vs the bits control ord_gh500k_none_s42, 3 seeds each. Problem indices are
intersected explicitly across all six cells (audit R5).

  1. accuracy among executing programs, all problems (the reported 57% vs 31%);
  2. the same restricted to problems where BOTH arms produced >= 1 executing
     program, pooled over seeds and also problem-weighted (each problem's
     conditional accuracy given execution, then averaged), with a paired
     bootstrap over problems for token - bits;
  3. difficulty of the problems each arm gets to run: prompt length (tokens),
     production-model correctness (mean of the 3 GH200 DDIM production seeds),
     and the number of <<calc>> steps in the gold solution.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cpu_reanalysis.common import (  # noqa: E402
    REPO, bootstrap_mean, cpu_guard, gold_answers, load_dir, per_problem, write_json,
)

cpu_guard()
os.environ.setdefault("HF_HOME", str(REPO / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

D = "results/tokens/eval_gh200/"
ARMS = {"token": D + "gsm8k_results_*tok_gh500k_token_ce_s42__ck000200000__s*.json",
        "bits": D + "gsm8k_results_*ord_gh500k_none_s42__ck000200000__s*.json"}
PROD = D + "gsm8k_results_*prod_anchor__ck000500000__s*.json"


def ratio_boot(num, den, n_boot=10000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(num.sum() / max(den.sum(), 1)), *map(float, np.percentile(r, [2.5, 97.5]))


def main():
    from data.tinygsm import get_tokenizer
    tok = get_tokenizer("HuggingFaceTB/SmolLM-135M")
    _, recs = gold_answers()
    cells = {a: [per_problem(d) for d in load_dir(p)] for a, p in ARMS.items()}
    files = {a: [d["_path"] for d in load_dir(p)] for a, p in ARMS.items()}
    prod = [per_problem(d) for d in load_dir(PROD)]
    assert all(len(v) == 3 for v in cells.values()) and len(prod) == 3
    common = sorted(set.intersection(*[set(p) for v in cells.values() for p in v], *[set(p) for p in prod]))

    ex = {a: np.array([[p[i][1] is not None for p in cells[a]] for i in common], float) for a in cells}
    co = {a: np.array([[p[i][0] for p in cells[a]] for i in common], float) for a in cells}
    prod_acc = np.array([np.mean([p[i][0] for p in prod]) for i in common])
    plen = np.array([len(tok(recs[i]["prompt"].strip(), add_special_tokens=False).input_ids) for i in common])
    steps = np.array([recs[i]["response_ground_truth"].count("<<") for i in common])

    res = {"files": files, "production_files": [d["_path"] for d in load_dir(PROD)],
           "n_common_problems": len(common)}
    allp = np.ones(len(common), bool)
    both = (ex["token"].sum(1) > 0) & (ex["bits"].sum(1) > 0)
    for name, mask in (("all_problems", allp), ("both_arms_execute", both)):
        r = {"n_problems": int(mask.sum())}
        for a in cells:
            r[a] = {"exec_programs": int(ex[a][mask].sum()), "correct": int(co[a][mask].sum()),
                    "acc_given_exec_pooled": ratio_boot(co[a][mask].sum(1), ex[a][mask].sum(1), seed=5)}
            if name == "both_arms_execute":
                cond = co[a][mask].sum(1) / ex[a][mask].sum(1)
                r[a]["acc_given_exec_problem_weighted"] = bootstrap_mean(cond, seed=6)
        if name == "both_arms_execute":
            ct = co["token"][mask].sum(1) / ex["token"][mask].sum(1)
            cb = co["bits"][mask].sum(1) / ex["bits"][mask].sum(1)
            rng = np.random.default_rng(7)
            idx = rng.integers(0, len(ct), size=(10000, len(ct)))
            dd = (ct - cb)[idx].mean(1)
            r["token_minus_bits_problem_weighted"] = {
                "delta": float((ct - cb).mean()), "ci95": list(map(float, np.percentile(dd, [2.5, 97.5]))),
                "p_two_sided": float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))}
            # Pooled ratio difference, paired over problems.
            nt, dt_, nb, db = co["token"][mask].sum(1), ex["token"][mask].sum(1), co["bits"][mask].sum(1), ex["bits"][mask].sum(1)
            rr = nt[idx].sum(1) / dt_[idx].sum(1) - nb[idx].sum(1) / db[idx].sum(1)
            r["token_minus_bits_pooled"] = {"delta": float(nt.sum() / dt_.sum() - nb.sum() / db.sum()),
                                            "ci95": list(map(float, np.percentile(rr, [2.5, 97.5])))}
        res[name] = r

    def prof(mask):
        return {"n_problems": int(mask.sum()),
                "prompt_tokens_mean": float(plen[mask].mean()), "prompt_tokens_median": float(np.median(plen[mask])),
                "production_acc_mean": float(prod_acc[mask].mean()),
                "gold_calc_steps_mean": float(steps[mask].mean())}
    res["difficulty"] = {
        "all": prof(allp),
        "token_executes_any": prof(ex["token"].sum(1) > 0),
        "bits_executes_any": prof(ex["bits"].sum(1) > 0),
        "both_execute": prof(both),
        "token_only": prof((ex["token"].sum(1) > 0) & (ex["bits"].sum(1) == 0)),
        "bits_only": prof((ex["bits"].sum(1) > 0) & (ex["token"].sum(1) == 0)),
        "neither": prof((ex["token"].sum(1) == 0) & (ex["bits"].sum(1) == 0)),
    }
    # Program-weighted: difficulty of the executing (problem, seed) programs themselves.
    for a in cells:
        w = ex[a].sum(1)
        res["difficulty"][f"{a}_executing_programs_weighted"] = {
            "n_programs": int(w.sum()),
            "prompt_tokens_mean": float((plen * w).sum() / w.sum()),
            "production_acc_mean": float((prod_acc * w).sum() / w.sum()),
            "gold_calc_steps_mean": float((steps * w).sum() / w.sum())}
    p = write_json("t5_selection.json", res)
    import json
    print(json.dumps({k: v for k, v in res.items() if k not in ("files", "production_files")}, indent=1))
    print("->", p)


if __name__ == "__main__":
    main()
