#experiments/guidance/compute_matched.py
"""Is guidance worth its extra forward passes?

CFG and AutoGuidance evaluate two branches per step, so at N steps they cost 2N
model evaluations while the baseline costs N. Every headline number in the
confirmation table compares guidance against a baseline given *half* the
compute, which flatters guidance and answers a question nobody deploying it has.

The compute-matched question is: spend the same 2N evaluations on two
independent baseline samples instead. Two ways to combine them:

  pass@2  -- correct if *either* sample is correct. Requires an oracle to say
             which one; not deployable. It is an UPPER BOUND on any realisable
             multi-sample scheme, so guidance beating pass@2 is a strong result
             and guidance losing to it is not by itself damning.
  maj@2   -- majority vote over executed answers. Deployable, but needs the
             answers recorded per problem, which older runs do not have.

This script computes the pass@k bound from per-problem correctness across seeds
(free -- the confirmation grids already ran three), and computes maj@k wherever
`per_problem.answer` is present.

    python -m experiments.guidance.compute_matched results/guidance
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

CONFIRM_GRIDS = ("cfg_confirm", "sg_confirm", "ag_confirm", "factorial_confirm",
                 "solver_control", "compute_control")


def load(path: str):
    """-> (correct_by_idx, answer_by_idx|None)"""
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        return {}, None
    pp = d.get("per_problem") or {}
    idx, cor = pp.get("idx"), pp.get("correct")
    if not idx or not cor:
        return {}, None
    correct = {int(i): int(c) for i, c in zip(idx, cor)}
    ans = pp.get("answer")
    answers = ({int(i): a for i, a in zip(idx, ans)} if ans else None)
    return correct, answers


def label(r: Dict) -> str:
    m = r.get("method", "")
    def g(k):
        try:
            return float(r.get(k) or 0)
        except ValueError:
            return 0.0
    if m == "CFG":
        return f"CFG w={g('guidance_scale'):g}"
    if m == "AG":
        return f"AG w={g('ag_scale'):g} bad={r.get('bad_step','?')}"
    if m in ("SG-prev", "SG-exact"):
        return f"{m} w={g('sg_scale'):g}"
    return m


def pass_at_k(per_seed: Dict[int, Dict[int, int]], seeds: List[int]) -> Dict[int, int]:
    shared = set.intersection(*[set(per_seed[s]) for s in seeds])
    return {i: max(per_seed[s][i] for s in seeds) for i in shared}


def maj_at_k(corr: Dict[int, Dict[int, int]], ans: Dict[int, Dict[int, object]],
             seeds: List[int]) -> Optional[Dict[int, int]]:
    if any(ans.get(s) is None for s in seeds):
        return None
    shared = set.intersection(*[set(ans[s]) for s in seeds])
    out = {}
    for i in shared:
        votes = [ans[s][i] for s in seeds if ans[s][i] is not None]
        if not votes:
            out[i] = 0
            continue
        # Ties break toward the first seed, which is the neutral (arbitrary but
        # not oracle-assisted) choice; an oracle tiebreak would be pass@k again.
        winner = collections.Counter(votes).most_common(1)[0][0]
        pick = next((s for s in seeds if ans[s][i] == winner), seeds[0])
        out[i] = corr[pick][i]
    return out


def boot(a: Dict[int, float], b: Dict[int, float], n_boot=20000, seed=0):
    ks = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in ks]
    m = len(d)
    if m < 20:
        return None
    pt = sum(d) / m
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        bs.append(sum(d[rng.randrange(m)] for _ in range(m)) / m)
    bs.sort()
    return pt, bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]


def mean(o) -> float:
    return sum(o.values()) / len(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()
    root = Path(args.results)
    rows = [r for r in csv.DictReader(open(root / "all_cells.csv"))
            if r.get("grid") in CONFIRM_GRIDS]

    corr = collections.defaultdict(dict)
    answ = collections.defaultdict(dict)
    nfe = {}
    for r in rows:
        c, a = load(r.get("result_file", ""))
        if not c:
            continue
        k, s = label(r), int(r.get("seed") or 0)
        corr[k][s], answ[k][s] = c, a
        try:
            nfe[k] = float(r.get("efficiency.nfe_per_sample") or 0)
        except ValueError:
            pass

    if "baseline" not in corr:
        raise SystemExit("no baseline rows")
    seeds = sorted(corr["baseline"])
    base_nfe = nfe.get("baseline", 512)
    pairs = list(itertools.combinations(seeds, 2))

    print(f"\nCompute-matched controls -- GSM8K test, seeds {seeds}\n")
    print(f"{'arm':<34}{'NFE/sample':>12}{'accuracy':>11}")
    print("-" * 57)
    b1 = sum(mean(corr["baseline"][s]) for s in seeds) / len(seeds)
    print(f"{'baseline pass@1':<34}{base_nfe:>12.0f}{b1:>11.4f}")
    p2v = [mean(pass_at_k(corr["baseline"], list(p))) for p in pairs]
    print(f"{'baseline pass@2 (ORACLE bound)':<34}{2*base_nfe:>12.0f}"
          f"{sum(p2v)/len(p2v):>11.4f}")
    m2 = maj_at_k(corr["baseline"], answ["baseline"], seeds[:2])
    print(f"{'baseline maj@2 (deployable)':<34}{2*base_nfe:>12.0f}"
          f"{mean(m2) if m2 else float('nan'):>11.4f}"
          f"{'' if m2 else '   [needs per_problem.answer]'}")
    for k in sorted(corr):
        if k == "baseline":
            continue
        a = sum(mean(corr[k][s]) for s in sorted(corr[k])) / len(corr[k])
        print(f"{k + ' pass@1':<34}{nfe.get(k,0):>12.0f}{a:>11.4f}")

    # Per-problem baseline pass@2, averaged over the seed pairs.
    bp2 = collections.defaultdict(list)
    for p in pairs:
        for i, v in pass_at_k(corr["baseline"], list(p)).items():
            bp2[i].append(v)
    bp2 = {i: sum(v) / len(v) for i, v in bp2.items()}

    # maj@2 per problem, averaged over seed pairs -- the DEPLOYABLE control.
    bm2 = collections.defaultdict(list)
    for p in pairs:
        mk = maj_at_k(corr["baseline"], answ["baseline"], list(p))
        if mk:
            for i, v in mk.items():
                bm2[i].append(v)
    bm2 = {i: sum(v) / len(v) for i, v in bm2.items()} if bm2 else None

    for ref, refname in ((bp2, "pass@2 (oracle bound)"),
                         (bm2, "maj@2 (deployable)")):
        if not ref:
            continue
        print(f"\nGuidance vs compute-matched baseline {refname} "
              f"(paired bootstrap, {args.n_boot} resamples):\n")
        for k in sorted(corr):
            if k == "baseline" or abs(nfe.get(k, 0) - 2 * base_nfe) > 1:
                continue
            arm = collections.defaultdict(list)
            for s, o in corr[k].items():
                for i, v in o.items():
                    arm[i].append(v)
            arm = {i: sum(v) / len(v) for i, v in arm.items()}
            res = boot(ref, arm, args.n_boot)
            if not res:
                continue
            pt, lo, hi = res
            verdict = ("guidance wins" if lo > 0 else
                       "baseline wins" if hi < 0 else "INCONCLUSIVE")
            print(f"  {k:<26} delta={pt:+.4f}  [{lo:+.4f},{hi:+.4f}]  -> {verdict}")
    print("\nNote: pass@2 needs an oracle to pick the right sample, so it is an\n"
          "upper bound. Guidance beating it is strong; guidance losing to it is\n"
          "not decisive until maj@2 is available.\n")


if __name__ == "__main__":
    main()
