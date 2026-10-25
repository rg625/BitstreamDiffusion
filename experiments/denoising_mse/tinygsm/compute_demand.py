#!/usr/bin/env python
"""
Where does extra computation / capacity pay off on TinyGSM -> GSM8K, and does it depend on problem difficulty?

All configurations share the paired sampler of traj_tinygsm.py (ddim_entropic, 256 steps = 255 churned Euler steps +
final denoise, gamma = 0.31, CoBit-S's schedule, self-conditioning carry, float32 RoPE) and, for a given seed, the
same initial noise (per problem) and churn noise (per batch and step), so every comparison is paired per problem.
The trajectory is split into 8 segments of 32 steps (segment 7 also contains the final denoise).

Experiment A, compute-demand curves (CoBit-S):
  base           CoBit-S everywhere
  sc2_seg<k>     +2 extra self-conditioning refreshes per step inside segment k (x0_hat <- D(x, sigma_next, x0_hat)),
                 i.e. +64 model calls, the compute unit of experiments/adaptive_compute (LM1B)
Experiment B, model switching (where does CoBit-M's capacity pay off):
  M              CoBit-M everywhere
  Mseg<k>        CoBit-M inside segment k, CoBit-S elsewhere
  MthenS<c>      CoBit-M for steps < c, CoBit-S after (c in 64, 128, 192)
  SthenM<c>      CoBit-S for steps < c, CoBit-M after
Model routing: each denoiser call is dispatched by its noise level; the churned evaluation sigmas are strictly
decreasing, so step i <-> sigma_hat_i, and the final denoise (sigma_min) belongs to the last step.
Difficulty: number of calculation steps ("<<...>>") in the GSM8K gold solution.
Optional global score temperature (--score-temp-tau, --score-temp-final; DDIMSampler, commit b71eb71) for every
configuration: outputs are tagged _st<tau>[u].
Output: runs/denoising_mse/tinygsm/compute_demand/<config>_seed<k>[_<tag>].json (per-problem correctness)
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import torch

import common as C
from gsm8k_eval import configure_stochastic
from traj_tinygsm import CHURN_SEED, MOD, XT_SEED, x_T

SEG = 32
CUTS = (64, 128, 192)
OUT = C.REPO / "runs/denoising_mse/tinygsm/compute_demand"


def all_configs():
    a = ["base"] + [f"sc2_seg{k}" for k in range(8)]
    b = ["M"] + [f"Mseg{k}" for k in range(8)] + [f"MthenS{c}" for c in CUTS] + [f"SthenM{c}" for c in CUTS]
    return a, b


def m_steps(cfg_name: str, n_steps: int):
    """Set of step indices (0 .. n_steps-1; the last is the final denoise) run by CoBit-M."""
    if cfg_name == "M":
        return set(range(n_steps))
    if cfg_name.startswith("Mseg"):
        k = int(cfg_name[4:])
        hi = n_steps if k == 7 else SEG * (k + 1)
        return set(range(SEG * k, hi))
    if cfg_name.startswith("MthenS"):
        return set(range(int(cfg_name[6:])))
    if cfg_name.startswith("SthenM"):
        return set(range(int(cfg_name[6:]), n_steps))
    return set()


class Router(torch.nn.Module):
    """Dispatch each denoiser call to CoBit-S or CoBit-M by the step its noise level belongs to."""

    def __init__(self, models, sig_eval, m_set):
        super().__init__()
        self.mS, self.mM = models["S"], models.get("M")
        self.edges = sig_eval  # strictly decreasing evaluation sigmas, one per step
        self.m_set = m_set
        self.calls = {"S": 0, "M": 0, "off_grid": 0}

    def step_of(self, sigma):
        s = float(sigma.reshape(-1)[0])
        # step i has the largest evaluation sigma <= s (the final denoise is the last entry)
        idx = min(int(np.searchsorted(-self.edges, -s * (1 + 1e-6), side="left")), len(self.edges) - 1)
        if abs(self.edges[idx] / s - 1) > 1e-4:  # e.g. an SC refresh at sigma_{i+1}: routed to the next lower step
            self.calls["off_grid"] += 1
        return idx

    def forward(self, x, sigma, x0_hat=None):
        use_m = self.step_of(sigma) in self.m_set
        self.calls["M" if use_m else "S"] += 1
        return (self.mM if use_m else self.mS)(x, sigma, x0_hat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all", help="comma list, 'A', 'B' or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.31)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)  # = the paired run (same churn seeds)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--score-temp-tau", type=float, default=1.0, help="global score temperature (DDIMSampler, b71eb71)")
    ap.add_argument("--score-temp-final", default="sharpened", choices=("sharpened", "unsharpened"))
    args = ap.parse_args()
    if abs(args.score_temp_tau - 1.0) > 1e-8:
        st = f"st{args.score_temp_tau:g}" + ("u" if args.score_temp_final == "unsharpened" else "")
        args.tag = f"{args.tag}_{st}" if args.tag else st
    A, B = all_configs()
    names = {"all": A + B, "A": A, "B": B}.get(args.configs, args.configs.split(","))
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import DDIMSampler

    dev = torch.device("cuda:0")
    cfgS, mS = C.load_model("S", dev)
    models = {"S": mS}
    if any(n for n in names if n not in A):
        _, mM = C.load_model("M", dev)
        models["M"] = mM
    configure_stochastic(cfgS, args.gamma, args.steps)
    run_dir = str(C.REPO / C.MODELS["S"]["run_dir"])
    test = C.GSM8KTest()
    n = len(test) if args.limit is None else min(args.limit, len(test))
    gold_steps = np.array([len(re.findall(r"<<", r["response_ground_truth"])) for r in test.records[:n]])
    OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(num_steps=args.steps, schedule="entropic", entropy_run_dir=run_dir, sigma_min_override=None,
              sigma_max_override=None, guidance_scale=0.0, sc_refresh_mode="carry", ati_eta=0.0, return_probs=True,
              progress=False, segment_len=SEG, score_temp_mode="global", score_temp_tau=args.score_temp_tau,
              score_temp_final=args.score_temp_final)
    probe = DDIMSampler(mS, ContinuousForwardProcess(cfgS), cfgS)
    sigmas = probe.sigmas.prepare(schedule="entropic", num_steps=args.steps, entropy_run_dir=run_dir)
    stc = probe.sigmas.resolve_stochastic_cfg(entropy_run_dir=run_dir)
    from diffusion.continuous.samplers import _compute_edm_gamma

    sig_eval = []
    for i in range(len(sigmas) - 1):
        g = _compute_edm_gamma(sigmas[i], num_intervals=len(sigmas) - 1, s_churn=stc.s_churn, s_tmin=stc.s_tmin,
                               s_tmax=stc.s_tmax) if stc.enabled else 0.0
        sig_eval.append(float(sigmas[i]) * (1 + g))
    sig_eval.append(float(sigmas[-1]))
    sig_eval = np.array(sig_eval)
    assert np.all(np.diff(sig_eval) < 0), "evaluation sigmas must be strictly decreasing for routing"
    n_steps = len(sig_eval)

    for name in names:
        tag = f"{name}_seed{args.seed}{('_' + args.tag) if args.tag else ''}{'' if args.limit is None else f'_n{n}'}"
        f = OUT / f"{tag}.json"
        if f.exists():
            print(f"[cd] {tag}: exists, skipped", flush=True)
            continue
        m_set = m_steps(name, n_steps)
        router = Router(models, sig_eval, m_set).eval()
        sampler = DDIMSampler(router, ContinuousForwardProcess(cfgS), cfgS)
        overrides = None
        if name.startswith("sc2_seg"):
            overrides = {int(name[7:]): {"sc_refresh": 2}}
        t0 = time.perf_counter()
        correct = []
        for s in range(0, n, args.batch):
            idx = list(range(s, min(s + args.batch, n)))
            x0p, pm, plens = test.batch(idx)
            x0p, pm = x0p.to(dev), pm.to(dev)
            S_ = x0p.shape[1]
            seed_off = 7_777_777 * args.seed
            xT = torch.where(pm, x0p, x_T([e + seed_off for e in idx], S_, sigmas[0], dev))
            init = dict(i=0, x=xT, x0_hat=torch.where(pm, x0p, torch.zeros_like(x0p)), sigmas=sigmas)
            churn = (CHURN_SEED * 1_000_003 + s // args.batch + 99_991 * args.seed) % MOD
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, probs = sampler.sample(len(idx), S_, conditioning_prefix_full=x0p, cond_prefix_mask=pm,
                                          churn_noise_seed=churn, init_state=init, segment_overrides=overrides, **kw)
            ok, _, _ = test.score((probs.float() >= 0.5).long(), plens, idx)
            correct += ok
        res = dict(config=name, seed=args.seed, gamma=args.gamma, steps=args.steps, n=n, accuracy=float(np.mean(correct)),
                   score_temp_tau=args.score_temp_tau, score_temp_final=args.score_temp_final,
                   calls=router.calls, m_steps=sorted(m_set), wall_sec=time.perf_counter() - t0,
                   per_problem_correct=[int(c) for c in correct], gold_steps=gold_steps.tolist())
        f.write_text(json.dumps(res))
        print(f"[cd] {tag}: acc {100 * res['accuracy']:.2f}%  calls S {router.calls['S']} M {router.calls['M']}  "
              f"{res['wall_sec'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
