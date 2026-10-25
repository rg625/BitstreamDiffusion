#!/usr/bin/env python
"""
GSM8K executable-code accuracy for a TinyGSM checkpoint, replicating the branch's evaluator
(origin/tasks/sudoku-tinygsm evaluation/tasks/gsm8k_eval.py + _task_common.sample_bits) on this repo's
DDIMSampler:
  prompt [BOS] question \\n clamped clean; ddim_entropic on the run's entropy tables, num_steps = --steps,
  sigma_min 0.002 (no terminal override), EDM churn s_churn = gamma (steps - 1), s_noise 1.003, full
  entropy-CDF window; self-conditioning carry, ATI off, no guidance; torch.manual_seed(seed) before every
  batch (as the branch does); bits = probs >= 0.5; scored with the vendored S-FLM sandbox.
--no-rope-fix keeps eager bf16 RoPE (the branch's eval path: use_compile=False), otherwise float32 RoPE
(the compiled training numerics).
Output: runs/denoising_mse/tinygsm/gsm8k/<model>_g<gamma>_s<steps>_<rope>.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import common as C


def configure_stochastic(cfg, gamma, steps, s_noise=1.003):
    from ml_collections import config_dict

    st = config_dict.ConfigDict()
    if gamma <= 0:
        st.enabled, st.s_churn, st.s_noise, st.window_mode = False, 0.0, 1.0, "deterministic"
    else:
        st.enabled, st.s_churn, st.s_noise, st.window_mode = True, float(gamma) * max(1, steps - 1), s_noise, "entropy_cdf"
    st.entropy_quantile_lo, st.entropy_quantile_hi = 0.0, 1.0
    st.entropy_fallback, st.s_tmin, st.s_tmax = "deterministic", None, None
    cfg.evaluation.stochastic = st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="S")
    ap.add_argument("--gamma", type=float, default=0.31)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-rope-fix", action="store_true")
    ap.add_argument("--sched", default=None, help="model key whose entropy-rate schedule to use (default: own)")
    args = ap.parse_args()

    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import DDIMSampler

    dev = torch.device("cuda:0")
    cfg, model = C.load_model(args.model, dev, rope_fix=not args.no_rope_fix)
    configure_stochastic(cfg, args.gamma, args.steps)
    sampler = DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    test = C.GSM8KTest()
    n = len(test) if args.limit is None else min(args.limit, len(test))
    correct, inv, tot = [], 0, 0
    t0 = time.perf_counter()
    for s in range(0, n, args.batch):
        idx = list(range(s, min(s + args.batch, n)))
        x0, pm, plens = test.batch(idx)
        x0, pm = x0.to(dev), pm.to(dev)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, probs = sampler.sample(len(idx), x0.shape[1], conditioning_prefix_full=x0, cond_prefix_mask=pm,
                                      num_steps=args.steps, schedule="entropic",
                                      entropy_run_dir=(str(C.REPO / C.MODELS[args.sched]["run_dir"]) if args.sched
                                                       else cfg.evaluation.entropy_run_dir),
                                      sigma_min_override=None, sigma_max_override=None, guidance_scale=0.0,
                                      sc_refresh_mode="carry", ati_eta=0.0, return_probs=True, progress=False)
        ok, i_, t_ = test.score((probs.float() >= 0.5).long(), plens, idx)
        correct += ok
        inv += i_
        tot += t_
        print(f"[gsm8k] {len(correct)}/{n} acc {100 * np.mean(correct):.2f}%", flush=True)
    c = np.array(correct, float)
    rng = np.random.default_rng(args.seed)
    boot = c[rng.integers(0, len(c), (10000, len(c)))].mean(1)
    res = dict(model=args.model, sched=args.sched or args.model, gamma=args.gamma, steps=args.steps, seed=args.seed, n=int(len(c)),
               rope="bf16_eager" if args.no_rope_fix else "fp32", accuracy=float(c.mean()), num_correct=int(c.sum()),
               ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               invalid_token_rate=inv / max(tot, 1), scoring_errors=int(getattr(test, "n_scoring_errors", 0)),
               sigma_data=float(cfg.diffusion.continuous.sigma_data),
               wall_sec=time.perf_counter() - t0, per_example_correct=[int(x) for x in correct])
    od = C.REPO / "runs/denoising_mse/tinygsm/gsm8k"
    od.mkdir(parents=True, exist_ok=True)
    f = od / (f"{args.model}_g{args.gamma}_s{args.steps}_{res['rope']}"
              f"{'' if not args.sched or args.sched == args.model else f'_sched{args.sched}'}"
              f"{'' if args.limit is None else f'_n{n}'}.json")
    f.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_example_correct"}, indent=1))


if __name__ == "__main__":
    main()
