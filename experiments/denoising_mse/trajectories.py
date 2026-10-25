#!/usr/bin/env python
"""
Paired generation trajectories of CoBit-S (130M) and CoBit-M (462M) under one shared sampler.

Sampler: the paper's OWT production sampler, identical for both models
  ddim_entropic, 256 NFE (255 sigma points: 254 churned Euler steps + final denoise), self-conditioning
  in "carry" mode, ATI off, no guidance, terminal sigma 0.08, EDM churn with gamma = 0.13
  (s_churn = 0.13 * 255 = 33.15, s_noise = 1.003, full entropy-CDF window). gamma = 0.13 is CoBit-S's
  Table-1 headline setting and CoBit-M's reported low-PPL 256-NFE point.
  Shared schedule: CoBit-S's entropy-rate table (assets/entropy_tables/owt) for BOTH models, so the
  integration sigmas are exactly the same (asserted).
RoPE: eager inference with the rotary angles in float32 (rope_fix.py), which reproduces the compiled
  production numerics; plain eager bf16 RoPE collapses positions > 256 (--no-rope-fix keeps it).
Randomness: for example e the initial state x_T = 0.5 + sigma_0 * N(0, I) is drawn from a generator
  seeded by e alone, and the churn noise of step i in batch b from a generator seeded by (b, i)
  (DDIMSampler churn_noise_seed). Both models therefore see exactly the same random numbers.
Recorded at every step i (per example), comparing the two runs:
  dx2   mean over bits of (x_S - x_M)^2, x = state after step i (at sigma_{i+1})
  dD2   mean over bits of (D_S - D_M)^2, D = denoiser output (bit probabilities) at step i
        (entry 254 is the final denoise at sigma = 0.08; x there is the state after step 253)
  tokdis fraction of the 1024 tokens whose rounded-D 16-bit codes differ
  HS, HM mean binary entropy [bits] of each model's D (how committed each model is)
plus the final samples (bits) of both models.
Modes: default = paired S vs M runs; --local [--ref S|M] = replay one model's trajectory and evaluate the
other on identical inputs; --truechain = ground-truth VE forward chains walked backwards (see
run_batch_truechain).
Batches of --batch examples; one output file per --chunk examples (resumable).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CONFIGS = {"S": "configs/owt/eval_750K_seed.py", "M": "configs/owt/eval_cobit_m_750K.py"}
GAMMA = 0.13
TARGET_NFE = 256
SCHEDULE_DIR = "assets/entropy_tables/owt"
XT_SEED = 20261004
CHURN_SEED = 20261005
OUT = REPO / "runs/denoising_mse/traj"
MOD = 2 ** 63 - 1


def production_spec():
    from evaluation.sampling_specs import build_sampling_specs
    from evaluation.utils import load_config

    cfg = load_config(str(REPO / CONFIGS["S"]))
    specs = build_sampling_specs(cfg=cfg, metric_cfg=cfg.evaluation.external_ppl)
    sel = [s for s in specs if abs(float(s["s_churn"]) - GAMMA * (TARGET_NFE - 1)) < 1e-9
           and int(s["target_nfe"]) == TARGET_NFE]
    assert len(sel) == 1, specs
    return dict(sel[0])


def load(model_key, spec, device):
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import DDIMSampler
    from evaluation.utils import load_checkpoint, load_config, unwrap_all
    from models import create_model
    from utils.ema import EMA

    cfg = load_config(str(REPO / CONFIGS[model_key]))
    cfg.device = str(device)
    st = cfg.evaluation.stochastic  # apply the spec exactly as GenerationDriver does
    for spec_k, cfg_k in {"stochastic_enabled": "enabled", "s_churn": "s_churn", "s_noise": "s_noise",
                          "window_mode": "window_mode", "entropy_quantile_lo": "entropy_quantile_lo",
                          "entropy_quantile_hi": "entropy_quantile_hi", "s_tmin": "s_tmin", "s_tmax": "s_tmax",
                          "entropy_fallback": "entropy_fallback"}.items():
        setattr(st, cfg_k, spec[spec_k])
    model = create_model(cfg).to(device)
    ema = EMA(unwrap_all(model), decay=0.0)
    load_checkpoint(model, ema, REPO / cfg.evaluation.checkpoint_path, device, apply_ema=True)
    model.eval()
    sampler = DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    return cfg, sampler


def sample_kwargs(spec, schedule_dir=SCHEDULE_DIR):
    return dict(schedule="entropic", num_steps=int(spec["num_steps"]),
                sigma_min_override=float(spec["terminal_sigma"]), sigma_max_override=None,
                entropic_blend_alpha=0.0, entropy_run_dir=REPO / schedule_dir,
                guidance_scale=float(spec["guidance_scale"]), sc_refresh_mode=str(spec["sc_refresh_mode"]),
                ati_eta=float(spec["ati_eta"]), return_probs=True, progress=False)


def x_T(examples, S, sigma0, device):
    out = torch.empty(len(examples), S, device=device)
    g = torch.Generator(device=device)
    for r, e in enumerate(examples):
        g.manual_seed((XT_SEED * 1_000_003 + int(e)) % MOD)
        out[r].normal_(generator=g)
    return out * float(sigma0) + 0.5


def bin_ent(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


@torch.no_grad()
def run_batch(samplers, kw, sigmas, examples, batch_idx, amp_dtype, device, bpt, S, repeat_S=False):
    """Run S then M (or S twice if repeat_S) on the same randomness; return per-step metrics."""
    B = len(examples)
    n = len(sigmas)  # 254 churned Euler steps + the final denoise (callback i = len(sigmas) - 1)
    xT = x_T(examples, S, sigmas[0], device)
    init = dict(i=0, x=xT, x0_hat=torch.zeros_like(xT), sigmas=sigmas)
    seed = (CHURN_SEED * 1_000_003 + int(batch_idx)) % MOD
    ref = dict(x=torch.empty(n, B, S, device=device), D=torch.empty(n, B, S, device=device))
    met = {k: torch.zeros(n, B, device=device) for k in ("dx2", "dD2", "tokdis", "HS", "HM")}
    sig_state = torch.zeros(n)
    gam = torch.zeros(n)

    def cb_ref(i, x, D, sigma, gamma, **_):
        ref["x"][i] = x
        ref["D"][i] = D.float()
        met["HS"][i] = bin_ent(D.float()).mean(-1)
        sig_state[i] = float(sigma)
        gam[i] = float(gamma)

    def cb_cmp(i, x, D, **_):
        D = D.float()
        met["dx2"][i] = ((x - ref["x"][i]) ** 2).mean(-1)
        met["dD2"][i] = ((D - ref["D"][i]) ** 2).mean(-1)
        a = (D > 0.5).view(B, -1, bpt)
        b = (ref["D"][i] > 0.5).view(B, -1, bpt)
        met["tokdis"][i] = (a != b).any(-1).float().mean(-1)
        met["HM"][i] = bin_ent(D).mean(-1)

    finals = {}
    for key, cb in (("S", cb_ref), ("S" if repeat_S else "M", cb_cmp)):
        tag = key if cb is cb_ref else ("S2" if repeat_S else "M")
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, probs = samplers[key].sample(
                B, S, conditioning_prefix_full=torch.zeros(B, S, device=device),
                cond_prefix_mask=torch.zeros(B, S, device=device, dtype=torch.bool),
                churn_noise_seed=seed, init_state={k: (v.clone() if torch.is_tensor(v) else v) for k, v in init.items()},
                step_callback=cb, **kw)
        finals[tag] = (probs.float() > 0.5).to(torch.uint8)
    other = "S2" if repeat_S else "M"
    fin_tokdis = (finals["S"].view(B, -1, bpt) != finals[other].view(B, -1, bpt)).any(-1).float().mean(-1)
    return ({k: v.cpu().numpy() for k, v in met.items()}, sig_state.numpy(), gam.numpy(),
            fin_tokdis.cpu().numpy(), {k: v.cpu().numpy() for k, v in finals.items()})


@torch.no_grad()
def run_batch_local(samplers, kw, sigmas, examples, batch_idx, amp_dtype, device, bpt, S, check=False, ref="S"):
    """Replay model `ref`'s trajectory (same randomness as run_batch) and, at every step, evaluate the other
    model on exactly ref's denoiser input: the churned state x_hat_i = x_{i-1} + sqrt(sig_hat_i^2 - sig_i^2) eps_i
    and ref's self-conditioning input (carry mode: ref's D from step i-1, zeros at step 0).
      ldD2    mean over bits of (D_ref - D_other(same input))^2
      ltokdis fraction of tokens whose rounded 16-bit codes differ
      lHo     mean binary entropy of the other model's D
    With check=True also re-evaluates ref on the rebuilt input; its max |D - D_ref| must be ~0."""
    from diffusion.continuous.logit_postprocess import _model_logits_continuous

    B = len(examples)
    n = len(sigmas)
    xT = x_T(examples, S, sigmas[0], device)
    init = dict(i=0, x=xT, x0_hat=torch.zeros_like(xT), sigmas=sigmas)
    seed = (CHURN_SEED * 1_000_003 + int(batch_idx)) % MOD
    met = {k: torch.zeros(n, B, device=device) for k in ("ldD2", "ltokdis", "lHo")}
    st = dict(x_prev=xT, cond=torch.zeros_like(xT), check=0.0)
    sR, sO = samplers[ref], samplers["M" if ref == "S" else "S"]

    def cb(i, x, D, sigma, gamma, eps, final, **_):
        if final or eps is None:
            x_state = st["x_prev"]
        else:
            sd = (sigma.float() ** 2 - sigmas[i].float() ** 2).clamp_min(0).sqrt()
            x_state = st["x_prev"] + sd * eps
        sig_B = sigma.float().expand(B)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            Dm = torch.sigmoid(_model_logits_continuous(sO.model, sO.cfg, x_state, sig_B, st["cond"]).float().view(B, S))
            if check:
                Dc = torch.sigmoid(_model_logits_continuous(sR.model, sR.cfg, x_state, sig_B,
                                                            st["cond"]).float().view(B, S))
                st["check"] = max(st["check"], float((Dc - D.float()).abs().max()))
        D = D.float()
        met["ldD2"][i] = ((Dm - D) ** 2).mean(-1)
        met["ltokdis"][i] = ((Dm > 0.5).view(B, -1, bpt) != (D > 0.5).view(B, -1, bpt)).any(-1).float().mean(-1)
        met["lHo"][i] = bin_ent(Dm).mean(-1)
        st["x_prev"] = x
        st["cond"] = D

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        sR.sample(B, S, conditioning_prefix_full=torch.zeros(B, S, device=device),
                  cond_prefix_mask=torch.zeros(B, S, device=device, dtype=torch.bool),
                  churn_noise_seed=seed, init_state=init, step_callback=cb, **kw)
    return {k: v.cpu().numpy() for k, v in met.items()}, st["check"]


@torch.no_grad()
def run_batch_truechain(samplers, x0, examples, sig_eval, amp_dtype, device, bpt):
    """Ground-truth trajectories: for each clean sequence x0 build the VE forward Markov chain over the
    sampler's evaluation noise levels (ascending l_0 < l_1 < ...: x(l_0) = x0 + l_0 eps_0,
    x(l_k) = x(l_{k-1}) + sqrt(l_k^2 - l_{k-1}^2) eps_k), then walk it backwards in sampler order. At step i
    each model denoises the TRUE state x(sig_eval[i]) with
      carry  : its own previous prediction as self-conditioning input (zeros at step 0), as in sampling
      zeros  : x0_hat = zeros (the no-self-conditioning training input; per-step marginals equal the static
               kernel, so this reproduces the static numbers and isolates the carry effect)
      onestep: its own zero-SC prediction at the previous (noisier) true state x(sig_eval[i-1]) as the
               self-conditioning input: x_t corrupted directly, x_{t+1} = x_t + sqrt(s_{t+1}^2 - s_t^2) eps
               (one forward transition), D(x_t, s_t, D(x_{t+1}, s_{t+1}, 0)); a one-step-history version of
               carry (equals zeros at step 0)
    Metrics per step and example (X in S, M; v in carry, zeros):
      mse_X_v    mean over bits (D - x0)^2;   terC_X_v  token error rate on non-pad tokens
      dD2_v      mean over bits (D_S - D_M)^2; tokdis_v fraction of tokens predicted differently"""
    from diffusion.continuous.logit_postprocess import _model_logits_continuous

    B, S = x0.shape
    n = len(sig_eval)
    lv = torch.flip(sig_eval, [0])                               # ascending
    inc = torch.cat([lv[:1], (lv[1:] ** 2 - lv[:-1] ** 2).clamp_min(0).sqrt()])
    states = torch.empty(n, B, S, device=device)
    g = torch.Generator(device=device)
    for r, e in enumerate(examples):
        g.manual_seed((XT_SEED * 7_919 + 104_729 * int(e) + 1) % MOD)
        eps = torch.randn(n, S, generator=g, device=device)
        states[:, r] = x0[r] + torch.cumsum(inc[:, None] * eps, 0)   # x(l_k), ascending order
    states = torch.flip(states, [0])                              # sampler order: high noise first
    content = (x0.view(B, -1, bpt).sum(-1) > 0).float()
    ntok = content.sum(1).clamp_min(1)
    VAR = ("carry", "zeros", "onestep")
    keys = [f"mse_{m}_{v}" for m in "SM" for v in VAR] + [f"terC_{m}_{v}" for m in "SM" for v in VAR] + \
           [f"{k}_{v}" for k in ("dD2", "tokdis") for v in VAR]
    met = {k: torch.zeros(n, B, device=device) for k in keys}
    cond = {m: torch.zeros(B, S, device=device) for m in "SM"}
    zeros = torch.zeros(B, S, device=device)
    prev_zero = {m: None for m in "SM"}      # each model's zero-SC prediction at the previous true state
    xt = x0.view(B, -1, bpt)
    for i in range(n):
        x = states[i]
        sig = sig_eval[i].float().expand(B)
        D = {}
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            for m in "SM":
                smp = samplers[m]
                for v, c in (("carry", cond[m]), ("zeros", zeros)):
                    D[(m, v)] = torch.sigmoid(_model_logits_continuous(smp.model, smp.cfg, x, sig, c).float().view(B, S))
                if prev_zero[m] is None:
                    D[(m, "onestep")] = D[(m, "zeros")]
                else:
                    D[(m, "onestep")] = torch.sigmoid(
                        _model_logits_continuous(smp.model, smp.cfg, x, sig, prev_zero[m]).float().view(B, S))
        for m in "SM":
            prev_zero[m] = D[(m, "zeros")]
            for v in VAR:
                d = D[(m, v)]
                met[f"mse_{m}_{v}"][i] = ((d - x0) ** 2).mean(-1)
                wrong = ((d > 0.5).float().view(B, -1, bpt) != xt).any(-1).float() * content
                met[f"terC_{m}_{v}"][i] = wrong.sum(1) / ntok
            cond[m] = D[(m, "carry")]
        for v in VAR:
            a, b = D[("S", v)], D[("M", v)]
            met[f"dD2_{v}"][i] = ((a - b) ** 2).mean(-1)
            met[f"tokdis_{v}"][i] = ((a > 0.5).view(B, -1, bpt) != (b > 0.5).view(B, -1, bpt)).any(-1).float().mean(-1)
    return {k: v.cpu().numpy() for k, v in met.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--schedule-dir", default=SCHEDULE_DIR,
                    help="entropy table defining the shared integration schedule (default: CoBit-S's)")
    ap.add_argument("--no-rope-fix", action="store_true",
                    help="keep eager bf16 RoPE (positions > 256 collapse; NOT the production numerics)")
    ap.add_argument("--ref", choices=("S", "M"), default="S",
                    help="--local: whose trajectory is replayed (the other model gets its inputs)")
    ap.add_argument("--truechain", action="store_true",
                    help="ground-truth VE forward chains of validation sequences, walked backwards")
    ap.add_argument("--local", action="store_true",
                    help="decomposition: evaluate CoBit-M on CoBit-S's own per-step inputs")
    ap.add_argument("--repeat-S", action="store_true",
                    help="determinism control: run CoBit-S twice instead of S then M")
    args = ap.parse_args()
    assert args.chunk % args.batch == 0

    if not args.no_rope_fix:
        import rope_fix

        rope_fix.apply()
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    spec = production_spec()
    kw = sample_kwargs(spec, args.schedule_dir)
    cfgS, sS = load("S", spec, device)
    samplers = {"S": sS}
    if not args.repeat_S:
        cfgM, sM = load("M", spec, device)
        samplers["M"] = sM
    sig = {k: s.sigmas.prepare(schedule="entropic", num_steps=kw["num_steps"], entropy_run_dir=kw["entropy_run_dir"],
                               sigma_min_override=kw["sigma_min_override"], sigma_max_override=None)
           for k, s in samplers.items()}
    for k in sig:
        assert torch.equal(sig[k], sig["S"]), "integration sigmas differ between models"
    sigmas = sig["S"]
    stoch = {k: vars(s.sigmas.resolve_stochastic_cfg(entropy_run_dir=kw["entropy_run_dir"],
                                                      sigma_min_override=kw["sigma_min_override"]))
             for k, s in samplers.items()}
    assert all(v == stoch["S"] for v in stoch.values()), stoch
    amp_dtype = torch.bfloat16 if str(cfgS.evaluation.amp_dtype) == "bf16" else torch.float16
    bpt = int(cfgS.data.bits_per_token)

    od = OUT / args.tag
    od.mkdir(parents=True, exist_ok=True)
    meta = dict(spec={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in spec.items()},
                gamma=GAMMA, schedule_dir=args.schedule_dir, rope_fix=not args.no_rope_fix, sigmas=sigmas.cpu().tolist(), stochastic=stoch["S"],
                n=args.n, batch=args.batch, chunk=args.chunk, xT_seed=XT_SEED, churn_seed=CHURN_SEED,
                repeat_S=args.repeat_S, amp_dtype=str(amp_dtype), configs=CONFIGS)
    (od / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[traj] {'S vs S (control)' if args.repeat_S else 'S vs M'}: n={args.n}, {len(sigmas)} sigmas "
          f"({sigmas[0]:.3f} -> {sigmas[-1]:.3f}), s_churn={stoch['S']['s_churn']}, window "
          f"[{stoch['S']['s_tmin']:.4f}, {stoch['S']['s_tmax']:.2f}]", flush=True)

    if args.truechain:
        import mse_curves as MC

        val = MC.ValBits(cfgS)
        seq_ids = MC.seq_indices(len(val), args.n)
        ref_run = sorted((OUT / "main").glob("chunk*.npz"))
        assert ref_run, "--truechain needs the main paired run for the sampler's evaluation sigmas"
        sig_eval = torch.from_numpy(np.load(ref_run[0])["sigma_state"].astype(np.float32)).to(device)
        assert bool((sig_eval[1:] < sig_eval[:-1]).all()), "evaluation sigmas must decrease"
    n_chunks = math.ceil(args.n / args.chunk)
    if args.max_chunks is not None:
        n_chunks = min(n_chunks, args.max_chunks)
    for c in range(n_chunks):
        f = od / f"chunk{c:04d}.npz"
        if f.exists():
            continue
        t0 = time.perf_counter()
        if args.truechain:
            parts_tc = []
            for b0 in range(c * args.chunk, min((c + 1) * args.chunk, args.n), args.batch):
                ids = seq_ids[b0:min(b0 + args.batch, args.n)]
                x0 = torch.stack([val[int(i)] for i in ids]).to(device, torch.float32)
                parts_tc.append(run_batch_truechain(samplers, x0, ids, sig_eval, amp_dtype, device, bpt))
            met = {k: np.concatenate([p_[k] for p_ in parts_tc], 1) for k in parts_tc[0]}
            tmp = od / f"chunk{c:04d}.tmp.npz"
            np.savez(tmp, examples=seq_ids[c * args.chunk:c * args.chunk + met["dD2_carry"].shape[1]],
                     sigma_state=sig_eval.cpu().numpy(), sigmas=sigmas.cpu().numpy(),
                     wall_sec=time.perf_counter() - t0, **met)
            os.replace(tmp, f)
            print(f"[traj-true] chunk {c + 1}/{n_chunks}: MSE carry S/M at step 150 "
                  f"{met['mse_S_carry'][150].mean():.4f}/{met['mse_M_carry'][150].mean():.4f}", flush=True)
            continue
        if args.local:
            lparts, worst = [], 0.0
            for b0 in range(c * args.chunk, min((c + 1) * args.chunk, args.n), args.batch):
                ex = list(range(b0, min(b0 + args.batch, args.n)))
                m_, chk = run_batch_local(samplers, kw, sigmas, ex, b0 // args.batch, amp_dtype, device, bpt,
                                          int(cfgS.data.sequence_len), check=(b0 == 0), ref=args.ref)
                lparts.append(m_)
                worst = max(worst, chk)
            met = {k: np.concatenate([p[k] for p in lparts], 1) for k in lparts[0]}
            tmp = od / f"chunk{c:04d}.tmp.npz"
            np.savez(tmp, examples=np.arange(c * args.chunk, c * args.chunk + met["ldD2"].shape[1]),
                     sigmas=sigmas.cpu().numpy(), wall_sec=time.perf_counter() - t0, check_max_abs=worst, **met)
            os.replace(tmp, f)
            print(f"[traj-local] chunk {c + 1}/{n_chunks}: ldD2 final {met['ldD2'][-1].mean():.3e}"
                  + (f", reconstruction check max|dD| = {worst:.2e}" if c == 0 else ""), flush=True)
            continue
        parts = []
        for b0 in range(c * args.chunk, min((c + 1) * args.chunk, args.n), args.batch):
            ex = list(range(b0, min(b0 + args.batch, args.n)))
            parts.append(run_batch(samplers, kw, sigmas, ex, b0 // args.batch, amp_dtype, device, bpt,
                                   int(cfgS.data.sequence_len), repeat_S=args.repeat_S))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        met = {k: np.concatenate([p[0][k] for p in parts], 1) for k in parts[0][0]}
        fin = {k: np.concatenate([p[4][k] for p in parts], 0) for k in parts[0][4]}
        tmp = od / f"chunk{c:04d}.tmp.npz"
        np.savez(tmp, examples=np.arange(c * args.chunk, c * args.chunk + met["dx2"].shape[1]),
                 sigma_state=parts[0][1], gamma=parts[0][2], sigmas=sigmas.cpu().numpy(), wall_sec=dt,
                 final_tokdis=np.concatenate([p[3] for p in parts]),
                 **met, **{f"bits_{k}": v for k, v in fin.items()})
        os.replace(tmp, f)
        m = met["dx2"]
        print(f"[traj] chunk {c + 1}/{n_chunks}: {m.shape[1]} examples in {dt:.0f}s; final dx2 "
              f"{m[-1].mean():.3e}, final token disagreement {np.concatenate([p[3] for p in parts]).mean():.3f}",
              flush=True)


if __name__ == "__main__":
    main()
