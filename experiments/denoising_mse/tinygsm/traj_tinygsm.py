#!/usr/bin/env python
"""
Trajectory experiments for TinyGSM (conditional: the question prefix is clamped clean everywhere), the
TinyGSM counterpart of ../trajectories.py. Sampler = the branch's GSM8K evaluator setting: ddim_entropic
on the run's entropy tables with num_steps = 256 (255 churned Euler steps + final denoise), sigma_min 0.002,
EDM churn s_churn = gamma * 255 (default gamma 0.31, CoBit-S's best at 256 NFE), s_noise 1.003, full
entropy-CDF window, self-conditioning carry, float32 RoPE. Both models share CoBit-S's schedule (--sched S).

Modes
  paired    --a S --b M   GSM8K test prompts; same initial noise (seeded per example) and churn noise (seeded
                          per batch and step) for both models. Per step: state gap dx2, denoiser gap dD2, token
                          disagreement over the answer region (all non-prompt tokens), each model's mean bit
                          entropy; final samples scored on GSM8K (per-example correctness, paired).
  local     --a REF --b OTHER   replay REF's trajectory; OTHER evaluated on REF's exact inputs (churned state,
                          re-clamped prompt; REF's self-conditioning input).
  truechain --models S[,M]   ground-truth VE forward chains of validation sequences (answer region only, prompt
                          clean) on the sampler's evaluation sigmas, walked from high to low noise; each model
                          denoises the true state with zero / one-step / carried self-conditioning; MSE vs x0
                          and token error over answer tokens; S-M disagreement if two models.
Output: runs/denoising_mse/tinygsm/traj_<tag>/chunk*.npz (+ meta.json)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import common as C
import mse_curves as MC
from gsm8k_eval import configure_stochastic

XT_SEED, CHURN_SEED, MOD = 20261006, 20261007, 2 ** 63 - 1


def bin_ent(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


def build(models, gamma, steps, sched_key, device):
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import DDIMSampler

    out = {}
    for m in models:
        cfg, model = C.load_model(m, device)
        configure_stochastic(cfg, gamma, steps)
        out[m] = DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    run = str(C.REPO / C.MODELS[sched_key]["run_dir"])
    kw = dict(num_steps=steps, schedule="entropic", entropy_run_dir=run, sigma_min_override=None,
              sigma_max_override=None, guidance_scale=0.0, sc_refresh_mode="carry", ati_eta=0.0,
              return_probs=True, progress=False)
    sig = {m: s.sigmas.prepare(schedule="entropic", num_steps=steps, entropy_run_dir=run) for m, s in out.items()}
    ref = next(iter(sig.values()))
    assert all(torch.equal(v, ref) for v in sig.values())
    return out, kw, ref


def x_T(examples, S, sigma0, device):
    out = torch.empty(len(examples), S, device=device)
    g = torch.Generator(device=device)
    for r, e in enumerate(examples):
        g.manual_seed((XT_SEED * 1_000_003 + int(e)) % MOD)
        out[r].normal_(generator=g)
    return out * float(sigma0) + 0.5


@torch.no_grad()
def run_paired(samplers, kw, sigmas, a, b, x0p, pm, examples, bidx):
    B, S = x0p.shape
    T = S // C.BPT
    n = len(sigmas)
    suf_tok = (~pm.view(B, T, C.BPT)[..., 0]).float()
    n_suf = suf_tok.sum(1).clamp_min(1)
    suf = (~pm).float()
    xT = torch.where(pm, x0p, x_T(examples, S, sigmas[0], x0p.device))
    init = dict(i=0, x=xT, x0_hat=torch.where(pm, x0p, torch.zeros_like(x0p)), sigmas=sigmas)
    seed = (CHURN_SEED * 1_000_003 + int(bidx)) % MOD
    ref = dict(x=torch.empty(n, B, S, device=x0p.device), D=torch.empty(n, B, S, device=x0p.device))
    met = {k: torch.zeros(n, B, device=x0p.device) for k in ("dx2", "dD2", "tokdis", "HA", "HB")}
    sig_state = np.zeros(n)

    def cb_a(i, x, D, sigma, **_):
        ref["x"][i], ref["D"][i] = x, D.float()
        sig_state[i] = float(sigma)
        met["HA"][i] = (bin_ent(D.float()) * suf).sum(1) / suf.sum(1)

    def cb_b(i, x, D, **_):
        D = D.float()
        met["dx2"][i] = (((x - ref["x"][i]) ** 2) * suf).sum(1) / suf.sum(1)
        met["dD2"][i] = (((D - ref["D"][i]) ** 2) * suf).sum(1) / suf.sum(1)
        dif = ((D > 0.5).view(B, T, C.BPT) != (ref["D"][i] > 0.5).view(B, T, C.BPT)).any(-1).float()
        met["tokdis"][i] = (dif * suf_tok).sum(1) / n_suf
        met["HB"][i] = (bin_ent(D) * suf).sum(1) / suf.sum(1)

    fin = {}
    for key, cb in ((a, cb_a), (b, cb_b)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, probs = samplers[key].sample(B, S, conditioning_prefix_full=x0p, cond_prefix_mask=pm,
                                            churn_noise_seed=seed,
                                            init_state={k: (v.clone() if torch.is_tensor(v) else v) for k, v in init.items()},
                                            step_callback=cb, **kw)
        fin[key if cb is cb_a else "B"] = (probs.float() >= 0.5).long()
    return {k: v.cpu().numpy() for k, v in met.items()}, fin[a], fin["B"], sig_state


@torch.no_grad()
def run_local(samplers, kw, sigmas, a, b, x0p, pm, examples, bidx, check=False):
    from diffusion.continuous.logit_postprocess import _model_logits_continuous

    B, S = x0p.shape
    T = S // C.BPT
    n = len(sigmas)
    suf = (~pm).float()
    suf_tok = suf.view(B, T, C.BPT)[..., 0]
    xT = torch.where(pm, x0p, x_T(examples, S, sigmas[0], x0p.device))
    init = dict(i=0, x=xT, x0_hat=torch.where(pm, x0p, torch.zeros_like(x0p)), sigmas=sigmas)
    seed = (CHURN_SEED * 1_000_003 + int(bidx)) % MOD
    met = {k: torch.zeros(n, B, device=x0p.device) for k in ("ldD2", "ltokdis", "lHo")}
    st = dict(x_prev=xT, cond=init["x0_hat"].clone(), check=0.0, sig=np.zeros(n))
    sR, sO = samplers[a], samplers[b]

    def cb(i, x, D, sigma, eps, final, **_):
        xs = st["x_prev"] if (final or eps is None) else \
            st["x_prev"] + (sigma.float() ** 2 - sigmas[i].float() ** 2).clamp_min(0).sqrt() * eps
        xs = torch.where(pm, x0p, xs)
        sB = sigma.float().expand(B)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Do = torch.where(pm, x0p, torch.sigmoid(_model_logits_continuous(sO.model, sO.cfg, xs, sB, st["cond"]).float().view(B, S)))
            if check:
                Dc = torch.where(pm, x0p, torch.sigmoid(_model_logits_continuous(sR.model, sR.cfg, xs, sB, st["cond"]).float().view(B, S)))
                st["check"] = max(st["check"], float((Dc - D.float()).abs().max()))
        D = D.float()
        met["ldD2"][i] = (((Do - D) ** 2) * suf).sum(1) / suf.sum(1)
        dif = ((Do > 0.5).view(B, T, C.BPT) != (D > 0.5).view(B, T, C.BPT)).any(-1).float()
        met["ltokdis"][i] = (dif * suf_tok).sum(1) / suf_tok.sum(1).clamp_min(1)
        met["lHo"][i] = (bin_ent(Do) * suf).sum(1) / suf.sum(1)
        st["x_prev"], st["cond"] = x, D
        st["sig"][i] = float(sigma)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        sR.sample(B, S, conditioning_prefix_full=x0p, cond_prefix_mask=pm, churn_noise_seed=seed,
                  init_state=init, step_callback=cb, **kw)
    return {k: v.cpu().numpy() for k, v in met.items()}, st["check"], st["sig"]


@torch.no_grad()
def run_truechain(samplers, x0, pm, ans_tok, examples, sig_eval):
    from diffusion.continuous.logit_postprocess import _model_logits_continuous

    B, S = x0.shape
    T = S // C.BPT
    n = len(sig_eval)
    lv = torch.flip(sig_eval, [0])
    inc = torch.cat([lv[:1], (lv[1:] ** 2 - lv[:-1] ** 2).clamp_min(0).sqrt()])
    states = torch.empty(n, B, S, device=x0.device)
    g = torch.Generator(device=x0.device)
    for r, e in enumerate(examples):
        g.manual_seed((XT_SEED * 7_919 + 104_729 * int(e) + 1) % MOD)
        eps = torch.randn(n, S, generator=g, device=x0.device)
        states[:, r] = torch.where(pm[r], x0[r], x0[r] + torch.cumsum(inc[:, None] * eps, 0))
    states = torch.flip(states, [0])
    ans = ans_tok.float()
    n_ans = ans.sum(1).clamp_min(1)
    xt = x0.view(B, T, C.BPT)
    models = list(samplers)
    VAR = ("carry", "zeros", "onestep")
    met = {f"{k}_{m}_{v}": torch.zeros(n, B, device=x0.device) for k in ("mse", "terC") for m in models for v in VAR}
    if len(models) == 2:
        met.update({f"{k}_{v}": torch.zeros(n, B, device=x0.device) for k in ("dD2", "tokdis") for v in VAR})
    zc = torch.where(pm, x0, torch.zeros_like(x0))
    cond = {m: zc.clone() for m in models}
    prev0 = {m: None for m in models}
    for i in range(n):
        x = states[i]
        sB = sig_eval[i].float().expand(B)
        D = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for m in models:
                smp = samplers[m]
                f = lambda c: torch.where(pm, x0, torch.sigmoid(  # noqa: E731
                    _model_logits_continuous(smp.model, smp.cfg, x, sB, c).float().view(B, S)))
                D[(m, "carry")], D[(m, "zeros")] = f(cond[m]), f(zc)
                D[(m, "onestep")] = D[(m, "zeros")] if prev0[m] is None else f(prev0[m])
        for m in models:
            prev0[m], cond[m] = D[(m, "zeros")], D[(m, "carry")]
            for v in VAR:
                p = D[(m, v)].view(B, T, C.BPT)
                met[f"mse_{m}_{v}"][i] = (((p - xt) ** 2).sum(-1) * ans).sum(1) / (n_ans * C.BPT)
                met[f"terC_{m}_{v}"][i] = ((((p > 0.5).float() != xt).any(-1).float()) * ans).sum(1) / n_ans
        if len(models) == 2:
            for v in VAR:
                a_, b_ = D[(models[0], v)].view(B, T, C.BPT), D[(models[1], v)].view(B, T, C.BPT)
                met[f"dD2_{v}"][i] = (((a_ - b_) ** 2).sum(-1) * ans).sum(1) / (n_ans * C.BPT)
                met[f"tokdis_{v}"][i] = (((a_ > 0.5) != (b_ > 0.5)).any(-1).float() * ans).sum(1) / n_ans
    return {k: v.cpu().numpy() for k, v in met.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("paired", "local", "truechain"), required=True)
    ap.add_argument("--a", default="S")
    ap.add_argument("--b", default="M")
    ap.add_argument("--models", default="S")
    ap.add_argument("--sched", default="S")
    ap.add_argument("--gamma", type=float, default=0.31)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--n", type=int, default=1319)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    models = args.models.split(",") if args.mode == "truechain" else list(dict.fromkeys([args.a, args.b]))
    samplers, kw, sigmas = build(models, args.gamma, args.steps, args.sched, dev)
    od = C.REPO / f"runs/denoising_mse/tinygsm/traj_{args.tag}"
    od.mkdir(parents=True, exist_ok=True)
    if args.mode == "truechain":
        ref_run = sorted((C.REPO / "runs/denoising_mse/tinygsm").glob("traj_*/chunk*.npz"))
        ref_run = [r for r in ref_run if "sigma_state" in np.load(r).files]
        assert ref_run, "--mode truechain needs a paired/local run for the sampler's evaluation sigmas"
        sig_eval = torch.from_numpy(np.load(ref_run[0])["sigma_state"].astype(np.float32)).to(dev)
        assert bool((sig_eval[1:] < sig_eval[:-1]).all())
        val = C.TinyGSMVal()
        ids_all = MC.seq_indices(len(val), args.n)
    else:
        test = C.GSM8KTest()
        ids_all = np.arange(min(args.n, len(test)))
    (od / "meta.json").write_text(json.dumps(dict(vars(args), models=models, sigmas=sigmas.cpu().tolist(),
                                                  stochastic=vars(samplers[models[0]].sigmas.resolve_stochastic_cfg(
                                                      entropy_run_dir=kw["entropy_run_dir"]))), indent=2, default=str))
    n_chunks = math.ceil(len(ids_all) / args.chunk)
    for c in range(n_chunks):
        f = od / f"chunk{c:04d}.npz"
        if f.exists():
            continue
        t0 = time.perf_counter()
        parts, extra = [], {}
        cid = ids_all[c * args.chunk:(c + 1) * args.chunk]
        for b0 in range(0, len(cid), args.batch):
            ex = cid[b0:b0 + args.batch]
            bidx = int(ex[0]) // args.batch
            if args.mode == "truechain":
                x0, pm, ans = val.batch(ex)
                parts.append(run_truechain(samplers, x0.to(dev), pm.to(dev), ans.to(dev), ex, sig_eval))
                continue
            x0p, pm, plens = test.batch(list(ex))
            x0p, pm = x0p.to(dev), pm.to(dev)
            if args.mode == "paired":
                m_, fa, fb, extra["sigma_state"] = run_paired(samplers, kw, sigmas, args.a, args.b, x0p, pm, ex, bidx)
                ca, _, _ = test.score(fa, plens, list(ex))
                cb_, _, _ = test.score(fb, plens, list(ex))
                extra.setdefault("correct_a", []).extend(ca)
                extra.setdefault("correct_b", []).extend(cb_)
                fin_dis = []
                for r in range(len(ex)):
                    ia, ib = C.bits_to_ids(fa[r:r + 1].cpu())[0, plens[r]:], C.bits_to_ids(fb[r:r + 1].cpu())[0, plens[r]:]
                    fin_dis.append(float((ia != ib).float().mean()))
                extra.setdefault("final_tokdis", []).extend(fin_dis)
                parts.append(m_)
            else:
                m_, chk, extra["sigma_state"] = run_local(samplers, kw, sigmas, args.a, args.b, x0p, pm, ex, bidx,
                                                          check=(b0 == 0 and c == 0))
                extra["check_max_abs"] = max(extra.get("check_max_abs", 0.0), chk)
                parts.append(m_)
        met = {k: np.concatenate([p_[k] for p_ in parts], 1) for k in parts[0]}
        tmp = od / f"chunk{c:04d}.tmp.npz"
        np.savez(tmp, examples=np.asarray(cid), sigmas=sigmas.cpu().numpy(), wall_sec=time.perf_counter() - t0,
                 **met, **{k: np.asarray(v) for k, v in extra.items()})
        os.replace(tmp, f)
        msg = f"[tg-{args.mode}] chunk {c + 1}/{n_chunks}: {time.perf_counter() - t0:.0f}s"
        if args.mode == "paired":
            msg += f"; acc a {np.mean(extra['correct_a']):.3f} b {np.mean(extra['correct_b']):.3f}"
        if args.mode == "local" and c == 0:
            msg += f"; reconstruction check max|dD| {extra['check_max_abs']:.2e}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
