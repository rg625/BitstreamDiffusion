#!/usr/bin/env python
"""
Adaptive-compute opportunity study for CoBit-S on LM1B (experiments/adaptive_compute/PLAN.md).

States   : the Run 1 reference trajectory of experiments/compute_allocation (seed 42, production
           γ=0.2 sampler), saved at the 8 segment starts (steps 0,32,…,224):
           runs/compute_allocation/batch0/states/seg{s}.pt. State n = sample n of that run.
Compute  : arm c adds c extra self-conditioning refreshes (x0_hat <- D(x, σ_next, x0_hat)) at every
           step of the 32-step window that starts at the checkpoint: +32·c model calls. The σ path,
           x update rule and churn noise are unchanged; c=1 equals the repo's `refined`
           self-conditioning mode inside the window (tested).
Rollouts : R continuations per (state, arm) with fresh churn noise (per-step seeded); all arms of a
           (checkpoint, block) share the same churn noise (common random numbers). States are
           processed in blocks of 200 × R rows; block k covers states 200k … 200k+199.

Subcommands
  rollouts   run (checkpoint, arm, block) cells; one output file per cell, finished cells skipped
  features   CPU: cheap state features per (state, checkpoint) from saved states and step logs
  analyze    CPU: heterogeneity, predictors, oracle allocation (analysis_ac.py)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/compute_allocation"))

import numpy as np  # noqa: E402

import diagnose as D  # noqa: E402  (Ctx, ledger helpers, constants)

STATES_DIR = REPO / "runs/compute_allocation/batch0/states"
STEP_LOGS = REPO / "runs/compute_allocation/batch0/step_logs.pt"
OUT = REPO / "runs/adaptive_compute"
RES = HERE / "results"
BLOCK = 200
ROLLOUT_BASE = 20261002
MOD = 2**63 - 1
SEG_LEN = D.SEG_LEN
LEDGERS = [REPO / "experiments/compute_allocation/results", RES]


def rollout_seed(s: int, k: int) -> int:
    return (ROLLOUT_BASE * 1_000_003 + 10_007 * int(s) + 101 * int(k) + 0xADA) % MOD


def used_hours() -> float:
    return sum(D._ledger_total_hours(p) for p in LEDGERS)


def check_cap(upcoming: float) -> None:
    u = used_hours()
    if u + upcoming > D.GPU_HOUR_CAP:
        raise SystemExit(f"STOP: GPU-hour cap ({D.GPU_HOUR_CAP} h) would be exceeded: used {u:.3f} h "
                         f"(both experiments) + upcoming {upcoming:.3f} h")


def cell_name(s: int, c: int, k: int) -> str:
    return f"s{s}_c{c}_k{k}"


def cmd_rollouts(args) -> None:
    import torch

    cells = [(s, c, k) for s in args.checkpoints for c in args.arms for k in args.blocks]
    todo = [x for x in cells if args.force or not (RES / "rollouts" / f"{cell_name(*x)}.json").exists()]
    print(f"[rollouts] {len(todo)} of {len(cells)} cells to run; GPU used so far {used_hours():.3f} h", flush=True)
    if not todo:
        return
    check_cap(0.01)
    ctx = D.Ctx(compile_model=not args.no_compile)
    R = int(args.rollouts)
    states = {}
    for (s, c, k) in todo:
        check_cap(0.02)
        t0 = time.perf_counter()
        if s not in states:
            states[s] = torch.load(STATES_DIR / f"seg{s}.pt", map_location="cpu")
        st = states[s]
        a, b = BLOCK * k, BLOCK * (k + 1)
        x = st["x"][a:b].repeat(R, 1)          # row = r*BLOCK + j
        x0 = st["x0_hat"][a:b].repeat(R, 1)
        B = x.size(0)
        init = dict(i=int(st["i"]), x=x.to(ctx.device), x0_hat=x0.to(ctx.device), sigmas=st["sigmas"])
        seed = rollout_seed(s, k)
        overrides = {s: {"sc_refresh": int(c)}} if c > 0 else None
        kw = ctx.sample_kwargs()
        ctx.model.calls = 0
        torch.cuda.synchronize()
        tg = time.perf_counter()
        with torch.random.fork_rng(devices=[0], enabled=True):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.autocast(device_type="cuda", enabled=True, dtype=ctx.amp_dtype):
                _, probs = ctx.sampler.sample(
                    B, ctx.S_bits,
                    conditioning_prefix_full=torch.zeros(B, ctx.S_bits, device=ctx.device),
                    cond_prefix_mask=torch.zeros(B, ctx.S_bits, device=ctx.device, dtype=torch.bool),
                    churn_noise_seed=seed, init_state=init,
                    segment_overrides=overrides, segment_len=SEG_LEN, **kw,
                )
        torch.cuda.synchronize()
        gen = time.perf_counter() - tg
        calls = ctx.model.calls
        bits = (probs.float() > 0.5).to(torch.uint8)
        sc = ctx.score(bits)
        total = time.perf_counter() - t0
        name = cell_name(s, c, k)
        shape = lambda v: np.asarray(v, dtype=float).reshape(R, BLOCK).tolist()  # noqa: E731
        meta = D._meta(ctx, experiment="adaptive_compute", checkpoint=s, start_step=int(st["i"]),
                       arm=int(c), block=k, states=[a, b], rollouts=R, rollout_seed=seed,
                       compute_unit="sc_refresh per step over the 32-step window after the checkpoint",
                       extra_nfe_per_sample=int(32 * c if s < 7 else 30 * c),
                       model_calls_per_row_from_checkpoint=int(calls),
                       generation_wall_sec=gen, scoring_wall_sec=sc["score_wall"], process_wall_sec=total)
        D._atomic_torch_save(OUT / "rollouts" / f"{name}.pt", dict(tokens=sc["tokens"], bits=bits.cpu()))
        D._atomic_json(RES / "rollouts" / f"{name}.json", dict(
            meta=meta, nll=shape(sc["nll"]), ntok=shape(sc["ntok"]), entropy=shape(sc["entropy"]),
            genppl=sc["genppl"], mean_entropy=sc["mean_entropy"]))
        D._ledger_add(RES, f"rollouts {name}", total)
        print(f"[rollouts] {name}: calls {calls} gen {gen:.1f}s score {sc['score_wall']:.1f}s "
              f"GenPPL {sc['genppl']:.2f} ent {sc['mean_entropy']:.4f}", flush=True)


# -----------------------------------------------------------------------------
# cheap state features
# -----------------------------------------------------------------------------
def _bin_ent(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def cmd_features(args) -> None:
    import torch

    logs = torch.load(STEP_LOGS, map_location="cpu")
    tok = logs["tok"].numpy().astype(np.int32)        # [255, N, 128] token guesses from D at step j
    bent = logs["bit_ent"].float().numpy()           # [255, N, 128] per-token mean bit entropy of D
    N = tok.shape[1]
    feats = {}
    for s in range(D.N_SEGMENTS):
        st = torch.load(STATES_DIR / f"seg{s}.pt", map_location="cpu")
        i = int(st["i"])
        sig = float(st["sigmas"][i])
        x = st["x"].numpy().astype(np.float64)       # [N, 1920]
        p = st["x0_hat"].numpy().astype(np.float64)  # D from step i-1 (zeros at step 0)
        f = {}
        f["log_sigma"] = np.full(N, math.log(sig))
        if i == 0:  # pure noise, no denoiser output yet: state stats from x only
            for name in ["H_mean", "H_tok_sd", "frac_uncertain_bits", "conf_mean", "conf_sd",
                         "frac_uncertain_tokens", "H_msb", "H_lsb", "dH", "d2H", "pred_change_1",
                         "pred_change_4", "frac_stable_8", "resid_rms"]:
                f[name] = np.zeros(N)
        else:
            h = _bin_ent(p).reshape(N, 128, 15)
            conf = np.abs(p - 0.5).reshape(N, 128, 15)
            tok_h = h.mean(-1)
            f["H_mean"] = h.mean((1, 2))
            f["H_tok_sd"] = tok_h.std(1)
            f["frac_uncertain_bits"] = ((p > 0.1) & (p < 0.9)).reshape(N, -1).mean(1)
            f["conf_mean"] = conf.mean((1, 2))
            f["conf_sd"] = conf.reshape(N, -1).std(1)
            f["frac_uncertain_tokens"] = (tok_h > 0.5).mean(1)
            f["H_msb"] = h[:, :, :5].mean((1, 2))
            f["H_lsb"] = h[:, :, 10:].mean((1, 2))
            j = i - 1  # step whose D is x0_hat
            H = bent.mean(2)  # [255, N]
            f["dH"] = H[j] - H[j - 1] if j >= 1 else np.zeros(N)
            f["d2H"] = (H[j] - 2 * H[j - 1] + H[j - 2]) if j >= 2 else np.zeros(N)
            f["pred_change_1"] = (tok[j] != tok[j - 1]).mean(1) if j >= 1 else np.zeros(N)
            f["pred_change_4"] = (tok[j] != tok[max(j - 4, 0)]).mean(1)
            lo = max(j - 8, 0)
            f["frac_stable_8"] = np.all(tok[lo:j + 1] == tok[j][None], axis=0).mean(1)
            f["resid_rms"] = np.sqrt(((x - p) ** 2).mean(1)) / sig
        f["x_rms_over_sigma"] = np.sqrt(((x - 0.5) ** 2).mean(1)) / sig
        feats[str(s)] = {k: [float(v) for v in arr] for k, arr in f.items()}
    D._atomic_json(RES / "features.json", dict(
        n_states=N, source=dict(states=str(STATES_DIR), step_logs=str(STEP_LOGS)),
        note="features of the state at the start of step 32s (before churn); x0_hat = D from step 32s-1",
        features=feats))
    print(f"[features] wrote {RES / 'features.json'} for {N} states x {D.N_SEGMENTS} checkpoints")


def cmd_confirm(args) -> None:
    """End-to-end policy run on all 1024 Run-1 samples from step 0 (same chunk/churn seeds as Run 1,
    so every sample is paired with its Run 1 reference). Policy = sc_refresh c in segments S."""
    sys.path.insert(0, str(REPO / "experiments/compute_allocation"))
    import analysis as CA

    name = f"policy_c{args.c}_segs{'-'.join(map(str, args.segments))}"
    out = RES / "confirm" / f"{name}.json"
    if out.exists() and not args.force:
        print(f"[confirm] {name} done")
        return
    check_cap(0.05)
    t0 = time.perf_counter()
    ctx = D.Ctx(compile_model=not args.no_compile)
    overrides = {s: {"sc_refresh": int(args.c)} for s in args.segments}
    run = ctx.run(seed=D.BASE_SEED, seeded_churn=True, segment_overrides=overrides)
    sc = ctx.score(run["bits"])
    total = time.perf_counter() - t0
    ref = CA._scores(REPO / "experiments/compute_allocation/results", 0, "reference")
    br = dict(nll=np.asarray(sc["nll"], float), ntok=np.asarray(sc["ntok"], float), ent=np.asarray(sc["entropy"], float))
    st = CA.paired_stats(ref, br)
    q_ref = -ref["nll"] / np.maximum(ref["ntok"], 1)
    q_br = -br["nll"] / np.maximum(br["ntok"], 1)
    dq = q_br - q_ref
    idx = np.random.default_rng(20261002).integers(0, dq.size, size=(2000, dq.size))
    meta = D._meta(ctx, experiment="adaptive_compute/confirm", policy=name, c=int(args.c), segments=args.segments,
                   nfe_per_sample=run["nfe_calls"] // 2, generation_wall_sec=run["wall"],
                   scoring_wall_sec=sc["score_wall"], process_wall_sec=total)
    D._atomic_json(out, dict(meta=meta, genppl=sc["genppl"], mean_entropy=sc["mean_entropy"],
                             paired_vs_run1_reference=st,
                             mean_dQ_per_sample=float(dq.mean()), mean_dQ_ci=[float(v) for v in np.percentile(dq[idx].mean(1), [2.5, 97.5])],
                             per_sample=dict(nll=sc["nll"], ntok=sc["ntok"], entropy=sc["entropy"])))
    D._ledger_add(RES, f"confirm {name}", total)
    print(f"[confirm] {name}: GenPPL {sc['genppl']:.3f} ent {sc['mean_entropy']:.4f} mean dQ {dq.mean():.4f} "
          f"dNLL/tok {st['d_nll_tok']:.4f} [{st['ci_lo']:.4f},{st['ci_hi']:.4f}] dEnt {st['d_entropy']:.4f}", flush=True)


def cmd_analyze(args) -> None:
    sys.path.insert(0, str(HERE))
    import analysis_ac

    analysis_ac.main(blocks=args.blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollouts")
    r.add_argument("--checkpoints", type=int, nargs="+", default=list(range(D.N_SEGMENTS)))
    r.add_argument("--arms", type=int, nargs="+", default=[0, 1, 2, 4])
    r.add_argument("--blocks", type=int, nargs="+", default=[0])
    r.add_argument("--rollouts", type=int, default=4)
    r.add_argument("--no-compile", action="store_true")
    r.add_argument("--force", action="store_true")
    cf = sub.add_parser("confirm")
    cf.add_argument("--c", type=int, required=True)
    cf.add_argument("--segments", type=int, nargs="+", required=True)
    cf.add_argument("--no-compile", action="store_true")
    cf.add_argument("--force", action="store_true")
    sub.add_parser("features")
    a = sub.add_parser("analyze")
    a.add_argument("--blocks", type=int, nargs="+", default=None)
    args = ap.parse_args()
    {"rollouts": cmd_rollouts, "features": cmd_features, "confirm": cmd_confirm, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
