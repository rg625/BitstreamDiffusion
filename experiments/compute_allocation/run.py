"""Adaptive-compute study on CoBit-S LM1B: counterfactual compute branches.

One CHUNK = n samples with their own starting noise. For a chunk:

  reference  production sampler (NFE 255, gamma 0.2, entropic grid, SC carry,
             bf16 autocast, terminal sigma 0.08) with keyed noise; saves the
             full sampler state at the start of each of 8 segments, per-step
             token guesses, and state features at each segment start.
  branches   from each saved state s and for each k in K: integrate segment s
             with (1+k) sub-steps per step (k*|segment| extra NFE, one "unit"
             = one extra pass over the 32-step segment), then the reference
             schedule to the end. Two continuation noise streams r=0,1 are run
             in one batch; r=0 reuses the reference's noise, so (k=0, r=0)
             replays the reference (noise-floor check).
  score      GPT-2-large per-sample NLL and unigram entropy for every rollout;
             the official sequential scorer on the reference (anchor).

Every output file is written once; a rerun skips what exists.

    python -m experiments.compute_allocation.run chunk --chunk 0 --n 256
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from evaluation.utils import load_config, load_checkpoint
from experiments.compute_allocation.branch_sampler import BranchingSampler, KeyedNoise, SamplerState
from diffusion.continuous.processes import ContinuousForwardProcess
from models import create_model
from utils.ema import EMA

CONFIG = "configs/lm1b/continuous/eval/rate_eval_seeds.py"
CKPT = ("runs/paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/"
        "checkpoints/step=001000000.pt")
ENTROPY_DIR = "assets/entropy_tables/lm1b"
TARGET_NFE = 256          # production spec: steps_for_target_nfe -> 255 grid points
NUM_STEPS = 255
GAMMA = 0.200
S_NOISE = 1.003
SIGMA_MIN = 0.08          # external_ppl.terminal_sigmas
N_SEG = 8
K_UNITS = (0, 1, 2, 4)
BASE_SEED = 20261019
STATE_DIR = Path("runs/compute_allocation/lm1b")            # large arrays (project storage)
OUT_DIR = Path("results/compute_allocation/lm1b_gh200")     # tables


def configure(cfg, device: str):
    """Apply the Table 1 gamma=0.200 spec exactly as GenerationDriver does."""
    cfg.device = device
    st = cfg.evaluation.stochastic
    st.enabled = True
    st.s_churn = float(GAMMA * max(1, TARGET_NFE - 1))
    st.s_noise = S_NOISE
    st.window_mode = "entropy_cdf"
    st.entropy_quantile_lo = 0.0
    st.entropy_quantile_hi = 1.0
    st.s_tmin = None
    st.s_tmax = None
    st.entropy_fallback = "deterministic"
    return cfg


def build(device: torch.device):
    cfg = configure(load_config(CONFIG), str(device))
    model = create_model(cfg).to(device)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", "").removeprefix("module."): v for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    del ck, state
    ema = EMA(model, decay=0.0)
    load_checkpoint(model, ema, Path(CKPT), device, apply_ema=True)
    model.eval()
    proc = ContinuousForwardProcess(cfg)
    sampler = BranchingSampler(model, proc, cfg, num_steps=NUM_STEPS, sigma_min=SIGMA_MIN,
                               entropy_run_dir=ENTROPY_DIR)
    return cfg, model, sampler


def _amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def _bits_to_ids(hard: torch.Tensor, bpt: int) -> torch.Tensor:
    B, S = hard.shape
    w = (2 ** torch.arange(bpt - 1, -1, -1, device=hard.device)).to(torch.int32)
    return (hard.view(B, S // bpt, bpt).to(torch.int32) * w).sum(-1)


def _h2(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


class FeatureLogger:
    """Per-step token guesses + state features at the segment starts.

    Everything here is computed from the FIRST denoiser call of each step, which
    every branch shares up to its point of divergence, so a feature at the start
    of segment s is information an allocator would have before deciding how much
    compute to give segment s.
    """

    def __init__(self, n: int, n_steps: int, bpt: int, n_tokens: int, checkpoints):
        self.bpt = bpt
        self.ckpts = set(checkpoints)
        self.ids = torch.zeros(n, n_steps, n_tokens, dtype=torch.int16)
        self.H = torch.zeros(n, n_steps)
        self.hist = deque(maxlen=9)  # (ids, hard bits, H) of the last steps
        self.features = {}
        self.tok_entropy = {}

    def __call__(self, step, sigma, x, D):
        p = D.float()
        h = _h2(p)                                   # [B,S] bits
        hard = p > 0.5
        ids = _bits_to_ids(hard, self.bpt)           # [B,T]
        Hm = h.mean(1)
        self.ids[:, step] = ids.to(torch.int16).cpu()
        self.H[:, step] = Hm.cpu()
        if step in self.ckpts:
            B, S = h.shape
            T = S // self.bpt
            tokH = h.view(B, T, self.bpt).sum(-1)    # bits per token
            top = max(1, int(round(0.1 * T)))
            conf = (2 * p - 1).abs()
            sig = float(sigma)
            f = {
                "sigma": torch.full((B,), sig), "log_sigma": torch.full((B,), math.log(sig)),
                "H_mean": Hm, "frac_unc_bits": (h > 0.5).float().mean(1),
                "conf_mean": conf.mean(1), "conf_var": conf.var(1),
                "tokH_mean": tokH.mean(1), "tokH_max": tokH.max(1).values,
                "tokH_top10_share": tokH.topk(top, 1).values.sum(1) / tokH.sum(1).clamp_min(1e-9),
                "frac_unc_tokens": (tokH > 0.5).float().mean(1),
                "resid": ((x.float() - p) ** 2).mean(1) / (sig ** 2),
            }
            for j in range(self.bpt):
                f[f"bitplaneH_{j}"] = h.view(B, T, self.bpt)[:, :, j].mean(1)
            have = len(self.hist)
            zeros = torch.zeros(B, device=p.device)
            if have >= 1:
                ids1, hard1, H1 = self.hist[-1]
                f["dH1"] = Hm - H1
                f["flip1"] = (ids != ids1).float().mean(1)
                f["bitflip1"] = (hard != hard1).float().mean(1)
            else:
                f["dH1"] = f["flip1"] = f["bitflip1"] = zeros
            if have >= 2:
                f["d2H"] = Hm - 2 * self.hist[-1][2] + self.hist[-2][2]
            else:
                f["d2H"] = zeros
            if have >= 8:
                ids8, _, H8 = self.hist[-8]
                f["dH8"] = Hm - H8
                f["flip8"] = (ids != ids8).float().mean(1)
            else:
                f["dH8"] = f["flip8"] = zeros
            f["has_hist"] = torch.full((B,), float(have >= 8))
            self.features[step] = {k: v.float().cpu() for k, v in f.items()}
            self.tok_entropy[step] = tokH.to(torch.float16).cpu()
        self.hist.append((ids, hard, Hm))


def reference(cfg, sampler, chunk: int, n: int, device, sdir: Path):
    f = sdir / "reference.pt"
    if f.exists():
        return torch.load(f, weights_only=False)
    segs = sampler.segments(N_SEG)
    starts = [a for a, _ in segs]
    bpt = int(cfg.data.bits_per_token)
    S = int(cfg.data.sequence_len)
    noise = KeyedNoise(BASE_SEED, [(chunk, 0, n)])
    st0 = sampler.initial_state(n, S, noise)
    log = FeatureLogger(n, sampler.n_steps, bpt, S // bpt, starts)
    t0 = time.perf_counter()
    with _amp(device):
        probs, saved, nfe = sampler.run(st0, noise, save_at=starts, hook=log)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    out = {
        "chunk": chunk, "n": n, "segments": segs, "nfe": nfe, "wall_s": wall,
        "bits": (probs > 0.5).to(torch.uint8).cpu(),
        "states": {s: SamplerState(s, v.x.cpu(), {b: t.cpu() for b, t in v.sc.items()}) for s, v in saved.items()},
        "features": log.features, "tok_entropy": log.tok_entropy,
        "step_ids": log.ids, "step_H": log.H,
        "sigmas": sampler.sigmas.cpu(),
    }
    tmp = f.with_suffix(".tmp")
    torch.save(out, tmp)
    os.replace(tmp, f)
    print(f"[ref] chunk {chunk}: n={n} nfe={nfe} wall={wall:.1f}s "
          f"({n * nfe / wall:.0f} sample-NFE/s)", flush=True)
    return out


def branches(cfg, sampler, ref, device, sdir: Path):
    chunk, n = ref["chunk"], ref["n"]
    for s, (a, b) in enumerate(ref["segments"]):
        st = ref["states"][a].to(device)
        both = SamplerState.cat([st, st])
        noise = KeyedNoise(BASE_SEED, [(chunk, 0, n), (chunk, 1, n)])
        for k in K_UNITS:
            f = sdir / f"branch_s{s}_k{k}.pt"
            if f.exists():
                continue
            factors = {i: 1 + k for i in range(a, b)}
            t0 = time.perf_counter()
            with _amp(device):
                probs, _, nfe = sampler.run(both, noise, factors=factors)
            if device.type == "cuda":
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            out = {"s": s, "k": k, "start_step": a, "seg": (a, b), "nfe": nfe, "wall_s": wall,
                   "bits": (probs > 0.5).to(torch.uint8).cpu()}   # rows [0,n) stream 0, [n,2n) stream 1
            tmp = f.with_suffix(".tmp")
            torch.save(out, tmp)
            os.replace(tmp, f)
            print(f"[branch] chunk {chunk} s={s} k={k}: nfe={nfe} wall={wall:.1f}s "
                  f"({2 * n * nfe / wall:.0f} sample-NFE/s)", flush=True)


def score(cfg, ref, device, sdir: Path, odir: Path):
    from experiments.compute_allocation.score import BatchedScorer, decode, unigram_entropy
    f = odir / f"scores_chunk{ref['chunk']}.npz"
    if f.exists():
        return
    chunk, n = ref["chunk"], ref["n"]
    rows, all_bits = [], []
    all_bits.append(ref["bits"])
    rows += [dict(s=-1, k=0, r=0, i=i, nfe=ref["nfe"]) for i in range(n)]
    for s in range(N_SEG):
        for k in K_UNITS:
            br = torch.load(sdir / f"branch_s{s}_k{k}.pt", weights_only=False)
            all_bits.append(br["bits"])
            rows += [dict(s=s, k=k, r=j // n, i=j % n, nfe=br["nfe"]) for j in range(2 * n)]
    bits = torch.cat(all_bits, 0)
    texts, ids = decode(cfg, bits)
    scorer = BatchedScorer(device)
    nll, cnt = scorer.score(texts)
    ent = unigram_entropy(ids)
    cols = {key: np.array([r[key] for r in rows]) for key in ("s", "k", "r", "i", "nfe")}
    np.savez_compressed(
        f, chunk=np.full(len(rows), chunk), nll_sum=nll, n_tok=cnt, entropy=ent,
        ids=ids.to(torch.int16).numpy(), **cols,
    )
    # Anchor: the official sequential scorer on the reference samples.
    from evaluation.external_perplexity import HFExternalPerplexityEvaluator
    ev = HFExternalPerplexityEvaluator(model_name="openai-community/gpt2-large", device=device,
                                       torch_dtype="bfloat16", attn_implementation="sdpa", use_amp=True)
    off = ev.score_texts(texts[:n])
    mine = float(np.exp(np.nansum(nll[:n]) / cnt[:n].sum()))
    anchor = {"chunk": chunk, "n": n, "genppl_official": off["external_ppl"], "genppl_batched": mine,
              "entropy_mean": float(ent[:n].mean()), "ref_nfe": ref["nfe"], "ref_wall_s": ref["wall_s"]}
    (odir / f"anchor_chunk{chunk}.json").write_text(json.dumps(anchor, indent=2))
    print(f"[anchor] {anchor}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["chunk"])
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(Path("hf_cache").resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sdir = STATE_DIR / f"chunk{args.chunk}"
    sdir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg, model, sampler = build(device)
    print(f"[setup] steps={sampler.n_steps} segments={sampler.segments(N_SEG)} "
          f"gamma_step={sampler.stoch.s_churn / sampler.n_steps:.4f} "
          f"churn_band=({sampler.stoch.s_tmin:.4g},{sampler.stoch.s_tmax:.4g})", flush=True)
    ref = reference(cfg, sampler, args.chunk, args.n, device, sdir)
    branches(cfg, sampler, ref, device, sdir)
    del model, sampler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    score(cfg, ref, device, sdir, OUT_DIR)


if __name__ == "__main__":
    main()
