#!/usr/bin/env python
"""
Expected denoising MSE vs noise level for a TinyGSM checkpoint (conditional: clean question prefix).

Same design as ../mse_curves.py (fixed per-sequence noise shared across the sigma grid and models, three
self-conditioning inputs), adapted to the training code's conditional path (origin/tasks/sudoku-tinygsm
trainers/trainer.py, cond.noise_prefix = False, p_uncond = 0):
  x      = x0 on the prompt (clamped clean), x0 + sigma eps on the suffix (answer + EOS + pad)
  nosc   : D(x, s, zeros with prompt := x0)                    rows without self-conditioning
  none   : D(x, s, None)                                       the SC bootstrap pass
  sc     : D(x, s, sigmoid(D(x, s, None)) with prompt := x0)   rows with self-conditioning
Per-sequence metrics (keys as in mse_curves.py, so the OWT analysis scripts apply):
  mse    per-bit MSE over the suffix bits (= the training loss region, incl. pad)
  mseC   per-bit MSE over the answer tokens (answer + EOS, no pad)
  berC / terC / tnllC  bit error rate, token error rate, token NLL [nats] over answer tokens
  pos_*  per-bit-in-token MSE over answer tokens (summed over sequences)
  tokpos_* / tokposR_*  answer-token errors summed per absolute position / position from the answer start, with
         tokpos_count / tokposR_count (as the OWT token-position sums)
Output: runs/denoising_mse/tinygsm/<tag>/<model>/chunk*.npz + meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import common as C
import mse_curves as MC

V = ("nosc", "sc", "none")
METRICS = ("mse", "mseC", "berC", "terC", "tnllC")


@torch.no_grad()
def run_chunk(logits_fn, x0, pm, ans_tok, eps, sigmas, amp_dtype):
    B, S = x0.shape
    G = len(sigmas)
    T = S // C.BPT
    out = {f"{m}_{v}": np.zeros((G, B)) for m in METRICS for v in V}
    pos = {f"pos_{v}": np.zeros((G, C.BPT)) for v in V}
    # token errors over answer tokens summed per absolute token position and per position relative to the
    # answer start (the prompt length varies), as the OWT tokpos sums
    for v in V:
        pos[f"tokpos_{v}"] = np.zeros((G, T))
        pos[f"tokposR_{v}"] = np.zeros((G, T))
    plen = pm.view(B, T, C.BPT)[..., 0].sum(1).long()
    rel = torch.arange(T, device=x0.device)[None] - plen[:, None]                  # [B, T]
    relidx = rel.clamp(0, T - 1)
    suf = (~pm).float()
    n_suf = suf.sum(1).clamp_min(1)
    ans = ans_tok.float()
    n_ans = ans.sum(1).clamp_min(1)
    xt = x0.view(B, T, C.BPT)
    zeros_c = torch.where(pm, x0, torch.zeros_like(x0))
    npy = lambda t: t.double().cpu().numpy()  # noqa: E731

    def record(v, g, logits):
        lg = logits.float().view(B, T, C.BPT)
        p = torch.sigmoid(lg)
        se = (p - xt) ** 2
        out[f"mse_{v}"][g] = npy((se.view(B, S) * suf).sum(1) / n_suf)
        se_t = se.sum(-1)
        out[f"mseC_{v}"][g] = npy((se_t * ans).sum(1) / (n_ans * C.BPT))
        wrong = ((p > 0.5).float() != xt).float()
        out[f"berC_{v}"][g] = npy((wrong.sum(-1) * ans).sum(1) / (n_ans * C.BPT))
        out[f"terC_{v}"][g] = npy(((wrong.sum(-1) > 0).float() * ans).sum(1) / n_ans)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(lg, xt, reduction="none").sum(-1)
        out[f"tnllC_{v}"][g] = npy((bce * ans).sum(1) / n_ans)
        pos[f"pos_{v}"][g] = npy(((se * ans[..., None]).sum(1) / n_ans[:, None]).sum(0))
        tw = (wrong.sum(-1) > 0).float() * ans                                        # [B, T]
        pos[f"tokpos_{v}"][g] = npy(tw.sum(0))
        pos[f"tokposR_{v}"][g] = npy(torch.zeros(T, device=x0.device).index_add_(0, relidx[ans > 0], tw[ans > 0]))

    for g, s in enumerate(sigmas):
        sigma = torch.full((B,), float(s), device=x0.device)
        x = torch.where(pm, x0, x0 + float(s) * eps)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            l_none = logits_fn(x, sigma, None)
            p_none = torch.sigmoid(l_none.float().view(B, S))
            l_sc = logits_fn(x, sigma, torch.where(pm, x0, p_none))
            l_nosc = logits_fn(x, sigma, zeros_c)
        record("none", g, l_none)
        record("sc", g, l_sc)
        record("nosc", g, l_nosc)
    pos["tokpos_count"] = npy(ans.sum(0))
    pos["tokposR_count"] = npy(torch.zeros(T, device=x0.device).index_add_(0, relidx[ans > 0], ans[ans > 0]))
    return out, pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="S")
    ap.add_argument("--n-seq", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=129)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="static")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--no-rope-fix", action="store_true")
    args = ap.parse_args()

    from diffusion.continuous.logit_postprocess import _model_logits_continuous

    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg, model = C.load_model(args.model, dev, rope_fix=not args.no_rope_fix)
    logits_fn = lambda x, s, c: _model_logits_continuous(model, cfg, x, s, c)  # noqa: E731
    val = C.TinyGSMVal()
    sigmas = MC.sigma_grid(args.grid)
    idx_all = MC.seq_indices(len(val), args.n_seq)
    od = C.REPO / f"runs/denoising_mse/tinygsm/{args.tag}/{args.model}"
    od.mkdir(parents=True, exist_ok=True)
    meta = dict(model=args.model, params=C.MODELS[args.model]["params"], checkpoint=C.MODELS[args.model]["ckpt"],
                entropy_run_dir=C.MODELS[args.model]["run_dir"], n_val_sequences=len(val), n_seq=len(idx_all),
                grid=args.grid, sigma_min=MC.SIG_MIN, sigma_max=MC.SIG_MAX, noise_seed=MC.NOISE_SEED,
                seq_len_bits=C.BLOCK * C.BPT, bits_per_token=C.BPT,
                sigma_data=float(cfg.diffusion.continuous.sigma_data), rope_fix=not args.no_rope_fix,
                task="tinygsm", loss_region="suffix (answer+EOS+pad)", content="answer tokens (answer+EOS)")
    (od / "meta.json").write_text(json.dumps(meta, indent=2))
    chunks = [idx_all[a:a + args.chunk] for a in range(0, len(idx_all), args.chunk)]
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    print(f"[tg-static] {args.model}: {len(idx_all)} of {len(val)} val seqs, grid {args.grid}, {len(chunks)} chunks",
          flush=True)
    for k, idx in enumerate(chunks):
        f = od / f"chunk{k:04d}.npz"
        if f.exists():
            continue
        t0 = time.perf_counter()
        seq, sums = {}, {}
        for a in range(0, len(idx), args.batch):
            bi = idx[a:a + args.batch]
            x0, pm, ans = val.batch(bi)
            x0, pm, ans = x0.to(dev), pm.to(dev), ans.to(dev)
            eps = MC.noise_for(torch.as_tensor(bi), x0.shape[1], dev)
            o, p = run_chunk(logits_fn, x0, pm, ans, eps, sigmas, torch.bfloat16)
            for key, val_ in o.items():
                seq.setdefault(key, []).append(val_)
            for key, val_ in p.items():
                sums[key] = sums.get(key, 0.0) + val_
        dt = time.perf_counter() - t0
        tmp = od / f"chunk{k:04d}.tmp.npz"
        np.savez(tmp, idx=idx, sigmas=sigmas, wall_sec=dt, **{key: np.concatenate(v, 1) for key, v in seq.items()},
                 **{key: (v if key.startswith("tokpos") else v / len(idx)) for key, v in sums.items()})
        os.replace(tmp, f)
        print(f"[tg-static] {args.model} chunk {k + 1}/{len(chunks)}: {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
