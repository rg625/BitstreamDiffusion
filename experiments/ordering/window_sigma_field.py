"""Realised per-token sigma field over REAL answer tokens, before any training.

Runs the exact training draw (`training_window_sigma`, the trainer's function)
on real TinyGSM validation examples and measures, over answer tokens only --
never padding -- what each arm actually does to the noise field:

  interior      fraction of draws a token is strictly inside its window
  n_active      answer tokens simultaneously inside the window (when any is)
  rho           Spearman(position, sigma) over active answer tokens:
                +1 = left resolves first, -1 = right first, 0 = no order
  adj_dlog      |d log sigma| between adjacent answer tokens, both active
  ks_vs_F       KS distance of active answer tokens' sigma from F (should be ~0)
  multiset      identical per-example sigma multiset across modes at this k
  steps/token   Euler steps inside the window at N=256, the matched budget

The old w=0.25 arms are included, re-measured the same way, for contrast.

    python experiments/ordering/window_sigma_field.py --out results/ordering/window_sigma_field.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from diffusion.continuous.ordering import (
    NoiseCDF,
    training_position_sigma,
    training_window_sigma,
)

DS = ("datasets/tinygsm/tinygsm_bs512_answeronly_trainonpad_filtered_"
      "HuggingFaceTB__SmolLM-135M_capall_vr0.01_vs42_validation")
PAD = 49152
BPT = 16
N_TOK = 512


def load(n):
    ids = np.fromfile(DS + "_ids.uint16", dtype=np.uint16).reshape(-1, N_TOK)[:n]
    plen = np.fromfile(DS + "_plen.int32", dtype=np.int32)[:n]
    ids = torch.from_numpy(ids.astype(np.int64))
    plen = torch.from_numpy(plen.astype(np.int64))
    pos = torch.arange(N_TOK).view(1, -1)
    prompt = pos < plen.view(-1, 1)
    content = (~prompt) & (ids != PAD)
    pm_bits = prompt.repeat_interleave(BPT, dim=1)
    return pm_bits, prompt, content


def spearman_rows(pos, val, mask):
    out = []
    for p, v, m in zip(pos, val, mask):
        if int(m.sum()) < 3:
            continue
        if float(v[m].double().std()) == 0.0:      # all tied: no order to measure
            continue
        a = p[m].double().argsort().argsort().double()
        b = v[m].double().argsort().argsort().double()
        out.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    return float(np.mean(out)) if out else float("nan")


def ks(x, F):
    if x.numel() == 0:
        return float("nan")
    u = F.cdf_of(x.double()).sort().values
    n = u.numel()
    grid = torch.arange(1, n + 1, dtype=torch.float64) / n
    return float(torch.maximum((grid - u).abs(), (u - (grid - 1.0 / n)).abs()).max())


def measure(sig_tok, keep_tok, lo, hi, prompt, content, F=None):
    """sig_tok [B,n] token sigma; interior = strictly between the atoms."""
    active = content & (sig_tok > lo * (1 + 1e-6)) & (sig_tok < hi * (1 - 1e-6))
    pos = torch.arange(N_TOK).expand_as(sig_tok)
    any_act = active.any(1)
    both = active[:, 1:] & active[:, :-1] & content[:, 1:] & content[:, :-1]
    dl = (sig_tok[:, 1:].log() - sig_tok[:, :-1].log()).abs()[both]
    # sigma span over the ACTIVE answer tokens of one draw
    span = [float(sig_tok[b][active[b]].max() / sig_tok[b][active[b]].min())
            for b in range(sig_tok.shape[0]) if int(active[b].sum()) >= 2]
    return {
        "interior": float(active.sum() / content.sum()),
        "clean_atom": float((content & (sig_tok <= lo * (1 + 1e-6))).sum() / content.sum()),
        "noisy_atom": float((content & (sig_tok >= hi * (1 - 1e-6))).sum() / content.sum()),
        "overlaps_answer": float(any_act.float().mean()),
        "n_active": float(active.sum(1)[any_act].float().mean()) if any_act.any() else 0.0,
        "rho": spearman_rows(pos, sig_tok, active),
        "adj_dlog": float(dl.mean()) if dl.numel() else float("nan"),
        "answer_span_median": float(np.median(span)) if span else float("nan"),
        "ks_vs_F": ks(sig_tok[active], F) if F is not None else float("nan"),
        "loss_tokens_frac": (float((keep_tok & content).sum() / content.sum())
                             if keep_tok is not None else 1.0),
        # supervised token-noise pairs per example per step, answer and canvas
        "loss_answer_per_ex": (float((keep_tok & content).sum(1).float().mean())
                               if keep_tok is not None else float(content.sum(1).float().mean())),
        "loss_canvas_per_ex": (float((keep_tok & ~prompt).sum(1).float().mean())
                               if keep_tok is not None else float((~prompt).sum(1).float().mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--draws", type=int, default=16, help="tau draws per example")
    ap.add_argument("--table", default="assets/noise_tables/tinygsm_window_F_v1.pt")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--out", default="results/ordering/window_sigma_field.json")
    args = ap.parse_args()

    F = NoiseCDF.load(args.table)
    pm_bits, prompt, content = load(args.n)
    B = prompt.shape[0]
    L_suf = (~prompt).sum(1).double()
    L_ans = content.sum(1).double()
    print(f"[field] {B} validation examples | answer tokens median {L_ans.median():.0f} "
          f"| generated canvas median {L_suf.median():.0f} "
          f"| padding share {1 - (L_ans / L_suf).median():.2f}")

    rep = lambda t: t.repeat(args.draws, 1)
    PM, PR, CO = rep(pm_bits), rep(prompt), rep(content)
    g = torch.Generator().manual_seed(0)
    tau = 1.0 - torch.rand(PR.shape[0], generator=g)
    lo, hi = F.sigma_lo, F.sigma_hi

    rows = []
    for k in (math.inf, 64, 16, 4):
        modes = ("none",) if math.isinf(k) else ("l2r", "r2l", "random")
        multisets = {}
        for mode in modes:
            sig, keep, sg = training_window_sigma(
                tau, n_bits=N_TOK * BPT, bits_per_token=BPT, mode=mode, k=k, noise=F,
                prefix_mask=PM, generator=torch.Generator().manual_seed(1))
            st, kt = sig[:, ::BPT], keep[:, ::BPT]
            m = measure(st, kt, lo, hi, PR, CO, F)
            m.update(design="window", mode=mode, k=k,
                     steps_per_token=float(args.steps * (1.0 if math.isinf(k) else
                                                         (k / (k + L_suf)).mean())))
            multisets[mode] = [st[b][~PR[b]].sort().values for b in range(0, st.shape[0], 97)]
            rows.append(m)
        if len(multisets) > 1:
            ref = multisets[modes[0]]
            same = all(all(torch.equal(a, b) for a, b in zip(ref, multisets[md])) for md in modes)
            for r in rows[-len(modes):]:
                r["multiset_identical_across_modes"] = bool(same)

    # the old design, measured identically, for contrast (its own log-linear map)
    for mode in ("l2r", "r2l", "random"):
        sg_old = F.icdf(tau)
        sig = training_position_sigma(sg_old, n_bits=N_TOK * BPT, bits_per_token=BPT,
                                      mode=mode, w=0.25, sigma_min=0.002, sigma_max=80.0,
                                      prefix_mask=PM, generator=torch.Generator().manual_seed(1))
        m = measure(sig[:, ::BPT], None, 0.002, 80.0, PR, CO, None)
        m.update(design="old_w0.25", mode=mode, k=None)
        rows.append(m)

    hdr = (f"{'design':10s} {'mode':6s} {'k':>4s} {'interior':>8s} {'n_act':>6s} {'rho':>6s} "
           f"{'adj|dlog|':>9s} {'span':>9s} {'KS(F)':>6s} {'loss%':>6s} {'sup/ex':>7s} {'steps/tok':>9s} multiset")
    print(hdr)
    for r in rows:
        kk = "inf" if r["k"] == math.inf else ("-" if r["k"] is None else f"{int(r['k'])}")
        print(f"{r['design']:10s} {r['mode']:6s} {kk:>4s} {r['interior']:8.3f} {r['n_active']:6.1f} "
              f"{r['rho']:+6.2f} {r['adj_dlog']:9.3f} {r['answer_span_median']:9.3g} "
              f"{r['ks_vs_F']:6.3f} {100*r['loss_tokens_frac']:5.1f}% {r['loss_canvas_per_ex']:7.1f} "
              f"{r.get('steps_per_token', float('nan')):9.1f} {r.get('multiset_identical_across_modes', '')}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n_examples": B, "draws": args.draws, "noise_table": args.table,
         "answer_tokens_median": float(L_ans.median()),
         "canvas_median": float(L_suf.median()), "rows": rows},
        indent=2, default=lambda x: "inf" if x == math.inf else str(x)))
    print(f"[field] wrote {args.out}")


if __name__ == "__main__":
    main()
