#!/usr/bin/env python
"""
Single-model (CoBit-S) analysis for TinyGSM; the S-vs-M parts are added once CoBit-M is available.

Inputs  runs/denoising_mse/tinygsm/static/S        (mse_tinygsm.py)
        runs/denoising_mse/tinygsm/traj_true_S     (traj_tinygsm.py --mode truechain --models S)
        runs/denoising_mse/tinygsm/gsm8k/*.json    (gsm8k_eval.py)
        runs/tasks/tinygsm/cobit_raw_binary_bits/entropy_*.pt  (the run's schedule)
Outputs experiments/denoising_mse/results_tinygsm/S/: fig_static.png, fig_rate_vs_sqrt.png,
        fig_truechain.png, curves.csv, summary.json

Conventions: training objective = 0.5 (no-SC + SC); "suffix" = the training loss region (answer + EOS + pad,
prompt clean); "answer" = answer + EOS tokens only. Diffusion time t = F(sigma), the CDF of the run's
entropy-rate schedule (t = 1 noise, t = 0 clean). Schedule rule (utils/schedule_controller.py): density per
log-sigma bin proportional to g(sigma) (MSE/sigma^2)^0.5, g = s^3/(s^3 + 0.1^3), where the MSE is the
training entropy metric over the loss region (suffix).
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R = REPO / "runs/denoising_mse/tinygsm"
RUN = REPO / "runs/tasks/tinygsm/cobit_raw_binary_bits"
OUT = HERE.parent / "results_tinygsm" / "S"
COL = {"S": "#2a78d6", "k": "#3d3d3a", "g": "#1baf7a", "o": "#eb6834", "v": "#4a3aa7", "gr": "#9a9994"}


def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def load_static():
    zs = [np.load(f) for f in sorted(glob.glob(str(R / "static/S/chunk*.npz")))]
    d = {"sigma": zs[0]["sigmas"]}
    for k in zs[0].files:
        if zs[0][k].ndim == 2:
            d[k] = np.concatenate([z[k] for z in zs], 1)
    for pre in ("mse", "mseC", "berC", "terC", "tnllC"):
        d[f"{pre}_train"] = 0.5 * (d[f"{pre}_nosc"] + d[f"{pre}_sc"])
    return d


def table():
    rd = lambda n: torch.load(RUN / n, map_location="cpu").numpy().astype(float)  # noqa: E731
    pdf, cdf, mids, edges = rd("entropy_pdf.pt"), rd("entropy_cdf.pt"), rd("entropy_sigmas.pt"), rd("entropy_edges.pt")
    cdf[-1] = 1.0
    return dict(pdf=pdf, cdf=cdf, mids=mids, edges=edges, dens=pdf / np.diff(np.log(edges)))


def unit(y, ls, act):
    y = np.where(act, np.clip(y, 0, None), 0.0)
    return y / np.trapz(y, ls)


def qstats(y, ls):
    c = np.concatenate([[0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(ls))])
    c /= c[-1]
    q = lambda p: float(np.exp(np.interp(p, c, ls)))  # noqa: E731
    return dict(sigma_peak=float(np.exp(ls[np.argmax(y)])), sigma_q05=q(0.05), sigma_median=q(0.5), sigma_q95=q(0.95))


def mean_se(a):
    return a.mean(1), a.std(1, ddof=1) / np.sqrt(a.shape[1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    summ = {}
    tab = table()
    # ---------------- GSM8K RoPE comparison
    g = {}
    for f in sorted(glob.glob(str(R / "gsm8k/S_*.json"))):
        dj = json.loads(Path(f).read_text())
        if dj["n"] == 1319:
            g[f"{dj['rope']}_g{dj['gamma']}_s{dj['steps']}"] = {k: dj[k] for k in ("accuracy", "num_correct", "ci95", "invalid_token_rate")}
    if {"bf16_eager_g0.31_s256", "fp32_g0.31_s256"} <= set(g):
        a = np.array(json.loads((R / "gsm8k/S_g0.31_s256_bf16_eager.json").read_text())["per_example_correct"])
        b = np.array(json.loads((R / "gsm8k/S_g0.31_s256_fp32.json").read_text())["per_example_correct"])
        rng = np.random.default_rng(0)
        bt = rng.integers(0, len(a), (10000, len(a)))
        diff = (b[bt] - a[bt]).mean(1)
        g["fp32_minus_bf16"] = dict(diff=float((b - a).mean()), ci95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                    only_fp32_correct=int(((b == 1) & (a == 0)).sum()), only_bf16_correct=int(((a == 1) & (b == 0)).sum()))
    summ["gsm8k"] = g

    # ---------------- static
    if (R / "static/S").exists():
        d = load_static()
        s, ls = d["sigma"], np.log(d["sigma"])
        N = d["mse_train"].shape[1]
        t = np.interp(ls, np.log(tab["mids"]), tab["cdf"], left=0, right=1)
        act = s >= float(tab["edges"][0])
        cols = dict(sigma=s, t=t)
        for pre in ("mse", "mseC", "terC", "tnllC", "berC"):
            for v in ("train", "nosc", "sc", "none"):
                cols[f"{pre}_{v}"], cols[f"{pre}_{v}_se"] = mean_se(d[f"{pre}_{v}"])
        rate = cols["mse_train"] / s ** 2
        g3 = s ** 3 / (s ** 3 + 0.1 ** 3)
        rule = unit(g3 * np.sqrt(rate), ls, act)
        dens = np.interp(ls, np.log(tab["mids"]), tab["dens"], left=0, right=0)
        dens_u = unit(dens, ls, act)
        cols.update(rate=rate, rule=rule, table_density=dens_u, rate_unit=unit(rate, ls, act),
                    sqrt_unit=unit(np.sqrt(rate), ls, act))
        with open(OUT / "curves.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(cols))
            for i in range(len(s)):
                w.writerow([f"{float(cols[k][i]):.9g}" for k in cols])
        summ["static"] = dict(
            n_sequences=int(N), sigma_data=0.399844765663147,
            rule_vs_table_TV=float(0.5 * np.trapz(np.abs(rule - dens_u), ls)),
            int_rate_suffix_nats_per_bit=float(np.trapz(rate[act], ls[act])),
            int_rate_answer_bits_per_token=float(np.trapz((cols["mseC_train"] / s ** 2)[act], ls[act]) * 16 / np.log(2)),
            at_max_noise=dict(terC=float(cols["terC_train"][-1]), tnllC_nats=float(cols["tnllC_train"][-1]),
                              mse_suffix=float(cols["mse_train"][-1]), mseC=float(cols["mseC_train"][-1])),
            # rate statistics on the main bump (sigma >= 0.1); a tiny error spike near sigma ~ 0.03 (MSE ~1e-5) is
            # hugely inflated by 1/sigma^2 and reported separately (the schedule's g(sigma) suppresses it)
            rate=qstats(unit(rate, ls, s >= 0.1), ls), sqrt_rate=qstats(unit(np.sqrt(rate), ls, s >= 0.1), ls),
            small_sigma_spike=dict(sigma=float(s[(s < 0.1)][np.argmax(cols["mseC_train"][s < 0.1])]),
                                   mseC=float(cols["mseC_train"][s < 0.1].max()),
                                   mse_suffix=float(cols["mse_train"][s < 0.1].max())),
            rule=qstats(rule, ls), table=qstats(dens_u, ls),
            se_rel_median=float(np.median((cols["mse_train_se"] / cols["mse_train"])[cols["mse_train"] > 1e-4])),
        )
        # fig static
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
        ax = axs[0]
        for k, lab, c in (("mse_train", "suffix (training loss region)", COL["S"]), ("mseC_train", "answer tokens", COL["o"])):
            ax.plot(s, cols[k], color=c, lw=1.8, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-7, 0.3)
        ax.set_xlabel("noise level σ"); ax.set_ylabel("E[per-bit MSE]"); ax.legend(frameon=False, fontsize=8)
        ax.set_title("(a) Denoising MSE, CoBit-S TinyGSM", loc="left")
        ax = axs[1]
        ax.plot(t, cols["terC_train"], color=COL["S"], lw=1.8, label="token error rate (answer)")
        ax.set_xlim(1, 0); ax.set_ylim(0, None); ax.set_xlabel("diffusion time t (sampling runs →)")
        ax.set_ylabel("token error rate"); ax.set_title("(b) Answer-token error vs t", loc="left")
        ax2 = axs[2]
        ax2.plot(t, cols["tnllC_train"] / np.log(2), color=COL["S"], lw=1.8)
        ax2.set_xlim(1, 0); ax2.set_ylim(0, None); ax2.set_xlabel("diffusion time t (sampling runs →)")
        ax2.set_ylabel("answer-token NLL [bits/token]"); ax2.set_title("(c) Answer-token NLL vs t", loc="left")
        for a_ in axs:
            style(a_)
        fig.suptitle(f"CoBit-S (250k) on TinyGSM validation, N = {N}, prompt clean (training objective)", x=0.01,
                     ha="left", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / "fig_static.png", dpi=160); plt.close(fig)
        # fig rate vs sqrt
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.plot(s, cols["rate_unit"], color=COL["S"], lw=2, label="entropy rate MSE/σ²")
        ax.plot(s, cols["sqrt_unit"], color=COL["o"], lw=2, label="sqrt-rate (MSE/σ²)^½")
        ax.plot(s, rule, color=COL["g"], lw=2, label="schedule rule g(σ)·(MSE/σ²)^½")
        ax.plot(s, dens_u, color="#1b1b1a", lw=1.1, ls="--", label="stored TinyGSM schedule table")
        ax.set_xscale("log"); ax.set_xlim(float(tab["edges"][0]), 80); ax.set_ylim(0, None)
        ax.set_xlabel("noise level σ"); ax.set_ylabel("density per unit log σ (unit area)")
        ax.legend(frameon=False, fontsize=8); style(ax)
        fig.tight_layout(); fig.savefig(OUT / "fig_rate_vs_sqrt.png", dpi=160); plt.close(fig)

    # ---------------- true chains
    tf = sorted(glob.glob(str(R / "traj_true_S/chunk*.npz")))
    if tf:
        zs = [np.load(f) for f in tf]
        TC = {k: np.concatenate([z[k] for z in zs], 1) for k in zs[0].files if zs[0][k].ndim == 2}
        ref = sorted(glob.glob(str(R / "traj_control_SS/chunk*.npz")))
        sh = np.load(ref[0])["sigma_state"].astype(float)
        tt = np.interp(np.log(sh), np.log(tab["mids"]), tab["cdf"], left=0, right=1)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for v, ls_, c in (("zeros", "--", COL["gr"]), ("onestep", "-.", COL["v"]), ("carry", "-", COL["S"])):
            axs[0].plot(tt, TC[f"mse_S_{v}"].mean(1), color=c, ls=ls_, lw=1.8, label=f"{v} SC")
        for v, c in (("onestep", COL["v"]), ("carry", COL["S"])):
            axs[1].plot(tt, TC[f"mse_S_{v}"].mean(1) - TC["mse_S_zeros"].mean(1), color=c, lw=1.8, label=f"{v} − zero")
        axs[1].axhline(0, color="#8a8984", lw=0.8)
        axs[0].set_ylabel("MSE vs true x₀ (answer tokens)"); axs[1].set_ylabel("MSE(with SC) − MSE(zero SC)")
        for a_ in axs:
            a_.set_xlim(1, 0); a_.set_xlabel("diffusion time t (sampling runs →)"); a_.legend(frameon=False, fontsize=8); style(a_)
        fig.suptitle(f"CoBit-S on ground-truth chains (TinyGSM validation, N = {TC['mse_S_zeros'].shape[1]})", x=0.01,
                     ha="left", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / "fig_truechain.png", dpi=160); plt.close(fig)
        cz, oz = TC["mse_S_carry"].mean(1) - TC["mse_S_zeros"].mean(1), TC["mse_S_onestep"].mean(1) - TC["mse_S_zeros"].mean(1)
        summ["truechain"] = dict(n=int(TC["mse_S_zeros"].shape[1]),
                                 carry_minus_zero_max=float(cz.max()), t_at=float(tt[np.argmax(cz)]), sigma_at=float(sh[np.argmax(cz)]),
                                 onestep_minus_zero_min=float(oz.min()), onestep_t_at_min=float(tt[np.argmin(oz)]),
                                 onestep_minus_zero_max=float(oz.max()))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
