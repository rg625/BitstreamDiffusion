#!/usr/bin/env python
"""
Aggregate runs/denoising_mse/<tag>/{S,M}/chunk*.npz (from mse_curves.py) into expected denoising
MSE curves, paired S-vs-M differences with standard errors, and figures.

Outputs (experiments/denoising_mse/results/<tag>/):
  curves.npz      every array below, plus per-bit-position means
  curves.csv      one row per sigma grid point (all bits; see columns)
  curves_content.csv   same, MSE restricted to non-pad code tokens
  curves_time_{S,M,own}.csv  MSE vs diffusion time t: both models on S's schedule, on M's
                  schedule, or each on its own
  summary.json    run metadata and scalar summaries
  fig_mse_sigma.png, fig_mse_time_{S,M,own}.png, fig_variants.png

Conventions
  variants: nosc = D(x, s, zeros), sc = D(x, s, sigmoid(D(x, s, None))), none = D(x, s, None),
            train = 0.5 * (nosc + sc), the per-sigma training objective (self_condition_prob 0.5).
  diff = S - M (positive: the 462M model has lower error). Paired per sequence (same x0, same eps).
  SE = sd over sequences / sqrt(N).
  Diffusion time t_X(sigma) = F_X(sigma), the entropy-rate CDF of model X
  (assets/entropy_tables/{owt,owt_medium}); t = 0 clean, t = 1 pure noise. The entropic sampler
  steps uniformly in t.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VARIANTS = ("train", "nosc", "sc", "none")
COL = {"S": "#2a78d6", "M": "#eb6834", "diff": "#3d3d3a"}
LABEL = {"S": "CoBit-S (130M)", "M": "CoBit-M (462M)"}

# Task switch (default OWT): DMSE_TASK=tinygsm runs the identical analysis on the TinyGSM runs.
import os as _os
TASK = _os.environ.get("DMSE_TASK", "owt").lower()
TASKCFG = {
    "owt": dict(tag_prefix="", results="results", tables={"S": "assets/entropy_tables/owt", "M": "assets/entropy_tables/owt_medium"},
                sig_lo=0.08, name="OWT"),
    "tinygsm": dict(tag_prefix="tinygsm/", results="results_tinygsm",
                    tables={"S": "runs/tasks/tinygsm/cobit_raw_binary_bits", "M": "assets/cobit_m_bits_s42"},
                    sig_lo=0.1, name="TinyGSM"),
}[TASK]
VLABEL = {"train": "training objective ½(no-SC + SC)", "nosc": "no self-conditioning",
          "sc": "self-conditioned", "none": "SC bootstrap pass (x̂=None)"}


def load_model(tag: str, m: str):
    tag = TASKCFG["tag_prefix"] + tag
    fs = sorted(glob.glob(str(REPO / f"runs/denoising_mse/{tag}/{m}/chunk*.npz")))
    if not fs:
        raise SystemExit(f"no chunks for {m}")
    zs = [np.load(f) for f in fs]
    d = dict(idx=np.concatenate([z["idx"] for z in zs]), sigmas=zs[0]["sigmas"],
             wall=float(sum(float(z["wall_sec"]) for z in zs)))
    n = np.array([len(z["idx"]) for z in zs], float)
    for pre in ("mse", "mseC"):
        for v in ("nosc", "sc", "none"):
            d[f"{pre}_{v}"] = np.concatenate([z[f"{pre}_{v}"] for z in zs], 1)   # [G, N]
        d[f"{pre}_train"] = 0.5 * (d[f"{pre}_nosc"] + d[f"{pre}_sc"])
    for v in ("nosc", "sc", "none"):
        d[f"pos_{v}"] = sum(z[f"pos_{v}"] * k for z, k in zip(zs, n)) / n.sum()  # [G, 16]
    d["pos_train"] = 0.5 * (d["pos_nosc"] + d["pos_sc"])
    d["meta"] = json.loads((REPO / f"runs/denoising_mse/{tag}/{m}/meta.json").read_text())
    d["meta"].setdefault("seq_len_bits", 16384)
    return d


def entropy_time(sig, table_dir):
    import torch

    c = torch.load(REPO / table_dir / "entropy_cdf.pt", map_location="cpu").numpy().astype(float)
    s = torch.load(REPO / table_dir / "entropy_sigmas.pt", map_location="cpu").numpy().astype(float)
    c[-1] = 1.0
    return np.interp(np.log(sig), np.log(s), c, left=0.0, right=1.0)


def mean_se(a):
    return a.mean(1), a.std(1, ddof=1) / np.sqrt(a.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    out = HERE / TASKCFG["results"] / args.tag
    out.mkdir(parents=True, exist_ok=True)

    D = {m: load_model(args.tag, m) for m in ("S", "M")}
    # pair on the common set of sequences (both runs use the same index list)
    common = np.intersect1d(D["S"]["idx"], D["M"]["idx"])
    for m in D:
        sel = np.searchsorted(D[m]["idx"], common)
        assert np.array_equal(D[m]["idx"][sel], common)
        for k in list(D[m]):
            if k.startswith("mse"):
                D[m][k] = D[m][k][:, sel]
    assert np.allclose(D["S"]["sigmas"], D["M"]["sigmas"])
    sig = D["S"]["sigmas"]
    N = len(common)
    sd = float(D["S"]["meta"]["sigma_data"])
    lam = (sig ** 2 + sd ** 2) / (sig ** 2 * sd ** 2)  # EDM loss weight
    t = {"S": entropy_time(sig, D["S"]["meta"]["entropy_run_dir"]),
         "M": entropy_time(sig, D["M"]["meta"]["entropy_run_dir"])}

    arr = dict(sigma=sig, edm_weight=lam, t_S=t["S"], t_M=t["M"], seq_idx=common)
    for pre in ("mse", "mseC"):
        for v in VARIANTS:
            for m in ("S", "M"):
                mu, se = mean_se(D[m][f"{pre}_{v}"])
                arr[f"{pre}_{v}_{m}"], arr[f"{pre}_{v}_{m}_se"] = mu, se
            dmu, dse = mean_se(D["S"][f"{pre}_{v}"] - D["M"][f"{pre}_{v}"])
            arr[f"{pre}_{v}_diff"], arr[f"{pre}_{v}_diff_se"] = dmu, dse
            arr[f"{pre}_{v}_reldiff"] = dmu / np.maximum(arr[f"{pre}_{v}_M"], 1e-300)
    for v in VARIANTS:
        for m in ("S", "M"):
            arr[f"pos_{v}_{m}"] = D[m][f"pos_{v}"]
    np.savez(out / "curves.npz", **arr)

    for pre, fname in (("mse", "curves.csv"), ("mseC", "curves_content.csv")):
        cols = ["sigma", "log_sigma", "t_S", "t_M", "edm_weight"]
        for v in VARIANTS:
            cols += [f"{v}_S", f"{v}_S_se", f"{v}_M", f"{v}_M_se", f"{v}_diff", f"{v}_diff_se",
                     f"{v}_reldiff"]
        with open(out / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for g in range(len(sig)):
                row = [sig[g], np.log(sig[g]), t["S"][g], t["M"][g], lam[g]]
                for v in VARIANTS:
                    row += [arr[f"{pre}_{v}_S"][g], arr[f"{pre}_{v}_S_se"][g], arr[f"{pre}_{v}_M"][g],
                            arr[f"{pre}_{v}_M_se"][g], arr[f"{pre}_{v}_diff"][g],
                            arr[f"{pre}_{v}_diff_se"][g], arr[f"{pre}_{v}_reldiff"][g]]
                w.writerow([f"{x:.9g}" for x in row])

    # --- diffusion-time views: per-sequence curves interpolated in log sigma onto a t grid ------
    # time_ref "S" / "M": both models on that model's schedule (a common time axis, so the
    # difference is a model difference); "own": each model on its own schedule (difference mixes
    # model quality with the ~7% shift between the two schedules' sigma(t)).
    tg = np.linspace(0.0, 1.0, 201)
    ls = np.log(sig)

    def ls_at_t(sched):
        tm = t[sched]
        keep = np.r_[True, np.diff(tm) > 1e-12]
        return np.interp(tg, tm[keep], ls[keep])

    timecurves = {}
    for ref in ("S", "M", "own"):
        tc = {}
        seqs = {}
        for m in ("S", "M"):
            ls_t = ls_at_t(m if ref == "own" else ref)
            tc[f"sigma_at_t_{m}"] = np.exp(ls_t)
            for v in VARIANTS:
                a = D[m][f"mse_{v}"]
                per_seq = np.stack([np.interp(ls_t, ls, a[:, n]) for n in range(N)], 1)  # [T, N]
                tc[f"{v}_{m}"], tc[f"{v}_{m}_se"] = mean_se(per_seq)
                seqs[(v, m)] = per_seq
        for v in VARIANTS:
            tc[f"{v}_diff"], tc[f"{v}_diff_se"] = mean_se(seqs[(v, "S")] - seqs[(v, "M")])
        timecurves[ref] = tc
        np.savez(out / f"curves_time_{ref}.npz", t=tg, **tc)
        with open(out / f"curves_time_{ref}.csv", "w", newline="") as f:
            w = csv.writer(f)
            cols = ["t", "sigma_at_t_S", "sigma_at_t_M"] + [f"{v}_{x}" for v in VARIANTS
                                                            for x in ("S", "S_se", "M", "M_se", "diff", "diff_se")]
            w.writerow(cols)
            for i in range(len(tg)):
                w.writerow([f"{tg[i]:.9g}"] + [f"{tc[c][i]:.9g}" for c in cols[1:]])

    # --- scalar summaries --------------------------------------------------------------------
    def integ(y):  # integral over log sigma (trapezoid)
        return float(np.trapz(y, ls))

    summ = dict(
        tag=args.tag, n_sequences=N, n_sigma=len(sig), sigma_range=[float(sig[0]), float(sig[-1])],
        bits_per_point=int(N * D["S"]["meta"]["seq_len_bits"]),
        gpu_wall_hours={m: D[m]["wall"] / 3600 for m in D}, meta={m: D[m]["meta"] for m in D},
        definitions=__doc__,
    )
    per_seq_int_diff = {v: np.trapz(D["S"][f"mse_{v}"] - D["M"][f"mse_{v}"], ls, axis=0) for v in VARIANTS}
    for v in VARIANTS:
        dmu, dse = arr[f"mse_{v}_diff"], arr[f"mse_{v}_diff_se"]
        z = dmu / np.maximum(dse, 1e-300)
        g_max = int(np.argmax(dmu))
        summ[v] = dict(
            int_dlogsigma_S=integ(arr[f"mse_{v}_S"]), int_dlogsigma_M=integ(arr[f"mse_{v}_M"]),
            int_dlogsigma_diff=integ(dmu),
            int_dlogsigma_diff_se=float(per_seq_int_diff[v].std(ddof=1) / np.sqrt(N)),
            max_diff=float(dmu[g_max]), max_diff_se=float(dse[g_max]), sigma_at_max_diff=float(sig[g_max]),
            max_reldiff=float(arr[f"mse_{v}_reldiff"][np.argmax(np.where(arr[f"mse_{v}_M"] > 1e-3, arr[f"mse_{v}_reldiff"], -np.inf))]),
            sigma_where_M_worse_significant=[float(s) for s in sig[z < -3]],
            sigma_where_M_better_significant=[float(s) for s in sig[z > 3]],
            median_se_rel_active=float(np.median((arr[f"mse_{v}_S_se"] / arr[f"mse_{v}_S"])[(sig > 0.15) & (sig < 20)])),
        )
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=float))

    plot(out, arr, timecurves, tg, N)
    print(f"[analyze] N={N} sequences, wrote {out}")


def _style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def plot(out, arr, tc, tg, N):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": "#0b0b0b",
                         "axes.titlecolor": "#0b0b0b", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    sig = arr["sigma"]

    def three_panel(x, get, xlabel, fname, logx, title, xsig):
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True,
                                gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
        for m in ("S", "M"):
            mu, se = get(m)
            axs[0].plot(x, mu, color=COL[m], lw=2, label=LABEL[m])
            axs[0].fill_between(x, mu - 2 * se, mu + 2 * se, color=COL[m], alpha=0.2, lw=0)
        axs[0].set_ylabel("E[ per-bit MSE ]  (log)")
        axs[0].set_yscale("log")
        axs[0].set_ylim(1e-6, 0.3)
        axs[0].legend(frameon=False, loc="upper left")
        mu, se = get("diff")
        mM = get("M")[0]
        # relative difference only in the sampler's range and where the MSE is non-negligible
        act = np.where((mM > 1e-3) & (xsig >= 0.08), 1.0, np.nan)
        for ax, y, s, lab in ((axs[1], mu, se, "ΔMSE = S − M"),
                              (axs[2], 100 * act * mu / mM, 100 * act * se / mM,
                               "relative Δ = (S − M)/M  [%]")):
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.fill_between(x, y - 2 * s, y + 2 * s, color=COL["diff"], alpha=0.22, lw=0, label="±2 SE")
            ax.plot(x, y, color=COL["diff"], lw=2, label="mean")
            ax.set_ylabel(lab)
        axs[1].legend(frameon=False, loc="upper right")
        axs[2].text(0.01, 0.03, "shown where σ ≥ 0.08 (sampler range) and MSE_M > 1e-3", transform=axs[2].transAxes,
                    color="#52514e", fontsize=7)
        for ax in axs:
            _style(ax)
            if logx:
                ax.set_xscale("log")
        axs[-1].set_xlabel(xlabel)
        axs[0].set_title(title, loc="left")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=160)
        plt.close(fig)

    def g_sigma(m, v="train", pre="mse"):
        if m == "diff":
            return arr[f"{pre}_{v}_diff"], arr[f"{pre}_{v}_diff_se"]
        return arr[f"{pre}_{v}_{m}"], arr[f"{pre}_{v}_{m}_se"]

    three_panel(sig, g_sigma, "noise level σ", "fig_mse_sigma.png", True,
                f"Denoising MSE vs σ, {TASKCFG['name']} val, N={N} sequences (training objective)", sig)
    for ref, desc in (("S", "CoBit-S's entropy-rate schedule (common axis)"),
                      ("M", "CoBit-M's entropy-rate schedule (common axis)"),
                      ("own", "each model's own entropy-rate schedule")):
        three_panel(tg, lambda m, c=tc[ref]: (c[f"train_{m}"], c[f"train_{m}_se"]),
                    f"diffusion time t = F(σ) under {desc}", f"fig_mse_time_{ref}.png", False,
                    f"Denoising MSE vs diffusion time, N={N} sequences (training objective)",
                    tc[ref]["sigma_at_t_M"])

    # variants: Δ vs sigma for each self-conditioning input, all bits and content-only
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    for j, v in enumerate(VARIANTS):
        for i, pre in enumerate(("mse", "mseC")):
            ax = axs[i, j]
            mu, se = g_sigma("diff", v, pre)
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.fill_between(sig, mu - 2 * se, mu + 2 * se, color=COL["diff"], alpha=0.22, lw=0)
            ax.plot(sig, mu, color=COL["diff"], lw=1.6)
            ax.set_xscale("log")
            _style(ax)
            if i == 0:
                ax.set_title(VLABEL[v], fontsize=8.5, loc="left")
            if j == 0:
                ax.set_ylabel(("all bits" if pre == "mse" else "content tokens only") + "\nΔMSE = S − M")
            if i == 1:
                ax.set_xlabel("σ")
    fig.suptitle(f"S − M denoising-MSE difference by self-conditioning input (±2 SE), N={N}",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_variants.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
