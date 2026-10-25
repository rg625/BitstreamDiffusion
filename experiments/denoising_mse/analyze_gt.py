#!/usr/bin/env python
"""
Same-input disagreement of CoBit-S and CoBit-M along three kinds of states, and the models' error along
ground-truth trajectories (trajectories.py --local [--ref M] and --truechain).

States
  S trajectories   runs/denoising_mse/traj/local      CoBit-S samples; CoBit-M fed S's exact inputs
  M trajectories   runs/denoising_mse/traj/localM     CoBit-M samples; CoBit-S fed M's exact inputs
  true chains      runs/denoising_mse/traj/truechain  VE forward chains of validation sequences walked
                                                      backwards; each model carries its own previous
                                                      prediction as self-conditioning input ("carry"), or
                                                      gets zeros (the training no-SC input; reproduces the
                                                      static curves)
Outputs (experiments/denoising_mse/results/gt/)
  fig_same_input.png   (D_S - D_M)^2 and token disagreement on identical inputs, vs diffusion time
  fig_truechain.png    MSE vs the true x0 along true chains: carry / one-step / zero SC vs static, and S - M
  fig_gray_curve.png   the surplus rate (MSE_S - MSE_M)/sigma^2 (the static "gray curve") recomputed on true
                       chains with zero, one-step (D(x_t, s_t, D(x_t+1, s_t+1, 0))) and carry SC
  fig_rates_all.png    densities over log sigma (unit area, sigma >= 0.08): same-input disagreement rates on
                       the three kinds of states, true-chain surplus rate (carry), static rates, token divergence
  gt_curves.csv, gt_summary.json
Bands: mean with central 68% / 95% of examples.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_traj as AT  # noqa: E402

C = {"S": "#2a78d6", "M": "#eb6834", "Straj": "#2a78d6", "Mtraj": "#eb6834", "true": "#1baf7a",
     "true0": "#8a8984", "surplus": "#9a9994", "div": "#3d3d3a"}
SIG_LO = AT.TASKCFG["sig_lo"]


def pband(ax, x, a, color, label, lw=1.7, ls="-", fill=True):
    if fill:
        q = np.percentile(a, [2.5, 16, 84, 97.5], axis=1)
        ax.fill_between(x, q[0], q[3], color=color, alpha=0.08, lw=0)
        ax.fill_between(x, q[1], q[2], color=color, alpha=0.20, lw=0)
    ax.plot(x, a.mean(1), color=color, lw=lw, ls=ls, label=label)


def stats(y, xs):
    o = np.argsort(xs)
    c = np.concatenate([[0], np.cumsum(0.5 * (y[o][1:] + y[o][:-1]) * np.diff(np.log(xs[o])))])
    c /= c[-1]
    q = lambda p: float(np.exp(np.interp(p, c, np.log(xs[o]))))  # noqa: E731
    return dict(sigma_peak=float(xs[np.argmax(y)]), sigma_median=q(0.5), sigma_q10=q(0.1), sigma_q90=q(0.9))


def main() -> None:
    out = HERE / AT.TASKCFG["results"] / "gt"
    out.mkdir(parents=True, exist_ok=True)
    LS, LM, TC, TR = AT.load("local"), AT.load("localM"), AT.load("truechain"), AT.load("main")
    sh = TR["sigma_state"].astype(float)
    t = AT.entropy_time(sh)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})

    # ---- Figure 1: same-input disagreement on the three kinds of states --------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    pband(axs[0], t, LS["ldD2"], C["Straj"], "along CoBit-S trajectories")
    pband(axs[0], t, LM["ldD2"], C["Mtraj"], "along CoBit-M trajectories")
    pband(axs[0], t, TC["dD2_carry"], C["true"], "along true chains (carry SC)")
    pband(axs[0], t, TC["dD2_zeros"], C["true0"], "along true chains (zero SC)", ls="--", fill=False)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-6, 1)
    axs[0].set_ylabel("same-input denoiser gap  mean (D_S − D_M)²")
    pband(axs[1], t, LS["ltokdis"], C["Straj"], "along CoBit-S trajectories")
    pband(axs[1], t, LM["ltokdis"], C["Mtraj"], "along CoBit-M trajectories")
    pband(axs[1], t, TC["tokdis_carry"], C["true"], "along true chains (carry SC)")
    pband(axs[1], t, TC["tokdis_zeros"], C["true0"], "along true chains (zero SC)", ls="--", fill=False)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("tokens predicted differently (same input)")
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axs:
        ax.set_xlim(1, 0)
        ax.set_xlabel("diffusion time t (CoBit-S schedule; sampling runs →)")
        AT.style(ax)
    fig.suptitle("How differently the two models think on identical inputs, along three kinds of states "
                 f"(N = {LS['ldD2'].shape[1]} / {LM['ldD2'].shape[1]} / {TC['dD2_carry'].shape[1]})",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_same_input.png", dpi=160)
    plt.close(fig)

    # ---- Figure 2: error vs the true x0 along true chains ---------------------------------------
    st_path = HERE / AT.TASKCFG["results"] / ("main" if AT.TASK == "owt" else "static") / "curves.npz"
    A = np.load(st_path)
    # the static curves must cover the same bits as the chain metrics: all bits on OWT, answer tokens on TinyGSM
    PRE = "mse" if AT.TASK == "owt" else "mseC"
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for m in ("S", "M"):
        pband(axs[0], t, TC[f"mse_{m}_carry"], C[m], f"CoBit-{m}, carry SC")
        pband(axs[0], t, TC[f"mse_{m}_onestep"], C[m], f"CoBit-{m}, one-step SC", ls="-.", fill=False, lw=1.3)
        pband(axs[0], t, TC[f"mse_{m}_zeros"], C[m], f"CoBit-{m}, zero SC", ls="--", fill=False)
    axs[0].set_ylabel("MSE vs true x₀ (per bit)")
    axs[0].set_title("(a) Error along true chains", loc="left")
    axs[0].legend(frameon=False, fontsize=8)
    for m in ("S", "M"):
        axs[1].plot(t, TC[f"mse_{m}_carry"].mean(1) - TC[f"mse_{m}_zeros"].mean(1), color=C[m], lw=1.8,
                    label=f"CoBit-{m}, carry − zero")
        axs[1].plot(t, TC[f"mse_{m}_onestep"].mean(1) - TC[f"mse_{m}_zeros"].mean(1), color=C[m], lw=1.3,
                    ls="-.", label=f"CoBit-{m}, one-step − zero")
    axs[1].axhline(0, color="#8a8984", lw=0.8)
    axs[1].set_ylabel("MSE(with SC) − MSE(zero SC)")
    axs[1].set_title("(b) Effect of carrying self-conditioning", loc="left")
    axs[1].legend(frameon=False, fontsize=8)
    dc = TC["mse_S_carry"] - TC["mse_M_carry"]
    dz = TC["mse_S_zeros"] - TC["mse_M_zeros"]
    do = TC["mse_S_onestep"] - TC["mse_M_onestep"]
    pband(axs[2], t, dc, C["true"], "true chains, carry SC")
    axs[2].plot(t, do.mean(1), color=C["true"], lw=1.4, ls="-.", label="true chains, one-step SC")
    axs[2].plot(t, dz.mean(1), color=C["true0"], lw=1.4, ls="--", label="true chains, zero SC")
    st_t = np.interp(np.log(sh), np.log(A["sigma"]), A[f"{PRE}_train_diff"])
    axs[2].plot(t, st_t, color=C["surplus"], lw=1.4, label="static (training mix)")
    axs[2].axhline(0, color="#8a8984", lw=0.8)
    axs[2].set_ylabel("MSE_S − MSE_M  (positive: M better)")
    axs[2].set_title("(c) CoBit-M's advantage along true chains", loc="left")
    axs[2].legend(frameon=False, fontsize=8)
    for ax in axs:
        ax.set_xlim(1, 0)
        ax.set_xlabel("diffusion time t (sampling runs →)")
        AT.style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_truechain.png", dpi=160)
    plt.close(fig)

    # ---- Figure 2b: the static surplus rate ("gray curve") under the three SC inputs -------------
    st_rate = A[f"{PRE}_train_diff"] / A["sigma"] ** 2
    variants = {"static (training-mix SC), the gray curve": (st_rate, A["sigma"], C["surplus"], "-", 2.0),
                "true chains, zero SC": (dz.mean(1) / sh ** 2, sh, C["true0"], "--", 1.4),
                "true chains, one-step SC  D(x_t, D(x_t+1))": (do.mean(1) / sh ** 2, sh, "#4a3aa7", "-", 2.0),
                "true chains, carry SC (full history)": (dc.mean(1) / sh ** 2, sh, C["true"], "-", 1.6)}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    gray = {}
    for lab, (y, xs, col, lsty, lw) in variants.items():
        a_ = xs >= SIG_LO
        axs[0].plot(xs[a_], y[a_], color=col, ls=lsty, lw=lw, label=lab)
        yy = AT.unit_area(y, np.log(xs), a_)
        axs[1].plot(xs, yy, color=col, ls=lsty, lw=lw, label=lab)
        o = np.argsort(xs[a_])
        gray[lab] = dict(integral_nats_per_bit=float(np.trapz(y[a_][o], np.log(xs[a_][o]))), **stats(yy, xs))
    # SE band for the one-step variant (per-sequence)
    se = (do / sh[:, None] ** 2).std(1, ddof=1) / np.sqrt(do.shape[1])
    m_ = sh >= SIG_LO
    axs[0].fill_between(sh[m_], (do.mean(1) / sh ** 2 - 2 * se)[m_], (do.mean(1) / sh ** 2 + 2 * se)[m_],
                        color="#4a3aa7", alpha=0.18, lw=0)
    axs[0].axhline(0, color="#8a8984", lw=0.8)
    axs[0].set_ylabel("surplus rate (MSE_S − MSE_M)/σ²  [nats/bit per unit log σ]")
    axs[0].set_title("(a) Absolute (area = surplus in the denoising bound)", loc="left")
    axs[1].set_ylabel("density per unit log σ (unit area, σ ≥ 0.08)")
    axs[1].set_title("(b) Shape only", loc="left")
    axs[0].legend(frameon=False, fontsize=8)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlim(SIG_LO, 80)
        ax.set_xlabel("noise level σ")
        AT.style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_gray_curve.png", dpi=160)
    plt.close(fig)

    # ---- Figure 3: densities over log sigma -------------------------------------------------------
    ls_ = np.log(sh)
    act = sh >= SIG_LO
    steps = slice(0, len(sh) - 1)
    curves = {
        "same-input rate, CoBit-S trajectories": (LS["ldD2"].mean(1) / sh ** 2, C["Straj"], "-", 1.8, sh),
        "same-input rate, CoBit-M trajectories": (LM["ldD2"].mean(1) / sh ** 2, C["Mtraj"], "-", 1.8, sh),
        "same-input rate, true chains (carry SC)": (TC["dD2_carry"].mean(1) / sh ** 2, C["true"], "-", 1.8, sh),
        "surplus rate on true chains (carry SC)": (dc.mean(1) / sh ** 2, C["true"], ":", 1.6, sh),
        "surplus rate on true chains (one-step SC)": (do.mean(1) / sh ** 2, "#4a3aa7", ":", 1.6, sh),
        "static surplus rate (MSE_S − MSE_M)/σ²": (A[f"{PRE}_train_diff"] / A["sigma"] ** 2, C["surplus"], "-", 1.4,
                                                  A["sigma"]),
        "static entropy rate MSE_S/σ²": (A[f"{PRE}_train_S"] / A["sigma"] ** 2, C["S"], "--", 1.2, A["sigma"]),
    }
    div = AT.slope(TR["tokdis"].mean(1)[steps], -ls_[steps])
    curves["token divergence, paired generation"] = (np.r_[div, 0.0], C["div"], "-", 2.2, sh)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    summ_rates = {}
    for lab, (y, col, lsty, lw, xs) in curves.items():
        a_ = xs >= SIG_LO
        yy = AT.unit_area(y, np.log(xs), a_)
        ax.plot(xs, yy, color=col, ls=lsty, lw=lw, label=lab)
        summ_rates[lab] = stats(yy, xs)
    ax.set_xscale("log")
    ax.set_xlim(SIG_LO, 80)
    ax.set_ylim(0, None)
    ax.set_xlabel("noise level σ")
    ax.set_ylabel("density per unit log σ (unit area, σ ≥ 0.08)")
    ax.legend(frameon=False, fontsize=8)
    AT.style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_rates_all.png", dpi=160)
    plt.close(fig)

    # ---- tables ----------------------------------------------------------------------------------
    cols = dict(step=np.arange(len(sh)), sigma=sh, t=t)
    for name, arr in (("ldD2_Straj", LS["ldD2"]), ("ldD2_Mtraj", LM["ldD2"]), ("dD2_true_carry", TC["dD2_carry"]),
                      ("dD2_true_zeros", TC["dD2_zeros"]), ("dD2_true_onestep", TC["dD2_onestep"]),
                      ("tokdis_Straj", LS["ltokdis"]),
                      ("tokdis_Mtraj", LM["ltokdis"]), ("tokdis_true_carry", TC["tokdis_carry"]),
                      ("tokdis_true_zeros", TC["tokdis_zeros"])):
        cols[f"{name}_mean"], cols[f"{name}_se"] = arr.mean(1), arr.std(1, ddof=1) / np.sqrt(arr.shape[1])
    for m in ("S", "M"):
        for v in ("carry", "zeros", "onestep"):
            for k in ("mse", "terC"):
                arr = TC[f"{k}_{m}_{v}"]
                cols[f"{k}_{m}_{v}_mean"], cols[f"{k}_{m}_{v}_se"] = arr.mean(1), arr.std(1, ddof=1) / np.sqrt(arr.shape[1])
    for nm, arr in (("carry", dc), ("onestep", do), ("zeros", dz)):
        cols[f"dmse_true_{nm}_mean"], cols[f"dmse_true_{nm}_se"] = arr.mean(1), arr.std(1, ddof=1) / np.sqrt(arr.shape[1])
    with open(out / "gt_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        for i in range(len(sh)):
            w.writerow([f"{float(cols[k][i]):.9g}" for k in cols])
    summ = dict(n=dict(Straj=int(LS["ldD2"].shape[1]), Mtraj=int(LM["ldD2"].shape[1]), true=int(TC["dD2_carry"].shape[1])),
                rates=summ_rates, gray_curve=gray)
    for nm, arr in (("Straj", LS["ltokdis"]), ("Mtraj", LM["ltokdis"]), ("true_carry", TC["tokdis_carry"]),
                    ("true_zeros", TC["tokdis_zeros"])):
        mu = arr.mean(1)
        summ[f"tokdis_peak_{nm}"] = dict(value=float(mu.max()), t=float(t[np.argmax(mu)]), sigma=float(sh[np.argmax(mu)]))
    (out / "gt_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
