"""Compact figures for the PDF report, drawn from the saved results (no recomputation).

Inputs : ../results/main/curves.npz, ../results/levels/{token_level_time,sequence_level_time}.csv,
         runs/denoising_mse/traj/{main,local} (via ../analyze_traj.py)
Outputs: fig1_mse_sigma.png, fig2_token_time.png, fig4_traj.png (this folder); fig3 and fig5 are
         copied from ../results/levels/fig_schedule_overlay.png and ../results/traj/fig_traj_vs_rate.png
Run in the `sedd` env: python experiments/denoising_mse/report/make_figs.py
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
COL = {"S": "#2a78d6", "M": "#eb6834", "diff": "#3d3d3a"}
LABEL = {"S": "CoBit-S (130M)", "M": "CoBit-M (462M)"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelcolor": "#0b0b0b",
                     "axes.titlecolor": "#0b0b0b", "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.family": "DejaVu Sans"})


def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def diff(ax, x, mu, se, ylabel):
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.fill_between(x, mu - 2 * se, mu + 2 * se, color=COL["diff"], alpha=0.25, lw=0, label="±2 SE")
    ax.plot(x, mu, color=COL["diff"], lw=1.7, label="mean S − M")
    ax.set_ylabel(ylabel)
    style(ax)


def read_csv(p):
    rows = list(csv.DictReader(open(p)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right")


if (RES / "main/curves.npz").exists() and (RES / "levels/token_level_time.csv").exists():
    # Figure 1: MSE vs sigma -------------------------------------------------------------------------
    a = np.load(RES / "main/curves.npz")
    s = a["sigma"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    ax = axs[0]
    for m in ("S", "M"):
        ax.plot(s, a[f"mse_train_{m}"], color=COL[m], lw=1.8, label=LABEL[m])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.3)
    ax.set_xlabel("noise level σ")
    ax.set_ylabel("E[per-bit MSE]")
    ax.axvline(0.08, color="#8a8984", lw=0.8, ls=":")
    ax.text(0.074, 2e-2, "sampler stops\nat σ = 0.08", fontsize=7, color="#52514e", ha="right")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    style(ax)
    panel_label(ax, "a")
    diff(axs[1], s, a["mse_train_diff"], a["mse_train_diff_se"], "ΔMSE = MSE_S − MSE_M")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("noise level σ")
    axs[1].axvline(0.08, color="#8a8984", lw=0.8, ls=":")
    axs[1].legend(frameon=False, loc="upper left", fontsize=8)
    panel_label(axs[1], "b")
    fig.tight_layout()
    fig.savefig(HERE / "fig1_mse_sigma.png", dpi=220)
    plt.close(fig)

    # Figure 2: token / sequence level vs diffusion time -------------------------------------------
    tk = read_csv(RES / "levels/token_level_time.csv")
    sq = read_csv(RES / "levels/sequence_level_time.csv")
    t = tk["t"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    diff(axs[0], t, 100 * tk["terC_diff"], 100 * tk["terC_diff_se"], "Δ token error rate [pp]")
    diff(axs[1], t, tk["tnllC_diff"], tk["tnllC_diff_se"], "Δ token NLL [bits/token]")
    axs[2].axhline(0.5, color="#8a8984", lw=0.8)
    axs[2].plot(sq["t"], sq["frac_seq_M_better"], color=COL["diff"], lw=1.7)
    axs[2].set_ylim(0, 1.02)
    axs[2].set_ylabel("fraction of sequences, M lower MSE")
    style(axs[2])
    axs[0].legend(frameon=False, loc="upper right", fontsize=8)
    for ax, lab in zip(axs, "abc"):
        ax.set_xlabel("diffusion time t (CoBit-S schedule)")
        panel_label(ax, lab)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_token_time.png", dpi=220)
    plt.close(fig)

# Figure 4: paired trajectories --------------------------------------------------------------
sys.path.insert(0, str(HERE.parent))
import analyze_traj as AT  # noqa: E402

T = AT.load("main")
L = AT.load("local")
common = np.intersect1d(T["examples"], L["examples"])
li = np.searchsorted(L["examples"], common)
sh = T["sigma_state"].astype(float)
sig_after = np.r_[T["sigmas"][1:], T["sigmas"][-1]].astype(float)
tt = AT.entropy_time(sh)
rel = np.sqrt(T["dx2"]) / sig_after[:, None]


def pband(ax, x, a, color, label, lw=1.6):
    q = np.percentile(a, [2.5, 16, 84, 97.5], axis=1)
    ax.fill_between(x, q[0], q[3], color=color, alpha=0.10, lw=0)
    ax.fill_between(x, q[1], q[2], color=color, alpha=0.24, lw=0)
    ax.plot(x, a.mean(1), color=color, lw=lw, label=label)


fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
pband(axs[0], tt, rel, COL["diff"], "S vs M")
axs[0].set_yscale("log")
axs[0].set_ylim(1e-4, 20)
axs[0].axhline(1, color="#8a8984", lw=0.8, ls="--")
axs[0].set_ylabel("state gap / σ")
pband(axs[1], tt, T["dD2"], COL["diff"], "own trajectories")
pband(axs[1], tt, L["ldD2"][:, li], "#4a3aa7", "same input")
axs[1].set_yscale("log")
axs[1].set_ylim(1e-6, 1)
axs[1].set_ylabel("denoiser gap  mean (D_S − D_M)²")
pband(axs[2], tt, T["tokdis"], COL["diff"], "own trajectories")
pband(axs[2], tt, L["ltokdis"][:, li], "#4a3aa7", "same input")
axs[2].set_ylim(0, 1)
axs[2].set_ylabel("tokens predicted differently")
axs[1].legend(frameon=False, loc="lower right", fontsize=8)
for ax, lab in zip(axs, "abc"):
    ax.set_xlim(1, 0)
    ax.set_xlabel("diffusion time t (sampling runs →)")
    style(ax)
    panel_label(ax, lab)
fig.tight_layout()
fig.savefig(HERE / "fig4_traj.png", dpi=220)
plt.close(fig)

print("wrote", sorted(p.name for p in HERE.glob("fig*.png")))
