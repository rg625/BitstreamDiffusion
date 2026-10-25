"""Single-panel versions of the source report's figures for the short PDF report.

Same data and plotting code as ../figs.py (Figure 1a, Figure 1 τ panel, Figure 3b); only titles are
dropped (captions live in the PDF) and sizes are set for a one-column A4 page.
Reads ../../results/analysis/{heterogeneity,oracle,additivity}.json; writes pdf_fig{1,2,3}.png here.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "results/analysis"
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
S = 8
seg = np.arange(S)
het = json.loads((A / "heterogeneity.json").read_text())["per_checkpoint"]
feats = json.loads((HERE.parents[1] / "results/features.json").read_text())["features"]
sigma = [float(np.exp(feats[str(s)]["log_sigma"][0])) for s in seg]
TICKS = [f"{s}\nσ={sigma[s]:.2f}" if sigma[s] < 10 else f"{s}\nσ={sigma[s]:.0f}" for s in seg]
ARMS = [("1", AQUA, "+1 (+32 NFE)"), ("2", ORANGE, "+2 (+64 NFE)"), ("4", BLUE, "+4 (+128 NFE)")]


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.axvspan(1.5, 3.5, color="#f1f5fb", zorder=0, linewidth=0)
    for j, (c, col, lab) in enumerate(ARMS):
        off = (j - 1) * 0.17
        m = np.array([het[str(s)]["arms"][c]["mean_M"] for s in seg])
        ci = np.array([het[str(s)]["arms"][c]["mean_M_ci"] for s in seg])
        ax.errorbar(seg + off, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="o", color=col, markersize=4.5,
                    elinewidth=1.3, label=lab)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(seg)
    ax.set_xticklabels(TICKS, fontsize=8)
    ax.set_xlabel("checkpoint (extra compute applied over the 32 steps after it)")
    ax.set_ylabel("mean gain in Q (nats/token)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "pdf_fig1.png", dpi=220)


def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for j, (c, col, lab) in enumerate(ARMS):
        off = (j - 1) * 0.17
        tau = np.array([het[str(s)]["arms"][c]["tau"] for s in seg])
        tci = np.array([het[str(s)]["arms"][c]["tau_ci"] for s in seg])
        ax.errorbar(seg + off, tau, yerr=[tau - tci[:, 0], tci[:, 1] - tau], fmt="o", color=col,
                    markersize=4.5, elinewidth=1.3, label=f"between-state spread τ, {lab.split(' ')[0]}")
    noise = np.array([het[str(s)]["arms"]["4"]["noise_sd_M_hat"] for s in seg])
    ax.plot(seg, noise, color=GRAY, linewidth=1.6, linestyle=(0, (4, 3)),
            label="noise sd of a 4-rollout estimate (+4)")
    ax.set_xticks(seg)
    ax.set_xticklabels(TICKS, fontsize=8)
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("nats/token")
    ax.legend(fontsize=7.8, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "pdf_fig2.png", dpi=220)


def fig3():
    orc = json.loads((A / "oracle.json").read_text())["4"]["results"]
    add = json.loads((A / "additivity.json").read_text())["policy_c4_segs2-3"]
    fr = sorted(orc, key=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = [("uniform", GRAY, "-", "uniform (random states and checkpoints)"),
              ("example_oracle", AQUA, "-", "example-level oracle (best samples)"),
              ("state_oracle(t x example)", BLUE, ":", "per-state oracle (cells picked on half A)"),
              ("temporal_oracle", ORANGE, "-", "timestep-only oracle (best checkpoints)"),
              ("learned_state+t(F)", BLUE, "--", "learned predictor (state + timestep)")]
    for key, col, ls, lab in styles:
        xs = [100 * orc[f][key]["extra_nfe_per_sample"] / 255 for f in fr]
        ys = [orc[f][key]["gain_per_sample"] for f in fr]
        ax.plot([0] + xs, [0] + ys, color=col, linestyle=ls, marker="o", markersize=3.5, linewidth=1.6, label=lab)
    x = 100 * (add["nfe_per_sample"] - 255) / 255
    ax.errorbar([x], [add["measured_dQ"]], yerr=[[add["measured_dQ"] - add["measured_ci"][0]],
                                                [add["measured_ci"][1] - add["measured_dQ"]]],
                fmt="*", color=INK, markersize=10, elinewidth=1.3, zorder=5,
                label="measured end-to-end run (+4 at checkpoints 2–3)")
    ax.annotate(f"{100 * add['ratio_measured_over_predicted']:.0f}% of the\nsummed prediction", (x, add["measured_dQ"]),
                textcoords="offset points", xytext=(10, 8), fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("extra compute (% of the 255-NFE baseline)")
    ax.set_ylabel("gain in Q per sample (nats/token)")
    ax.legend(fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "pdf_fig3.png", dpi=220)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote", HERE)
