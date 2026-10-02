"""Figures 1–4 of PLAN.md §14 from results/analysis/*.json|npz. Writes fig1_*.png … fig4_*.png here."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent / "results/analysis"
FEAT = HERE.parent / "results/features.json"
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID, SURF = ("#2a78d6", "#eb6834", "#1baf7a", "#8a8984",
                                                    "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb")
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
S = 8
het = json.loads((A / "heterogeneity.json").read_text())
pc = het["per_checkpoint"]
arr = np.load(A / "arrays.npz")
seg = np.arange(S)
ARM_COL = {"1": AQUA, "2": ORANGE, "4": BLUE}


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    for j, (c, col) in enumerate(ARM_COL.items()):
        off = (j - 1) * 0.18
        m = np.array([pc[str(s)]["arms"][c]["mean_M"] for s in seg])
        ci = np.array([pc[str(s)]["arms"][c]["mean_M_ci"] for s in seg])
        ax.errorbar(seg + off, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="o", color=col, markersize=4.5,
                    elinewidth=1.3, label=f"+{c} (+{32*int(c)} NFE)")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(seg)
    ax.set_xlabel("checkpoint s (state at step 32s)")
    ax.set_ylabel("mean marginal value  E[Q(+c) − Q(+0)]  (nats/token)")
    ax.set_title("a. Average value of extra compute (95% CI over states)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for j, (c, col) in enumerate(ARM_COL.items()):
        off = (j - 1) * 0.18
        tau = np.array([pc[str(s)]["arms"][c]["tau"] for s in seg])
        tci = np.array([pc[str(s)]["arms"][c]["tau_ci"] for s in seg])
        ax.errorbar(seg + off, tau, yerr=[tau - tci[:, 0], tci[:, 1] - tau], fmt="o", color=col,
                    markersize=4.5, elinewidth=1.3, label=f"between-state sd τ, +{c}")
    noise = np.array([pc[str(s)]["arms"]["4"]["noise_sd_M_hat"] for s in seg])
    ax.plot(seg, noise, color=GRAY, linewidth=1.6, linestyle=(0, (4, 3)), label="noise sd of a 4-rollout M̂ (+4)")
    ax.axhline(0.01, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(S - 0.6, 0.0105, "0.01 threshold", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(seg)
    ax.set_xlabel("checkpoint s")
    ax.set_ylabel("nats/token")
    ax.set_title("b. Within-timestep spread of marginal value vs noise")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "fig1_heterogeneity.png", dpi=160)

    # 1b: marginal value vs a state statistic, one panel per checkpoint (small multiples)
    feats = json.loads(FEAT.read_text())["features"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
    MA, MB, rs = arr["MA_c4"], arr["MB_c4"], arr["rows_s"]
    N = int((rs == 0).sum())
    for s in seg:
        ax = axes.flat[s]
        stat = np.asarray(feats[str(s)]["H_mean" if s > 0 else "x_rms_over_sigma"])[:N]
        m = 0.5 * (MA[rs == s] + MB[rs == s])
        ax.scatter(stat, m, s=7, color=BLUE, alpha=0.55, linewidths=0)
        ax.axhline(0, color=INK2, linewidth=0.7)
        nz = pc[str(s)]["arms"]["4"]["noise_sd_M_hat"]
        ax.axhspan(-nz, nz, color=GRAY, alpha=0.15, linewidth=0)
        ax.set_title(f"s={s}  τ={pc[str(s)]['arms']['4']['tau']:.3f}", fontsize=9)
        ax.set_xlabel("mean bit entropy of x̂₀" if s > 0 else "rms(x−½)/σ (no x̂₀ yet)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("M̂(+4), 4 rollouts")
    fig.suptitle("Figure 1b. Marginal value of +4 vs state, by checkpoint (gray band = ±1 noise sd)",
                 fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fig1b_state_scatter.png", dpi=160)


def fig2():
    pr = json.loads((A / "predictors.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, arm in zip(axes, ["1", "4"]):
        models = list(pr[arm]["models"].keys())
        y = np.arange(len(models))
        v = np.array([pr[arm]["models"][m]["spearman_within_timestep"] for m in models])
        ci = np.array([pr[arm]["models"][m]["spearman_within_ci"] for m in models])
        ax.errorbar(v, y, xerr=[v - ci[:, 0], ci[:, 1] - v], fmt="o", color=BLUE, markersize=5, elinewidth=1.3,
                    label="within-timestep Spearman ρ (CV, fit on A, scored on B)")
        vp = np.array([pr[arm]["models"][m]["spearman_pooled"] for m in models])
        ax.scatter(vp, y, marker="D", s=22, color=ORANGE, zorder=3, label="pooled ρ (includes between-timestep)")
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("rank correlation with held-out M̂_B")
        ax.set_title(f"Figure 2{'ab'[arm == '4']}. Predicting the value of +{arm}")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "fig2_predictors.png", dpi=160)


def fig3_fig4():
    orc = json.loads((A / "oracle.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    styles = [("uniform", GRAY, "-", "uniform (random cells)"),
              ("temporal_oracle", ORANGE, "-", "temporal oracle (best checkpoints)"),
              ("example_oracle", AQUA, "-", "example-level oracle (best samples)"),
              ("state_oracle(t x example)", BLUE, "-", "state oracle (best cells, chosen on half A)"),
              ("learned_state+t(F)", BLUE, "--", "learned predictor (state + t)")]
    for ax, arm in zip(axes, ["1", "4"]):
        res = orc[arm]["results"]
        fr = sorted(res, key=float)
        for key, col, ls, lab in styles:
            xs = [100 * res[f][key]["extra_nfe_per_sample"] / 255 for f in fr]
            ys = [res[f][key]["gain_per_sample"] for f in fr]
            ax.plot([0] + xs, [0] + ys, color=col, linestyle=ls, marker="o", markersize=4, linewidth=1.6, label=lab)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xlabel("extra compute (% of the 255-NFE baseline)")
        ax.set_ylabel("quality gain per sample, Σ M̂_B (nats/token)")
        ax.set_title(f"Figure 3{'ab'[arm == '4']}. Fixed-budget allocation, unit = +{arm} per cell")
    axes[0].legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(HERE / "fig3_oracle.png", dpi=160)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    res = orc["4"]["results"]["0.250"]
    for j, (key, col, lab) in enumerate([("state_oracle(t x example)", BLUE, "state oracle"),
                                         ("learned_state+t(F)", ORANGE, "learned (state + t)")]):
        share = np.array(res[key]["share_cells_by_checkpoint"])
        ax.bar(seg + (j - 0.5) * 0.38, share, width=0.36, color=col, label=lab)
    ax.axhline(0.25, color=GRAY, linewidth=1.6, linestyle=(0, (4, 3)), label="uniform (25% of cells)")
    ax.set_xticks(seg)
    ax.set_xlabel("checkpoint s")
    ax.set_ylabel("fraction of states given +4")
    ax.set_title("Figure 4. Where a 25%-of-cells budget goes (+4 unit)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(HERE / "fig4_allocation.png", dpi=160)


if __name__ == "__main__":
    fig1()
    if (A / "predictors.json").exists():
        fig2()
        fig3_fig4()
    print("figures written to", HERE)
