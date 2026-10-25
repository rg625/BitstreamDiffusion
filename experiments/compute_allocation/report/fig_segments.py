"""Per-segment effects (PLAN §13.2). Reads results/analysis/run2_analysis.json -> segment_effects.png."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "results/analysis/run2_analysis.json"
BLUE, ORANGE, GRAY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = json.loads(SRC.read_text())
rows = d["rows"]
S = 8
seg = np.arange(S)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1]))

ax = axes[0]
for kind, color, off, label in [("refine", BLUE, -0.12, "refine[s]: 4 sub-steps per step (+96 NFE)"),
                                ("nochurn", ORANGE, 0.12, "nochurn[s]: γ = 0 in segment")]:
    m = np.array([rows[f"{kind}_{s}"]["d_nll_tok"] for s in seg])
    lo = np.array([rows[f"{kind}_{s}"]["ci_lo"] for s in seg])
    hi = np.array([rows[f"{kind}_{s}"]["ci_hi"] for s in seg])
    flag = np.array([rows[f"{kind}_{s}"]["entropy_flag"] for s in seg])
    ax.errorbar(seg + off, m, yerr=[m - lo, hi - m], fmt="none", ecolor=color, elinewidth=1.4, capsize=0)
    ax.scatter(seg[~flag] + off, m[~flag], s=34, color=color, zorder=3, label=label, edgecolors=SURF, linewidths=1.5)
    if flag.any():
        ax.scatter(seg[flag] + off, m[flag], s=34, facecolors=SURF, edgecolors=color, linewidths=1.6, zorder=3)
ax.axhline(0, color=INK2, linewidth=0.8)
for y in (0.01, -0.01):
    ax.axhline(y, color=GRAY, linewidth=0.9, linestyle=(0, (4, 3)))
ax.text(S - 0.5, 0.0105, "±0.01 materiality", ha="right", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(seg)
ax.set_xlabel("segment s (steps 32s … 32s+31)")
ax.set_ylabel("ΔNLL/token vs reference (nats)\n> 0 = worse")
ax.set_title("a. Paired effect of changing one segment (95% bootstrap CI)")
ax.legend(loc="best", fontsize=7.5)
if any(rows[f"{k}_{s}"]["entropy_flag"] for k in ("refine", "nochurn") for s in seg):
    ax.text(0.01, 0.02, "hollow marker = entropy flag (|Δentropy| > 0.02)", transform=ax.transAxes, fontsize=7, color=INK2)

ax = axes[1]
for kind, color, off in [("refine", BLUE, -0.12), ("nochurn", ORANGE, 0.12)]:
    m = np.array([rows[f"{kind}_{s}"]["d_entropy"] for s in seg])
    lo = np.array([rows[f"{kind}_{s}"]["d_entropy_ci"][0] for s in seg])
    hi = np.array([rows[f"{kind}_{s}"]["d_entropy_ci"][1] for s in seg])
    ax.errorbar(seg + off, m, yerr=[m - lo, hi - m], fmt="none", ecolor=color, elinewidth=1.4)
    ax.scatter(seg + off, m, s=30, color=color, zorder=3, edgecolors=SURF, linewidths=1.5)
ax.axhline(0, color=INK2, linewidth=0.8)
for y in (0.02, -0.02):
    ax.axhline(y, color=GRAY, linewidth=0.9, linestyle=(0, (4, 3)))
ax.text(S - 0.5, 0.0205, "±0.02 flag", ha="right", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(seg)
ax.set_xlabel("segment s")
ax.set_ylabel("Δ unigram entropy (nats)")
ax.set_title("b. Entropy change")

fig.tight_layout()
fig.savefig(HERE / "segment_effects.png", dpi=160)
print("wrote", HERE / "segment_effects.png")
