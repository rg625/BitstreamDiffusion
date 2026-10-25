"""Token-level breakdown (PLAN §13.4). Reads results/analysis/run2_analysis.json -> token_breakdown.png."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "results/analysis/run2_analysis.json"
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
d = json.loads(SRC.read_text())
tb = d["token_breakdowns_batch0"]
series = [("refine_4", BLUE, "refine[4]"), ("nochurn_5", ORANGE, "nochurn[5]")]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

ax = axes[0]
labels = list(tb["refine_4"]["by_ref_freq_rank"].keys())
x = np.arange(len(labels))
for j, (k, c, lab) in enumerate(series):
    v = [tb[k]["by_ref_freq_rank"][l] for l in labels]
    ax.bar(x + (j - 0.5) * 0.36, v, width=0.34, color=c, label=lab)
ax.set_xticks(x)
ax.set_xticklabels([f"rank {l}" for l in labels], fontsize=8)
ax.set_ylabel("fraction of tokens changed vs reference")
ax.set_title("a. By frequency of the reference token")
ax.legend(fontsize=8)

ax = axes[1]
for k, c, lab in series:
    v = tb[k]["by_position_bin16"]
    ax.plot(np.arange(8), v, color=c, marker="o", markersize=5, linewidth=1.6, label=lab)
ax.set_xticks(np.arange(8))
ax.set_xticklabels([f"{16*i}-{16*i+15}" for i in range(8)], fontsize=7, rotation=30)
ax.set_xlabel("token position in the 128-token sequence")
ax.set_title("b. By sequence position")

ax = axes[2]
for k, c, lab in series:
    v = tb[k]["bitflip_by_position_msb_first"]
    ax.plot(np.arange(15), v, color=c, marker="o", markersize=4, linewidth=1.6, label=lab)
ax.set_xticks(np.arange(15))
ax.set_xlabel("bit position (0 = most significant)")
ax.set_ylabel("fraction of bits flipped")
ax.set_title("c. By bit position")
for ax in axes[1:]:
    ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(HERE / "token_breakdown.png", dpi=160)
print("wrote", HERE / "token_breakdown.png")
