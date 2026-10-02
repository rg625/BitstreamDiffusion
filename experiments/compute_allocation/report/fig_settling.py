"""Settling-time figure (PLAN §13.1). Reads results/analysis/run1_settling.json -> settling.png."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "results/analysis/run1_settling.json"
BLUE, ORANGE, GRAY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = json.loads(SRC.read_text())
S = 8
seg = np.arange(S)
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

# (a) share of tokens whose final settling falls in each segment vs uniform compute share
ax = axes[0]
share = np.array(d["settle_share_per_segment"])
uni = np.array(d["uniform_share_per_segment"])
ax.bar(seg, share, width=0.72, color=BLUE, label="tokens settling in segment")
ax.hlines(uni, seg - 0.42, seg + 0.42, color=GRAY, linewidth=2, label="compute share (model calls)")
for s, v in zip(seg, share):
    ax.text(s, v + 0.006, f"{v:.0%}" if v >= 0.005 else f"{v:.1%}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(seg)
ax.set_xlabel("segment (32 steps each; 7 = 30 steps + final denoise)")
ax.set_ylabel("fraction")
ax.set_title("a. Where final token decisions are made")
ax.legend(loc="upper left", fontsize=7.5)

# (b) by step: final-settle histogram and per-step guess-change rate
ax = axes[1]
h = np.array(d["settle_hist_by_step"], dtype=float)
h /= h.sum()
steps = np.arange(h.size)
ax.plot(steps, h, color=BLUE, linewidth=1.6, label="tokens settling at step (fraction)")
for b in range(1, S):
    ax.axvline(32 * b - 0.5, color=GRID, linewidth=0.8, zorder=0)
ax.set_xlim(0, h.size - 1)
ax.set_xlabel("step (0 = σ 76.8 … 254 = final denoise at σ 0.08)")
ax.set_ylabel("fraction of tokens")
ax.set_title("b. Final settling by step")
ax.text(0.02, 0.97, f"median settle step {d['median_settle_step']:.0f}\n"
        f"{d['overall_frac_revised_after_provisional']:.0%} of tokens change again\nafter 8 unchanged steps",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK2)

# (d) per-step guess-change rate (all tokens)
ax = axes[3]
cr = np.array(d["change_rate_by_step"])
ax.plot(np.arange(1, cr.size + 1), cr, color=BLUE, linewidth=1.6)
for b in range(1, S):
    ax.axvline(32 * b - 0.5, color=GRID, linewidth=0.8, zorder=0)
ax.set_xlim(0, cr.size)
ax.set_xlabel("step")
ax.set_ylabel("fraction of tokens whose guess changes")
ax.set_title("d. Guess changes per step")

# (c) bit position: mean final-settle step per bit (MSB first)
ax = axes[2]
bp = d["bit_position"]
bits = np.array([r["bit"] for r in bp])
mean = np.array([r["mean_settle_step"] for r in bp])
never = np.array([r["frac_never_changed"] for r in bp])
ax.plot(bits, mean, color=BLUE, marker="o", markersize=5, linewidth=1.6, label="mean settle step")
ax.set_xticks(bits)
ax.set_xlabel("bit position (0 = most significant)")
ax.set_ylabel("step")
ax.set_ylim(0, 254)
for b, m, n in zip(bits, mean, never):
    if n > 0.2:
        ax.annotate(f"{n:.0%} never\nchange", (b, m), textcoords="offset points", xytext=(9, 3), fontsize=7, color=INK2)
ax.set_title("c. Settling by bit position")

fig.tight_layout()
fig.savefig(HERE / "settling.png", dpi=160)
print("wrote", HERE / "settling.png")
