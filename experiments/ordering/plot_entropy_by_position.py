#!/usr/bin/env python
"""Plot empirical token entropy against position, from entropy_by_position.py.

Three stacked panels rather than two y-axes on one: entropy (bits) and example
support (%) are different measures, and overlaying them on twin axes invites
reading a crossing point that does not exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
SURFACE = "#fcfcfb"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/ordering/entropy_by_position/entropy_by_position.json")
    ap.add_argument("--out", default="results/ordering/figures/entropy_by_position.png")
    ap.add_argument("--min-support", type=float, default=0.10,
                    help="stop the answer-frame panels below this example fraction")
    args = ap.parse_args()

    d = json.loads(Path(args.json).read_text())
    n_ex = d["n_examples"]

    ab = d["absolute"]
    ab_pos = np.array([r["pos"] for r in ab])
    ab_h = np.array([r["H_mm"] for r in ab])
    ab_pad = np.array([r["pad_frac"] for r in ab])

    al = d["aligned"]
    al_pos = np.array([r["pos"] for r in al])
    al_h = np.array([r["H_mm_nopad"] for r in al])
    al_sup = np.array([r["n_nopad"] for r in al], dtype=float) / n_ex

    keep = al_sup >= args.min_support
    cut = int(al_pos[keep].max()) if keep.any() else int(al_pos.max())

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1, 1, 0.55], "hspace": 0.38})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=MUTED, linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=9)

    # (1) absolute position -------------------------------------------------
    ax = axes[0]
    ax.plot(ab_pos, ab_h, color=BLUE, linewidth=2.0)
    ax.set_title("Token entropy by ABSOLUTE position in the 512-token block",
                 color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel("H (bits)", color=INK2, fontsize=10)
    ax.set_xlabel("position in block", color=INK2, fontsize=10)
    ax.set_xlim(0, 511)
    # where answers typically begin
    pl = d.get("prompt_len_mean")
    if pl is None:   # JSON written before that field was recorded
        try:
            from experiments.ordering.entropy_by_position import CACHE, STEM
            pl = float(np.fromfile(CACHE / f"{STEM}_{d['split']}_plen.int32",
                                   dtype=np.int32).mean())
        except Exception:
            pl = None
    if pl:
        ax.axvline(pl, color=MUTED, linewidth=1.5, linestyle="--")
        ax.annotate(f"mean answer start ({pl:.0f})", xy=(pl + 10, 3.2),
                    color=INK2, fontsize=9)

    # (2) answer-relative position -----------------------------------------
    ax = axes[1]
    ax.plot(al_pos[:cut + 1], al_h[:cut + 1], color=ORANGE, linewidth=2.0)
    zero = al_pos[(al_h <= 1e-9) & (al_pos < 64)]
    if zero.size:
        z = int(zero.max())
        ax.axvspan(-0.5, z + 0.5, color=ORANGE, alpha=0.10, linewidth=0)
        # Verified by decoding the modal token at each of these positions: all
        # 11 are the same in 100% of examples.
        ax.annotate(f"H = 0 for the first {z + 1} tokens \u2014 every answer opens\n"
                    f"with the identical signature:\n"
                    f"def simple_math_problem() -> int:",
                    xy=(z + 8, max(al_h[:cut + 1]) * 0.30), color=INK2, fontsize=9,
                    family="monospace" if False else None)
    ax.set_title("Token entropy by position WITHIN THE ANSWER (padding excluded)",
                 color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel("H (bits)", color=INK2, fontsize=10)
    ax.set_xlim(0, cut)

    # (3) support -----------------------------------------------------------
    ax = axes[2]
    ax.fill_between(al_pos[:cut + 1], 100 * al_sup[:cut + 1], color=MUTED, alpha=0.55,
                    linewidth=0)
    ax.plot(al_pos[:cut + 1], 100 * al_sup[:cut + 1], color=INK2, linewidth=1.5)
    ax.set_title("share of examples whose answer reaches this position",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.set_ylabel("% of examples", color=INK2, fontsize=10)
    ax.set_xlabel("position within the answer (tokens)", color=INK2, fontsize=10)
    ax.set_xlim(0, cut)
    ax.set_ylim(0, 100)

    fig.text(0.008, 0.005,
             f"TinyGSM train split, {n_ex:,} examples, SmolLM-135M tokenizer "
             f"(V=49,153). Miller-Madow corrected. Panels 2-3 stop at "
             f"{100*args.min_support:.0f}% support.",
             color=INK2, fontsize=8)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
