#!/usr/bin/env python
"""
Where does CoBit-M's extra capacity pay off, without and with the global score temperature?
Compares the S/M model-switching study at tau = 1 (results_tinygsm/compute_demand/summary.json) and at tau = <tag>
(results_tinygsm/compute_demand_<tag>/summary.json, from analyze_compute_demand.py --tag <tag>).

Panels: (1) accuracy gain of CoBit-M in one 32-step segment over CoBit-S, (2) the same as a share of the full CoBit-M
gain, (3) M-first / M-last switching curves, (4) full CoBit-M gain per difficulty stratum.
Outputs: results_tinygsm/compute_demand_<tag>/{fig_compare_switching.png, compare.json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results_tinygsm"
GROUPS = ["all", "≤2 steps", "3–4 steps", "5+ steps"]
COL = {"base": "#9a9994", "temp": "#2a78d6"}


def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="st0.6")
    a = ap.parse_args()
    tau = a.tag[2:].rstrip("u")
    S = {"base": json.loads((RES / "compute_demand/summary.json").read_text())["gsm8k"],
         "temp": json.loads((RES / f"compute_demand_{a.tag}/summary.json").read_text())["gsm8k"]}
    lab = {"base": "τ = 1 (no temperature)", "temp": f"τ = {tau} (score temperature)"}
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
    out = {}
    for j, (k, g) in enumerate(S.items()):
        off = -0.12 if k == "base" else 0.12
        B = g["B"]["all"]
        ks = np.array(B["segments"])
        d, lo, hi = (100 * np.array(B[x]) for x in ("delta", "lo", "hi"))
        axs[0].errorbar(ks + off, d, yerr=[d - lo, hi - d], fmt="o-", color=COL[k], ms=3.5, capsize=2, label=lab[k])
        fr = g["B_fraction"]["all"]
        axs[1].bar(ks + off * 1.6, fr["fraction"], width=0.38, color=COL[k], label=lab[k])
        ps = g["prefix_suffix"]["all"]
        for key, mk, ls in (("m_first", "o", "-"), ("m_last", "s", "--")):
            xs = sorted(map(int, ps[key]))
            axs[2].plot(xs, [ps[key][str(x)] for x in xs], marker=mk, ls=ls, color=COL[k], ms=4,
                        label=f"{lab[k]}, CoBit-M {'first' if key == 'm_first' else 'last'}")
        full = [100 * g["B_fraction"][n]["full_gain"] for n in GROUPS]
        axs[3].bar(np.arange(4) + off * 1.6, full, width=0.38, color=COL[k], label=lab[k])
        acc = g["accuracy"]
        out[k] = dict(acc_S={n: acc["base"][n] for n in GROUPS}, acc_M={n: acc["M"][n] for n in GROUPS},
                      full_gain={n: g["B_fraction"][n]["full_gain"] for n in GROUPS},
                      seg_delta=B["delta"], seg_share=fr["fraction"], sum_share=fr["sum_fraction"],
                      m_first=ps["m_first"], m_last=ps["m_last"])
    axs[0].axhline(0, color="#8a8984", lw=0.8)
    axs[0].set_title("CoBit-M in one 32-step segment: Δ vs CoBit-S [pp]", loc="left", fontsize=9)
    axs[1].axhline(0, color="#8a8984", lw=0.8)
    axs[1].set_title("same, as share of the full CoBit-M gain", loc="left", fontsize=9)
    for ax in axs[:2]:
        ax.set_xticks(range(8))
        ax.set_xlabel("segment (0 = pure noise)")
    axs[2].set_title("switching: accuracy vs steps run by CoBit-M", loc="left", fontsize=9)
    axs[2].set_xticks([0, 64, 128, 192, 256])
    axs[2].set_xlabel("steps run by CoBit-M")
    axs[2].set_ylabel("GSM8K accuracy [%]")
    axs[3].set_xticks(range(4))
    axs[3].set_xticklabels(GROUPS, fontsize=8)
    axs[3].set_title("full CoBit-M gain over CoBit-S by difficulty [pp]", loc="left", fontsize=9)
    axs[0].legend(frameon=False, fontsize=7.5)
    axs[2].legend(frameon=False, fontsize=6.5)
    for ax in axs:
        style(ax)
    fig.tight_layout()
    od = RES / f"compute_demand_{a.tag}"
    fig.savefig(od / "fig_compare_switching.png", dpi=160)
    plt.close(fig)
    (od / "compare.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, "S", {n: round(100 * x, 1) for n, x in v["acc_S"].items()}, "M", {n: round(100 * x, 1) for n, x in v["acc_M"].items()})
        print("   seg Δ", [round(100 * x, 1) for x in v["seg_delta"]], "shares", [round(x, 2) for x in v["seg_share"]],
              "Σ", round(v["sum_share"], 2))
        print("   M first", v["m_first"], " M last", v["m_last"])


if __name__ == "__main__":
    main()
