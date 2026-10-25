#!/usr/bin/env python
"""
Entropy rate vs its square root (the schedule's "sqrt-rate" target), per model.

For each model X, from the measured training-objective MSE (results/main/curves.npz):
  rate_X(σ)      = MSE_X / σ²                       I-MMSE information rate per unit log σ
  sqrt-rate_X(σ) = rate_X^0.5                       cfg.train.entropy_target = "sqrt-rate"
  rule_X(σ)      = g(σ) · rate_X^0.5,  g = σ³/(σ³ + 0.1³)   the trainer's schedule rule
  table_X(σ)     the stored schedule density (assets/entropy_tables/{owt,owt_medium})
plus the S − M surplus rate (MSE_S − MSE_M)/σ² for reference. Every curve is a density over log σ
with unit area on σ ≥ 0.08 (the schedules put no mass below it; this also keeps CoBit-M's
σ ≈ 0.007 error spike, where MSE/σ² is huge, out of the normalisation).

Outputs: results/levels/fig_rate_vs_sqrt.png, results/levels/rate_vs_sqrt.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

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
OUT = HERE / TASKCFG["results"] / "levels"
SIG_LO = TASKCFG["sig_lo"]
TABLES = TASKCFG["tables"]
LABEL = {"S": "CoBit-S (130M)", "M": "CoBit-M (462M)"}
C = {"rate": "#2a78d6", "sqrt": "#eb6834", "rule": "#1baf7a", "table": "#1b1b1a", "surplus": "#9a9994"}


def unit_area(y, ls, act):
    y = np.where(act, np.clip(y, 0, None), 0.0)
    return y / np.trapz(y, ls)


def table_density(d, ls):
    rd = lambda n: torch.load(REPO / d / n, map_location="cpu").numpy().astype(float)  # noqa: E731
    pdf, mids, edges = rd("entropy_pdf.pt"), rd("entropy_sigmas.pt"), rd("entropy_edges.pt")
    dens = pdf / np.diff(np.log(edges))
    return np.interp(ls, np.log(mids), dens, left=0, right=0)


def stats(y, ls, sig):
    c = np.concatenate([[0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(ls))])
    c /= c[-1]
    q = lambda p: float(np.exp(np.interp(p, c, ls)))  # noqa: E731
    return dict(sigma_peak=float(sig[np.argmax(y)]), sigma_q05=q(0.05), sigma_median=q(0.5), sigma_q95=q(0.95))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = np.load(HERE / TASKCFG["results"] / ("main" if TASK == "owt" else "static") / "curves.npz")
    sig = a["sigma"]
    ls = np.log(sig)
    act = sig >= SIG_LO
    g = sig ** 3 / (sig ** 3 + 0.1 ** 3)
    surplus = unit_area((a["mse_train_S"] - a["mse_train_M"]) / sig ** 2, ls, act)
    cols = dict(sigma=sig, surplus_rate=surplus)
    summary = {"surplus_rate": stats(surplus, ls, sig)}

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, m in zip(axs, ("S", "M")):
        rate = a[f"mse_train_{m}"] / sig ** 2
        curves = {
            "rate": (unit_area(rate, ls, act), "entropy rate  MSE/σ²", "-", 2.0),
            "sqrt": (unit_area(np.sqrt(np.clip(rate, 0, None)), ls, act), "sqrt-rate  (MSE/σ²)^½", "-", 2.0),
            "rule": (unit_area(g * np.sqrt(np.clip(rate, 0, None)), ls, act),
                     "schedule rule  g(σ)·(MSE/σ²)^½", "-", 2.0),
            "table": (unit_area(table_density(TABLES[m], ls), ls, act), "stored schedule table", "--", 1.1),
        }
        ax.plot(sig, surplus, color=C["surplus"], lw=1.4, label="surplus rate (MSE_S − MSE_M)/σ²", zorder=1)
        for k, (y, lab, lsty, lw) in curves.items():
            ax.plot(sig, y, color=C[k], lw=lw, ls=lsty, label=lab, zorder=3 if k == "table" else 2)
            cols[f"{k}_{m}"] = y
            summary[f"{k}_{m}"] = stats(y, ls, sig)
        ax.set_xscale("log")
        ax.set_xlim(SIG_LO, sig[-1])
        ax.set_ylim(0, None)
        ax.set_xlabel("noise level σ")
        ax.set_title(LABEL[m], loc="left")
        ax.grid(True, color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#8a8984")
        ax.tick_params(colors="#52514e", labelsize=8)
    axs[0].set_ylabel(f"density per unit log σ (unit area on σ ≥ {SIG_LO:g})")
    axs[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Entropy rate vs its square root (the schedule's sqrt-rate target), from the measured "
                 f"training-objective MSE, {TASKCFG['name']}, N=4096", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rate_vs_sqrt.png", dpi=170)
    plt.close(fig)

    keys = list(cols)
    with open(OUT / "rate_vs_sqrt.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(sig)):
            w.writerow([f"{cols[k][i]:.9g}" for k in keys])
    (OUT / "rate_vs_sqrt.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
