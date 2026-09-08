"""Figures for the Fri 4 -> Tue 8 Sep report. Values are read from result files
where they exist; the few hardcoded numbers are measurements quoted in the text."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160, "savefig.bbox": "tight"})
C0, C1, C2 = "#1f77b4", "#d62728", "#2ca02c"


def f1():
    scr = {}
    for f in glob.glob("results/ordering/screen/*.json"):
        d = json.load(open(f))
        if d.get("steps") == 256 and d.get("order_mode") in ("none", "l2r"):
            scr[float(d.get("order_w") or 0.0)] = d["accuracy"]
    w = sorted(scr)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(w, [scr[x] for x in w], "o-", color=C0, lw=1.6, ms=5)
    ax.axhline(scr.get(0.0, 0.176), ls="--", c="grey", lw=1, label="control (w=0)")
    ax.set_xlabel("ordering strength $w$"); ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Ordering dose-response (l2r, n=250)")
    ax.legend(frameon=False, fontsize=8)
    ax.annotate("collapse", xy=(0.25, 0.005), xytext=(0.45, 0.06), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    fig.savefig(f"{OUT}/ordering_dose.pdf"); plt.close(fig)


def f2():
    d = json.load(open("results/ordering/confirm_summary.json"))
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for i, (arm, c) in enumerate((("l2r w=0.1", C0), ("random w=0.1", C1))):
        rows = [r for r in d if r["arm"] == arm]
        x = np.arange(len(rows)) + i * 0.18 - 0.09
        y = [r["delta"] for r in rows]
        lo = [r["delta"] - r["ci95"][0] for r in rows]
        hi = [r["ci95"][1] - r["delta"] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=c, ms=5, capsize=3,
                    lw=1.3, label=arm)
    ax.axhline(0, c="k", lw=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"seed {i}" for i in range(3)])
    ax.set_ylabel(r"$\Delta$ accuracy vs control")
    ax.set_title("Ordering confirmation (n=1319, paired)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(f"{OUT}/ordering_confirm.pdf"); plt.close(fig)


def f3():
    pts = [(5_000, 0.0), (10_000, 0.0), (200_000, 0.2024), (500_000, 0.164)]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=C0, lw=1.6, ms=6)
    ax.axvspan(10_000, 200_000, color="orange", alpha=0.15)
    ax.text(4.2e4, 0.105, "lift-off\nunmeasured", fontsize=8, ha="center")
    ax.axvline(12_000, ls=":", c=C1, lw=1.4)
    ax.text(13_000, 0.185, "runs diverge\nbefore ~12k", fontsize=7.5, color=C1)
    ax.set_xscale("log"); ax.set_xlabel("training steps"); ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Accuracy exists only above the trainable range")
    fig.savefig(f"{OUT}/accuracy_vs_steps.pdf"); plt.close(fig)


def f4():
    d = json.load(open("results/objective/stability_summary.json"))["runs"]
    sm = sorted(r["break_step"] for r in d if r["arm"] == "binary_sm")
    ce = sorted(r["break_step"] for r in d if r["arm"] == "binary_ce")
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    ax.plot(sm, [1] * len(sm), "o", color=C1, ms=8, label="binary\\_sm")
    ax.plot(ce, [2] * len(ce), "o", color=C2, ms=8, label="binary\\_ce")
    ax.plot([11278], [1.5], "D", color="k", ms=7, label="production-era commit")
    ax.set_yticks([1, 1.5, 2]); ax.set_yticklabels(["SM", "036a2b5", "CE"])
    ax.set_ylim(0.6, 2.4); ax.set_xlabel("step at which the run diverged")
    ax.set_title("Time-to-divergence (3 seeds/arm)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(f"{OUT}/break_steps.pdf"); plt.close(fig)


def f5():
    a = json.load(open("results/objective/offline_probe_wide_250k.json"))["arms"]["production"][0]
    b = json.load(open("results/objective/offline_probe_wide_500k.json"))["arms"]["production"][0]
    s = [r["sigma"] for r in a["grid"]]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for e, lab, c in ((a, "250k", C0), (b, "500k", C1)):
        y = [r["grad_survival"] for r in e["grid"]]
        lo = [r["grad_survival"] - r["grad_survival_ci95"][0] for r in e["grid"]]
        hi = [r["grad_survival_ci95"][1] - r["grad_survival"] for r in e["grid"]]
        ax.errorbar(s, y, yerr=[lo, hi], fmt="o-", color=c, ms=4, capsize=2, lw=1.3, label=lab)
    ax.set_xscale("log"); ax.set_xlabel(r"$\sigma$"); ax.set_ylabel("gradient survival")
    ax.set_title(r"Low-$\sigma$ collapse (descriptive)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(f"{OUT}/low_sigma.pdf"); plt.close(fig)


for fn in (f1, f2, f3, f4, f5):
    try:
        fn(); print("ok", fn.__name__)
    except Exception as e:
        print("FAILED", fn.__name__, type(e).__name__, e)
