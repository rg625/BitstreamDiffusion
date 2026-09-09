import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 160, "savefig.bbox": "tight"})
C0, C1, C2 = "#1f77b4", "#d62728", "#2ca02c"

def f_guidance():
    d = sorted(json.load(open("results/guidance/confirm_summary.json")),
               key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    y = np.arange(len(d))
    lo = [r["delta"]-r["ci95_low"] for r in d]; hi = [r["ci95_high"]-r["delta"] for r in d]
    ax.errorbar([r["delta"] for r in d], y, xerr=[lo, hi], fmt="none",
                capsize=3, lw=1.3, ecolor="grey")
    for i, r in enumerate(d):
        ax.plot(r["delta"], i, "o", color=(C2 if r["delta"] > 0 else C1), ms=6)
    ax.axvline(0, c="k", lw=.9)
    ax.set_yticks(y); ax.set_yticklabels([r["config"] for r in d], fontsize=7.5)
    ax.set_xlabel(r"$\Delta$ accuracy vs baseline 0.1385")
    ax.set_title("Guidance: paired confirmation (n=1069, 3 seeds)")
    fig.savefig(f"{OUT}/guidance.pdf"); plt.close(fig)

def f_ordering_inf():
    d = json.load(open("results/ordering/confirm_summary.json"))
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for i, (arm, c) in enumerate((("l2r w=0.1", C0), ("random w=0.1", C1))):
        rows = [r for r in d if r["arm"] == arm]
        x = np.arange(len(rows)) + i*0.18 - 0.09
        y = [r["delta"] for r in rows]
        lo = [r["delta"]-r["ci95"][0] for r in rows]; hi = [r["ci95"][1]-r["delta"] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=c, ms=5, capsize=3, lw=1.3, label=arm)
    ax.axhline(0, c="k", lw=.9)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"seed {i}" for i in range(3)])
    ax.set_ylabel(r"$\Delta$ accuracy vs control")
    ax.set_title("Ordering at inference (n=1319, 3 seeds)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(f"{OUT}/ordering_inference.pdf"); plt.close(fig)

def f_lr():
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    runs = [("py3.9   lr 3e-4", 6356, C1), ("py3.10  lr 3e-4", 5346, C1),
            ("036a2b5 lr 3e-4", 11278, C1), ("py3.10  lr 3e-5", 20000, C2)]
    for i, (nm, v, c) in enumerate(runs):
        ax.barh(i, v, color=c, height=.6)
        ax.text(v+500, i, "diverged" if c == C1 else "stable, target reached",
                va="center", fontsize=8)
    ax.set_yticks(range(len(runs))); ax.set_yticklabels([r[0] for r in runs], fontsize=8)
    ax.set_xlabel("training step reached"); ax.set_xlim(0, 30000)
    ax.set_title("Divergence is a learning-rate effect, not the interpreter")
    fig.savefig(f"{OUT}/lr_stability.pdf"); plt.close(fig)

def f_steps():
    pts = [(5_000, 0.0), (10_000, 0.0), (200_000, .2024), (500_000, .164)]
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=C0, lw=1.6, ms=6)
    ax.axvspan(10_000, 200_000, color="orange", alpha=.15)
    ax.text(4.2e4, .105, "lift-off\nunmeasured", fontsize=8, ha="center")
    ax.set_xscale("log"); ax.set_xlabel("training steps"); ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Accuracy vs training budget")
    fig.savefig(f"{OUT}/accuracy_vs_steps.pdf"); plt.close(fig)

for fn in (f_guidance, f_ordering_inf, f_lr, f_steps):
    try: fn(); print("ok", fn.__name__)
    except Exception as e: print("FAIL", fn.__name__, type(e).__name__, e)
