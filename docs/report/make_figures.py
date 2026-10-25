#!/usr/bin/env python
"""Regenerate every figure and number in the supervisor report from recorded data.

Nothing here is hand-written: point estimates come from the per-problem outcome
vectors in runs/guidance/, and every interval is a paired bootstrap over
problems (numpy-vectorised, so the whole report rebuilds in seconds).
"""
from __future__ import annotations
import json, glob, re, collections, itertools, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("runs/guidance"); OUT = Path("docs/report/fig"); OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0); NBOOT = 20000

CFG, AG, SG, REF = "#1d4ed8", "#c2410c", "#047857", "#6b7688"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})

def load(p):
    d = json.loads(Path(p).read_text()); pp = d["per_problem"]
    c = {int(i): int(x) for i, x in zip(pp["idx"], pp["correct"])}
    a = ({int(i): v for i, v in zip(pp["idx"], pp["answer"])} if pp.get("answer") else None)
    return d, c, a

def collect(pattern, seed_rx):
    C, A, D = {}, {}, {}
    for f in glob.glob(pattern):
        m = re.search(seed_rx, f)
        if not m: continue
        s = int(m.group(1)); d, c, a = load(f); C[s], A[s], D[s] = c, a, d
    return C, A, D

def seedmean(C):
    acc = collections.defaultdict(list)
    for o in C.values():
        for i, v in o.items(): acc[i].append(v)
    return {i: float(np.mean(v)) for i, v in acc.items()}

def paired(a, b, nboot=NBOOT):
    """Paired bootstrap over problems of mean(b) - mean(a)."""
    ks = sorted(set(a) & set(b))
    d = np.array([b[k] - a[k] for k in ks], dtype=float)
    idx = RNG.integers(0, len(d), size=(nboot, len(d)))
    bs = np.sort(d[idx].mean(axis=1))
    return float(d.mean()), float(bs[int(.025 * nboot)]), float(bs[int(.975 * nboot)])

def acc(o): return float(np.mean(list(o.values())))

NUM = {}

# ---------------------------------------------------------------- fig 1: gamma
arms = {}
for f in glob.glob(str(ROOT / "stoch_confirm/*.json")):
    m = re.search(r"stc_g([0-9p]+)_(\w+?)_s(\d\d)", f)
    g = float(m.group(1).replace("p", ".")); meth = m.group(2); s = int(m.group(3))
    _, c, _ = load(f); arms.setdefault((meth, g), {})[s] = c
GAM = [0.0, 0.2, 0.3]
series = [("baseline", "base", REF, "--"), ("CFG w=12", "cfg12", CFG, "-"),
          ("AutoGuidance w=15", "ag15", AG, "-"), ("SG-prev w=2", "sgprev2", SG, "-")]
fig, ax = plt.subplots(figsize=(5.4, 3.3))
for lab, key, col, ls in series:
    ys = [acc(seedmean(arms[(key, g)])) for g in GAM]
    ax.plot(GAM, ys, ls, color=col, marker="o", ms=5, lw=1.8, label=lab, zorder=3)
    NUM[f"gamma_{key}"] = ys
ax.set_xlabel(r"stochastic churn $\gamma$   (no extra forward passes)")
ax.set_ylabel("exact match"); ax.set_xticks(GAM); ax.set_ylim(0, .30)
ax.legend(loc="lower left", fontsize=7.4, bbox_to_anchor=(0.01, 0.01))
ax.annotate(f"{NUM['gamma_base'][-1]:.4f}", (0.3, NUM['gamma_base'][-1]),
            textcoords="offset points", xytext=(6, 4), color=REF, fontsize=7.5)
ax.annotate(f"{NUM['gamma_sgprev2'][-1]:.4f}", (0.3, NUM['gamma_sgprev2'][-1]),
            textcoords="offset points", xytext=(6, -3), color=SG, fontsize=7.5)
fig.savefig(OUT / "fig1_gamma.pdf"); plt.close(fig)
for g in GAM[1:]:
    NUM[f"churn_gain_g{g}"] = paired(seedmean(arms[("base", 0.0)]), seedmean(arms[("base", g)]))
for key in ("cfg12", "ag15", "sgprev2"):
    for g in GAM:
        NUM[f"vs_base_{key}_g{g}"] = paired(seedmean(arms[("base", g)]), seedmean(arms[(key, g)]))

# ------------------------------------------------- fig 2: CFG scale, both gammas
cfg0 = {}
for f in glob.glob(str(ROOT / "cfg_confirm/*.json")):
    m = re.search(r"cfgconf_w(\d+)_s(\d\d)", f)
    if not m: continue
    cfg0.setdefault(float(m.group(1)), {})[int(m.group(2))] = load(f)[1]
# cfg_confirm stores its baseline as w=0 rather than a separate "base" cell.
b0 = cfg0.pop(0.0)
cfg3 = {}
for f in glob.glob(str(ROOT / "churn_anatomy/*chn_cfg*.json")):
    m = re.search(r"chn_cfg([0-9.]+)_s(\d\d)\.json", f)
    cfg3.setdefault(float(m.group(1)), {})[int(m.group(2))] = load(f)[1]
b3, _, _ = collect(str(ROOT / "stoch_confirm/*stc_g0p3_base_s*.json"), r"base_s(\d\d)")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, (data, base, lab, ttl) in zip(axes, [
        (cfg0, b0, r"$\gamma=0$  (512 steps)", "deterministic"),
        (cfg3, b3, r"$\gamma=0.3$  (256 steps)", "stochastic")]):
    ws = sorted(data); ys = [acc(seedmean(data[w])) for w in ws]
    ba = acc(seedmean(base))
    ax.axhline(ba, color=REF, ls="--", lw=1.4)
    ax.plot(ws, ys, "-o", color=CFG, ms=5, lw=1.8)
    ax.annotate("no guidance", (ws[0], ba), textcoords="offset points",
                xytext=(2, 4), color=REF, fontsize=7.2)
    best = ws[int(np.argmax(ys))]
    ax.plot([best], [max(ys)], "o", ms=10, mfc="none", mec=CFG, mew=1.6)
    # Annotate BELOW-RIGHT of the peak: above collides with the panel title.
    ax.annotate(f"$w^*={best:g}$", (best, max(ys)), textcoords="offset points",
                xytext=(11, -4), color=CFG, fontsize=8.5, fontweight="bold")
    lo, hi = min(ys + [ba]), max(ys)
    ax.set_ylim(lo - .05 * (hi - lo), hi + .18 * (hi - lo))   # headroom for the title
    ax.set_title(lab, fontsize=8.5, pad=7); ax.set_xlabel("CFG scale $w$")
    NUM[f"cfgscale_{ttl}"] = list(zip(ws, ys)); NUM[f"cfgbase_{ttl}"] = ba
axes[0].set_ylabel("exact match")
fig.savefig(OUT / "fig2_cfgscale.pdf"); plt.close(fig)
for w in sorted(cfg3):
    NUM[f"cfg_g03_w{w:g}"] = paired(seedmean(b3), seedmean(cfg3[w]))

# ------------------------------------------------------------ fig 3: solver
sol = {}
for f in glob.glob(str(ROOT / "solver_control/*.json")):
    m = re.search(r"solv_(\w+?)_seed(\d\d)", f)
    sol.setdefault(m.group(1), {})[int(m.group(2))] = load(f)[1]
LBL = [("ddim512", "DDIM  512 steps", 512, REF), ("heun256", "Heun  256 steps", 511, "#8b96a8"),
       ("sgprev512", "SG-prev  512 steps", 512, SG), ("ddim1024", "DDIM  1024 steps", 1024, REF),
       ("heun512", "Heun  512 steps", 1023, "#8b96a8")]
fig, ax = plt.subplots(figsize=(5.6, 2.6))
ys = [acc(seedmean(sol[k])) for k, *_ in LBL]
ax.barh(range(len(LBL)), ys, color=[c for *_, c in LBL], height=.62, zorder=3)
ax.set_yticks(range(len(LBL)))
ax.set_yticklabels([f"{n}\n({fw} fwd)" for _, n, fw, _ in LBL], fontsize=7.6)
ax.invert_yaxis(); ax.set_xlabel("exact match"); ax.set_xlim(0, .22)
for i, v in enumerate(ys):
    ax.text(v + .004, i, f"{v:.4f}", va="center", fontsize=7.8)
fig.savefig(OUT / "fig3_solver.pdf"); plt.close(fig)
NUM["solver"] = {k: acc(seedmean(sol[k])) for k, *_ in LBL}
NUM["sg_vs_heun"] = paired(seedmean(sol["heun256"]), seedmean(sol["sgprev512"]))
NUM["heun_vs_ddim"] = paired(seedmean(sol["ddim512"]), seedmean(sol["heun256"]))
NUM["ddim1024_vs_512"] = paired(seedmean(sol["ddim512"]), seedmean(sol["ddim1024"]))

# ------------------------------------------------- fig 4: voting composition
def majk(C, A, ss):
    ks = set.intersection(*[set(A[s]) for s in ss]); out = {}
    for i in ks:
        v = [A[s][i] for s in ss if A[s][i] is not None]
        if not v: out[i] = 0; continue
        w = collections.Counter(v).most_common(1)[0][0]
        pick = next((s for s in ss if A[s][i] == w), ss[0]); out[i] = C[pick][i]
    return out
def aggk(C, A, k):
    a = collections.defaultdict(list)
    for ss in itertools.combinations(sorted(C), k):
        for i, v in majk(C, A, list(ss)).items(): a[i].append(v)
    return {i: float(np.mean(v)) for i, v in a.items()}
bC, bA, _ = collect(str(ROOT / "stoch_confirm/*stc_g0p3_base_s*.json"), r"base_s(\d\d)")
cC, cA, _ = collect(str(ROOT / "churn_anatomy/*chn_cfg2_s*.json"), r"_s(\d\d)\.json")
fig, ax = plt.subplots(figsize=(5.0, 3.0))
for C, A, col, lab, per in ((bC, bA, REF, "baseline", 256), (cC, cA, CFG, "CFG $w=2$", 512)):
    ks = [1, 2, 3]
    ys = [acc(seedmean(C))] + [acc(aggk(C, A, k)) for k in (2, 3)]
    ax.plot([k * per for k in ks], ys, "-o", color=col, ms=5, lw=1.8, label=lab)
    for k, y in zip(ks, ys):
        ax.annotate(f"maj@{k}", (k * per, y), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=6.8, color=col)
    NUM[f"vote_{lab}"] = list(zip([k * per for k in ks], ys))
ax.set_xscale("log", base=2); ax.set_xlabel("total forward passes")
ax.set_ylabel("exact match"); ax.legend(fontsize=8)
fig.savefig(OUT / "fig4_voting.pdf"); plt.close(fig)
NUM["vote_add_base"] = paired(seedmean(bC), aggk(bC, bA, 2))
NUM["vote_add_cfg"] = paired(seedmean(cC), aggk(cC, cA, 2))
NUM["cfgmaj2_vs_basemaj2"] = paired(aggk(bC, bA, 2), aggk(cC, cA, 2))
NUM["cfgmaj2_vs_basemaj3"] = paired(aggk(bC, bA, 3), aggk(cC, cA, 2))

# ---------------------------------------------------------- fig 5: factorial
rows = [r for r in csv.DictReader(open("results/guidance/all_cells.csv"))
        if r["grid"] == "factorial_confirm"]
cell = collections.defaultdict(list)
for r in rows:
    f = lambda k: float(r[k] or 0)
    parts = []
    if f("guidance_scale") > 0: parts.append("CFG")
    if f("ag_scale") > 0: parts.append("AG")
    if f("sg_scale") != 0: parts.append("SG-" + r["sg_variant"])
    cell["+".join(parts) or "baseline"].append(f("accuracy"))
items = sorted(((k, float(np.mean(v))) for k, v in cell.items()), key=lambda t: -t[1])
fig, ax = plt.subplots(figsize=(5.6, 3.4))
cols = [("#b91c1c" if v < .01 else CFG if "CFG" in k else AG if "AG" in k
         else SG if "SG" in k else REF) for k, v in items]
ax.barh(range(len(items)), [v for _, v in items], color=cols, height=.66, zorder=3)
ax.axvline(cell["baseline"] and float(np.mean(cell["baseline"])), color=REF, ls="--", lw=1.2)
ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _ in items], fontsize=7.4)
ax.invert_yaxis(); ax.set_xlabel(r"exact match  ($\gamma=0$, 1319 problems, 3 seeds)")
for i, (_, v) in enumerate(items):
    ax.text(v + .003, i, f"{v:.4f}", va="center", fontsize=7)
fig.savefig(OUT / "fig5_factorial.pdf"); plt.close(fig)
NUM["factorial"] = items

# ---------------------------------------------------------------- fig 6: NFE
nfe = collections.defaultdict(dict)
for r in csv.DictReader(open("results/guidance/all_cells.csv")):
    if r["grid"] != "nfe": continue
    nfe[r["method"]][int(r["steps"])] = (float(r["accuracy"]),
                                         float(r["efficiency.nfe_per_sample"]))
fig, ax = plt.subplots(figsize=(5.4, 3.1))
for meth, col, ls in (("baseline", REF, "--"), ("SG-prev", SG, "-"),
                      ("CFG", CFG, "-"), ("AG", AG, "-")):
    if meth not in nfe: continue
    pts = sorted(nfe[meth].items())
    ax.plot([p[1][1] for p in pts], [p[1][0] for p in pts], ls, color=col,
            marker="o", ms=4, lw=1.6, label=meth)
ax.set_xscale("log", base=2); ax.set_xlabel("model evaluations per sample")
ax.set_ylabel(r"exact match  ($\gamma=0$, 250 problems)"); ax.legend(fontsize=7.6)
fig.savefig(OUT / "fig6_nfe.pdf"); plt.close(fig)

json.dump({k: v for k, v in NUM.items()}, open("docs/report/numbers.json", "w"),
          indent=1, default=float)
print("figures ->", OUT)
for k in ("churn_gain_g0.3", "sg_vs_heun", "heun_vs_ddim", "cfg_g03_w2",
          "vote_add_base", "vote_add_cfg", "cfgmaj2_vs_basemaj3"):
    if k in NUM: print(f"  {k:<24}", np.round(NUM[k], 4))
