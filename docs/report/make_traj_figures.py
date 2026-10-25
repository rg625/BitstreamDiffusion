#!/usr/bin/env python
"""Trajectory figures for the SG divergence / CFG fix-break analysis.

Regenerates from runs/guidance/rb_traj/*.npz. x-axis is the ORIGINAL diffusion
step (logged rows are stride-4), never the logged row index.
"""
import glob, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/report/fig"); OUT.mkdir(parents=True, exist_ok=True)
REF, CFG, AG, SG = "#6b7688", "#1d4ed8", "#c2410c", "#047857"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})

def get(t):
    z = np.load(glob.glob(f'runs/guidance/rb_traj/*{t}*.npz')[0])
    d = json.loads(Path(glob.glob(f'runs/guidance/rb_traj/*{t}*.json')[0]).read_text())
    pp = d["per_problem"]
    return z, {int(i): int(c) for i, c in zip(pp["idx"], pp["correct"])}

zb, cb = get("traj_baseline"); steps = zb["step"]
arms = [("baseline", zb, REF, "--"), ("SG-prev w=0.125", get("traj_sgprev0.125")[0], SG, "-"),
        ("SG-prev w=2", get("traj_sgprev2")[0], "#b91c1c", "-")]

# ---- fig 7: SG divergence, four panels -------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
panels = [("ratio", "guidance / drift"), ("bit_entropy", "bit entropy"),
          ("sat_lt_001", "saturation  p<0.01"), ("s_norm", "drift norm  ‖s‖")]
for ax, (feat, lab) in zip(axes, panels):
    for name, z, col, ls in arms:
        if feat not in z.files: continue
        med = np.median(z[feat], axis=0)
        q1, q3 = np.percentile(z[feat], 25, axis=0), np.percentile(z[feat], 75, axis=0)
        ax.plot(steps, med, ls, color=col, lw=1.7, label=name)
        ax.fill_between(steps, q1, q3, color=col, alpha=.15, lw=0)
    ax.set_xlabel("diffusion step"); ax.set_title(lab, fontsize=9)
    if feat in ("ratio", "s_norm"): ax.set_yscale("log")
axes[0].axhline(1.0, color="k", lw=.8, ls=":")
axes[0].legend(fontsize=7.2, loc="upper left")
fig.savefig(OUT / "fig7_sg_divergence.pdf"); plt.close(fig)

# ---- fig 8: event timing for SG w=2 ----------------------------------------
z2 = get("traj_sgprev2")[0]
def sustained(M, thr, run=5):
    bad = M > thr; out = np.full(bad.shape[0], -1)
    for p in range(bad.shape[0]):
        c = 0
        for t in range(bad.shape[1]):
            c = c + 1 if bad[p, t] else 0
            if c >= run: out[p] = t - run + 1; break
    return out
def zdev(z, feat):
    b = zb[feat]; m = np.median(b, 0); mad = np.median(np.abs(b - m), 0) * 1.4826
    sd = np.where(mad > 1e-12, mad, np.std(b, 0) + 1e-12)
    return np.abs((z[feat] - m) / sd)
events = [("‖s‖ anomaly", sustained(zdev(z2, "s_norm"), 2.00)),
          ("‖x‖ anomaly", sustained(zdev(z2, "x_norm"), 2.75)),
          ("ratio > 1", sustained(z2["ratio"], 1.0)),
          ("saturation fails", sustained(zdev(z2, "sat_lt_001"), 3.00)),
          ("entropy fails", sustained(zdev(z2, "bit_entropy"), 5.50))]
fig, ax = plt.subplots(figsize=(6.4, 2.7))
for i, (name, idx) in enumerate(events):
    hit = steps[idx[idx >= 0]]
    ax.scatter(hit, np.full(len(hit), i) + np.random.uniform(-.14, .14, len(hit)),
               s=7, alpha=.45, color="#b91c1c")
    ax.scatter([np.median(hit)], [i], s=70, marker="|", color="k", zorder=5)
ax.set_yticks(range(len(events))); ax.set_yticklabels([e[0] for e in events], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlabel("diffusion step at which the event first fires (sustained)")
ax.set_title("SG-prev w=2: event ordering across 64 trajectories", fontsize=9)
fig.savefig(OUT / "fig8_sg_event_timing.pdf"); plt.close(fig)

# ---- fig 9: the runaway, g and s vs baseline drift --------------------------
bs = np.median(zb["s_norm"], axis=0)
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for name, z, col in (("SG-prev w=0.125", get("traj_sgprev0.125")[0], SG),
                     ("SG-prev w=2", z2, "#b91c1c")):
    ax.plot(steps, np.median(z["g_norm"], 0) / np.maximum(bs, 1e-12), "-", color=col,
            lw=1.7, label=f"{name}  ‖g‖")
    ax.plot(steps, np.median(z["s_norm"], 0) / np.maximum(bs, 1e-12), "--", color=col,
            lw=1.4, label=f"{name}  ‖s‖")
ax.axhline(1.0, color="k", lw=.8, ls=":")
ax.set_yscale("log"); ax.set_xlabel("diffusion step")
ax.set_ylabel("norm ÷ baseline drift at same step"); ax.legend(fontsize=7.2)
ax.set_title("Both guidance and the model's own drift inflate together", fontsize=9)
fig.savefig(OUT / "fig9_runaway.pdf"); plt.close(fig)

# ---- fig 10: CFG fix/break, with honest group sizes -------------------------
zc, cc = get("traj_cfg2"); ids = zc["problem_idx"]
grp = {"FIX": [], "BREAK": [], "CC": [], "WW": []}
for r, i in enumerate(ids):
    b, c = cb[int(i)], cc[int(i)]
    grp[("CC" if c else "BREAK") if b else ("FIX" if c else "WW")].append(r)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
for ax, feat, lab in zip(axes, ["ratio", "bit_entropy", "sat_lt_001"],
                         ["guidance / drift", "bit entropy", "saturation p<0.01"]):
    for k, col in (("CC", "#15803d"), ("WW", REF), ("FIX", CFG), ("BREAK", "#b91c1c")):
        if not grp[k]: continue
        ax.plot(steps, np.median(zc[feat][grp[k]], 0), "-", color=col, lw=1.6,
                label=f"{k} (n={len(grp[k])})")
    ax.set_xlabel("diffusion step"); ax.set_title(lab, fontsize=9)
    if feat == "ratio": ax.set_yscale("log")
axes[0].legend(fontsize=7)
fig.suptitle("CFG w=2 fix/break — FIX n=5 and BREAK n=2 are far too small to separate",
             fontsize=9, y=1.04)
fig.savefig(OUT / "fig10_cfg_fixbreak.pdf"); plt.close(fig)
print("wrote fig7-fig10 ->", OUT)
