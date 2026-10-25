"""Inspect continuous-DGC vs DP-DGC sigma grids at several NFEs (CPU only).

    python scripts/dgc/dp_dgc_schedules.py OUT_PREFIX [--table assets/entropy_tables/owt_medium_dgc]
        [--sigma_min 0.08] [--K 8] [--weight increment|length_squared] [--nfes 32,64,128,256,512]

Prints, per NFE, the DP block allocation next to the continuous grid's step count in the same blocks (blocks
uniform in log sigma^2, high sigma first), and writes OUT_PREFIX.csv (every sigma of every grid) and
OUT_PREFIX.png (log sigma against step index, one panel per NFE). Also prints the notebook's Theorem-1 finite-step
bound of each grid (lower = better by the DP's own objective; not a model metric). NFE -> sampler steps uses the same
accounting as the evaluation (ddim, self-conditioning carry, final readout).
"""
import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusion.continuous.dp_dgc import WEIGHTS, DGCIncrement, block_quantities  # noqa: E402
from diffusion.continuous.samplers import SigmaSchedule  # noqa: E402
from evaluation.nfe import steps_for_target_nfe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--table", default="assets/entropy_tables/owt_medium_dgc")
ap.add_argument("--sigma_min", type=float, default=0.08)   # OWT terminal sigma (10^-1.10)
ap.add_argument("--sigma_max", type=float, default=80.0)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--weight", default="increment", choices=WEIGHTS)
ap.add_argument("--nfes", default="32,64,128,256,512")
a = ap.parse_args()

cfg = NS(diffusion=NS(continuous=NS(sigma_min=0.002, sigma_max=a.sigma_max)),
         evaluation=NS(entropic_blend_alpha=0.0, dp_dgc_blocks=a.K, dp_dgc_weight=a.weight))
sch = SigmaSchedule(None, cfg, torch.device("cpu"))
ld = lambda f: torch.load(f"{a.table}/{f}", map_location="cpu", weights_only=True).double().numpy()
sig, cdf, pdf = ld("entropy_sigmas.pt"), ld("entropy_cdf.pt"), ld("entropy_pdf.pt")
cdf[-1] = 1.0

rows, grids = [], {}
for nfe in [int(x) for x in a.nfes.split(",")]:
    steps, _ = steps_for_target_nfe(framework="continuous_score", sampler_name="ddim_dp_dgc", target_nfe=nfe,
                                    self_condition=True, sc_refresh_mode="carry")
    kw = dict(num_steps=steps, entropy_run_dir=a.table, sigma_min_override=a.sigma_min)
    cont = sch.prepare(schedule="entropic", **kw).double().numpy()
    dp = sch.prepare(schedule="dp_dgc", **kw).double().numpy()
    lo, hi = float(cont[-1]), float(cont[0])
    _, _, edges, m = block_quantities(sig, cdf, lo, hi, a.K, pdf=pdf, weight=a.weight)
    inc = DGCIncrement(sig, pdf, lo, hi)
    # intervals of the continuous grid per block, by the log-variance midpoint of each interval
    mid = np.log(cont[:-1]) + np.log(cont[1:])
    cont_alloc = np.histogram(mid, bins=edges)[0][::-1].tolist()
    dp_alloc = next(v[1] for k, v in sch._dp_dgc_cache.items() if k[:3] == (a.table, steps, a.K) and k[-1] == a.weight)
    print(f"NFE={nfe:4d} ({steps - 1} intervals)  DP   {dp_alloc}")
    print(f"{'':27s}cont {cont_alloc}   (continuous share x intervals: "
          f"{[round(x, 1) for x in (m[::-1] * (steps - 1))]})")
    print(f"{'':27s}Theorem-1 bound: continuous {inc.finite_step_bound(cont):.4g}   DP {inc.finite_step_bound(dp):.4g}")
    grids[nfe] = (cont, dp)
    for name, g in (("continuous_dgc", cont), ("dp_dgc", dp)):
        rows += [(nfe, name, i, float(s)) for i, s in enumerate(g)]

with open(f"{a.out}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nfe", "schedule", "index", "sigma"])
    w.writerows(rows)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, len(grids), figsize=(3.0 * len(grids), 2.8), squeeze=False)
for ax, (nfe, (cont, dp)) in zip(axes[0], grids.items()):
    x = np.linspace(0, 1, len(cont))
    ax.plot(x, cont, color="#2a78d6", lw=2, label="continuous DGC")
    ax.plot(x, dp, color="#eb6834", lw=2, ls="--", label=f"DP-DGC (K={a.K})")
    ax.set_yscale("log")
    ax.set_title(f"NFE {nfe}", fontsize=9, loc="left")
    ax.set_xlabel("step / total", fontsize=8)
    ax.grid(alpha=0.25)
axes[0][0].set_ylabel("sigma", fontsize=8)
axes[0][0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(f"{a.out}.png", dpi=180)
print("wrote", f"{a.out}.csv", f"{a.out}.png")
