"""Figures for the TinyGSM CoBit-S report: copies the analysis figures and draws the OWT-vs-TinyGSM comparison.
Run in the `sedd` env after tinygsm/analyze_S.py and ../analyze.py (OWT)."""
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
for src, dst in (("results_tinygsm/S/fig_static.png", "fig1_static.png"),
                 ("results_tinygsm/S/fig_rate_vs_sqrt.png", "fig3_rate_vs_sqrt.png"),
                 ("results_tinygsm/S/fig_truechain.png", "fig4_truechain.png")):
    shutil.copy(EXP / src, HERE / dst)


def unit(y, ls, act):
    y = np.where(act, np.clip(y, 0, None), 0.0)
    return y / np.trapz(y, ls)


def table_dens(d, ls):
    rd = lambda n: torch.load(REPO / d / n, map_location="cpu").numpy().astype(float)  # noqa: E731
    pdf, mids, edges = rd("entropy_pdf.pt"), rd("entropy_sigmas.pt"), rd("entropy_edges.pt")
    return np.interp(ls, np.log(mids), pdf / np.diff(np.log(edges)), left=0, right=0)


a = np.load(EXP / "results/main/curves.npz")
so, lso = a["sigma"], np.log(a["sigma"])
r = list(csv.DictReader(open(EXP / "results_tinygsm/S/curves.csv")))
c = {k: np.array([float(x[k]) for x in r]) for k in r[0]}
st, lst = c["sigma"], np.log(c["sigma"])
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(so, unit(a["mse_train_S"] / so ** 2, lso, so >= 0.1), color="#2a78d6", lw=2, label="OWT: entropy rate MSE/σ²")
ax.plot(so, unit(table_dens("assets/entropy_tables/owt", lso), lso, so >= 0.08), color="#2a78d6", lw=1.2, ls="--",
        label="OWT: schedule")
ax.plot(st, unit(c["rate"], lst, st >= 0.1), color="#eb6834", lw=2, label="TinyGSM: entropy rate MSE/σ² (training loss region)")
ax.plot(st, unit(table_dens("runs/tasks/tinygsm/cobit_raw_binary_bits", lst), lst, st >= 0.002), color="#eb6834", lw=1.2,
        ls="--", label="TinyGSM: schedule")
ax.set_xscale("log"); ax.set_xlim(0.1, 80); ax.set_ylim(0, None)
ax.set_xlabel("noise level σ"); ax.set_ylabel("density per unit log σ (unit area)")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="#e4e3df", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(HERE / "fig2_owt_vs_tinygsm.png", dpi=180); plt.close(fig)
print("ok")
