"""Report 4: Sigma-style ("global") score temperature on TinyGSM -> GSM8K. CPU only.

Reads every result JSON under results/scoretemp/eval_gh200 (top level only; the
invalid_* and gate/ subdirectories are excluded by construction), the cells
submitted by scripts/hpc/arch/scoretemp_gh200.sh. Paired deltas are bootstrapped
over problems: every cell in a group saw the same 1319 problems with seed 0.

    python docs/pdf_reports/build_scoretemp.py   # -> docs/pdf_reports/report_score_temperature.pdf
"""
from __future__ import annotations

import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdfkit_common import (  # noqa: E402
    BAD, H1, H2, HERE, INK, INK2, MUTED, SERIES, P, PageBreak, Spacer, build, bullets, callout,
    figure, mm, mpl_style, plt, table,
)

REPO = HERE.parents[1]
OUT = REPO / "results" / "scoretemp" / "eval_gh200"
LABEL = re.compile(r"__(?P<arm>[a-z0-9]+)_w(?P<w>[\d.]+)_t(?P<tau>[\d.]+)_g(?P<g>[\d.]+)_n(?P<n>\d+)")
TAUS = [1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # dataviz blue 100->700


def load(sub=""):
    cells = {}
    for f in glob.glob(str(OUT / sub / "gsm8k_results_*.json")):
        m = LABEL.search(f)
        if not m:
            continue
        r = json.load(open(f))
        pp = r["per_problem"]
        r["_c"] = dict(zip(pp["idx"], pp["correct"]))
        cells[(m["arm"], float(m["w"]), float(m["tau"]), float(m["g"]), int(m["n"]))] = r
    return cells


def paired(a, b, n_boot=10000, seed=0):
    common = sorted(set(a["_c"]) & set(b["_c"]))
    d = np.array([a["_c"][i] - b["_c"][i] for i in common], float)
    boots = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)


def acc(r):
    return 100 * r["accuracy"]


def fmt_d(d):
    return f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]"


def churn_stability():
    """Per-step contraction factor 1 - r/tau of the Euler step with EDM churn (Karras, 1024)."""
    N, smin, smax, rho, gam = 1024, 0.002, 80.0, 7.0, 0.41
    t = np.linspace(0, 1, N)
    s = (smax ** (1 / rho) + t * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    r = 1 - s[1:] / (s[:-1] * (1 + gam))
    return float(r.min()), float(r.max()), float(np.median(r))


# ------------------------------------------------------------------ figures
SERIES_DEF = [  # (key, legend label, direct label, colour) -- fixed order, colour follows the entity
    (("m250", 0.0, 0.41, 1024), "CoBit-M 462M, unguided, step 250k (Karras)", "CoBit-M", SERIES[0]),
    (("cfg425", 1.0, 0.41, 1024), "CoBit-S 130M, CFG checkpoint at w=1, step 425k (Karras)", "CoBit-S CFG", SERIES[1]),
    (("s500", 0.0, 0.41, 1024), "CoBit-S 130M, unguided, step 500k ({sched})", "CoBit-S", SERIES[2]),
]
S425_SHA = "9a6817c7c4871104acecceedf8aee1621bb3f3b3a2a0576a8cd394e43a9588dd"
S500_SHA = "65cedca93bbd7c1c00c5cf8c4990b39945b9f9d08755ed19922aecbdf8ff24a3"
# Filled in after hashing the Drive copy of step=000425000.pt against the upload (12 Oct).
DRIVE425_NOTE = ("the copy in the shared Drive folder is byte-identical to it (same SHA-256), so this is the "
                 "Drive artifact itself; the June handoff records that the 29.42% came from a main-run 425k "
                 "that was never saved to the shared folder")


def series(cells, key):
    arm, w, g, n = key
    return {tau: cells[(arm, w, tau, g, n)] for tau in TAUS if (arm, w, tau, g, n) in cells}


def sched_of(s):
    for r in s.values():
        return (r.get("provenance") or {}).get("schedule") or r.get("schedule") or "?"
    return "?"


def fig_tau(cells):
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.axvspan(0.04, 0.145, color="#f4f3ef", zorder=0)
    ax.text(0.076, 47.5, "Karras: churn step\nunstable (τ < ~0.15)", ha="center", va="top", fontsize=7, color=INK2)
    offs = {"m250": (0, 7), "cfg425": (0, -13), "s500": (-10, 6)}
    for key, label, short, col in SERIES_DEF:
        s = series(cells, key)
        if not s:
            continue
        label = label.format(sched=sched_of(s))
        x = sorted(s, reverse=True)
        y = [acc(s[t]) for t in x]
        ax.vlines(x, [100 * s[t]["ci95_low"] for t in x], [100 * s[t]["ci95_high"] for t in x],
                  color=col, lw=1.2, alpha=0.6)
        ax.plot(x, y, "-o", color=col, label=label, ms=5.5, mec="white", mew=1.2, zorder=3)
        best = max(x, key=lambda t: acc(s[t]))
        ax.annotate(f"{short} {acc(s[best]):.1f}%", (best, acc(s[best])), xytext=offs[key[0]],
                    textcoords="offset points", ha="right" if key[0] == "s500" else "center", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1.15, 0.042)  # sharper to the right; room for the left-hand labels
    ax.set_xticks(TAUS)
    ax.set_xticklabels([f"{t:g}" for t in TAUS])
    ax.minorticks_off()
    ax.set_ylim(-1, 48)
    ax.set_xlabel("score temperature τ (drift ÷ τ at every σ)  →  sharper")
    ax.set_ylabel("GSM8K accuracy, % (95% CI)")
    ax.set_title("τ alone with churn: 1319 problems, 1024 steps, seed 0")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=7, handlelength=1.6)
    return fig


def fig_grid(cells):
    ws = [1, 2, 3, 5, 9, 12]
    taus = [1, 0.5, 0.3, 0.2, 0.1]
    M = np.full((len(ws), len(taus)), np.nan)
    for i, w in enumerate(ws):
        for j, t in enumerate(taus):
            r = cells.get(("cfg425", float(w), float(t), 0.41, 1024))
            if r is not None:
                M[i, j] = acc(r)
    vmax = np.nanmax(M)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    from matplotlib.colors import ListedColormap
    ax.imshow(M, cmap=ListedColormap(SEQ), vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(ws)):
        for j in range(len(taus)):
            if np.isnan(M[i, j]):
                continue
            light = M[i, j] < 0.55 * vmax
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color=INK if light else "white", fontweight="bold" if M[i, j] == vmax else "normal")
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_yticks(range(len(ws)))
    ax.set_yticklabels([f"{w:g}" for w in ws])
    ax.set_xlabel("score temperature τ  →  sharper")
    ax.set_ylabel("CFG weight w  (1 = no guidance)")
    ax.set_title("CFG checkpoint: accuracy % over (w, τ), churn, 1024 steps")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return fig


# ------------------------------------------------------------------ tables
def tau_table(s):
    base = s.get(1)
    rows = [["τ", "accuracy %", "95% CI", "Δ vs τ=1, paired (95% CI)", "programs that ran %"]]
    for t in sorted(s, reverse=True):
        r = s[t]
        rows.append([f"{t:g}", f"{acc(r):.1f}", f"[{100 * r['ci95_low']:.1f}, {100 * r['ci95_high']:.1f}]",
                     "—" if t == 1 or base is None else fmt_d(paired(r, base)),
                     f"{100 * r.get('executable_rate', float('nan')):.1f}"])
    best = max(s, key=lambda t: acc(s[t]))
    return table(rows, col_widths=[14 * mm, 24 * mm, 28 * mm, 52 * mm, 34 * mm],
                 highlight_rows=[1 + sorted(s, reverse=True).index(best)])


def best(s):
    t = max(s, key=lambda t: acc(s[t]))
    return t, acc(s[t]), paired(s[t], s[1])


# ------------------------------------------------------------------ section 7: blocks A-E (13 Oct)
def _cell_row(r, base=None):
    d = fmt_d(paired(r, base)) if base is not None else "—"
    return [f"{acc(r):.1f}", f"{100 * r.get('executable_rate', float('nan')):.1f}",
            f"{100 * r.get('invalid_token_rate', float('nan')):.2f}", d]


SER5 = list(SERIES) + ["#e87ba4"]   # dataviz reference palette, slot 5 (magenta), fixed order


def fig_blocks(series_list):
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for lab, s, col, ls in series_list:
        if not s:
            continue
        x = sorted(s, reverse=True)
        ax.plot(x, [acc(s[t]) for t in x], ls, color=col, label=lab, ms=5, mec="white", mew=1.0, lw=1.8)
    ax.set_xscale("log")
    ax.set_xlim(1.15, 0.085)
    ax.set_xticks([1, 0.7, 0.5, 0.3, 0.2, 0.1])
    ax.set_xticklabels(["1", "0.7", "0.5", "0.3", "0.2", "0.1"])
    ax.minorticks_off()
    ax.set_ylim(-1, 45)
    ax.set_xlabel("score temperature τ  →  sharper")
    ax.set_ylabel("GSM8K accuracy, %")
    ax.set_title("CoBit-S unguided 500k: the original curve and the four fixes (1319 problems)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7)
    return fig


def blocks_section(cells, title="7  Fixing the collapse below τ = 0.7 (runs of 13 Oct)", sec_cells="Section 4",
                   sec_mode="Section 5"):
    B = load("../blocks_gh200")
    if not B:
        return [], None
    S = lambda arm, g, n=1024, src=B: {k[2]: r for k, r in src.items() if k[0] == arm and k[3] == g and k[4] == n}
    orig_c, orig_d = S("s500", 0.41, src=cells), S("s500", 0.0, src=cells)
    A_c, A_d, Bn = S("s500a", 0.41), S("s500a", 0.0), S("s500b", 0.0)
    C = {g: S("s500c", g) for g in (0.1, 0.2, 0.3)}
    E, EU, D = S("s500e", 0.41), S("s500eu", 0.41), S("s500d", 0.41)
    base = orig_c.get(1.0)
    cands = []
    for name, ser, ref in ([("A churn, unsharpened last step", A_c, base), ("A deterministic, unsharpened", A_d, orig_d.get(1.0)),
                            ("B ancestral", Bn, Bn.get(1.0)), ("E σ_min=0.1", E, E.get(1.0)), ("E σ_min=0.1, unsharpened", EU, E.get(1.0))]
                           + [(f"C γ={g}", C[g], C[g].get(1.0)) for g in C] + [("D logit temperature", D, base)]):
        for t, r in ser.items():
            if t != 1.0:
                cands.append((acc(r), name, t, r))
    best = max(cands, key=lambda c: c[0]) if cands else None
    out = [PageBreak(), H1(title),
           P("All cells: CoBit-S unguided 500k EMA, entropic grid, 1024 steps, 1319 problems, seed 0, paired with the "
             "cells of " + sec_cells + ". A: the last step (onto σ_min, the readout) uses the untempered drift. B: the VE "
             "ancestral sampler (exact Gaussian bridge, no churn). C: lower churn γ, last step unsharpened. D: "
             "DeepMind's logit temperature T (softmax over the valid codewords with the full logit, every step). E: the "
             "trajectory ends at σ_min = 0.1 instead of 0.002 (see entropy_profile.pdf).")]
    gbest = max(C, key=lambda g: max((acc(r) for r in C[g].values()), default=0)) if any(C.values()) else None
    out.append(figure(fig_blocks([
        ("original: churn γ=0.41, last step sharpened", orig_c, SER5[0], "-o"),
        ("A: churn γ=0.41, last step unsharpened", {**({1.0: base} if base else {}), **A_c}, SER5[1], "-o"),
        ("B: ancestral, last step unsharpened", Bn, SER5[2], "-o"),
        (f"C: churn γ={gbest}, last step unsharpened" if gbest else "C", C.get(gbest, {}), SER5[3], "-o"),
        ("E: end at σ_min = 0.1, churn γ=0.41", E, SER5[4], "-o"),
    ]), "scoretemp_blocks", 165, "Accuracy per cell (each 1319 problems). A's τ=1 point is the original τ=1 cell (the "
       "two settings coincide at τ=1)."))
    rows = [["sampler", "τ", "last step sharpened: acc / run / invalid", "unsharpened: acc / run / invalid",
             "unsharpened Δ vs τ=1 (paired)"]]
    for lab, orig, new in (("churn γ=0.41", orig_c, A_c), ("deterministic", orig_d, A_d)):
        for t in (0.5, 0.3, 0.2):
            if t in new:
                o = orig.get(t)
                rows.append([lab, f"{t:g}", " / ".join(_cell_row(o)[:3]) if o else "—", " / ".join(_cell_row(new[t])[:3]),
                             fmt_d(paired(new[t], orig[1.0])) if 1.0 in orig else "—"])
    out += [H2("A — unsharpened last step"), table(rows, col_widths=[26 * mm, 10 * mm, 45 * mm, 45 * mm, 44 * mm]), Spacer(1, 3)]
    if Bn:
        out += [H2("B — ancestral sampler (no churn), last step unsharpened"), tau_table(Bn), Spacer(1, 3)]
    if any(C.values()):
        taus = [1.0, 0.5, 0.3, 0.2, 0.1]
        crow = [["γ \\ τ"] + [f"{t:g}" for t in taus] + ["best Δ vs own τ=1"]]
        for g in (0.1, 0.2, 0.3):
            c = C[g]
            bt = max((t for t in c if t != 1.0), key=lambda t: acc(c[t]), default=None)
            crow.append([f"{g:g}"] + [f"{acc(c[t]):.1f}" if t in c else "—" for t in taus]
                        + [f"τ={bt:g}: {fmt_d(paired(c[bt], c[1.0]))}" if bt is not None and 1.0 in c else "—"])
        if orig_c:
            crow.append(["0.41 (orig.)"] + [f"{acc(orig_c[t]):.1f}" if t in orig_c else "—" for t in taus] + ["sharpened last step"])
        out += [H2("C — churn γ × τ, last step unsharpened (accuracy %)"),
                table(crow, col_widths=[22 * mm] + [17 * mm] * 5 + [53 * mm]), Spacer(1, 3)]
    if E or EU:
        erow = [["end point / last step", "τ", "accuracy %", "programs run %", "invalid tokens %", "Δ vs τ=1 at σ_min 0.1"]]
        for lab, ser in (("σ_min 0.1, sharpened", E), ("σ_min 0.1, unsharpened", EU)):
            for t in sorted(ser, reverse=True):
                erow.append([lab, f"{t:g}"] + _cell_row(ser[t], E.get(1.0) if t != 1.0 else None))
        out += [H2("E — end the trajectory at σ_min = 0.1"),
                table(erow, col_widths=[38 * mm, 10 * mm, 22 * mm, 26 * mm, 26 * mm, 48 * mm]), Spacer(1, 3)]
    if D:
        drow = [["logit temperature T", "accuracy %", "programs run %", "invalid tokens %", "Δ vs churn τ=1 (paired)"]]
        for t in sorted(D, reverse=True):
            drow.append([f"{t:g}"] + _cell_row(D[t], base))
        out += [H2("D — DeepMind logit temperature (churn γ=0.41)"),
                table(drow, col_widths=[32 * mm, 26 * mm, 28 * mm, 28 * mm, 56 * mm]), Spacer(1, 3)]
    else:
        out.append(P("<i>D (logit temperature) pending: first measured on a 64-problem smoke run for cost.</i>", "small"))
    v = lambda ser, t: acc(ser[t]) if t in ser else float("nan")
    obs = []
    if A_c and base:
        bt = max(A_c, key=lambda t: acc(A_c[t]))
        obs.append(f"<b>The churn collapse was the last step.</b> With it unsharpened, churn γ=0.41 keeps working to τ=0.2 "
                   f"({v(A_c, 0.5):.1f} / {v(A_c, 0.3):.1f} / {v(A_c, 0.2):.1f}% at τ=0.5 / 0.3 / 0.2, against 2.7 / 0.0 / 0.0% "
                   f"before) and invalid tokens fall back to 0. Best: τ={bt:g}, {fmt_d(paired(A_c[bt], base))} pts over τ=1.")
    if A_d:
        obs.append(f"<b>The deterministic collapse is not.</b> Unsharpening the last step leaves it unchanged "
                   f"({v(A_d, 0.5):.1f}% at τ=0.5): it is the mode/length collapse of {sec_mode}, as predicted.")
    if Bn:
        obs.append(f"<b>Ancestral</b> is stable to τ=0.2 ({v(Bn, 0.2):.1f}%, {fmt_d(paired(Bn[0.2], Bn[1.0]))} pts over its own "
                   f"τ=1) but starts much lower ({v(Bn, 1.0):.1f}% at τ=1, little noise) and drops back at τ=0.1 "
                   f"({v(Bn, 0.1):.1f}%). It never beats churn.")
    if any(C.values()):
        obs.append(f"<b>Lower γ extends the range</b>: at τ=0.1, γ=0.1 / 0.2 score {v(C[0.1], 0.1):.1f} / {v(C[0.2], 0.1):.1f}% while "
                   f"γ=0.3 and 0.41 diverge, as the stability rule predicts. But no γ beats γ=0.41 with an unsharpened last "
                   "step at its best τ; lower γ buys range, not accuracy.")
    if E:
        obs.append(f"<b>Ending at σ_min = 0.1</b> alone recovers τ=0.5–0.3 ({v(E, 0.5):.1f} / {v(E, 0.3):.1f}%) but a sharpened "
                   f"last step still corrupts bits at τ=0.2 ({v(E, 0.2):.1f}%). With the last step unsharpened, ending at 0.1 "
                   "or 0.002 gives identical answers: once that step is untempered, it decides nothing.")
    if D:
        bd = max(D, key=lambda t: acc(D[t]))
        obs.append(f"<b>DeepMind's logit temperature</b> is stable at every T (no invalid tokens), but helps far less: best "
                   f"T={bd:g}, {acc(D[bd]):.1f}% ({fmt_d(paired(D[bd], base))} pts over churn τ=1).")
    out += [H2("Observations"), bullets(obs, "small"),
            P("One seed throughout; the best cells should get seeds 1–2 before the numbers are quoted externally.", "small")]
    head = None
    if best and base:
        head = callout(f"<b>Best fixed cell: {best[1]}, τ={best[2]:g} — {best[0]:.1f}%</b> vs 28.1% for the original "
                       f"sampler at τ=1 ({fmt_d(paired(best[3], base))} pts, paired) and 32.4% for the original best (τ=0.7). "
                       "Details in Section 7.", "keep")
    return out, head


def main():
    mpl_style()
    cells = load()
    diag = load("diag")
    anc = load("anchor500")
    rmin, rmax, rmed = churn_stability()

    m = series(cells, ("m250", 0.0, 0.41, 1024))
    c1 = series(cells, ("cfg425", 1.0, 0.41, 1024))
    s5 = series(cells, ("s500", 0.0, 0.41, 1024))
    s5d = series(cells, ("s500", 0.0, 0.0, 1024))
    s5h = series(cells, ("s500", 0.0, 0.41, 512))
    sc = series(cells, ("s425", 0.0, 0.41, 1024))
    sc512 = series(cells, ("s425", 0.0, 0.41, 512))
    sd = series(cells, ("s425", 0.0, 0.0, 1024))
    raw = {k[2]: r for k, r in diag.items() if k[0] == "s425raw"}
    a_ent = next((r for k, r in anc.items() if k[0] == "s500ent"), None)
    a_kar = next((r for k, r in anc.items() if k[0] == "s500kar"), None)
    s5_sched = sched_of(s5) if s5 else (sched_of(s5d) if s5d else "pending")
    have5 = 1 in s5 and len(s5) >= 5

    mb, cb = best(m), best(c1)
    grid = {(w, t): cells[("cfg425", float(w), float(t), 0.41, 1024)]
            for w in (1, 2, 3, 5, 9, 12) for t in (1, 0.5, 0.3, 0.2, 0.1)
            if ("cfg425", float(w), float(t), 0.41, 1024) in cells}
    gbest = max(grid, key=lambda k: acc(grid[k]))
    w_only = max((w for w in (1, 2, 3, 5, 9, 12) if (w, 1) in grid), key=lambda w: acc(grid[(w, 1)]))

    story = [
        P("Sigma-style score temperature on CoBit", "title"),
        P("TinyGSM → GSM8K, executable-code accuracy · results of the experiments in "
          "“Testing Sigma-style score temperature on CoBit-S” (11 Oct 2026) · run 12–13 Oct 2026 on "
          "Isambard GH200 · CPU-only analysis · revised with Experiment 1 on the 500k checkpoint", "subtitle"),
    ]
    if have5:
        b5 = best(s5)
        sig5 = b5[2][1] > 0
        story.append(callout(
            "<b>Experiment 1 (unguided CoBit-S, 500k, EMA):</b> "
            f"{acc(s5[1]):.1f}% at τ=1 → <b>{b5[1]:.1f}% at τ={b5[0]:g}</b> "
            f"({fmt_d(b5[2])} pts, paired; churn, 1024 steps, {s5_sched} schedule). "
            + ("Global score temperature helps the main unguided model too; below τ=0.7 it collapses, most likely "
               "because the entropic σ grid's final step cannot take a drift that large (Section 5; fixed in Section 7), "
               "rather than because the model stops benefiting. " if sig5 and b5[0] != 1 else
               "Score temperature does not improve the main unguided model at any τ tested. ")
            + "<b>Across the two other checkpoints</b> it lifts CoBit-M (462M) from "
            f"{acc(m[1]):.1f}% to {mb[1]:.1f}% (τ={mb[0]:g}, {fmt_d(mb[2])}) and the CoBit-S CFG checkpoint at "
            f"w=1 from {acc(c1[1]):.1f}% to {cb[1]:.1f}% (τ={cb[0]:g}, {fmt_d(cb[2])}). <b>CFG adds nothing on "
            f"top</b> (best (w, τ) = ({gbest[0]}, {gbest[1]:g}), {acc(grid[gbest]):.1f}%). <b>Sigma's τ ≤ 0.1 gives "
            "0% with churn</b>: our EDM-churn Euler step is numerically unstable there, which says nothing about "
            "the readout."))
    else:
        story.append(callout("<b>Experiment 1 on the 500k checkpoint: pending.</b>", "warn"))
    sched_cause = bool(a_ent and a_kar and abs(acc(a_ent) - acc(a_kar)) > 5)
    if a_ent and a_kar:
        story.append(callout(
            "<b>Correction to the first version.</b> It blamed a stale EMA for the unguided checkpoint's low "
            "scores. That was wrong: the EMA is complete and was updating normally (Section 4). "
            + (f"The cause was the σ schedule: on the 500k EMA at τ=1, entropic scores {acc(a_ent):.1f}% and Karras "
               f"{acc(a_kar):.1f}% ({fmt_d(paired(a_ent, a_kar))} pts, paired)." if sched_cause else
               f"Nor is it the σ schedule: the 500k EMA at τ=1 scores {acc(a_ent):.1f}% (entropic) and "
               f"{acc(a_kar):.1f}% (Karras), reproducing the ~29% of June. The failure is specific to the uploaded "
               "425k file" + (" " + DRIVE425_NOTE if DRIVE425_NOTE else "") + "; Experiment 1 now runs on the "
               "500k checkpoint."), "warn"))

    story += [H1("1  What was run"),
              P("One GPU per cell, all 1319 GSM8K test problems, seed 0, EMA weights, DDIM/Euler integrator, "
                "<font face='Mono'>--score_temp_mode global</font>: the PF-ODE drift is divided by τ at every σ after "
                "the CFG mix, i.e. D<sub>τ</sub> = x + (D − x)/τ, exactly Eq. (3) of the note. Self-conditioning and "
                "the final readout use the untempered posterior, as in Sigma. τ=1 is bit-identical to the previous "
                "sampler (unit-tested). Churn is EDM γ=0.41 per step over the whole σ range."),
              table([
                  ["arm", "checkpoint", "σ schedule", "cells"],
                  ["Exp. 1 — τ alone", "CoBit-S unguided, step 500k (Drive last.pt)", s5_sched,
                   "τ ∈ {1, .7, .5, .3, .2, .1, .05} × {churn @1024, γ=0 @1024, churn @512}"],
                  ["schedule anchor", "same, τ=1, churn @1024", "entropic vs Karras", "2"],
                  ["Exp. 2 — (w, τ)", "CoBit-S p<sub>uncond</sub>=0.1, step 425k", "Karras",
                   "w ∈ {1, 2, 3, 5, 9, 12} × τ ∈ {1, .5, .3, .2, .1} + (1, 0.05); churn @1024"],
                  ["462M — τ alone", "CoBit-M unguided, step 250k (of a 500k cosine)", "Karras",
                   "τ ∈ {1, .7, .5, .3, .2, .1, .05}; churn @1024"],
                  ["superseded", "CoBit-S unguided 425k upload (EMA; raw-weight diagnostic)", "Karras",
                   "the first Exp. 1 attempt; appendix"],
              ], col_widths=[27 * mm, 52 * mm, 23 * mm, 68 * mm]),
              Spacer(1, 4),
              H2("Differences from the note's protocol, and why"),
              bullets([
                  "<b>Additions:</b> a w=12 column (the earlier GSM8K CFG sweep was still rising at w=7–20; the "
                  "note's “flat” sweep was the Sudoku one) and a 512-step control. The γ=0 check was run for every τ.",
                  "<b>Schedules.</b> Exp. 2 and CoBit-M use Karras; Exp. 1 uses the schedule its anchor picked. "
                  "Every comparison is paired within one setting against its own τ=1 cell, so the conclusions hold, "
                  "but absolute accuracies are only comparable within an arm.",
                  "<b>One seed.</b> Intervals are bootstrap over problems; paired deltas use the same problems "
                  "and seed.",
                  "<b>Checkpoint.</b> Exp. 1 runs on the 500k <font face='Mono'>last.pt</font> from the shared Drive "
                  f"folder (global_step 500000, raw + EMA weights, SHA-256 {S500_SHA[:16]}…), the same run as the "
                  "uploaded 425k, 75k steps later.",
                  "<b>w=1</b> runs through the two-branch CFG path (algebraically the conditional model).",
              ], "small"),
              PageBreak()]

    story += [H1("2  τ alone"),
              figure(fig_tau(cells), "scoretemp_tau", 165,
                     "Each point is one cell (1319 problems); bars are bootstrap 95% CIs. Schedules differ between "
                     "arms (legend), so compare shapes, not heights. In the shaded band the churn step is unstable "
                     "(Section 5) and every cell scores 0%.")]
    if s5:
        story += [H2(f"CoBit-S unguided, step 500k — churn, 1024 steps ({s5_sched})"), tau_table(s5), Spacer(1, 3)]
    story += [H2("CoBit-M 462M, unguided, step 250k — churn, 1024 steps (Karras)"), tau_table(m), Spacer(1, 3),
              H2("CoBit-S CFG checkpoint at w=1 (no guidance), step 425k — churn, 1024 steps (Karras)"),
              tau_table(c1)]
    arms = [("CoBit-M", m), ("the CFG checkpoint", c1)] + ([("the 500k unguided model", s5)] if have5 else [])
    rob = [n for n, s_ in arms if all(paired(s_[t], s_[1])[1] > 0 for t in s_ if 0.2 <= t <= 0.7)]
    nonrob = [n for n, _ in arms if n not in rob]
    obs = []
    if rob:
        obs.append("<b>The gain is real where sampling is stable.</b> For " + ", ".join(rob) +
                   ", every τ tested in 0.2–0.7 beats τ=1 with a paired CI excluding zero."
                   + (f" For {', '.join(nonrob)} at least one τ in that range does not (see its table)." if nonrob else ""))
    elif nonrob:
        obs.append("For " + ", ".join(nonrob) + " at least one τ in 0.2–0.7 does not beat τ=1 (see the tables).")
    obs += [
        f"<b>Programs run more often when sharpened</b> (CoBit-M {100 * m[1]['executable_rate']:.0f}% → "
        f"{100 * m[mb[0]]['executable_rate']:.0f}%): most of the gain is fewer broken programs.",
        "<b>Model size.</b> CoBit-M gains " + f"{mb[2][0]:+.1f} pts and the CFG checkpoint {cb[2][0]:+.1f}"
        + (f"; the 500k unguided CoBit-S {best(s5)[2][0]:+.1f}" if have5 else "")
        + ". These are different checkpoints (scale, optimiser, conditioning dropout, schedule), not one model at "
        "several quality levels.",
        "<b>Where sampling breaks most likely depends on the σ grid, not the model.</b> On Karras (CoBit-M, CFG checkpoint) "
        "the churn step fails below τ≈0.15. On the entropic grid (500k unguided) the final step, σ 0.157 → 0.002, "
        "covers 99% of the remaining σ at once and fails below τ≈0.5 (Section 5), so that curve stops at 0.7.",
    ]
    story += [H2("Observations"), bullets(obs, "small")]

    sig_txt = ", ".join(
        f"({w}, {t:g}) → {acc(cells[('cfg425', float(w), float(t), 0.41, 1024)]):.1f}%"
        for w, t in [(2, 0.1), (9, 1), (1, 0.05)] if ("cfg425", float(w), float(t), 0.41, 1024) in cells)
    story += [H1("3  CFG × τ on the CFG checkpoint"),
              figure(fig_grid(cells), "scoretemp_grid", 120,
                     "Accuracy % per cell, churn, 1024 steps, Karras, 1319 problems. w=1 is the conditional model. "
                     "Bold = best cell."),
              bullets([
                  f"<b>τ does the work; w does not add to it.</b> Best cell: w={gbest[0]}, τ={gbest[1]:g} "
                  f"({acc(grid[gbest]):.1f}%). CFG alone peaks at w={w_only} ({acc(grid[(w_only, 1)]):.1f}%, "
                  f"{fmt_d(paired(grid[(w_only, 1)], grid[(1, 1)]))} pts over w=1). Every w>1 row at its best τ is "
                  "at or below the w=1 row at its best τ.",
                  "<b>Guidance moves the breaking point.</b> At τ=0.2 accuracy falls from "
                  f"{acc(grid[(1, 0.2)]):.1f}% (w=1) to {acc(grid[(3, 0.2)]):.1f}% (w=3) and ≈0 for w≥9: "
                  "CFG and τ both enlarge the drift, so their combination has to stay inside the stable range.",
                  f"<b>Sigma's settings:</b> {sig_txt}. Their optimum lies in the region our sampler cannot reach.",
                  "<b>Where the branches disagree.</b> The per-step CFG direction ‖D<sub>c</sub> − D<sub>u</sub>‖ (rms "
                  "over free bits) is ≈0.18 at σ>10 for every w, 0.12–0.20 at σ∈[1,10], ≤0.09 at σ∈[0.1,1] and 0 "
                  "below: guidance acts mostly while the sample is still choosing its mode. A weak unconditional "
                  "branch (10% dropout on a 130M model) is a plausible reason CFG adds little.",
              ], "small")]

    story += [H1("4  Experiment 1 on the unguided CoBit-S model")]
    if a_ent and a_kar:
        story += [H2("Schedule anchor (500k EMA, τ=1, churn, 1024 steps)"),
                  table([["schedule", "accuracy %", "95% CI", "programs that ran %"],
                         ["entropic", f"{acc(a_ent):.1f}", f"[{100 * a_ent['ci95_low']:.1f}, {100 * a_ent['ci95_high']:.1f}]",
                          f"{100 * a_ent['executable_rate']:.1f}"],
                         ["Karras", f"{acc(a_kar):.1f}", f"[{100 * a_kar['ci95_low']:.1f}, {100 * a_kar['ci95_high']:.1f}]",
                          f"{100 * a_kar['executable_rate']:.1f}"]],
                        col_widths=[30 * mm, 26 * mm, 30 * mm, 36 * mm]), Spacer(1, 4)]
    if s5d:
        story += [H2(f"500k EMA — deterministic (γ=0), 1024 steps ({s5_sched})"), tau_table(s5d), Spacer(1, 3)]
    if s5h:
        story += [H2(f"500k EMA — churn, 512 steps ({s5_sched}): step-count control"), tau_table(s5h), Spacer(1, 3)]
        if s5:
            d512 = [abs(acc(s5h[t]) - acc(s5[t])) for t in s5 if t in s5h and t >= 0.2]
            story.append(P(f"512 vs 1024 steps differ by at most {max(d512):.1f} pts at any τ≥0.2"
                           + (", so the τ gain is not a step-size correction." if max(d512) <= 2.5 else
                              "; see the two tables for where they part.") if d512 else "", "small"))
    story += [H2("What went wrong with the first attempt (425k upload)"),
              P("The uploaded 425k (SHA-256 " + S425_SHA[:16] + "…) scored "
                + (f"{acc(sc[1]):.1f}% at τ=1 with churn and {acc(sd[1]):.1f}% deterministic" if sc and sd else "very low")
                + " under Karras. The first version of this report blamed a stale EMA. A CPU check disproves that: "
                "the EMA holds every trained tensor (only the fixed <font face='Mono'>time_fn._freq</font> buffer is "
                "absent, as in the healthy CFG checkpoint), is finite, and moved with the weights between 350k and "
                "425k (median relative change 31% for EMA vs 33% for raw; EMA-to-raw distance median 8.9%). "
                + ("The anchor above identifies the σ schedule as the cause." if sched_cause else
                   "The anchor above rules out the σ schedule as well: the 500k file works under both. The cause "
                   "is specific to the uploaded 425k file" + (" (" + DRIVE425_NOTE + ")" if DRIVE425_NOTE else "")
                   + " and is not identified.")
                + (f" Its raw weights did score higher than its EMA under Karras ({acc(raw[1.0]):.1f}% vs "
                   f"{acc(sc512[1]):.1f}% at 512 steps), consistent with the observation that running weights "
                   "work better for this model." if 1.0 in raw and 1 in sc512 else "")),
              H1("5  Where sampling breaks: the step size, not the readout")]
    low = [r for k, r in cells.items() if k[3] > 0 and k[2] <= 0.1]
    lo_acc = max(acc(r) for r in low)
    lo_ex = max(100 * r.get("executable_rate", 0) for r in low)
    story += [P("With churn γ=0.41 each step first raises σ to σ̂ = 1.41σ, then takes an Euler step to the next "
                "σ. The step contracts the distance to the denoiser output by a factor 1 − r/τ with "
                f"r = 1 − σ<sub>next</sub>/σ̂ ∈ [{rmin:.2f}, {rmax:.2f}] on a 1024-step grid (median {rmed:.2f}). "
                f"It is stable only while |1 − r/τ| &lt; 1, i.e. τ &gt; r/2 ≈ {rmax / 2:.2f}. At τ=0.2 the factor "
                f"is ≈{1 - rmed / 0.2:.1f} (overshooting but stable); at τ=0.1 it is ≈{1 - rmed / 0.1:.1f}, the state "
                "grows geometrically, and the diagnostics show it going non-finite by step 67 (σ≈78), before any "
                f"token has formed. All {len(low)} churn cells at τ≤0.1 score "
                + ("0% with no program running." if lo_acc == 0 and lo_ex == 0 else
                   f"at most {lo_acc:.1f}% ({lo_ex:.1f}% of programs running).")),
              P("<b>The entropic grid has one much larger step.</b> CoBit's entropic schedule (used for the 500k arm) "
                "ends at σ=0.157 and then jumps to σ=0.002 in a single step, r=0.987, while every other step has "
                "r≤0.13 (≈0.30 with churn). For that final step the same condition gives τ &gt; 0.49, with or without "
                "churn: at τ=0.5 the step overshoots the denoiser output by ~2× just before readout and flips bits. "
                "This matches the 500k collapse at τ≤0.5 in both its churn and deterministic tables, while Karras "
                "(largest deterministic step r=0.024) lets the other arms go down to τ=0.2. Leaving the final step "
                "unsharpened — Sigma reads out from an unsharpened call — removes it: Section 7 confirms this on the real model."),
              P("So the note's proposed reading (“if the executable rate collapses at τ ≳ 0.3, the bitstream readout "
                "is the limit”) cannot be applied to churn cells: the collapse is numerical. Sigma samples with DDPM "
                "ancestral steps, which have no churn jump; whether CoBit stays stable at τ≤0.1 under such a step "
                "is untested (the rolling sampler's η=1 bridge is the natural place to test it)."),
              H1("6  What this does and does not show"),
              bullets([
                  "<b>Does:</b> constant score temperature over the whole trajectory — the operation our earlier A1 "
                  f"sweep did not test — at τ≈0.2–0.3 with churn, on {len(rob)} of {len(arms)} checkpoints tested "
                  "with paired CIs excluding zero across that range. The earlier A1 negative (late-σ only) does not "
                  "transfer to this operation.",
                  "<b>Does not:</b> reach Sigma's regime (τ≤0.1, w≥2); test CFG fairly (our unconditional branch "
                  "comes from 10% dropout on a 130M model, Sigma's from attention masking at 3B); or separate model "
                  "quality from model identity (one checkpoint per model, one seed).",
                  "<b>Earlier related result:</b> the FKC drift-only β sweep (K=1, no weights) was constant scaling "
                  "for τ≥0.77 and showed +4 pts at τ=0.83; consistent with Section 2.",
              ], "small"),
              P("The fixes proposed here (unsharpened last step, ancestral sampler, lower γ, logit temperature, "
                "ending at σ_min = 0.1) were run on 13 Oct: Section 7.", "small")]
    sec7, head7 = blocks_section(cells)
    story += sec7
    if head7 is not None:
        story.insert(2, head7)
    story += [PageBreak(),
              H1("Appendix — superseded first attempt: unguided CoBit-S 425k upload (Karras, EMA)")]
    for s_, title in [(sc, "churn, 1024 steps"), (sc512, "churn, 512 steps"), (sd, "deterministic (γ=0), 1024 steps")]:
        if s_:
            story += [H2(title), tau_table(s_), Spacer(1, 3)]
    if raw:
        story += [P("Raw-weight diagnostic, churn, 512 steps: " + ", ".join(
            f"τ={t:g} → {acc(r):.1f}%" for t, r in sorted(raw.items(), reverse=True)) + ".", "small")]
    story += [Spacer(1, 6),
              P(f"Data: {len(cells)} result files in results/scoretemp/eval_gh200 (+{len(anc)} in anchor500/, "
                f"+{len(diag)} in diag/; the 14 silently-deterministic first-run cells are kept in "
                "invalid_s425_churn_silently_deterministic/ and unused). Sweep driver "
                "scripts/hpc/arch/scoretemp_gh200.sh; code snapshot logs/arch/scoretemp_provenance/. Built by "
                "docs/pdf_reports/build_scoretemp.py.", "small")]

    path = build(HERE / "report_score_temperature.pdf", story,
                 "Sigma-style score temperature on CoBit · 13 Oct 2026", "score temperature")
    print(path)


if __name__ == "__main__":
    main()
