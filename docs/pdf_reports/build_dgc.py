"""Report: DGC sigma grid vs the entropy-rate grid on TinyGSM (GSM8K), Sudoku and OWT. CPU only.

Reads the cells submitted by scripts/dgc/tasks_ab.sh (results/dgc/{gsm8k,sudoku}, and the OWT
runs/.../dgc_ab/<arm>/seed42_c<cell>/results.jsonl), plus the stored entropy-grid CoBit-S cells of
the score-temperature study, which the gate showed are reproduced exactly (128/128 problems).
Paired deltas are bootstrapped over problems (GSM8K) or over (seed, puzzle) pairs (Sudoku).

    python docs/pdf_reports/build_dgc.py   # -> docs/pdf_reports/report_dgc_sampler.pdf
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdfkit_common import (  # noqa: E402
    PageBreak, H1, H2, HERE, INK2, MUTED, SERIES, P, Spacer, build, bullets, callout, figure, mm, mpl_style,
    plt, table,
)

REPO = HERE.parents[1]
GSM = REPO / "results" / "dgc" / "gsm8k"
SUD = REPO / "results" / "dgc" / "sudoku"
OWT_RUNS = {
    "l": REPO / "runs/paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/dgc_sweep",
    "s": REPO / "runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_1M/dgc_ab",
    "m": REPO / "runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_medium_24x1024/dgc_ab",
}
_B = "results/scoretemp/blocks_gh200/gsm8k_results_stochastic_g{g}_w0.0_s1024_sd0.3998_ema1___{arm}_w0_t{t}_g{g}_n1024_gtau{t}_ufin.json"
STORED_S500 = {  # (tau, gamma) -> entropy-grid CoBit-S 500k cell of the score-temperature study (seed 0, 1319)
    (1.0, 0.41): REPO / "results/scoretemp/eval_gh200/gsm8k_results_stochastic_g0.41_w0.0_s1024_sd0.3998_ema1___s500_w0_t1_g0.41_n1024.json",
    (0.3, 0.41): REPO / _B.format(g="0.41", arm="s500a", t="0.3"),
    (0.2, 0.41): REPO / _B.format(g="0.41", arm="s500a", t="0.2"),
    (0.5, 0.41): REPO / _B.format(g="0.41", arm="s500a", t="0.5"),
    (0.3, 0.2): REPO / _B.format(g="0.2", arm="s500c", t="0.3"),
    (0.3, 0.3): REPO / _B.format(g="0.3", arm="s500c", t="0.3"),
}
for _g in ("0.1", "0.2", "0.3"):   # tau = 1 churn sweep of block C (no temperature suffix at tau 1)
    STORED_S500[(1.0, float(_g))] = REPO / (f"results/scoretemp/blocks_gh200/gsm8k_results_stochastic_g{_g}_w0.0_s1024"
                                            f"_sd0.3998_ema1___s500c_w0_t1_g{_g}_n1024.json")
PAPER_OWT = {  # (model, nfe, gamma) -> (GenPPL, entropy), paper Tables 1-2, seed 42, 1024 samples
    ("s", 256, 0.13): (27.06, 5.26),
    ("m", 256, 0.21): (19.48, 5.40), ("m", 384, 0.24): (13.06, 5.33), ("m", 512, 0.26): (9.87, 5.25),
}
OWT_REAL = (15.07, 5.44)
LM1B_REAL = (53.13, 4.343)   # Georgios's same-code pipeline (results/dgc/lm1b_sweeps); no LM1B cache here
LM1B = (("l", "CoBit-S LM1B"),)
LM1B_NFES = (128, 256, 384, 512)
PART2_FINDINGS = (
    "<b>Part 2 findings (T = 1, best churn per grid).</b> "
    "(1) <b>TinyGSM: no gain.</b> CoBit-S: the standard grid's best is 29.0% (churn 0.3), DGC's best 27.9% "
    "(churn 0.41); DGC is equal or lower at every churn. CoBit-M: 33.8% (standard, churn 0.3) vs 34.0% (DGC, "
    "churn 0.3 and 0.41), +0.2. Only one paired difference excludes zero (CoBit-S, churn 0.3: -2.4). "
    "(2) <b>Sudoku: the hard-puzzle gain holds best-vs-best</b>: 70.3% vs 66.2% (+4.1, both at gamma 0.5), with "
    "DGC ahead at every gamma from 0.1 to 0.5; medium +0.4, easy -0.2. "
    "(3) <b>LM1B, both grids swept in one pipeline: the OWT pattern again.</b> At 128 NFE DGC is far better at "
    "matched entropy (about 82 vs 94-107 GenPPL at entropy 4.34). At 256 and 512 NFE the two grids do not overlap "
    "in entropy: DGC sits 0.02-0.05 higher, the standard grid reaches lower GenPPL (512 NFE: 43.6 at 4.24 vs "
    "46.4 at 4.295). Across all NFE <= 512, DGC is ahead at every entropy from 4.28 up: 48.3 vs 59.5 GenPPL at "
    "entropy >= 4.30, and 61.0 vs 86.8 near the real-data entropy (>= 4.34); the standard grid's lowest GenPPL "
    "comes only at lower entropy. Our DGC points reproduce Georgios's sweep (512 NFE: 46.4 / 4.295 vs 46.3 / 4.291), and our "
    "standard 256-NFE point matches the paper (59.6 vs 59.76), not his 57.3.")
# Standard-grid CoBit-S frontier points at higher churn (README / paper Table 14), 256 NFE, seed 42
S_FRONTIER = [(0.175, 34.0, 5.31), (0.18, 34.35, 5.32)]
COBIT, DGC = SERIES[0], SERIES[1]   # categorical slots 1-2, fixed: standard grid blue, DGC orange
rng = np.random.default_rng(0)


def boot_ci(d, n_boot=10000):
    d = np.asarray(d, float)
    b = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def fmt_d(d):
    return f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]"


# ------------------------------------------------------------------ GSM8K
GSM_LABEL = re.compile(r"__dgc_(?P<ck>s500|m250)_(?P<arm>cobit|dgc|dpdgc)_t(?P<tau>[\d.]+)_g(?P<g>[\d.]+)_n(?P<n>\d+)_l(?P<l>\d+)(_gtau[\d.]+_ufin)?\.json$")


def load_gsm():
    cells = {}
    for f in glob.glob(str(GSM / "gsm8k_results_*.json")):
        m = GSM_LABEL.search(f)
        if not m or int(m["l"]) != 1319:
            continue
        cells[(m["ck"], m["arm"], float(m["tau"]), float(m["g"]), int(m["n"]))] = json.load(open(f))
    for (tau, g), f in STORED_S500.items():
        if f.exists():
            cells.setdefault(("s500", "cobit", tau, g, 1024), json.load(open(f)))
    for r in cells.values():
        pp = r["per_problem"]
        r["_c"] = dict(zip(pp["idx"], pp["correct"]))
    return cells


T1_SETTINGS = [(1024, 1.0, g) for g in (0.1, 0.2, 0.3, 0.41)]
MAIN_SETTINGS = [(1024, 1.0, 0.41), (1024, 0.3, 0.41), (256, 1.0, 0.41)]
SCREEN_SETTINGS = [(1024, 0.3, 0.2), (1024, 0.3, 0.3), (1024, 0.2, 0.41), (1024, 0.5, 0.41), (1024, 0.7, 0.41)]


def gsm_rows(cells, settings=MAIN_SETTINGS, models=(("s500", "CoBit-S unguided 500k"), ("m250", "CoBit-M 250k"))):
    rows, deltas = [], []
    for ck, name in models:
        for n, tau, g in settings:
            a, b = cells.get((ck, "cobit", tau, g, n)), cells.get((ck, "dgc", tau, g, n))
            if a is None and b is None:
                continue
            setting = f"{n} steps, churn {g:g}, " + ("tau 1" if tau == 1 else f"tau {tau:g}, unsharpened last step")
            d = None
            if a and b:
                common = sorted(set(a["_c"]) & set(b["_c"]))
                d = boot_ci([b["_c"][i] - a["_c"][i] for i in common])
                deltas.append((f"{name.split()[0]} {n}, tau {tau:g}" + ("" if g == 0.41 else f", churn {g:g}"), d))
            acc = lambda r: f"{100 * r['accuracy']:.1f}" if r else "pending"
            inv = lambda r: f"{100 * r.get('invalid_token_rate', 0):.3f}" if r else "-"
            rows.append([name, setting, acc(a), acc(b), fmt_d(d) if d else "-", f"{inv(a)} / {inv(b)}"])
    return rows, deltas


# ------------------------------------------------------------------ Sudoku
SUD_LABEL = re.compile(r"sudoku_results_(?P<d>easy|medium|hard)_(?P<s>deterministic|stochastic)_g(?P<g>[\d.]+)_s(?P<steps>\d+)_.*_(?P<arm>cobit_dgcchurn|dgc_stdchurn|cobit|dpdgc|dgc)_seed(?P<seed>\d+)_n2000\.json$")


def load_sudoku(steps=180):
    """(difficulty, gamma, arm, seed) -> result, for one step count (180 = every Part 1-2 cell)."""
    cells = {}
    for f in glob.glob(str(SUD / "*.json")):
        m = SUD_LABEL.search(f)
        if m and int(m["steps"]) == steps:
            cells[(m["d"], float(m["g"]), m["arm"], int(m["seed"]))] = json.load(open(f))
    return cells


def sudoku_rows(cells):
    rows, deltas = [], []
    for diff in ("easy", "medium", "hard"):
        for g in (0.0, 0.4):
            seeds = sorted({k[3] for k in cells if k[0] == diff and k[1] == g})
            if not seeds:
                continue
            accs = {arm: [100 * cells[(diff, g, arm, s)]["exact_match_accuracy"] for s in seeds
                          if (diff, g, arm, s) in cells] for arm in ("cobit", "dgc")}
            pairs = []
            for s in seeds:
                a, b = cells.get((diff, g, "cobit", s)), cells.get((diff, g, "dgc", s))
                if a and b:
                    pairs += [int(y) - int(x) for x, y in zip(a["per_puzzle_exact"], b["per_puzzle_exact"])]
            d = boot_ci(pairs) if pairs else None
            if d:
                deltas.append((f"Sudoku {diff}, " + ("det." if g == 0 else f"gamma {g:g}"), d))
            ms = lambda v: f"{np.mean(v):.1f} ({', '.join(f'{x:.1f}' for x in v)})" if v else "pending"
            rows.append([diff, "deterministic" if g == 0 else f"churn gamma {g:g}", ms(accs["cobit"]),
                         ms(accs["dgc"]), fmt_d(d) if d else "-", str(len(seeds))])
    return rows, deltas


def sudoku_sweep_rows(cells):
    """Per difficulty and grid: accuracy (mean of seeds 42-44) at every gamma, and each grid's best gamma."""
    rows = []
    for diff in ("easy", "medium", "hard"):
        for arm, name in (("cobit", "standard"), ("dgc", "DGC")):
            accs = {}
            for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                v = [100 * cells[(diff, g, arm, sd)]["exact_match_accuracy"] for sd in (42, 43, 44)
                     if (diff, g, arm, sd) in cells]
                if len(v) == 3:
                    accs[g] = float(np.mean(v))
            if not accs:
                continue
            best = max(accs, key=accs.get)
            rows.append([diff, name] + [f"{accs[g]:.1f}" if g in accs else "-" for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
                        + [f"{accs[best]:.1f} (gamma {best:g})"])
    return rows


# ------------------------------------------------------------------ OWT
TAG = re.compile(r"tnfe-(?P<nfe>\d+)_.*?(?:_stoch-ch(?P<ch>[\d.]+))?(?:-n|$)")


def load_owt():
    """(model, nfe, gamma, arm) -> {genppl, entropy (seed means), n, per-seed}, from every non-smoke OWT run."""
    per = {}
    for model, root in OWT_RUNS.items():
        for f in glob.glob(str(root / "*" / "seed*" / "results.jsonl")):
            if "smoke" in f:
                continue
            arm = Path(f).parent.parent.name
            seed = int(re.match(r"seed(\d+)", Path(f).parent.name).group(1))
            for line in open(f):
                r = json.loads(line)
                if r.get("metric") not in ("gen_full_external_ppl", "gen_full_token_unigram_entropy"):
                    continue
                m = TAG.search(r.get("tag", ""))
                if not m:
                    continue
                nfe = int(m["nfe"])
                gamma = round(float(m["ch"]) / (nfe - 1), 3) if m["ch"] else 0.0
                per.setdefault((model, nfe, gamma, arm), {}).setdefault(seed, {})[r["metric"]] = float(r["value"])
    out = {}
    for k, seeds in per.items():
        ok = {sd: v for sd, v in seeds.items() if len(v) == 2}
        if ok:
            out[k] = {met: float(np.mean([v[met] for v in ok.values()]))
                      for met in ("gen_full_external_ppl", "gen_full_token_unigram_entropy")}
            out[k]["n"] = len(ok)
            out[k]["seeds"] = ok
    return out


def s42(owt, key):
    """Seed-42 values of one OWT point: the seed every arm and every point has, so tables built from it are paired."""
    v = owt.get(key)
    return v["seeds"].get(42, {}) if v else {}


def owt_rows(owt):
    rows = []
    for (model, nfe, gamma), (pp, ph) in PAPER_OWT.items():
        a, b = s42(owt, (model, nfe, gamma, "cobit")), s42(owt, (model, nfe, gamma, "dgc"))
        g = lambda r, k, f: f.format(r[k]) if k in r else "pending"
        dppl = f"{b['gen_full_external_ppl'] - a['gen_full_external_ppl']:+.2f}" if \
            "gen_full_external_ppl" in a and "gen_full_external_ppl" in b else "-"
        dh = f"{b['gen_full_token_unigram_entropy'] - a['gen_full_token_unigram_entropy']:+.3f}" if \
            "gen_full_token_unigram_entropy" in a and "gen_full_token_unigram_entropy" in b else "-"
        rows.append([("CoBit-S" if model == "s" else "CoBit-M"), f"{nfe} / {gamma:g}", f"{pp:.2f} / {ph:.2f}",
                     g(a, "gen_full_external_ppl", "{:.2f}") + " / " + g(a, "gen_full_token_unigram_entropy", "{:.3f}"),
                     g(b, "gen_full_external_ppl", "{:.2f}") + " / " + g(b, "gen_full_token_unigram_entropy", "{:.3f}"),
                     dppl, dh])
    return rows


# ------------------------------------------------------------------ figures
def forest(deltas, title):
    mpl_style()
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(deltas) + 0.9))
    y = np.arange(len(deltas))[::-1]
    for yi, (lab, (m, lo, hi)) in zip(y, deltas):
        ax.plot([lo, hi], [yi, yi], color=DGC, lw=2, solid_capstyle="round")
        ax.plot([m], [yi], "o", ms=6, color=DGC, mec="white", mew=1.5)
        ax.text(hi + 0.3, yi, f"{m:+.1f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [d[0] for d in deltas], fontsize=7.5)
    ax.set_xlabel("DGC minus standard grid, accuracy points (paired 95% CI)", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return fig


def owt_frontier(owt):
    mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, model, name in ((axes[0], "s", "CoBit-S (130M)"), (axes[1], "m", "CoBit-M (462M)")):
        for (m, nfe, gamma), _ in PAPER_OWT.items():
            if m != model:
                continue
            a, b = s42(owt, (m, nfe, gamma, "cobit")), s42(owt, (m, nfe, gamma, "dgc"))
            pts = [(r["gen_full_token_unigram_entropy"], r["gen_full_external_ppl"], c)
                   for r, c in ((a, COBIT), (b, DGC)) if "gen_full_external_ppl" in r and "gen_full_token_unigram_entropy" in r]
            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=MUTED, lw=0.8, zorder=1)
            for x, yv, c in pts:
                ax.plot([x], [yv], "o", ms=7, color=c, mec="white", mew=1.5, zorder=2)
            if pts:
                ax.annotate(f"{nfe} NFE", (pts[0][0], pts[0][1]), textcoords="offset points", xytext=(6, -2),
                            fontsize=7, color=INK2)
        if model == "s":
            for g, ppl, h in S_FRONTIER:
                ax.plot([h], [ppl], "o", ms=7, mfc="white", mec=COBIT, mew=1.5, zorder=2)
            ax.annotate("std grid, gamma .175/.18\n(paper)", (S_FRONTIER[-1][2], S_FRONTIER[-1][1]),
                        textcoords="offset points", xytext=(-75, -24), fontsize=6.5, color=INK2)
        ax.plot([OWT_REAL[1]], [OWT_REAL[0]], marker="*", ms=9, color=INK2, ls="none")
        ax.annotate("real OWT", (OWT_REAL[1], OWT_REAL[0]), textcoords="offset points", xytext=(5, 3), fontsize=7, color=INK2)
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_xlabel("unigram entropy (higher = more diverse)", fontsize=8)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("GenPPL (GPT-2-large, lower = better)", fontsize=8)
    h = [plt.Line2D([], [], marker="o", ls="none", color=c, ms=7) for c in (COBIT, DGC)]
    fig.legend(h, ["standard entropy-rate grid", "DGC grid"], loc="upper center", ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def curve(owt, model, nfe, arm):
    """The gamma sweep of one grid at one NFE, sorted by entropy: [(entropy, genppl, gamma)]."""
    pts = [(v["gen_full_token_unigram_entropy"], v["gen_full_external_ppl"], g)
           for (m, n, g, a), v in owt.items()
           if (m, n, a) == (model, nfe, arm)]
    return sorted(pts)


def matched(owt, model, nfe):
    """GenPPL of each grid interpolated (linearly in entropy) at three entropies spanning their overlap."""
    a, b = curve(owt, model, nfe, "cobit"), curve(owt, model, nfe, "dgc")
    if len(a) < 2 or len(b) < 2:
        return None
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if hi <= lo:
        return (lo, hi, [])
    out = []
    for h in (lo + 0.25 * (hi - lo), 0.5 * (lo + hi), lo + 0.75 * (hi - lo)):
        pa = np.interp(h, [x[0] for x in a], [x[1] for x in a])
        pb = np.interp(h, [x[0] for x in b], [x[1] for x in b])
        out.append((h, pa, pb))
    return (lo, hi, out)


def fronts_rows(owt, models=(("s", "CoBit-S"), ("m", "CoBit-M")), nfes=(256, 384, 512)):
    rows = []
    for model, name in models:
        for nfe in nfes:
            r = matched(owt, model, nfe)
            if r is None:
                continue
            lo, hi, pts = r
            if not pts:
                rows.append([name, str(nfe), f"no overlap ({lo:.3f} &gt; {hi:.3f})", "-", "-", "-"])
                continue
            for h, pa, pb in pts:
                rows.append([name, str(nfe), f"{h:.3f}", f"{pa:.2f}", f"{pb:.2f}", f"{pb - pa:+.2f}"])
    return rows


ENTROPY_LEVELS = {"s": (5.20, 5.25, 5.30, 5.35, 5.40, 5.44), "m": (5.30, 5.34, 5.37, 5.40, 5.44),
                  "l": (4.28, 4.30, 4.32, 4.34)}


def best_front_rows(owt, models=(("s", "CoBit-S"), ("m", "CoBit-M"))):
    """Per grid, the lowest GenPPL among all swept points (any NFE <= 512, any gamma) with entropy >= h."""
    rows = []
    for model, name in models:
        for h in ENTROPY_LEVELS[model]:
            cells = []
            for arm in ("cobit", "dgc"):
                pts = [(v["gen_full_external_ppl"], n, g) for (m, n, g, a), v in owt.items()
                       if m == model and a == arm and v["gen_full_token_unigram_entropy"] >= h]
                cells.append(f"{min(pts)[0]:.1f} ({min(pts)[1]}, {min(pts)[2]:g})" if pts else "not reached")
            rows.append([name, f"&gt;= {h:.2f}"] + cells)
    return rows


def fronts_figure(owt, models=(("s", "CoBit-S"), ("m", "CoBit-M")), nfes=(256, 384, 512), real=OWT_REAL, real_label="real OWT"):
    mpl_style()
    fig, axes = plt.subplots(len(models), len(nfes), figsize=(7.4, 2.6 * len(models) + 0.2), squeeze=False)
    for i, (model, name) in enumerate(models):
        for j, nfe in enumerate(nfes):
            ax = axes[i][j]
            for arm, c in (("cobit", COBIT), ("dgc", DGC)):
                pts = sorted(curve(owt, model, nfe, arm), key=lambda t: t[2])   # join in gamma order
                if not pts:
                    continue
                xs, ys, gs = zip(*pts)
                ax.plot(xs, ys, color=c, lw=2, zorder=2)
                ax.plot(xs, ys, "o", ms=6, color=c, mec="white", mew=1.5, zorder=3)
                for x, y, g in (pts[0], pts[-1]):
                    ax.annotate(f"{g:g}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6, color=INK2)
            ax.plot([real[1]], [real[0]], marker="*", ms=8, color=INK2, ls="none", zorder=4)
            ax.set_title(f"{name}, {nfe} NFE", fontsize=8, loc="left")
            ax.tick_params(labelsize=6.5)
            ax.grid(alpha=0.25)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel("GenPPL", fontsize=7.5)
            if i == len(models) - 1:
                ax.set_xlabel("unigram entropy", fontsize=7.5)
    h = [plt.Line2D([], [], marker="o", color=c, ms=6, lw=2) for c in (COBIT, DGC)] + \
        [plt.Line2D([], [], marker="*", color=INK2, ms=8, ls="none")]
    fig.legend(h, ["standard entropy-rate grid", "DGC grid", real_label], loc="upper center", ncol=3,
               fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if len(models) > 1 else 0.88))
    return fig


MECH_ARMS = [("cobit", "standard", "standard"), ("cobit_dgcchurn", "standard", "DGC"),
             ("dgc_stdchurn", "DGC", "standard"), ("dgc", "DGC", "DGC")]
OWT_MECH_ARMS = [("cobit", "standard", "standard"), ("cobit_swapchurn", "standard", "DGC"),
                 ("dgc_swapchurn", "DGC", "standard"), ("dgc", "DGC", "DGC")]


def mech_sudoku_rows(cells, diff="hard", g=0.15):
    base = {k[3]: v for k, v in cells.items() if k[:3] == (diff, g, "cobit")}
    rows = []
    for arm, steps, churn in MECH_ARMS:
        runs = {k[3]: v for k, v in cells.items() if k[:3] == (diff, g, arm)}
        if not runs:
            continue
        accs = [100 * runs[sd]["exact_match_accuracy"] for sd in sorted(runs)]
        pairs = [int(y) - int(x) for sd in sorted(runs) if sd in base
                 for x, y in zip(base[sd]["per_puzzle_exact"], runs[sd]["per_puzzle_exact"])]
        d = fmt_d(boot_ci(pairs)) if pairs and arm != "cobit" else "-"
        rows.append([steps, churn, f"{np.mean(accs):.1f} ({', '.join(f'{a:.1f}' for a in accs)})", d])
    return rows


def mech_owt_rows(owt, model="s", nfe=256):
    rows = []
    for g in (0.05, 0.13):
        for arm, steps, churn in OWT_MECH_ARMS:
            v = s42(owt, (model, nfe, g, arm))
            if "gen_full_external_ppl" in v:
                rows.append([f"{g:g}", steps, churn, f"{v['gen_full_external_ppl']:.2f}",
                             f"{v['gen_full_token_unigram_entropy']:.3f}"])
    return rows


def churn_placement_figure():
    """Cumulative churn sum(gamma_i) from sigma_max down to sigma, for each grid (CPU, from the tables)."""
    import torch
    from types import SimpleNamespace as NS
    sys.path.insert(0, str(REPO))
    from diffusion.continuous.samplers import SigmaSchedule
    cfg = NS(diffusion=NS(continuous=NS(sigma_min=0.002, sigma_max=80.0)), evaluation=NS(entropic_blend_alpha=0.0))
    sch = SigmaSchedule(None, cfg, torch.device("cpu"))
    cases = [("OWT CoBit-S, 256 NFE, gamma 0.13", "assets/entropy_tables/owt", "assets/entropy_tables/owt_dgc", 256, 0.08, 0.13),
             ("Sudoku hard, 180 steps, gamma 0.4", "/scratch/u6ya/raricu.u6ya/sudoku_drive/hard/cobit_raw_binary_bits",
              "assets/entropy_tables/tasks/sudoku_hard_dgc", 180, 0.002, 0.4)]
    mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, (title, std, dgc, n, smin, g) in zip(axes, cases):
        for d, c, lab in ((std, COBIT, "standard grid"), (dgc, DGC, "DGC grid")):
            sg = sch.prepare(schedule="entropic", num_steps=n, entropy_run_dir=REPO / d if not d.startswith("/") else d,
                             sigma_min_override=smin).double().numpy()[:-1]
            cum = np.cumsum(np.full(len(sg), min(g, np.sqrt(2) - 1)))
            ax.plot(sg, cum / cum[-1], color=c, lw=2, label=lab)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("sigma (sampling runs from left to right)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("fraction of total churn applied", fontsize=7.5)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


DP_NFES = (32, 64, 128, 256, 384, 512)
DP_ARMS = (("cobit", "standard"), ("dgc", "continuous DGC"), ("dp_dgc_K8_lensq", "DP-DGC (K=8)"))  # Part 3 runs
DP4_FINDINGS = (   # cells run 23 Oct
    "<b>Part 4 findings (run 23 Oct): the notebook's G_k does not change Part 3's conclusion.</b> Its allocations are "
    "within a few steps of Part 3's (identical at NFE 32) and so are the samples: 385.9 at NFE 32 (continuous DGC "
    "361.7, worse), and from NFE 64 to 512 between 0.9 and 4.5 GenPPL below continuous DGC but always at lower "
    "entropy (-0.012 to -0.056), i.e. a move along the front, not past it. The DP's 15x lower Theorem-1 bound at NFE "
    "32 again does not show up in GenPPL.")
DP_FINDINGS = (
    "<b>Part 3 findings: no evidence that the finite-NFE DP helps.</b> "
    "(1) <b>Low NFE:</b> at NFE 32 DP-DGC is worse than continuous DGC (GenPPL 385.9 vs 361.7 at the same entropy); "
    "at 64 and 128 it is 5.0 and 1.8 GenPPL lower but also 0.011 and 0.007 lower in entropy, within one-seed noise "
    "at GenPPL 70-170. "
    "(2) <b>High NFE:</b> DP-DGC is 0.5-1.1 GenPPL lower and 0.016-0.043 lower in entropy, a move along the front: "
    "at 256, 384 and 512 NFE a continuous-DGC point from the Part 1 gamma sweep beats it on both axes at once "
    "(e.g. 512 NFE: 14.9 / 5.399 at gamma 0.3 vs DP 16.2 / 5.385). "
    "(3) <b>Why so little changes:</b> with K = 8 the exponential term is nearly linear in the blocks that hold the "
    "steps (S_k / n_k ~ 0.17 even at NFE 32), and the near-empty high-sigma blocks have G_k ~ 0, so the DP solution "
    "is the continuous allocation plus one forced step in each high-sigma block and log-uniform steps within blocks. "
    "The finite-NFE problem as posed here is not where the low-NFE losses come from. "
    "One seed, gamma fixed at 0.2, S_k and G_k derived without the reference notebook.")


def dp_rows(owt, g=0.2, arms=DP_ARMS):
    rows = []
    for nfe in DP_NFES:
        vals = [s42(owt, ("m", nfe, g, arm)) for arm, _ in arms]
        if not any(vals):
            continue
        cell = lambda v: (f"{v['gen_full_external_ppl']:.1f} / {v['gen_full_token_unigram_entropy']:.3f}"
                          if "gen_full_external_ppl" in v else "pending")
        d = (f"{vals[-1]['gen_full_external_ppl'] - vals[1]['gen_full_external_ppl']:+.1f} / "
             f"{vals[-1]['gen_full_token_unigram_entropy'] - vals[1]['gen_full_token_unigram_entropy']:+.3f}"
             if "gen_full_external_ppl" in vals[1] and "gen_full_external_ppl" in vals[-1] else "-")
        rows.append([str(nfe)] + [cell(v) for v in vals] + [d])
    return rows


def dp_alloc_rows(table="assets/entropy_tables/owt_medium_dgc", K=8, lo=0.08, weight="length_squared", bound=False):
    import torch
    sys.path.insert(0, str(REPO))
    from diffusion.continuous.dp_dgc import DGCIncrement, block_quantities, dp_dgc_sigmas
    from diffusion.continuous.samplers import SigmaSchedule
    from types import SimpleNamespace as NS
    from evaluation.nfe import steps_for_target_nfe
    ld = lambda f: torch.load(REPO / table / f, map_location="cpu", weights_only=True).double().numpy()
    sig, cdf, pdf = ld("entropy_sigmas.pt"), ld("entropy_cdf.pt").copy(), ld("entropy_pdf.pt")
    cdf[-1] = 1.0
    hi = min(80.0, float(sig[-1]))
    inc = DGCIncrement(sig, pdf, lo, hi)
    sch = SigmaSchedule(None, NS(diffusion=NS(continuous=NS(sigma_min=0.002, sigma_max=80.0)),
                                 evaluation=NS(entropic_blend_alpha=0.0)), torch.device("cpu"))
    rows = []
    for nfe in DP_NFES:
        steps, _ = steps_for_target_nfe(framework="continuous_score", sampler_name="ddim_dp_dgc", target_nfe=nfe,
                                        self_condition=True, sc_refresh_mode="carry")
        s, alloc = dp_dgc_sigmas(sig, cdf, steps, lo, hi, K, pdf=pdf, weight=weight)
        if bound:
            cont = sch.prepare(schedule="entropic", num_steps=steps, entropy_run_dir=REPO / table,
                               sigma_min_override=lo).double().numpy()
            rows.append([str(nfe), str(steps - 1), ", ".join(map(str, alloc)),
                         f"{inc.finite_step_bound(cont):.3g}", f"{inc.finite_step_bound(s):.3g}"])
            continue
        m = block_quantities(sig, cdf, lo, hi, K, pdf=pdf, weight=weight)[3][::-1]
        rows.append([str(nfe), str(steps - 1), ", ".join(map(str, alloc)),
                     ", ".join(f"{x:.1f}" for x in m * (steps - 1))])
    return rows


def dp_schedule_figure(csv_path=REPO / "results/dgc/dp_dgc/schedules_owt_medium_K8.csv"):
    import csv as _csv
    data = {}
    for r in _csv.DictReader(open(csv_path)):
        data.setdefault(int(r["nfe"]), {}).setdefault(r["schedule"], []).append(float(r["sigma"]))
    mpl_style()
    fig, axes = plt.subplots(1, len(data), figsize=(7.4, 2.2), squeeze=False)
    for ax, (nfe, d) in zip(axes[0], sorted(data.items())):
        for name, c, ls in (("continuous_dgc", COBIT, "-"), ("dp_dgc", DGC, "--")):
            y = d[name]
            ax.plot(np.linspace(0, 1, len(y)), y, color=c, lw=1.8, ls=ls)
        ax.set_yscale("log")
        ax.set_title(f"NFE {nfe}", fontsize=8, loc="left")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
        ax.set_xlabel("step / total", fontsize=6.5)
    axes[0][0].set_ylabel("sigma", fontsize=7)
    h = [plt.Line2D([], [], color=COBIT, lw=1.8), plt.Line2D([], [], color=DGC, lw=1.8, ls="--")]
    fig.legend(h, ["continuous DGC", "DP-DGC (K=8)"], loc="upper center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


P5_ARMS = (("cobit", "standard"), ("dgc", "continuous DGC"), ("dpdgc", "DP-DGC"))
P5_FINDINGS = (
    "<b>Part 5 findings: the finite-NFE DP grid does not raise pass@1; on Sudoku it lowers it.</b> "
    "(1) <b>GSM8K:</b> at 64, 128 and 256 steps, on CoBit-S and CoBit-M, every DP-DGC difference is within "
    "+/-1.4 points and every 95% CI contains zero (vs continuous DGC and vs standard). The three grids are within "
    "noise of each other at these step counts. "
    "(2) <b>Sudoku hard:</b> continuous DGC beats the standard grid at every step count (+2.3 to +4.1). DP-DGC loses "
    "1.0-2.4 points of that against continuous DGC at every step count, each CI excluding zero, and sits at the "
    "standard grid's level at 32-64 steps. "
    "(3) <b>Likely reason, untested:</b> Part 1's mechanism test put the Sudoku gain in DGC's low-sigma churn placement. "
    "The DP keeps the block totals but spaces steps log-uniformly within blocks and spends two forced steps at "
    "sigma ~0.03 and ~0.008, below where the table has density, so it changes exactly the low-sigma part the "
    "gain comes from. Invalid-token rates are unchanged (<= 0.002%).")


def p5_gsm_rows(gsm):
    rows = []
    for ck, name in (("s500", "CoBit-S"), ("m250", "CoBit-M")):
        for n in (64, 128, 256):
            r = {arm: gsm.get((ck, arm, 1.0, 0.41, n)) for arm, _ in P5_ARMS}
            if not any(r.values()):
                continue
            def d(a, b):
                if not (r[a] and r[b]):
                    return "-"
                common = sorted(set(r[a]["_c"]) & set(r[b]["_c"]))
                return fmt_d(boot_ci([r[a]["_c"][i] - r[b]["_c"][i] for i in common]))
            acc = lambda x: f"{100 * x['accuracy']:.1f} ({100 * x.get('invalid_token_rate', 0):.3f})" if x else "pending"
            rows.append([name, str(n)] + [acc(r[a]) for a, _ in P5_ARMS] + [d("dpdgc", "dgc"), d("dpdgc", "cobit")])
    return rows


def p5_sudoku_rows(diff="hard", g=0.5):
    rows = []
    for st in (32, 64, 90, 180):
        cells = load_sudoku(st)
        accs, ok = {}, False
        for arm, _ in P5_ARMS:
            v = [100 * cells[(diff, g, arm, sd)]["exact_match_accuracy"] for sd in (42, 43, 44) if (diff, g, arm, sd) in cells]
            accs[arm] = v
            ok = ok or bool(v)
        if not ok:
            continue
        def d(a, b):
            pairs = [int(y) - int(x) for sd in (42, 43, 44) if (diff, g, a, sd) in cells and (diff, g, b, sd) in cells
                     for x, y in zip(cells[(diff, g, b, sd)]["per_puzzle_exact"], cells[(diff, g, a, sd)]["per_puzzle_exact"])]
            return fmt_d(boot_ci(pairs)) if pairs else "-"
        ms = lambda v: f"{np.mean(v):.1f}" if len(v) == 3 else ("pending" if not v else f"{np.mean(v):.1f} (n={len(v)})")
        rows.append([str(st)] + [ms(accs[a]) for a, _ in P5_ARMS] + [d("dpdgc", "dgc"), d("dpdgc", "cobit")])
    return rows


# ------------------------------------------------------------------ story
# ---------------------------------------------------------------- Part 6: gamma-dependent Euler grid (23 Oct)
GE = REPO / "results" / "dgc" / "gamma_euler"
GE_ARMS = (("table", "corrected table"), ("gamma", "gamma-dependent"), ("dgc_profile", "ordinary DGC"),
           ("log_uniform", "log-uniform"), ("karras", "Karras"))
GE_COL = dict(zip([a for a, _ in GE_ARMS], SERIES + ["#8a5cd1"]))
GE_PROD_G = {"lm1b": (0.185, 0.2), "owt": (0.13, 0.175, 0.18)}
P6_FINDINGS = (
    "<b>Part 6 findings: the gamma-dependent grid does not help; the profile-based ordinary DGC grid does, at low NFE.</b> "
    "(1) <b>The profile is nearly flat.</b> On both checkpoints A ~ 2.25 and B ~ 1 (a negligible Jacobian) except "
    "for a dip at sigma ~0.3-1, so sqrt(A / (S^2 B + eta)) is close to log-uniform at S = 1-12 and only concentrates "
    "on the dip at 256 calls with production churn (S = 33-51). "
    "(2) <b>Production sampler (DDIM + EDM churn).</b> The gamma grid ties log-uniform at 64 and 128 calls, both far "
    "behind the corrected table (LM1B +65 / +35 GenPPL; OWT +242-262 / +123-138). At 256 calls it is level on LM1B "
    "(-2.0 +/- 0.7 GenPPL at gamma 0.185, but entropy -0.025 to -0.039) and worse on OWT (+7 to +12, entropy not higher). "
    "(3) <b>Ordinary DGC from the same profile, density sqrt(q), is the best grid at low NFE.</b> 64 calls: LM1B -20 / -21 "
    "GenPPL, OWT -23 / -67 / -78 at the three gammas, entropy equal or higher. 128 calls: LM1B -7.7 at +0.02 entropy; "
    "OWT level (+1.7 to -5.3, CIs ~ +/-3) at +0.03-0.04 entropy. 256 calls: LM1B level at +0.03 entropy; OWT +5 GenPPL at +0.05-0.06 entropy, "
    "i.e. the Part 1 shift of the front toward higher entropy (at matched entropy ~5.30, DGC gamma 0.13 gives 31.7 vs the "
    "table's 33.2 at gamma 0.175). "
    "(4) <b>Literal Euler (theory sweep) is far from production quality</b> (LM1B 230-480, OWT 500-1100 GenPPL; null "
    "self-conditioning, small churn). Under it the corrected table is best, gamma ~ log-uniform, Karras worse, and "
    "the DGC grid collapses (one 3.2-nat step from sigma 80): the grid ranking under the surrogate's own solver does not "
    "predict the ranking under the production sampler. "
    "Sanity check: the corrected table at 256 calls, gamma 0.13, gives OWT 26.5 / 5.241 against the paper's 27.06 / 5.26."
)


def ge_cells():
    """{(data, run, arm, T): summary cell} from scripts/dgc/gamma_euler_collect.py; gamma_* arms map to 'gamma'."""
    sys.path.insert(0, str(REPO / "scripts" / "dgc"))
    import gamma_euler_collect as gc
    out = {}
    for c in gc.summarise(gc.load()):
        arm = "gamma" if c["arm"].startswith("gamma_") else c["arm"]
        out[(c["data"], c["run"], arm, c["T"])] = c
    return out


def _ppl_h(c, d=True):
    if c is None or "genppl" not in c:
        return "-"
    s = f"{c['genppl']:.1f} / {c['entropy']:.3f}"
    if d and "d_genppl" in c:
        s += f"<br/>({c['d_genppl']:+.1f}&plusmn;{c['d_genppl_ci']:.1f} / {c['d_entropy']:+.3f})"
    return s


def ge_rows(cells, data, runs):
    rows = []
    for T in (64, 128, 256):
        for arm, name in GE_ARMS:
            r = [str(T) if arm == "table" else "", name]
            r += [P(_ppl_h(cells.get((data, run, arm, T))), "small") for run in runs]
            if any(x != "-" for x in r[2:]):
                rows.append(r)
    return rows


def ge_profile_figure():
    mpl_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for data, ls in (("lm1b", "-"), ("owt", "--")):
        f = GE / "profiles" / f"{data}_cobit_s.npz"
        if not f.exists():
            continue
        z = np.load(f)
        u = 0.5 * (z["log_sigma_edges"][1:] + z["log_sigma_edges"][:-1]) / np.log(10)
        for key, col, lab in (("A", SERIES[0], "A = E|J-1.5I|^2/d"), ("B", SERIES[1], "B = E|J-I|^2/d")):
            axes[0].plot(u, z[key], ls, color=col, lw=1.4, label=f"{lab} ({data.upper()})")
            axes[0].fill_between(u, z[f"{key}_ci_lo"], z[f"{key}_ci_hi"], color=col, alpha=0.2, lw=0)
        axes[1].plot(u, np.sqrt(z["q_increment"]), ls, color=SERIES[2], lw=1.4, label=f"sqrt(q) ({data.upper()})")
    axes[0].set_title("forward profile (95% CI)", fontsize=8.5, loc="left")
    axes[1].set_title("ordinary DGC density sqrt(q)", fontsize=8.5, loc="left")
    g = GE / "grids"
    for k, (arm, name) in enumerate(GE_ARMS):
        f = g / ("lm1b_gamma_g0.2.npz" if arm == "gamma" else f"lm1b_{arm}.npz")
        if f.exists():
            s = np.load(f)["N127"]
            axes[2].plot(np.log10(s), np.full(len(s), k), "|", ms=7, color=GE_COL[arm])
    f = g / "lm1b_gamma_S4.npz"
    if f.exists():
        axes[2].plot(np.log10(np.load(f)["N127"]), np.full(128, len(GE_ARMS)), "|", ms=7, color=GE_COL["gamma"])
    axes[2].set_yticks(range(len(GE_ARMS) + 1), [n for _, n in GE_ARMS[:1]] + ["gamma (g=0.2)"] +
                       [n for _, n in GE_ARMS[2:]] + ["gamma (S=4)"], fontsize=6.5)
    axes[2].set_title("LM1B grids, 128 calls", fontsize=8.5, loc="left")
    for ax in axes:
        ax.set_xlabel("log10 sigma", fontsize=7.5)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=5.5, frameon=False)
    axes[1].legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    return fig


def ge_front_figure(cells, data):
    gs = GE_PROD_G[data]
    mpl_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=False)
    for ax, T in zip(axes, (64, 128, 256)):
        for arm, name in GE_ARMS:
            pts = [(cells[k]["entropy"], cells[k]["genppl"], cells[k].get("entropy_ci", 0), cells[k].get("genppl_ci", 0))
                   for g in gs for k in [(data, f"ddim_grid_g{g:g}", arm, T)] if k in cells and "genppl" in cells[k]]
            if not pts:
                continue
            x, y, xe, ye = map(np.array, zip(*sorted(pts)))
            ax.errorbar(x, y, xerr=np.nan_to_num(xe), yerr=np.nan_to_num(ye), fmt="o-", ms=4, lw=1, capsize=0,
                        color=GE_COL[arm], label=name)
        ax.set_title(f"{T} model calls", fontsize=8.5, loc="left")
        ax.set_xlabel("unigram entropy", fontsize=7.5)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("GenPPL (GPT-2-large)", fontsize=7.5)
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def part6(story):
    cells = ge_cells()
    if not cells:
        return story
    story += [PageBreak(), H1("Part 6: gamma-dependent Euler grid (CoBit-S LM1B and OWT)"),
              P("Added 23 Oct, from the handoff euler_dgc_derivation.pdf. Grid density per unit log sigma "
                "p = sqrt(A / (S^2 B + eta)), S = N gamma, with A = E||J - 1.5 I||^2 / d and B = E||J - I||^2 / d "
                "measured forward-only on held-out data (2048 sequences, one Brownian path each, 64 log-sigma bins on "
                "[0.08, 80], EMA weights, null self-conditioning). Every arm is a precomputed grid with exact endpoints "
                "0.08 and 80, inverted with the cumulative mass at bin edges. NFE is the number of model calls actually "
                "made (64/128/256 calls = 63/127/255 intervals; the legacy NFE label counts one more). "
                "1024 samples, seeds 42-46 paired across arms; cells show seed-mean GenPPL / entropy and, in "
                "brackets, the paired difference from the corrected table with a t 95% interval over seeds.", "caption")]
    if P6_FINDINGS:
        story.append(callout(P6_FINDINGS, "key"))
    story.append(figure(ge_profile_figure(), "ge_profiles", 170))
    story.append(P("Held-out data: LM1B test split through the training pipeline (detokenise, bert-base-uncased, "
                   "[SEP] boundaries, 128-token blocks, 15-bit codes); OWT train[-100000:] (the documents held out "
                   "from training) through GPT-2 ids, the bpe16 code tokenizer and 16-bit codes. The K = 128 profile "
                   "(same sequences, fresh path), averaged onto the K = 64 bins, agrees to within 4% for A and B and "
                   "20% for q, but about half its bins fall outside the K = 64 sequence-bootstrap CIs: the bin-width "
                   "dependence of the finite-bin estimator is visible, though small next to the grid differences.", "caption"))
    for data, name in (("lm1b", "LM1B"), ("owt", "OWT")):
        th = ge_rows(cells, data, ("euler_lambda_S1", "euler_lambda_S4"))
        if th:
            story += [H2(f"{name}: theory sweep, literal Euler at fixed S"),
                      table([["calls", "grid", "S = 1 (gamma = 1/N)", "S = 4 (gamma = 4/N)"]] + th,
                            col_widths=[12 * mm, 32 * mm, 60 * mm, 60 * mm]),
                      P("x_next = x + (1 + lambda) sigma Delta (D - x) / sigma^2 + sqrt(2 lambda sigma Delta) z, "
                        "lambda_i = gamma sigma_i / Delta_i; the denoiser is evaluated before noise is added; null "
                        "self-conditioning (the closure the profile was measured with).", "caption")]
        gs = GE_PROD_G[data]
        pr = ge_rows(cells, data, tuple(f"ddim_grid_g{g:g}" for g in gs))
        if pr:
            w = 120 / len(gs)
            story += [H2(f"{name}: production sampler (DDIM + EDM churn, carry self-conditioning)"),
                      table([["calls", "grid"] + [f"gamma = {g:g}" for g in gs]] + pr,
                            col_widths=[12 * mm, 32 * mm] + [w * mm] * len(gs)),
                      P("The EDM transfer: each grid replaces only the sigma points; churn gamma acts at every step "
                        "(s_churn = gamma x intervals). The gamma-dependent grid uses S = N x gamma of its column.",
                        "caption"),
                      figure(ge_front_figure(cells, data), f"ge_{data}_fronts", 170)]
    return story


# ---------------------------------------------------------------- Part 7: TinyGSM gamma-Euler grids (23 Oct)
GE_GSM = re.compile(r"__ge_(?P<ck>s500|m250)_(?P<arm>table|dgc_profile|gamma_g[\d.]+)_t1_g(?P<g>[\d.]+)_n(?P<n>\d+)_l1319\.json$")
P7_FINDINGS = (
    "<b>Part 7 findings: on GSM8K neither profile-based grid raises pass@1; the gamma grid lowers it.</b> "
    "(1) <b>The TinyGSM profile is flat.</b> A = 2.24-2.25, B = 0.997-1.001 on both checkpoints: about three quarters "
    "of the generated positions are padding the model predicts exactly from the prompt, so the per-coordinate "
    "average is dominated by J ~ 0 (max q 0.005, vs ~0.36 on LM1B). The gamma grid is then log-uniform on "
    "[0.002, 80] apart from the churn term, and it loses at every step count on both models: -1.9 to -7.2 points, "
    "7 of 8 CIs exclude zero. "
    "(2) <b>Ordinary DGC from the profile ties the corrected table</b> (within +/-1.4 points in 7 of 8 cells; CoBit-S "
    "256 steps -2.7 [-4.9, -0.5]); its best cell, CoBit-M 1024 steps, 35.4% vs 34.0% (+1.4 [-0.8, +3.6]). Unlike "
    "LM1B and OWT there is no low-NFE gain: sqrt(q) is shape-only, and its shape is close to the entropy-rate table's. "
    "(3) The corrected (edge-paired) table matches the legacy grid within ~1.3 points. "
    "(4) <b>CoBit-M tau 0.3 second seed:</b> 39.0% vs seed 0's 38.1% (+1.0 [-0.7, +2.7]); mean 38.6%, so tau 0.3's "
    "~5-point gain over tau 1 (33.3%) is not a sampling-seed artefact. "
    "Possible follow-up (not run): profile the answer tokens only (exclude padding from gen_mask) for the gamma grid."
)


def load_ge_gsm():
    cells = {}
    for f in glob.glob(str(GSM / "gsm8k_results_*__ge_*.json")):
        m = GE_GSM.search(f)
        if m:
            r = json.load(open(f))
            r["_c"] = dict(zip(r["per_problem"]["idx"], r["per_problem"]["correct"]))
            cells[(m["ck"], "gamma" if m["arm"].startswith("gamma") else m["arm"], int(m["n"]))] = r
    return cells


def p7_rows(ge, legacy):
    rows = []
    for ck, name in (("s500", "CoBit-S 500k"), ("m250", "CoBit-M 250k")):
        for n in (64, 128, 256, 1024):
            t, d, g = (ge.get((ck, a, n)) for a in ("table", "dgc_profile", "gamma"))
            if not (t or d or g):
                continue
            leg = legacy.get((ck, "cobit", 1.0, 0.41, n))
            acc = lambda r: f"{100 * r['accuracy']:.1f}" if r else "-"

            def dd(b):
                if not (t and b):
                    return "-"
                common = sorted(set(t["_c"]) & set(b["_c"]))
                return fmt_d(boot_ci([b["_c"][i] - t["_c"][i] for i in common]))
            rows.append([name if n == 64 else "", str(n), acc(leg), acc(t), acc(d), acc(g), dd(d), dd(g)])
    return rows


def p7_temp_rows(gsm):
    rows = []
    s0 = gsm.get(("m250", "cobit", 0.3, 0.41, 1024))
    s1 = [json.load(open(f)) for f in glob.glob(str(GSM / "gsm8k_results_*__dgc_m250_cobit_t0.3_g0.41_n1024_l1319_seed1*.json"))]
    t1 = gsm.get(("m250", "cobit", 1.0, 0.41, 1024))
    for lab, r in (("tau 1 (seed 0)", t1), ("tau 0.3, unsharpened last step, seed 0", s0),
                   ("tau 0.3, unsharpened last step, seed 1", s1[0] if s1 else None)):
        if r:
            rows.append([lab, f"{100 * r['accuracy']:.1f}", f"[{100 * r['ci95_low']:.1f}, {100 * r['ci95_high']:.1f}]"])
    if s0 and s1:
        a = dict(zip(s0["per_problem"]["idx"], s0["per_problem"]["correct"]))
        b = dict(zip(s1[0]["per_problem"]["idx"], s1[0]["per_problem"]["correct"]))
        common = sorted(set(a) & set(b))
        mean2 = 100 * np.mean([(a[i] + b[i]) / 2 for i in common])
        rows.append(["tau 0.3, mean of seeds 0 and 1", f"{mean2:.1f}", f"seed 1 - seed 0: {fmt_d(boot_ci([b[i] - a[i] for i in common]))}"])
    return rows


def part7(story, gsm):
    ge = load_ge_gsm()
    trows = p7_temp_rows(gsm)
    if not ge and not trows:
        return story
    story += [PageBreak(), H1("Part 7: TinyGSM with the profile-based grids")]
    if ge:
        story += [P("Added 23 Oct. Part 6's grids on the prompted task: forward profile on 2048 held-out TinyGSM validation "
                    "sequences (prompt clean and clamped into the self-condition, answer and padding noised, EMA, the "
                    "eval's sigma_data), 64 log-sigma bins on [0.002, 80], the eval's full band. Production sampler "
                    "(DDIM + EDM churn 0.41, carry, T = 1), seed 0, all 1319 GSM8K problems. The gamma-dependent grid "
                    "uses S = (steps - 1) x 0.41. Steps = model calls with the carry closure.", "caption")]
        if P7_FINDINGS:
            story.append(callout(P7_FINDINGS, "key"))
        story += [table([["model", "steps", "legacy grid %", "corrected table %", "ordinary DGC %", "gamma grid %",
                          "DGC - table", "gamma - table"]] + p7_rows(ge, gsm),
                        col_widths=[19 * mm, 13 * mm, 17 * mm, 20 * mm, 18 * mm, 17 * mm, 31 * mm, 31 * mm]),
                  P("Legacy grid: the earlier standard entropic-grid cells (Parts 1, 2 and 5; midpoint-paired CDF, one "
                    "fewer model call). Differences are paired over problems, bootstrap 95% CI.", "caption")]
    if trows:
        story += [H2("CoBit-M 250k: second seed at tau 0.3 (1024 steps, churn 0.41, standard grid)"),
                  table([["setting", "accuracy %", "95% CI / paired difference"]] + trows,
                        col_widths=[75 * mm, 25 * mm, 65 * mm]),
                  P("Seed 1 repeats the Part 1 cell (score temperature tau 0.3, global mode, unsharpened last step) with a "
                    "different sampling seed, to separate the tau effect from sampling noise.", "caption")]
    return story


def main():
    gsm, sud, owt = load_gsm(), load_sudoku(), load_owt()
    grows, gdel = gsm_rows(gsm)
    srows, sdel = sudoku_rows(sud)
    orows = owt_rows(owt)

    story = [H1("DGC sampling grid on TinyGSM, Sudoku and OWT"),
             P("CoBit checkpoints, one GH200 per cell, 20-21 Oct 2026, about 30 GPU-h. One sampling seed for GSM8K, "
               "three for Sudoku and for the OWT points that decide the front comparison.", "caption"), Spacer(1, 4)]
    story.append(callout(
        "<b>Findings.</b> "
        "(1) <b>OWT: DGC moves the GenPPL-entropy front toward higher entropy.</b> With gamma swept on both grids in "
        "one pipeline (3 seeds at the deciding points), CoBit-S is level at low entropy and DGC is 2-8 GenPPL "
        "better above entropy ~5.22 at the same NFE. CoBit-M's two grids never overlap in entropy at a given NFE: "
        "allowing any NFE &lt;= 512, the standard grid is better below entropy 5.34 (13.0 vs 14.7), DGC is better "
        "above 5.37 (14.7 vs 18.4), and only DGC reaches 5.40 (15.3) or the real-data entropy 5.44 (20.0). "
        "(2) <b>GSM8K: no gain.</b> At the standard settings every difference is within +/-2 points. DGC prefers "
        "less churn (0.3 instead of 0.41), but its best cell (35.2%) ties the standard grid's best (35.3%). "
        "(3) <b>Sudoku: one real gain</b>, hard puzzles with churn, 66.1% -&gt; 70.0% (+3.9 [+2.8, +5.0]). "
        "(4) <b>Mechanism: two effects pulling opposite ways.</b> On Sudoku the gain comes from where the churn "
        "acts (DGC's churn on the standard steps: +2.2; DGC's steps with the standard churn: -0.2). On OWT the "
        "extra entropy comes from where the steps are (DGC's steps with the standard churn give the highest "
        "entropy), while DGC's low-sigma churn lowers both entropy and GenPPL.", "key"))
    story.append(callout("<b>What changed and what didn't.</b> Only the sigma grid: the DGC step density "
                         "(Wainwright, arXiv:2607.26285), rebuilt from each checkpoint's own entropy-rate table by "
                         "scripts/dgc/build_dgc_tables.py, replaces the entropy-rate grid. Checkpoint, sampler, churn "
                         "gamma, NFE, seeds and metrics are those of each task's existing protocol. Because the churn "
                         "window is read from the same table, per-step gamma is unchanged but its placement follows "
                         "the grid. DGC moves steps from sigma &gt; 5 to sigma 0.2-1 and puts more steps below 0.17 "
                         "(TinyGSM 1024 steps: 7 -&gt; 47 below 0.17, 209 -&gt; 52 above 5; last real step 0.157 -&gt; 0.093)."))

    story += [H2("TinyGSM -> GSM8K (1319 problems, seed 0, EMA)"),
              table([["checkpoint", "setting", "standard %", "DGC %", "DGC - std (95% CI)", "invalid tok % (std / DGC)"]] + grows,
                    col_widths=[30 * mm, 50 * mm, 17 * mm, 15 * mm, 32 * mm, 26 * mm]),
              P("CoBit-S 1024-step standard-grid cells are the stored score-temperature cells "
                "(results/scoretemp); the gate reran the first 128 problems and matched every answer. "
                "CoBit-M's stored cells used the Karras grid, so both of its arms were run here.", "caption")]
    if gdel:
        story.append(figure(forest(gdel, "GSM8K: paired accuracy difference"), "dgc_gsm_forest", 150))

    story += [H2("Sudoku (2000 validation puzzles, 180 steps, raw weights, sigma_data 0.5)"),
              table([["difficulty", "sampler", "standard % (seeds 42-44)", "DGC % (seeds 42-44)", "DGC - std (95% CI)", "n"]] + srows,
                    col_widths=[18 * mm, 28 * mm, 40 * mm, 40 * mm, 36 * mm, 8 * mm]),
              P("Raw weights because the 20k-step EMA scores 0%; gamma 0.4 is where the stored gamma sweep "
                "plateaus. CI pools the three seeds' paired puzzles.", "caption")]
    if sdel:
        story.append(figure(forest(sdel, "Sudoku: paired exact-match difference"), "dgc_sudoku_forest", 150))

    story += [H2("OWT at the paper's gamma (1024 samples, seed 42, terminal sigma 0.08)"),
              table([["model", "NFE / gamma", "paper GenPPL / H", "standard GenPPL / H", "DGC GenPPL / H", "dGenPPL", "dH"]] + orows,
                    col_widths=[18 * mm, 22 * mm, 28 * mm, 32 * mm, 32 * mm, 18 * mm, 16 * mm]),
              P("Same pipeline for both arms, without the real-data pass (no OWT cache on Isambard); real OWT "
                "reference 15.07 / 5.44 from the paper. GenPPL is a corpus-level number from one seed: the "
                "paper's seed-to-seed sd is about 0.6 (CoBit-S, 256 NFE). Lower GenPPL at lower entropy is a "
                "move along the frontier, not an improvement, so read both columns. The mechanism section shows "
                "where DGC's extra entropy comes from: its step placement, not its churn placement.", "caption"),
              figure(owt_frontier(owt), "dgc_owt_frontier", 165,
                     "Each grey segment joins the two grids at the same NFE and gamma. Hollow blue points: the "
                     "standard grid at higher churn, from the paper (CoBit-S only).")]

    frows = fronts_rows(owt)
    if frows:
        story += [H2("OWT Pareto fronts: gamma swept on both grids"),
                  P("Both grids swept over gamma {0.05, 0.1, 0.2, 0.3, 0.41} at NFE 256, 384 and 512 (plus the "
                    "paper's gamma above), same pipeline, 1024 samples, seed 42 plus seeds 43-44 at the 24 points that "
                    "decide the comparison. The comparison that matters is "
                    "GenPPL at equal entropy: each grid's sweep is interpolated linearly in entropy and compared "
                    "at three entropies spanning the range both grids reach.", "body"),
                  figure(fronts_figure(owt), "dgc_owt_fronts", 165,
                         "Each line is one grid's gamma sweep at fixed NFE, joined in gamma order; the end points are labelled "
                         "with gamma."),
                  table([["model", "NFE", "entropy", "standard GenPPL", "DGC GenPPL", "DGC - std"]] + frows,
                        col_widths=[22 * mm, 14 * mm, 22 * mm, 30 * mm, 26 * mm, 22 * mm]),
                  P("Points with seeds 43-44 are 3-seed means; the rest are seed 42. Neighbouring points move by "
                    "about 1-2 GenPPL from seed noise alone.", "caption"),
                  H2("Best front over NFE <= 512"),
                  P("At fixed NFE the CoBit-M curves of the two grids do not overlap in entropy (gamma barely moves "
                    "CoBit-M's entropy), so the fair comparison is the best each grid can do at a required entropy, "
                    "allowing any swept NFE and gamma.", "body"),
                  table([["model", "entropy", "standard: GenPPL (NFE, gamma)", "DGC: GenPPL (NFE, gamma)"]] + best_front_rows(owt),
                        col_widths=[22 * mm, 22 * mm, 58 * mm, 58 * mm])]

    trows, tdel = gsm_rows(gsm, SCREEN_SETTINGS, (("s500", "CoBit-S unguided 500k"),))
    if any(r[3] != "pending" for r in trows):
        story += [H2("TinyGSM: DGC with re-tuned churn and temperature (CoBit-S unguided 500k)"),
                  table([["checkpoint", "setting", "standard %", "DGC %", "DGC - std (95% CI)", "invalid tok % (std / DGC)"]] + trows,
                        col_widths=[30 * mm, 50 * mm, 17 * mm, 15 * mm, 32 * mm, 26 * mm]),
                  P("Standard-grid partners are the stored score-temperature cells (blocks A and C), except tau 0.7, "
                    "which was run here on both grids.", "caption")]

    mrows_s, mrows_o = mech_sudoku_rows(sud), mech_owt_rows(owt)
    story += [H2("Mechanism: where the steps are vs where the churn acts"),
              P("With full-band churn every step gets the same gamma, so the churn per unit log-sigma follows the "
                "step density: moving the steps moves the churn. The swapped arms separate the two. The steps stay "
                "on one table's grid while gamma_i = gamma * p_other(sigma_i) / p_grid(sigma_i) (capped at "
                "sqrt(2)-1), so the churn density follows the other table (COBIT_CHURN_TABLE in DDIMSampler).", "body"),
              figure(churn_placement_figure(), "dgc_churn_placement", 160,
                     "Cumulative churn along the trajectory. The dotted line is sigma = 1: the DGC grid applies "
                     "most of its churn below it, the standard grid much less.")]
    if mrows_s:
        story += [table([["steps", "churn placement", "hard Sudoku % (seeds 42-44)", "vs standard/standard (95% CI)"]] + mrows_s,
                        col_widths=[22 * mm, 30 * mm, 60 * mm, 50 * mm]),
                  P("Hard Sudoku, churn gamma 0.15, chosen so the cap barely binds in the swapped arms.", "caption")]
    if mrows_o:
        story += [table([["gamma", "steps", "churn placement", "GenPPL", "entropy"]] + mrows_o,
                        col_widths=[18 * mm, 24 * mm, 32 * mm, 24 * mm, 24 * mm]),
                  P("OWT CoBit-S, 256 NFE, seed 42, 1024 samples.", "caption")]

    # ---------------- Part 2 (21 Oct, evening): everything at T = 1, appended; nothing above changes
    story += [PageBreak(), H1("Part 2: everything at T = 1"),
              P("Added 21 Oct (evening), 8.3 GPU-h (39.3 GPU-h for the whole study). No score temperature anywhere; each "
                "grid gets its own best churn, so the comparisons are best-vs-best. Everything above is unchanged.", "caption")]
    missing = [lab for lab, ok in (
        ("the LM1B standard-grid sweep at 384 NFE", bool(curve(owt, "l", 384, "cobit"))),
        ("the CoBit-M DGC cell at churn 0.1", ("m250", "dgc", 1.0, 0.1, 1024) in gsm)) if not ok]
    if PART2_FINDINGS:
        story.append(callout(PART2_FINDINGS + ("" if not missing else
                     " <b>Interim:</b> " + " and ".join(missing) + " were lost to compute-node faults and are "
                     "re-running; the best-front rows can still change."), "key"))
    t1rows, _ = gsm_rows(gsm, T1_SETTINGS)
    if any(r[3] != "pending" for r in t1rows):
        story += [H2("At T = 1: TinyGSM with churn re-tuned for each grid"),
                  table([["checkpoint", "setting", "standard %", "DGC %", "DGC - std (95% CI)", "invalid tok % (std / DGC)"]] + t1rows,
                        col_widths=[30 * mm, 50 * mm, 17 * mm, 15 * mm, 32 * mm, 26 * mm]),
                  P("No score temperature. Best-vs-best means each grid's best churn row, not the same row.", "caption")]
    swrows = sudoku_sweep_rows(sud)
    if swrows:
        story += [H2("At T = 1: Sudoku with gamma swept on both grids"),
                  table([["difficulty", "grid", "det.", "0.1", "0.2", "0.3", "0.4", "0.5", "best"]] + swrows,
                        col_widths=[20 * mm, 20 * mm, 14 * mm, 14 * mm, 14 * mm, 14 * mm, 14 * mm, 14 * mm, 30 * mm]),
                  P("Exact-match %, mean of seeds 42-44, 2000 puzzles, 180 steps, raw weights.", "caption")]

    lrows = fronts_rows(owt, LM1B, LM1B_NFES)
    if lrows:
        story += [H2("LM1B Pareto fronts: gamma swept on both grids, this pipeline"),
                  P("Released CoBit-S LM1B checkpoint, NFE 128-512 x gamma {0.05, 0.1, 0.2, 0.3, 0.41}, both grids, "
                    "seed 42, 1024 samples, GH200. Unlike the LM1B sweep in results/dgc/lm1b_sweeps (DGC only, compared "
                    "with the paper's points from another pipeline), both grids here come from one pipeline. The real-data "
                    "marker is the same-code measurement from that sweep (53.13 / 4.343); this cluster has no LM1B cache.", "body"),
                  figure(fronts_figure(owt, LM1B, LM1B_NFES, LM1B_REAL, "real LM1B"), "dgc_lm1b_fronts", 165,
                         "Each line is one grid's gamma sweep at fixed NFE, joined in gamma order."),
                  table([["model", "NFE", "entropy", "standard GenPPL", "DGC GenPPL", "DGC - std"]] + lrows,
                        col_widths=[28 * mm, 14 * mm, 22 * mm, 30 * mm, 26 * mm, 22 * mm]),
                  table([["model", "entropy", "standard: GenPPL (NFE, gamma)", "DGC: GenPPL (NFE, gamma)"]]
                        + best_front_rows(owt, LM1B), col_widths=[28 * mm, 22 * mm, 55 * mm, 55 * mm]),
                  P("One seed (42). Second table: lowest GenPPL among all swept points with at least that entropy.", "caption")]

    # ---------------- Part 3 (22 Oct): finite-NFE DP-DGC, appended; nothing above changes
    drows = dp_rows(owt)
    if drows:
        story += [PageBreak(), H1("Part 3: finite-NFE DP-DGC (CoBit-M OWT)"),
                  P("Added 22 Oct. Question: does solving the finite-NFE DGC allocation problem with a DP beat the "
                    "continuous DGC grid at low NFE? Only the sigma grid changes: same CoBit-M checkpoint, DGC table, "
                    "endpoints (76.76 to 0.08), DDIM sampler, churn gamma 0.2 at every NFE, T = 1, seed 42, 1024 "
                    "samples.", "caption")]
        if DP_FINDINGS:
            story.append(callout(DP_FINDINGS, "key"))
        story += [H2("The DP grid"),
                  P("K = 8 blocks uniform in log sigma^2. DP minimises sum_k G_k (exp(S_k / n_k) - 1) over integers "
                    "n_k >= 1 with sum n_k = N - 1 (the sampler's interval count), with S_k the block width in "
                    "log sigma^2 and G_k = m_k^2 / S_k, m_k the DGC table mass in the block, so the large-N optimum is "
                    "the continuous DGC allocation (n_k proportional to m_k). Each block is then filled with n_k "
                    "log-uniform steps. The reference notebook (DGC_five_delta_schedulers.ipynb) was not available; "
                    "S_k and G_k are this derivation, not the notebook's.", "body"),
                  table([["NFE", "intervals", "DP allocation (high sigma first)", "continuous share x intervals"]]
                        + dp_alloc_rows(), col_widths=[14 * mm, 18 * mm, 62 * mm, 72 * mm]),
                  figure(dp_schedule_figure(), "dgc_dp_schedules", 165,
                         "The DP grid keeps >= 1 step in each of the near-empty high-sigma blocks, where the "
                         "continuous grid jumps from 76 to about 2 in one step, and is log-uniform within blocks."),
                  H2("Model metric: GenPPL / entropy"),
                  table([["NFE", "standard", "continuous DGC", "DP-DGC (K=8)", "DP - continuous"]] + drows,
                        col_widths=[14 * mm, 34 * mm, 34 * mm, 34 * mm, 36 * mm]),
                  P("GenPPL (GPT-2-large) / unigram entropy, seed 42 for every cell, so the three grids share initial "
                    "noise. Standard and continuous-DGC cells at 256-512 NFE are the seed-42 cells of the Part 1 sweep. "
                    "One seed: about 1 GenPPL is within noise.", "caption")]

    # ---------------- Part 4 (22 Oct): DP-DGC with the notebook's G_k, appended; nothing above changes
    arms4 = DP_ARMS + (("dp_dgc_K8", "DP-DGC, notebook G_k"),)
    if any(s42(owt, ("m", n, 0.2, "dp_dgc_K8")) for n in DP_NFES):
        story += [PageBreak(), H1("Part 4: DP-DGC with the reference notebook's G_k"),
                  P("Added 22 Oct, after DGC_five_delta_schedulers.ipynb became available. Part 3's G_k was a derivation "
                    "(m_k^2 / S_k); the notebook defines G_k = 1/2 int_block h'(v) dr, the block's DGC increment. "
                    "With the table's density p(u) ∝ sqrt(h'), this is G_k ∝ int_block p(u)^2 du. Blocks, S_k, the DP and "
                    "the within-block geometric steps were already the notebook's. Same protocol as Part 3.", "caption")]
        if DP4_FINDINGS:
            story.append(callout(DP4_FINDINGS, "key"))
        story += [table([["NFE", "intervals", "DP allocation, notebook G_k (high sigma first)",
                          "Theorem-1 bound, continuous", "Theorem-1 bound, DP"]]
                        + dp_alloc_rows(weight="increment", bound=True),
                        col_widths=[13 * mm, 17 * mm, 62 * mm, 37 * mm, 37 * mm]),
                  P("Theorem-1 bound (the notebook's finite-step KL surrogate, sum over intervals of ((hi/lo)^2 - 1) G) "
                    "on each grid, up to the arbitrary scale of the table; the DP minimises it over blockwise-geometric "
                    "grids. It is a surrogate, not the model metric.", "caption"),
                  table([["NFE", "standard", "continuous DGC", "DP (Part 3 G_k)", "DP (notebook G_k)", "notebook DP - continuous"]]
                        + dp_rows(owt, arms=arms4),
                        col_widths=[12 * mm, 27 * mm, 27 * mm, 27 * mm, 29 * mm, 34 * mm]),
                  P("GenPPL / unigram entropy, seed 42, churn gamma 0.2, T = 1, 1024 samples.", "caption")]

    # ---------------- Part 5 (22 Oct): pass@1 at low NFE with DP-DGC on the tasks, appended
    g5, s5 = p5_gsm_rows(gsm), p5_sudoku_rows()
    if g5 or s5:
        story += [PageBreak(), H1("Part 5: pass@1 at low NFE on the tasks"),
                  P("Added 22 Oct. Can the finite-NFE DP grid (K = 8, the notebook's G_k) raise single-sample accuracy "
                    "at low step counts? Standard vs continuous DGC vs DP-DGC, T = 1. On these tables the grid runs to "
                    "sigma 0.002 and the DGC density is ~0 below sigma ~0.1, so the DP's n_k >= 1 puts single steps at "
                    "sigma ~0.03 and ~0.008 where the continuous grid jumps from ~0.1 to 0.002.", "caption")]
        if P5_FINDINGS:
            story.append(callout(P5_FINDINGS, "key"))
        if g5:
            story += [H2("GSM8K (1319 problems, seed 0, churn 0.41, EMA)"),
                      table([["model", "steps", "standard % (inv. tok %)", "continuous DGC %", "DP-DGC %",
                              "DP - continuous", "DP - standard"]] + g5,
                            col_widths=[17 * mm, 12 * mm, 31 * mm, 29 * mm, 26 * mm, 27 * mm, 27 * mm]),
                      P("Accuracy is pass@1 (one sample per problem); paired 95% CIs over problems. Invalid-token rate "
                        "in brackets. CoBit-S 256-step standard and continuous cells are Part 1's.", "caption")]
        if s5:
            story += [H2("Sudoku hard (2000 puzzles, gamma 0.5, raw weights, seeds 42-44)"),
                      table([["steps", "standard %", "continuous DGC %", "DP-DGC %", "DP - continuous", "DP - standard"]] + s5,
                            col_widths=[14 * mm, 25 * mm, 30 * mm, 25 * mm, 36 * mm, 36 * mm]),
                      P("Exact match, mean of three seeds; paired CIs pool the seeds' puzzles. The 180-step standard and "
                        "continuous cells are Part 2's.", "caption")]

    # ---------------- Part 6 (23 Oct): gamma-dependent Euler grid, appended
    story = part6(story)
    story = part7(story, gsm)   # Part 7 (23 Oct): TinyGSM gamma-Euler grids and the CoBit-M tau 0.3 seed

    story += [H2("Provenance"),
              bullets(["Launcher scripts/dgc/tasks_ab.sh (phases gate, main-tasks, main-owt, sweep-owt, seeds-owt, "
                       "tgsm-screen, mechanism); cell scripts scripts/hpc/arch/{scoretemp_eval,dgc_sudoku,dgc_owt}.slurm.",
                       "Swapped churn: COBIT_CHURN_TABLE in diffusion/continuous/samplers.py (DDIMSampler only; off "
                       "unless set). LM1B sweep from Georgios: results/dgc/lm1b_sweeps/.",
                       "Part 2: phases t1, lm1b-gate and lm1b-sweep; LM1B config configs/lm1b/continuous/eval/dgc_sweep.py "
                       "and launcher scripts/hpc/arch/dgc_lm1b.slurm. Jobs died at start-up on faulty compute nodes "
                       "(bus error or I/O error loading torch/libstdc++ on nid010003, nid010029, nid010033, nid010038, "
                       "nid010040) and were resubmitted with those nodes excluded (DGC_EXCLUDE); others were held at "
                       "launch and released.",
                       "Part 3: schedule dp_dgc (diffusion/continuous/dp_dgc.py; tests/test_dp_dgc.py), OWT config "
                       "SCHEDULE=dp_dgc, phases dp-smoke and dp-owt; schedules dumped by scripts/dgc/dp_dgc_schedules.py.",
                       "Part 5: phase dp-tasks (grid arm dpdgc in gsm8k and dgc_sudoku launchers; SU_STEPS for Sudoku).",
                       "Part 6: held-out data scripts/dgc/build_{lm1b,owt}_heldout_bits.py; profile "
                       "scripts/dgc/profile_forward_increments.py (diffusion/continuous/forward_profile.py); grids "
                       "scripts/dgc/build_gamma_grid.py (diffusion/continuous/gamma_grid.py); runs configs/gamma_euler_eval.py "
                       "via scripts/hpc/arch/gamma_euler_eval.slurm and manifests from scripts/dgc/gamma_euler_plan.sh; "
                       "samplers euler_lambda / ddim_grid; tests/test_gamma_euler.py. Outputs in results/dgc/gamma_euler/.",
                       "Part 7: held-out data scripts/dgc/build_tinygsm_heldout_bits.py; profile with --task; grids and "
                       "cells scripts/dgc/tinygsm_ge.sh (gsm8k_eval --grid_file); results results/dgc/gsm8k/*__ge_*.json.",
                       "Code was uncommitted at submission; diff and table hashes in logs/arch/dgc_provenance/.",
                       "DGC tables: assets/entropy_tables/{owt,owt_medium}_dgc and assets/entropy_tables/tasks/*_dgc."])]
    out = HERE / "report_dgc_sampler.pdf"
    build(out, story, "DGC sampling grid on CoBit tasks", "DGC vs entropy-rate sigma grid")
    print("wrote", out)


if __name__ == "__main__":
    main()
