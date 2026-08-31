#experiments/guidance/analyse.py
"""Statistics and figures for the guidance study, from the aggregated CSVs.

Every number and every panel in the report is produced here, from
`all_cells.csv` / `per_step.csv` written by `experiments.guidance.aggregate`.
Nothing is hand-edited, so a rerun after new cells land regenerates the whole
analysis consistently.

    python -m experiments.guidance.analyse results/guidance --out results/guidance/figures

Statistical stance
------------------
Accuracy on a fixed problem set is a sum of Bernoulli outcomes on *the same
problems* across methods, so comparisons are paired. Where per-problem outcomes
are available we use a paired bootstrap over problems, which removes
problem-difficulty variance and is far tighter than comparing two independent
intervals. Where only aggregate accuracy is available we fall back to a Wilson
interval and say so. No improvement is claimed from overlapping intervals
alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def read_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def f(row: Dict, key: str) -> Optional[float]:
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval -- correct near 0 and 1, unlike the normal approx."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap(a: Sequence[int], b: Sequence[int], n_boot: int = 10000,
                     seed: int = 0) -> Dict[str, float]:
    """Bootstrap the paired difference mean(b) - mean(a) over problems.

    Resamples *problems* (not outcomes), keeping each method's result on a given
    problem together, so the interval reflects only the method difference.
    """
    if len(a) != len(b) or not a:
        return {}
    rng = random.Random(seed)
    n = len(a)
    diffs = [b[i] - a[i] for i in range(n)]
    obs = sum(diffs) / n
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    # Two-sided bootstrap p-value for "no difference".
    p_val = 2 * min(
        sum(1 for x in boots if x <= 0) / n_boot,
        sum(1 for x in boots if x >= 0) / n_boot,
    )
    return {
        "delta": obs, "ci95_low": lo, "ci95_high": hi,
        "p_value": min(1.0, p_val), "n_pairs": n,
        "n_discordant": sum(1 for d in diffs if d != 0),
    }


def per_problem_outcomes(result_file: str) -> Optional[List[int]]:
    """Per-problem correctness, if the result JSON recorded it.

    `sample_records` is capped at 100 entries by the evaluator, so this is only
    usable for paired tests when the evaluation set is that small; otherwise the
    analysis falls back to unpaired intervals and labels the comparison as such.
    """
    try:
        r = json.loads(Path(result_file).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    recs = r.get("sample_records") or []
    if not recs or len(recs) < int(r.get("num_examples") or 0):
        return None
    return [1 if x.get("correct") else 0 for x in sorted(recs, key=lambda z: z["idx"])]


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _errorbar(ax, xs, ys, los, his, label, marker="o"):
    yerr = [[y - lo for y, lo in zip(ys, los)], [hi - y for y, hi in zip(ys, his)]]
    ax.errorbar(xs, ys, yerr=yerr, marker=marker, capsize=3, label=label, linewidth=1.5)


def plot_scale_curve(rows: List[Dict], key: str, out: Path, *, title: str, xlabel: str):
    """Accuracy vs a guidance scale, with intervals. One line per bad checkpoint
    where relevant, so AutoGuidance's badness axis is visible."""
    plt = _plt()
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        w = f(r, key)
        if w is None:
            continue
        g = r.get("bad_step") or "-"
        groups[str(g)].append(r)
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for g, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: f(r, key) or 0)
        xs = [f(r, key) for r in rs]
        ys = [f(r, "accuracy") for r in rs]
        los = [f(r, "ci95_low") or y for r, y in zip(rs, ys)]
        his = [f(r, "ci95_high") or y for r, y in zip(rs, ys)]
        if any(v is None for v in ys):
            continue
        _errorbar(ax, xs, ys, los, his, f"bad@{g}" if g != "-" else "accuracy")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(groups) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analyse] {out}")


def plot_ag_heatmap(rows: List[Dict], out: Path):
    """Bad-checkpoint quality x AG scale. The question is whether there is an
    *optimal* badness, so both axes must be shown together."""
    plt = _plt()
    cells = [(r.get("bad_step"), f(r, "ag_scale"), f(r, "accuracy"))
             for r in rows if f(r, "ag_scale")]
    cells = [(int(b), w, a) for b, w, a in cells if b not in (None, "", "None") and a is not None]
    if not cells:
        return
    bads = sorted({b for b, _, _ in cells})
    ws = sorted({w for _, w, _ in cells})
    grid = [[float("nan")] * len(ws) for _ in bads]
    for b, w, a in cells:
        grid[bads.index(b)][ws.index(w)] = a

    fig, ax = plt.subplots(figsize=(1.1 * len(ws) + 3, 0.7 * len(bads) + 2.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(ws)), [f"{w:g}" for w in ws])
    ax.set_yticks(range(len(bads)), [f"{b//1000}k" for b in bads])
    ax.set_xlabel("AutoGuidance scale $w_{ag}$")
    ax.set_ylabel("bad checkpoint (training step)")
    ax.set_title("AutoGuidance: badness x scale")
    for i in range(len(bads)):
        for j in range(len(ws)):
            if not math.isnan(grid[i][j]):
                ax.text(j, i, f"{grid[i][j]:.3f}", ha="center", va="center",
                        fontsize=7, color="w")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analyse] {out}")


def plot_quality_vs_compute(rows: List[Dict], out: Path):
    """The Pareto view. Guidance methods cost different numbers of model
    evaluations per step, so equal NFE is not equal compute -- plotting against
    measured cost is what makes the comparison fair."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, xkey, xlabel, logx in (
        (axes[0], "steps", "sampler steps (NFE)", True),
        (axes[1], "nfe_total", "denoiser evaluations / sample", True),
        (axes[2], "efficiency.wall_clock_s", "wall clock (s)", True),
    ):
        by_method: Dict[str, List[Dict]] = defaultdict(list)
        for r in rows:
            if f(r, xkey) and f(r, "accuracy") is not None:
                by_method[r.get("method", "?")].append(r)
        for m, rs in sorted(by_method.items()):
            rs = sorted(rs, key=lambda r: f(r, xkey))
            ax.plot([f(r, xkey) for r in rs], [f(r, "accuracy") for r in rs],
                    marker="o", label=m, linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy")
        if logx:
            ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Quality vs compute")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analyse] {out}")


def plot_diversity_quality(rows: List[Dict], out: Path):
    """Guidance buys quality with diversity; this is where that price is read."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ykey, ylabel in ((axes[0], "diversity.distinct_2", "distinct-2"),
                             (axes[1], "diversity.token_entropy", "token entropy (nats)")):
        by_method: Dict[str, List[Dict]] = defaultdict(list)
        for r in rows:
            if f(r, ykey) is not None and f(r, "accuracy") is not None:
                by_method[r.get("method", "?")].append(r)
        for m, rs in sorted(by_method.items()):
            ax.scatter([f(r, ykey) for r in rs], [f(r, "accuracy") for r in rs],
                       label=m, s=28, alpha=0.8)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Diversity vs quality")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analyse] {out}")


def plot_vs_sigma(per_step: List[Dict], out_dir: Path):
    """Trajectory diagnostics as a function of sigma.

    Several questions are about shape, not average: does self-guidance
    destabilise at low sigma? does the guidance/score ratio blow up? does strong
    guidance saturate the bit posterior?
    """
    plt = _plt()
    panels = [
        ("bit_entropy_mean", "mean bit entropy (nats)"),
        ("guidance_over_score", "||guidance|| / ||score||"),
        ("cfg_dir_rms", "||CFG direction||"),
        ("ag_dir_rms", "||AG direction||"),
        ("sg_dir_rms", "||SG direction||"),
        ("frac_p_gt_0.99", "fraction p > 0.99"),
    ]
    by_method: Dict[str, List[Dict]] = defaultdict(list)
    for r in per_step:
        by_method[r.get("method", "?")].append(r)
    if not by_method:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (key, label) in zip(axes.ravel(), panels):
        plotted = False
        for m, rs in sorted(by_method.items()):
            pts = [(f(r, "sigma"), f(r, key)) for r in rs]
            pts = sorted((s, v) for s, v in pts if s is not None and v is not None)
            if not pts:
                continue
            ax.plot([s for s, _ in pts], [v for _, v in pts], marker=".",
                    linewidth=1.2, markersize=3, label=m)
            plotted = True
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if not plotted:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Guidance diagnostics vs noise level")
    fig.tight_layout()
    out = out_dir / "diagnostics_vs_sigma.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analyse] {out}")


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------

def paired_table(rows: List[Dict], baseline_method: str = "baseline") -> List[Dict]:
    """Paired comparison of each cell against the matching baseline.

    Cells are matched on everything that is not the guidance policy (sampler,
    NFE, seed, evaluation size), so the difference is attributable to guidance
    and nothing else.
    """
    def key(r):
        return (r.get("sampler"), r.get("steps"), r.get("seed"),
                r.get("num_examples"), r.get("gamma"))

    baselines = {key(r): r for r in rows if r.get("method") == baseline_method}
    out = []
    for r in rows:
        if r.get("method") == baseline_method:
            continue
        b = baselines.get(key(r))
        if b is None:
            continue
        rec = {
            "method": r.get("method"),
            "cfg_scale": r.get("guidance_scale"),
            "ag_scale": r.get("ag_scale"),
            "sg_scale": r.get("sg_scale"),
            "sg_variant": r.get("sg_variant"),
            "bad_step": r.get("bad_step"),
            "steps": r.get("steps"),
            "n": r.get("num_examples"),
            "acc": f(r, "accuracy"),
            "acc_baseline": f(b, "accuracy"),
        }
        rec["delta"] = (rec["acc"] or 0) - (rec["acc_baseline"] or 0)

        oa = per_problem_outcomes(b.get("result_file", ""))
        ob = per_problem_outcomes(r.get("result_file", ""))
        if oa and ob and len(oa) == len(ob):
            rec.update({f"paired_{k}": v for k, v in
                        paired_bootstrap(oa, ob).items()})
            rec["comparison"] = "paired_bootstrap"
        else:
            n = int(rec["n"] or 0)
            _, lo1, hi1 = wilson(int(round((rec["acc"] or 0) * n)), n)
            _, lo0, hi0 = wilson(int(round((rec["acc_baseline"] or 0) * n)), n)
            rec.update({"acc_lo": lo1, "acc_hi": hi1,
                        "baseline_lo": lo0, "baseline_hi": hi0})
            # Only assert separation when the intervals genuinely do not overlap.
            rec["separated"] = (lo1 > hi0) or (lo0 > hi1)
            rec["comparison"] = "unpaired_wilson"
        out.append(rec)
    return sorted(out, key=lambda x: -(x.get("delta") or 0))


def print_paired(table: List[Dict]) -> None:
    if not table:
        print("[analyse] no paired comparisons available")
        return
    print(f"\n{'method':18} {'w':>5} {'ag':>4} {'sg':>4} {'nfe':>5} "
          f"{'acc':>7} {'base':>7} {'delta':>8}  {'95% CI / sep':>22} {'test'}")
    print("-" * 108)
    for r in table:
        if r.get("comparison") == "paired_bootstrap":
            ci = f"[{r.get('paired_ci95_low', 0):+.4f},{r.get('paired_ci95_high', 0):+.4f}]"
        else:
            ci = "separated" if r.get("separated") else "overlapping"
        print(f"{r['method']:18} {_g(r['cfg_scale']):>5} {_g(r['ag_scale']):>4} "
              f"{_g(r['sg_scale']):>4} {_g(r['steps']):>5} "
              f"{(r['acc'] or 0):>7.4f} {(r['acc_baseline'] or 0):>7.4f} "
              f"{(r['delta'] or 0):>+8.4f}  {ci:>22} {r.get('comparison')}")


def factorial_effects(rows: List[Dict]) -> Dict[str, float]:
    """Main effects and interactions from the 2x2x2 factorial cells.

    Estimated as differences of cell means over the factorial design, which is
    the standard effect decomposition; with one observation per cell there is no
    within-cell error term, so these are point estimates only and are reported
    as such.
    """
    cells = {}
    for r in rows:
        if not str(r.get("result_file", "")).count("fact"):
            continue
        c = (float(r.get("guidance_scale") or 0) > 0,
             float(r.get("ag_scale") or 0) > 0,
             float(r.get("sg_scale") or 0) != 0)
        a = f(r, "accuracy")
        if a is not None:
            cells[c] = a
    if len(cells) < 8:
        return {}

    def m(**fix) -> float:
        sel = [v for k, v in cells.items()
               if all(k[i] == fix[n] for i, n in enumerate(("cfg", "ag", "sg")) if n in fix)]
        return sum(sel) / len(sel) if sel else float("nan")

    eff = {
        "main_cfg": m(cfg=True) - m(cfg=False),
        "main_ag": m(ag=True) - m(ag=False),
        "main_sg": m(sg=True) - m(sg=False),
        "int_cfg_ag": (m(cfg=True, ag=True) - m(cfg=True, ag=False)) -
                      (m(cfg=False, ag=True) - m(cfg=False, ag=False)),
        "int_cfg_sg": (m(cfg=True, sg=True) - m(cfg=True, sg=False)) -
                      (m(cfg=False, sg=True) - m(cfg=False, sg=False)),
        "int_ag_sg": (m(ag=True, sg=True) - m(ag=True, sg=False)) -
                     (m(ag=False, sg=True) - m(ag=False, sg=False)),
    }
    three = ((cells[(True, True, True)] - cells[(True, True, False)]) -
             (cells[(True, False, True)] - cells[(True, False, False)])) - \
            ((cells[(False, True, True)] - cells[(False, True, False)]) -
             (cells[(False, False, True)] - cells[(False, False, False)]))
    eff["int_cfg_ag_sg"] = three
    return eff


def _g(v) -> str:
    try:
        x = float(v)
        return f"{int(x)}" if x == int(x) else f"{x:g}"
    except (TypeError, ValueError):
        return "-"


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="directory containing all_cells.csv / per_step.csv")
    ap.add_argument("--out", default=None, help="directory for figures (default <results>/figures)")
    args = ap.parse_args()

    root = Path(args.results)
    rows = read_csv(root / "all_cells.csv")
    per_step = read_csv(root / "per_step.csv")
    if not rows:
        raise SystemExit(f"no all_cells.csv in {root}; run experiments.guidance.aggregate first")

    out = Path(args.out or (root / "figures"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"[analyse] {len(rows)} cells, {len(per_step)} per-step records")

    cfg_rows = [r for r in rows if f(r, "guidance_scale") is not None
                and not f(r, "ag_scale") and not f(r, "sg_scale")]
    plot_scale_curve(cfg_rows, "guidance_scale", out / "cfg_scale.png",
                     title="Classifier-free guidance", xlabel="CFG scale $w$")

    ag_rows = [r for r in rows if f(r, "ag_scale")]
    plot_scale_curve(ag_rows, "ag_scale", out / "ag_scale.png",
                     title="AutoGuidance", xlabel="AG scale $w_{ag}$")
    plot_ag_heatmap(ag_rows, out / "ag_heatmap.png")

    sg_rows = [r for r in rows if f(r, "sg_scale")]
    plot_scale_curve(sg_rows, "sg_scale", out / "sg_scale.png",
                     title="Self-guidance", xlabel="SG scale $w_{sg}$")

    plot_quality_vs_compute(rows, out / "quality_vs_compute.png")
    plot_diversity_quality(rows, out / "diversity_vs_quality.png")
    plot_vs_sigma(per_step, out)

    table = paired_table(rows)
    print_paired(table)
    (root / "paired_comparisons.json").write_text(json.dumps(table, indent=2))
    print(f"[analyse] wrote {root / 'paired_comparisons.json'}")

    eff = factorial_effects(rows)
    if eff:
        print("\n=== factorial effects (accuracy, point estimates) ===")
        for k, v in eff.items():
            print(f"  {k:16} {v:+.4f}")
        (root / "factorial_effects.json").write_text(json.dumps(eff, indent=2))
    else:
        print("\n[analyse] factorial grid incomplete; interaction effects not estimated")


if __name__ == "__main__":
    main()
