#!/usr/bin/env python
"""
Compute demand on TinyGSM, stratified by problem difficulty (strata: <=2, 3-4, 5+ calculation steps).

Static (no sampling; the 4096 validation problems of runs/denoising_mse/tinygsm/static, answer region, training
objective = 1/2 (nosc + sc)):
  difficulty = computation lines in the solution code (an assignment whose right-hand side has an arithmetic operator)
  per stratum: entropy rate MSE/sigma^2 per model, surplus rate (MSE_S - MSE_M)/sigma^2, bootstrap bands over
  problems, and the surplus mass in each 32-step segment of the sampler (CoBit-S schedule, gamma 0.31, 256 steps)
GSM8K (compute_demand.py, per-problem correctness; difficulty = "<<...>>" calculations in the gold solution):
  A  accuracy gain of +2 SC refreshes per step in segment k over CoBit-S          (compute demand per segment)
  B  accuracy gain of CoBit-M in segment k only over CoBit-S, as a fraction of the full CoBit-M gain
     prefix / suffix switches (CoBit-M for the first / last c steps)
Per-problem results are averaged over the seeds available, then bootstrapped over problems (paired: the same
problems, initial noise and churn noise in every configuration).
Outputs: results_tinygsm/compute_demand/{fig_*.png, summary.json, *.csv}
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = REPO / "runs/denoising_mse/tinygsm"
OUT = HERE.parent / "results_tinygsm/compute_demand"
TAG = ""  # set by --tag (e.g. st0.6): analyse runs tagged _<tag> (compute_demand.py --score-temp-tau), output to *_<tag>
SIG_LO = 0.1  # as the TinyGSM analyses: excludes CoBit-M's sigma ~ 0.03 error spike
STRATA = [("≤2 steps", 0, 2), ("3–4 steps", 3, 4), ("5+ steps", 5, 99)]
SC = ["#2a78d6", "#eb6834", "#1baf7a"]
SEG, NSEG, NB = 32, 8, 2000
RNG = np.random.default_rng(0)
OPS = re.compile(r"^\s*[A-Za-z_]\w*\s*=\s*(.+)$")


def stratum(n):
    return next(k for k, (_, lo, hi) in enumerate(STRATA) if lo <= n <= hi)


def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def boot_mean(x, axis_n=0):
    """Mean and 95% bootstrap band over the leading axis (problems)."""
    n = x.shape[axis_n]
    idx = RNG.integers(0, n, size=(NB, n))
    bs = np.stack([x[i].mean(0) for i in idx[:400]])  # 400 resamples for curves (cheap enough, smooth)
    return x.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


# ------------------------------------------------------------------ static, validation problems
def n_calc_lines(code: str) -> int:
    body = code.split("'''")[-1] if code.count("'''") >= 2 else code
    n = 0
    for line in body.splitlines():
        m = OPS.match(line)
        if m and re.search(r"[-+*/%]", m.group(1)) and not re.fullmatch(r"\s*-?[\d.]+\s*", m.group(1)):
            n += 1
    return n


def schedule():
    meta = json.loads((RUNS / "traj_paired_SM/meta.json").read_text())
    return np.array(meta["sigmas"], dtype=float)


def static_part(summary):
    import common as C

    val = C.TinyGSMVal()
    tok = C.get_tokenizer()
    data = {}
    for m in ("S", "M"):
        fs = sorted((RUNS / "static" / m).glob("chunk*.npz"))
        zs = [np.load(f) for f in fs]
        data[m] = dict(idx=np.concatenate([z["idx"] for z in zs]), sig=zs[0]["sigmas"].astype(float),
                       mse=np.concatenate([0.5 * (z["mseC_nosc"] + z["mseC_sc"]) for z in zs], 1).T)  # [N, G]
    assert np.array_equal(data["S"]["idx"], data["M"]["idx"])
    idx, sig = data["S"]["idx"], data["S"]["sig"]
    calc = np.array([n_calc_lines(tok.decode(np.asarray(val.ids[i][val.plen[i]:val.plen[i] + val.alen[i]]).tolist()))
                     for i in idx])
    alen = np.asarray(val.alen)[idx]
    st = np.array([stratum(c) for c in calc])
    ls = np.log(sig)
    act = sig >= SIG_LO
    rS, rM = data["S"]["mse"] / sig ** 2, data["M"]["mse"] / sig ** 2
    sur = rS - rM
    sch = schedule()
    # sampler segment k covers sigma in [sigmas[32(k+1)], sigmas[32k]] (the last one down to sigma_min)
    seg_edges = [sch[min(SEG * k, len(sch) - 1)] for k in range(NSEG)] + [sch[-1]]

    def seg_mass(y):
        y = np.where(act, np.clip(y, 0, None), 0.0)
        out = []
        for k in range(NSEG):
            hi, lo = seg_edges[k], seg_edges[k + 1]
            g = np.linspace(np.log(lo), np.log(hi), 400)
            out.append(np.trapz(np.interp(g, ls, y), g))
        out = np.array(out)
        return out / out.sum()

    res = dict(n=int(len(idx)), calc_lines_hist=np.bincount(calc).tolist(), strata={})
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
    ymax = np.zeros(4)
    rows = []
    for k, (name, _, _) in enumerate(STRATA):
        sel = st == k
        mS, loS, hiS = boot_mean(rS[sel])
        mM, loM, hiM = boot_mean(rM[sel])
        mD, loD, hiD = boot_mean(sur[sel])
        for ax, (mu, lo, hi), ttl in zip(axs, ((mS, loS, hiS), (mM, loM, hiM), (mD, loD, hiD)),
                                         ("CoBit-S  MSE/σ²", "CoBit-M  MSE/σ²", "surplus  (MSE_S − MSE_M)/σ²")):
            ax.plot(sig, mu, color=SC[k], lw=1.6, label=f"{name} (n={sel.sum()}, {alen[sel].mean():.0f} answer tok)")
            ax.fill_between(sig, lo, hi, color=SC[k], alpha=0.2, lw=0)
            ax.set_title(ttl, loc="left")
        for j, y in enumerate((hiS, hiM, hiD)):
            ymax[j] = max(ymax[j], y[act].max())
        shape = np.where(act, np.clip(mD, 0, None), 0.0)
        shape = shape / np.trapz(shape, ls)
        axs[3].plot(sig, shape, color=SC[k], lw=1.6, label=name)
        ymax[3] = max(ymax[3], shape.max())
        axs[3].set_title("surplus shape (unit area over log σ)", loc="left")
        tot = lambda y: float(np.trapz(np.where(act, y, 0), ls))  # noqa: E731
        nbits = alen[sel].mean() * C.BPT
        sm = seg_mass(mD)
        res["strata"][name] = dict(
            n=int(sel.sum()), mean_answer_tokens=float(alen[sel].mean()), mean_calc_lines=float(calc[sel].mean()),
            rate_integral_S_per_bit=tot(mS), rate_integral_M_per_bit=tot(mM), surplus_integral_per_bit=tot(mD),
            surplus_integral_per_problem=tot(mD) * nbits,
            surplus_peak_sigma=float(sig[act][np.argmax(mD[act])]),
            surplus_segment_mass=sm.tolist(), rate_S_segment_mass=seg_mass(mS).tolist())
        for g in range(len(sig)):
            rows.append([name, f"{sig[g]:.6g}", f"{mS[g]:.6g}", f"{mM[g]:.6g}", f"{mD[g]:.6g}", f"{loD[g]:.6g}",
                         f"{hiD[g]:.6g}"])
    for j, ax in enumerate(axs):
        ax.set_xscale("log")
        ax.set_xlim(SIG_LO, sig[-1])
        ax.set_ylim(0, 1.08 * ymax[j])
        ax.set_xlabel("noise level σ")
        style(ax)
    axs[0].set_ylabel("per answer bit, per unit log σ")
    for e in seg_edges[1:-1]:
        axs[3].axvline(e, color="#c9c8c3", lw=0.6, ls=":")
    axs[2].legend(frameon=False, fontsize=8)
    fig.suptitle("TinyGSM validation, entropy rate and surplus rate by difficulty (computation lines in the "
                 "solution), answer region, training objective, 95% bootstrap bands", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_static_strata.png", dpi=160)
    plt.close(fig)
    with open(OUT / "static_strata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stratum", "sigma", "rate_S", "rate_M", "surplus", "surplus_lo", "surplus_hi"])
        w.writerows(rows)
    summary["static"] = res
    return res


# ------------------------------------------------------------------ GSM8K
def load_gsm8k():
    by = {}
    for f in glob.glob(str(RUNS / "compute_demand/*_seed*.json")):
        name = Path(f).stem
        if re.search(r"_seed\d+" + (f"_{re.escape(TAG)}" if TAG else "") + "$", name) is None:
            continue
        r = json.loads(Path(f).read_text())
        by.setdefault(r["config"], {})[r["seed"]] = np.array(r["per_problem_correct"], float)
        gold = np.array(r["gold_steps"])
    if not by:
        return None, None
    return by, gold


def per_problem(by, cfg, seeds):
    return np.mean([by[cfg][s] for s in seeds], 0)


def delta_ci(a, b):
    """Mean of b - a with a 95% paired bootstrap CI over problems."""
    d = b - a
    n = len(d)
    bs = d[RNG.integers(0, n, size=(NB, n))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def gsm8k_part(summary, static_res):
    by, gold = load_gsm8k()
    if by is None or "base" not in by:
        print("no GSM8K compute-demand results yet")
        return
    st = np.array([stratum(g) for g in gold])
    groups = [("all", np.ones_like(st, bool))] + [(n, st == k) for k, (n, _, _) in enumerate(STRATA)]
    res = {"strata_n": {n: int(m.sum()) for n, m in groups}}

    def seeds_for(cfgs):
        s = set(by["base"])
        for c in cfgs:
            s &= set(by.get(c, {}))
        return sorted(s)

    acc = {}
    for c in by:
        sd = seeds_for([c])
        acc[c] = dict(seeds=sd, **{n: float(per_problem(by, c, sd)[m].mean()) for n, m in groups})
    res["accuracy"] = acc

    # A: SC refreshes per segment
    segA = [f"sc2_seg{k}" for k in range(NSEG) if f"sc2_seg{k}" in by]
    # B: model switching per segment
    segB = [f"Mseg{k}" for k in range(NSEG) if f"Mseg{k}" in by]
    sch = schedule()
    tmid = np.array([(SEG * k + SEG / 2) / (len(sch) - 1) for k in range(NSEG)])  # fraction of steps

    def seg_deltas(cfgs):
        out = {}
        for n, m in groups:
            rows = []
            for c in cfgs:
                sd = seeds_for([c])
                rows.append(delta_ci(per_problem(by, "base", sd)[m], per_problem(by, c, sd)[m]))
            out[n] = np.array(rows)
        return out

    if segA or segB:
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
        for ax, cfgs, ttl in ((axs[0], segA, "A  +2 SC refreshes in one 32-step segment (CoBit-S, +64 NFE)"),
                              (axs[1], segB, "B  CoBit-M in one 32-step segment, CoBit-S elsewhere")):
            if not cfgs:
                continue
            ks = np.array([int(c[-1]) for c in cfgs])
            d = seg_deltas(cfgs)
            for j, (n, _) in enumerate(groups):
                col = "#1b1b1a" if n == "all" else SC[j - 1]
                off = (j - 1.5) * 0.08
                ax.errorbar(ks + off, 100 * d[n][:, 0], yerr=100 * np.abs(d[n][:, 1:].T - d[n][:, 0]), fmt="o-",
                            color=col, ms=3.5, lw=1.4 if n == "all" else 1.0, capsize=2,
                            label=f"{n} (n={res['strata_n'][n]})")
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.set_xticks(range(NSEG))
            ax.set_xticklabels([f"{k}\n{SEG * k}–{min(SEG * (k + 1), len(sch)) - 1}" for k in range(NSEG)], fontsize=7)
            ax.set_xlabel("segment (steps; 0 = pure noise)")
            ax.set_ylabel("Δ GSM8K accuracy vs CoBit-S [pp]")
            ax.set_title(ttl, loc="left")
            style(ax)
            res["A" if cfgs is segA else "B"] = {n: dict(segments=ks.tolist(), delta=v[:, 0].tolist(),
                                                         lo=v[:, 1].tolist(), hi=v[:, 2].tolist())
                                                 for n, v in d.items()}
        axs[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"GSM8K (1319, γ=0.31, 256 steps, CoBit-S schedule), where extra compute / capacity pays off, "
                     f"by difficulty (gold calculation steps); paired 95% CIs; seeds {seeds_for(segA + segB)}",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "fig_gsm8k_segments.png", dpi=160)
        plt.close(fig)

    # B vs static surplus: fraction of the full CoBit-M gain recovered per segment vs surplus mass per segment
    if segB and "M" in by and static_res is not None:
        fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), sharey=True)
        res["B_fraction"] = {}
        for j, (n, m) in enumerate(groups):
            ax = axs[j]
            sd = seeds_for(segB + ["M"])
            base = per_problem(by, "base", sd)[m]
            full = per_problem(by, "M", sd)[m].mean() - base.mean()
            fr = np.array([(per_problem(by, c, sd)[m].mean() - base.mean()) / full for c in segB])
            ks = np.array([int(c[-1]) for c in segB])
            ax.bar(ks - 0.2, fr, width=0.4, color="#2a78d6", label="share of full CoBit-M gain (GSM8K)")
            if n == "all":
                sm = np.mean([static_res["strata"][s]["surplus_segment_mass"] for s, _, _ in STRATA], 0)
            else:
                sm = np.array(static_res["strata"][n]["surplus_segment_mass"])
            ax.bar(np.arange(NSEG) + 0.2, sm, width=0.4, color="#9a9994",
                   label="share of static surplus (MSE_S − MSE_M)/σ², val")
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.set_xticks(range(NSEG))
            ax.set_xlabel("segment (0 = pure noise)")
            ax.set_title(f"{n}: full gain {100 * full:+.1f} pp, Σ segments {100 * fr.sum() * full:+.1f} pp",
                         loc="left", fontsize=9)
            style(ax)
            res["B_fraction"][n] = dict(full_gain=float(full), fraction=fr.tolist(), sum_fraction=float(fr.sum()),
                                        static_surplus_share=np.asarray(sm).tolist())
        axs[0].set_ylabel("share")
        axs[0].legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        fig.savefig(OUT / "fig_gsm8k_switch_vs_surplus.png", dpi=160)
        plt.close(fig)

    # prefix / suffix switches
    pre = [c for c in ("MthenS64", "MthenS128", "MthenS192") if c in by]
    suf = [c for c in ("SthenM64", "SthenM128", "SthenM192") if c in by]
    if (pre or suf) and "M" in by:
        fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), sharey=False)
        res["prefix_suffix"] = {}
        for j, (n, m) in enumerate(groups):
            ax = axs[j]
            sd = seeds_for(pre + suf + ["M"])
            a = lambda c: 100 * per_problem(by, c, sd)[m].mean()  # noqa: E731
            xs_pre = [0] + [int(c[6:]) for c in pre] + [256]
            ys_pre = [a("base")] + [a(c) for c in pre] + [a("M")]
            xs_suf = [0] + [256 - int(c[6:]) for c in suf][::-1] + [256]
            ys_suf = [a("base")] + [a(c) for c in suf][::-1] + [a("M")]
            ax.plot(xs_pre, ys_pre, "o-", color="#2a78d6", label="CoBit-M first (high noise), then CoBit-S")
            ax.plot(xs_suf, ys_suf, "s-", color="#eb6834", label="CoBit-S first, then CoBit-M (low noise)")
            ax.set_xticks([0, 64, 128, 192, 256])
            ax.set_xlabel("steps run by CoBit-M")
            ax.set_title(f"{n} (n={int(m.sum())})", loc="left", fontsize=9)
            style(ax)
            res["prefix_suffix"][n] = dict(m_first=dict(zip(map(str, xs_pre), ys_pre)),
                                           m_last=dict(zip(map(str, xs_suf), ys_suf)))
        axs[0].set_ylabel("GSM8K accuracy [%]")
        axs[0].legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        fig.savefig(OUT / "fig_gsm8k_prefix_suffix.png", dpi=160)
        plt.close(fig)

    with open(OUT / "gsm8k_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "seeds"] + [n for n, _ in groups])
        for c in sorted(acc):
            w.writerow([c, " ".join(map(str, acc[c]["seeds"]))] + [f"{100 * acc[c][n]:.2f}" for n, _ in groups])
    summary["gsm8k"] = res


def main() -> None:
    import argparse

    global OUT, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="run tag, e.g. st0.6 for the score-temperature switching study")
    TAG = ap.parse_args().tag
    if TAG:
        OUT = OUT.parent / f"compute_demand_{TAG}"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    summary = {}
    sr = static_part(summary)
    gsm8k_part(summary, sr)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.get("static", {}).items() if k != "strata"}, indent=1))
    for n, v in summary.get("static", {}).get("strata", {}).items():
        print(n, {k: (round(x, 4) if isinstance(x, float) else x if isinstance(x, int) else [round(y, 3) for y in x]) for k, x in v.items()})
    if "gsm8k" in summary:
        print(json.dumps(summary["gsm8k"].get("accuracy"), indent=0)[:3000])


if __name__ == "__main__":
    main()
