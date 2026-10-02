"""Report 3: temporal ordering of denoising (L2R, R2L, random). CPU only.

Tables and charts are read from results/ordering/*.json; nothing is re-run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdfkit_common import (  # noqa: E402
    BAD, GOOD, H1, H2, HERE, INK2, MUTED, SERIES, KeepTogether, P, PageBreak, Spacer, build,
    bullets, callout, figure, mm, mpl_style, plt, table,
)

REPO = HERE.parents[1]
R = REPO / "results" / "ordering"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def load_window():
    w = json.loads((R / "window_summary.json").read_text())
    own = {}
    for r in w:
        if r["decoder"] == "own" and r["nfe"] == 256:
            own[(r["mode"], float(r["k"]), r["step"])] = r
    std = {(r["mode"], float(r["k"]), r["step"]): r for r in w if r["decoder"] == "standard"}
    return own, std


def welch(a, b):
    """Welch interval on the difference of two 2-seed means (as analyze_window.py)."""
    from scipy import stats
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else 1
    t = stats.t.ppf(0.975, df)
    d = a.mean() - b.mean()
    return d, d - t * se, d + t * se


def seeds(r):
    return [r["acc_seed"][s] for s in sorted(r["acc_seed"])]


def fig_clocks():
    """Illustrative: per-token noise level t_j under window l2r, k=16."""
    L, k = 60, 16
    pos = np.arange(L)
    r = L - 1 - pos                       # l2r: first token gets the highest priority
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    taus = [0.9, 0.7, 0.5, 0.3, 0.1]
    ramp = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]
    for tau, c in zip(taus, ramp):
        t = np.clip(tau * (1 + L / k) - r / k, 0, 1)
        ax.plot(pos, t, color=c, lw=2)
        x = int(np.argmax(t > 0.5)) if (t > 0.5).any() else L - 1
        ax.text(min(x + 1, L - 6), 0.52, f"τ={tau}", color=INK2, fontsize=7)
    ax.set_xlabel("token position in the answer (left → right)")
    ax.set_ylabel("noise level t (1 = pure noise)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Window L2R ordering, k = 16: a band of ~16 tokens is being denoised at any time")
    return fig


def fig_window_curve(own):
    steps = sorted({s for (_, _, s) in own})
    arms = [("none", float("inf"), "uniform control (no ordering)", MUTED, "-"),
            ("l2r", 64, "L2R, k=64", SERIES[0], "-"),
            ("random", 64, "random, k=64", SERIES[2], "-"),
            ("r2l", 64, "R2L, k=64", SERIES[1], "-")]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for mode, k, lab, c, ls in arms:
        y = [100 * own[(mode, k, s)]["acc"] for s in steps]
        lw = 2.4 if mode == "none" else 2.0
        ax.plot([s / 1000 for s in steps], y, color=c, ls=ls, lw=lw, marker="o", ms=4)
        ax.text(steps[-1] / 1000 + 4, y[-1], f"{lab}  {y[-1]:.1f}%", color=INK2, fontsize=7,
                va="center")
    lm = [100 * own[("lm64", float("inf"), s)]["acc"] for s in steps]
    ax.plot([s / 1000 for s in steps], lm, color="#b5b4ae", ls="--", lw=1.6)
    ax.text(steps[-1] / 1000 + 4, lm[-1] - 0.35, f"supervision-matched control  {lm[-1]:.1f}%",
            color=MUTED, fontsize=7, va="center")
    ax.set_xlim(40, 330)
    ax.set_xticks([s / 1000 for s in steps])
    ax.set_xlabel("training steps (thousands)")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("Window study, k = 64: L2R is the best order, but the unordered model stays ahead")
    return fig


def main():
    mpl_style()
    own, std = load_window()
    conf = json.loads((R / "confirm_summary.json").read_text())
    curve = json.loads((R / "train_curve_gh200.json").read_text())

    st = []
    st.append(P("Time-ordering of denoising: L2R, R2L and random", "title"))
    st.append(P("TinyGSM → GSM8K, CoBit bits model. What we tried, how, and whether it beats "
                "denoising every token at once. Report written 2 Oct 2026 from results on disk; "
                "no GPU was used.", "subtitle"))

    st.append(callout(
        "<b>Bottom line.</b> No ordering beat plain simultaneous denoising at any budget we tested. "
        "Ordering at decode time only (on a normally trained model) does nothing or hurts. "
        "When the model is <i>trained</i> with an order, <b>left-to-right is clearly the best "
        "order</b> (better than random, much better than right-to-left), and it learns 2–6× more per "
        "supervised token than a fair control. But the ordinary unordered model still wins at the "
        "same number of steps: <b>9.6% vs 8.0%</b> at 250k steps, and the gap was not closing."))

    # ------------------------------------------------------------------ idea
    st.append(H1("1. The idea, in plain words"))
    st.append(P("Normally CoBit denoises all 512 tokens of the answer together: at every step "
                "every token sits at the same noise level σ. An autoregressive model instead "
                "writes left to right. The hypothesis was that giving each token its own clock, "
                "so that earlier tokens become clean before later ones, would let the model "
                "condition on a settled beginning, like an AR model, and improve reasoning."))
    st.append(P("<b>How it is implemented.</b> Each token j gets its own noise level σ<sub>j</sub> "
                "(all 16 bits of a token share it). A priority decides who goes first:"))
    st.append(bullets([
        "<b>L2R</b>: the first answer token has the highest priority (resolves first).",
        "<b>R2L</b>: the last position resolves first.",
        "<b>random</b>: a fresh random order for every example.",
        "<b>uniform control</b>: no ordering, every token at the same σ (the normal model).",
    ]))
    st.append(P("Two versions of the clock were used:"))
    st.append(table([
        ["Design", "Formula", "Used in", "Problem"],
        ["<b>Spread w</b> (old)", "t<sub>j</sub> = clip(t(1+w) − w·u<sub>j</sub>), u<sub>j</sub> ∈ [0,1] "
         "= priority normalised over the whole generated suffix",
         "decode-only tests; w=0.25 training to 500k",
         "The suffix is ~74% padding, so the real answer occupied a tiny slice of u: at w=0.25 "
         "93–99% of answer tokens were active at once. It never really ordered the answer."],
        ["<b>Window k</b> (new)", "t<sub>j</sub> = clip(τ(1+L/k) − r<sub>j</sub>/k), r<sub>j</sub> = "
         "absolute rank in tokens; σ<sub>j</sub> = F<super>−1</super>(t<sub>j</sub>) through one frozen "
         "noise curve shared by all arms",
         "window study, 26 runs", "Fixes the scale: about k tokens are \"in flight\" at a time; "
         "k=∞ is the uniform model."],
    ], col_widths=[24 * mm, 52 * mm, 34 * mm, 60 * mm]))
    st.append(Spacer(1, 4))
    st.append(figure(fig_clocks(), "ord_clocks", 150,
                     "Illustration (computed from the window formula, not data): as the global time "
                     "τ runs from 1 to 0, a band about k tokens wide moves from left to right; tokens "
                     "to its left are already clean, tokens to its right are still noise."))
    st.append(P("In the window design, training puts loss only on tokens inside the moving band, "
                "so an ordered model sees far fewer supervised token–noise pairs per step than the "
                "uniform model (about 56 vs 462 per example at k=64, 4 at k=4). To separate "
                "<i>order</i> from <i>less supervision</i>, we added <b>supervision-matched controls</b> "
                "(\"lm\"): uniform σ, but each token supervised with the same probability as a window "
                "arm. Every comparison uses the same number of network evaluations (256)."))

    # ------------------------------------------------------------------ experiments
    st.append(H1("2. What we ran, in order"))
    st.append(H2("A. Ordering only at decode time, on the production model (8 Sep)"))
    st.append(P("Question: can we just decode the normally trained 500k model in L2R order? "
                "Full test set (1319 problems), 3 seeds, 256 steps, paired comparison."))
    by = {}
    for r in conf:
        by.setdefault(r["arm"], []).append(r)
    rows = [["Arm", "Seed 0", "Seed 1", "Seed 2", "Mean change vs control", "Significant seeds"]]
    ctrl = sorted({(r["seed"], r["control_accuracy"]) for r in conf})
    rows.append(["control (no order)"] + [pct(a, 2) + "%" for _, a in ctrl] + ["—", "—"])
    for arm in ("l2r w=0.1", "random w=0.1"):
        rs = sorted(by[arm], key=lambda r: r["seed"])
        md = np.mean([r["delta"] for r in rs])
        rows.append([arm.replace("l2r", "L2R")] + [pct(r["accuracy"], 2) + "%" for r in rs]
                    + [f"{100 * md:+.2f} pp", f"{sum(r['significant'] for r in rs)}/3"])
    rows.append(["L2R or random, w ≥ 0.25", "0.0%", "", "", "collapses to 0", "(n=250 screen)"])
    st.append(table(rows, col_widths=[38 * mm, 18 * mm, 18 * mm, 18 * mm, 40 * mm, 30 * mm]))
    st.append(P("Verdict: <b>closed, negative.</b> A model trained with one shared σ does not "
                "tolerate different σ per token: a small spread does nothing, random order hurts "
                "(−3.4 pp, every seed), and a larger spread breaks generation entirely. "
                "(An early run of this test scored 0.000 everywhere because of a sampler bug that "
                "returned the second-to-last denoise; it was caught by the production anchor and "
                "fixed before these numbers.)", "body"))

    st.append(H2("B. Fine-tuning the production model with ordering, w=0.25 (8–10 Sep)"))
    st.append(P("Inconclusive and not used. At lr 3e-4 the runs diverged; at 3e-5 the control "
                "arm itself degenerated to 0% at 5k and 30k steps, so there was nothing to compare "
                "against. The first evaluation was also confounded: each arm decoded with its own "
                "re-fitted noise schedule."))

    st.append(H2("C. Training from scratch with ordering, w=0.25, to 500k steps (Sep 10–22)"))
    rows = [["Arm (1 training seed)", "Decoded with its own order", "Δ vs control", "Decoded uniformly",
             "Δ vs control"]]
    agg = {}
    for r in curve:
        if r["ckpt"] in ("last.pt", "step=000500000.pt") and r["run"].startswith("ord_fs500k"):
            agg.setdefault((r["run"], r["decode"]), []).append(r)
    ctrl500 = np.mean(sorted({(r["seed"], r["control_accuracy"]) for v in agg.values() for r in v
                              }, key=lambda x: x[0]), axis=0)[1] if agg else float("nan")
    names = {"l2r": "L2R", "r2l": "R2L", "random": "random"}
    for arm in ("l2r", "r2l", "random"):
        run = f"ord_fs500k_{arm}_w0.25_s42"
        m, u = agg.get((run, "matched"), []), agg.get((run, "uniform"), [])
        f = lambda rs: (f"{100 * np.mean([r['accuracy'] for r in rs]):.1f}%",  # noqa: E731
                        f"{100 * np.mean([r['delta'] for r in rs]):+.1f} pp "
                        f"({sum(r['significant'] for r in rs)}/3 sig.)")
        rows.append([names[arm], *f(m), *f(u)])
    rows.append(["uniform control", f"{100 * ctrl500:.1f}%", "—", f"{100 * ctrl500:.1f}%", "—"])
    st.append(table(rows, col_widths=[34 * mm, 34 * mm, 34 * mm, 32 * mm, 34 * mm]))
    st.append(P("At first this read as \"L2R does nothing\" and \"random helps\". The 24 Sep audit "
                "showed the design never ordered the answer: normalising the priority over a "
                "suffix that is ~74% padding gave the whole answer only a ~2× σ range (adjacent "
                "tokens differed by 0.6%). L2R and R2L were effectively simultaneous denoising "
                "with a global noise shift; only random actually perturbed the answer, and its "
                "arm also trained at systematically lower noise. <b>These results are not "
                "evidence about ordering in either direction</b>, and they rest on one training "
                "seed. The one solid fact: ordering-trained models are captive to their decode "
                "schedule (L2R decoded uniformly falls to ~4%, R2L to 0%)."))

    st.append(H2("D. The window study: the real test (24–26 Sep, Isambard)"))
    st.append(P("26 runs from scratch: uniform control; L2R, R2L, random at k = 64, 16, 4; "
                "supervision-matched controls at the same k; each with training seeds 42 and 43. "
                "lr 1e-4, batch 512, 4 GH200 each. Stopped by budget at 250k evaluated steps "
                "(~2,780 GPU-h). Each model is decoded with its own window sampler, NFE 256, "
                "full 1319 problems."))
    st.append(figure(fig_window_curve(own), "ord_window_curve", 160,
                     "GSM8K accuracy (mean of 2 training seeds) vs training steps, k = 64. Source: "
                     "results/ordering/window_summary.json."))
    rows = [["Arm at 250k", "k = 64", "k = 16", "k = 4"]]
    for mode, lab in (("l2r", "L2R"), ("random", "random"), ("r2l", "R2L")):
        rows.append([lab] + [pct(own[(mode, k, 250000)]["acc"]) + "%" for k in (64, 16, 4)])
    rows.append(["supervision-matched control"] +
                [pct(own[(f"lm{k}", float("inf"), 250000)]["acc"]) + "%" for k in (64, 16, 4)])
    rows.append(["<b>uniform control (k = ∞)</b>", f"<b>{pct(own[('none', float('inf'), 250000)]['acc'])}%</b>",
                 "", ""])
    st.append(table(rows, col_widths=[56 * mm, 30 * mm, 30 * mm, 30 * mm]))
    st.append(Spacer(1, 4))

    rows = [["Contrast at 250k (pp, Welch 95% over 2 training seeds)", "k = 64", "k = 16"]]
    for lab, b in (("L2R − R2L", "r2l"), ("L2R − random", "random"), ("L2R − matched control", "lm")):
        cells = []
        for k in (64, 16):
            bkey = (f"lm{k}", float("inf"), 250000) if b == "lm" else (b, k, 250000)
            d, lo, hi = welch(seeds(own[("l2r", k, 250000)]), seeds(own[bkey]))
            sig = lo > 0 or hi < 0
            txt = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            cells.append(f"<b>{txt}</b>" if sig else txt)
        rows.append([lab] + cells)
    cells = []
    for k in (64, 16):
        d, lo, hi = welch(seeds(own[("l2r", k, 250000)]), seeds(own[("none", float("inf"), 250000)]))
        txt = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        cells.append(f"<b>{txt}</b>" if (lo > 0 or hi < 0) else txt)
    rows.append(["L2R − uniform control"] + cells)
    st.append(table(rows, col_widths=[84 * mm, 40 * mm, 40 * mm]))
    st.append(P("Bold = interval excludes 0. With only 2 training seeds these intervals have "
                "about 1–2 degrees of freedom, so most are wide; at 200k L2R also beat random at "
                "k=16 and the matched control at both k.", "small"))

    st.append(H1("3. What it means"))
    st.append(H2("What worked"))
    st.append(bullets([
        "<b>Order matters, in the autoregressive direction.</b> Among ordered models L2R is best "
        "at every k where anything is learned (L2R > random > R2L). This is the first evidence "
        "in the project that the <i>order itself</i>, not just noise heterogeneity, changes what "
        "the model learns.",
        "<b>L2R learns more per supervised token.</b> Against the supervision-matched control it "
        "is ~2× better at k=64 (8.0% vs 4.3%) and ~6× at k=16 (6.5% vs 1.1%).",
    ]))
    st.append(H2("What did not"))
    st.append(bullets([
        "<b>No ordering beats the uniform model at equal training steps.</b> L2R k=64 trails by "
        "−1.5 pp [−2.0, −1.1] at 250k, and the gap is not shrinking (−1.3 at 200k).",
        "<b>Narrow windows fail:</b> k=4 is ~1% in every arm, including its matched control.",
        "<b>Decode-time ordering of a normal model</b> does nothing (small spread) or hurts "
        "(random, larger spread).",
        "<b>Ordered models only work with their own sampler:</b> window-trained L2R/R2L score 0% "
        "with the standard decoder.",
        "<b>Random order is never the best order</b>, so the \"mixed-noise helps\" idea from the "
        "w=0.25 study is not supported by the proper test.",
    ]))
    st.append(H2("Why (the explanations on record)"))
    st.append(bullets([
        "<b>Supervision volume.</b> An ordered model supervises 8–116× fewer token–noise pairs per "
        "step; at this budget the uniform model's sheer volume of training signal wins.",
        "<b>Too few denoising steps per token at decode.</b> With 256 evaluations a token spends "
        "only ~31 steps inside a k=64 window and ~2 inside k=4. The NFE-1024 cells that would test "
        "this were never run.",
        "<b>Schedule captivity.</b> A model trained under one per-token noise pattern breaks under "
        "another, which is also why decode-only ordering failed.",
    ]))

    st.append(H1("4. How much to trust this"))
    st.append(bullets([
        "<b>Only 2 training seeds</b> in the window study, no correction for the many contrasts "
        "(3 k × 4 contrasts × 5 checkpoints). The w=0.25 study had 1 seed.",
        "<b>Training bugs found afterwards</b> (fixed 1 Oct, not yet re-run): GPUs other than "
        "rank 0 trained with σ<sub>data</sub> 0.5 instead of the ≈0.40 estimate (T1); all GPUs drew the same noise (T3); "
        "seeds 42 and 43 saw the same data order (T4), so the seed intervals understate variance. "
        "These affect all arms alike, but the conclusions are not confirmed on the fixed trainer.",
        "<b>Random-order evaluations reused the same permutations in every batch</b> (T5), so "
        "\"random\" was less random than intended.",
        "<b>Missing cells:</b> no 300k and no NFE-1024 evaluations, although the 300k checkpoints "
        "exist for all 26 runs.",
        "R2L and random rank the whole canvas including padding (the answer length is unknown at "
        "sampling), so R2L resolves the padding first; that partly explains why it is worst.",
    ]))

    st.append(H1("5. What to keep and what to do next"))
    st.append(callout(
        "<b>Keep:</b> uniform simultaneous denoising as the default. <b>Keep the window code</b> "
        "(committed 1 Oct with its frozen noise table): it is the correct way to test ordering. "
        "<b>Keep the finding</b> that L2R is the right direction and is more data-efficient per "
        "supervised token. <b>Drop:</b> decode-time ordering of uniform models, the w-spread "
        "design, and R2L/random as candidates.", "keep"))
    st.append(P("If ordering is revisited (GPU estimates; nothing has been run):"))
    st.append(bullets([
        "First confirm the training fixes (12k-step test, ~20 GPU-h for 3 seeds).",
        "Evaluate the existing 300k checkpoints and NFE-1024 for L2R k=16/64 (~65 GPU-h): does "
        "giving each token more decode steps close the gap?",
        "Test a supervision rule that also trains tokens not yet in the window (the AR-like "
        "signal), so ordered models are not starved of supervision.",
        "More training seeds (with T4 fixed) before claiming any small difference.",
    ]))
    st.append(Spacer(1, 6))
    st.append(P("<b>Sources.</b> docs/experiments_ordering_objective.md (A1–A3, 20/24/26 Sep sections), "
                "docs/report_since_friday/report.tex, docs/report_meeting/report.tex, CODEBASE.md "
                "(T1–T5, R5), docs/cpu_reanalysis.md; results/ordering/confirm_summary.json, "
                "summary.json, train_curve_gh200.json, train_summary_500k_gh200.json, "
                "window_summary.json, window_sigma_field.json. Welch intervals recomputed from the "
                "per-seed accuracies in window_summary.json with the formula of "
                "experiments/ordering/analyze_window.py.", "small"))

    out = HERE / "report_time_ordering.pdf"
    build(out, st, "Time-ordering (L2R, R2L, random) — CoBit on TinyGSM/GSM8K",
          "Temporal ordering experiments")
    print(out)


if __name__ == "__main__":
    main()
