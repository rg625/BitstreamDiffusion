"""Consolidated report: every temperature experiment on CoBit TinyGSM -> GSM8K (11-13 Oct 2026). CPU only.

Pulls figures/tables from build_scoretemp.py, build_sampler_math.py and build_entropy_profile.py, plus the
earlier A1 / FKC numbers in results/fkc_temperature_all_cells.csv, into one narrative for Georgios.

    python docs/pdf_reports/build_temperature_study.py   # -> docs/pdf_reports/temperature_study_cobit.pdf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_entropy_profile as be  # noqa: E402
import build_sampler_math as bm  # noqa: E402
import build_scoretemp as bs  # noqa: E402
from pdfkit_common import (  # noqa: E402
    H1, H2, HERE, P, PageBreak, Spacer, build, bullets, callout, figure, mm, mpl_style, table,
)

REPO = HERE.parents[1]
acc, paired, fmt_d, series, tau_table = bs.acc, bs.paired, bs.fmt_d, bs.series, bs.tau_table
M = bm.M


def ci(r):
    return f"[{100 * r['ci95_low']:.1f}, {100 * r['ci95_high']:.1f}]"


def prior_results():
    rows = list(csv.DictReader(open(REPO / "results/fkc_temperature_all_cells.csv")))
    a1 = [["κ-temperature τ (A1, late-σ only)", "accuracy %"]]
    for r in rows:
        if r["sweep"] == "tau_s1024":
            a1.append([r["beta"].replace("tau=", ""), r["acc"]])
    fk = [["τ = 1/β (whole trajectory)", "accuracy %"]]
    for r in rows:
        if r["K"] == "1" and r["steps"] == "512" and r["policy"] in ("never", "ess") and r["sweep"] in (
                "sweep_K32_s512_n256", "sweep_beta_n256", "sweep_K1_drift_512") and r["n"] == "256":
            fk.append([f"{1 / float(r['beta']):.2f}", r["acc"]])
    return a1, fk


def main():
    mpl_style()
    cells = bs.load()
    m = series(cells, ("m250", 0.0, 0.41, 1024))
    c1 = series(cells, ("cfg425", 1.0, 0.41, 1024))
    s5 = series(cells, ("s500", 0.0, 0.41, 1024))
    s5d = series(cells, ("s500", 0.0, 0.0, 1024))
    s5h = series(cells, ("s500", 0.0, 0.41, 512))
    anc = bs.load("anchor500")
    a_ent = next(r for k, r in anc.items() if k[0] == "s500ent")
    a_kar = next(r for k, r in anc.items() if k[0] == "s500kar")
    B = bs.load("../blocks_gh200")
    Sb = lambda arm, g: {k[2]: r for k, r in B.items() if k[0] == arm and k[3] == g and k[4] == 1024}
    A_c = Sb("s500a", 0.41)
    D = Sb("s500d", 0.41)
    best_t = max(A_c, key=lambda t: acc(A_c[t]))
    mb, cb, b5 = bs.best(m), bs.best(c1), bs.best(s5)
    grid = {(w, t): cells[("cfg425", float(w), float(t), 0.41, 1024)] for w in (1, 2, 3, 5, 9, 12)
            for t in (1, 0.5, 0.3, 0.2, 0.1) if ("cfg425", float(w), float(t), 0.41, 1024) in cells}
    gbest = max(grid, key=lambda k: acc(grid[k]))
    bd = max(D, key=lambda t: acc(D[t]))
    a1, fk = prior_results()
    G = bm.grids()

    story = [P("Temperature scaling for CoBit on TinyGSM → GSM8K", "title"),
             P("All experiments and results, 11–13 October 2026 · CoBit-S (130M), its CFG variant and CoBit-M (462M) · "
               "GSM8K executable-code accuracy, all 1319 test problems · Isambard GH200 · prepared for Georgios", "subtitle"),
             callout(
                 "<b>Main result.</b> Sigma-style score temperature (the score divided by τ at every noise level) works on "
                 "CoBit once the sampler's last step is left unsharpened. On the main unguided CoBit-S model (500k) "
                 f"accuracy goes from <b>{acc(s5[1]):.1f}% to {acc(A_c[best_t]):.1f}%</b> at τ={best_t:g} "
                 f"({fmt_d(paired(A_c[best_t], s5[1]))} pts, paired 95% CI). On CoBit-M (250k) it goes from {acc(m[1]):.1f}% to "
                 f"{mb[1]:.1f}% (τ={mb[0]:g}, {fmt_d(mb[2])}). Recipe: EDM churn γ=0.41, τ≈0.3, last step untempered."),
             table([["finding", "evidence"],
                    ["Score temperature helps every checkpoint we tested", f"+{(acc(A_c[best_t]) - acc(s5[1])):.1f} (CoBit-S 500k), "
                     f"+{mb[2][0]:.1f} (CoBit-M), +{cb[2][0]:.1f} (CoBit-S CFG ckpt) pts, all CIs exclude 0"],
                    ["Classifier-free guidance adds nothing on top", f"best (w, τ) = ({gbest[0]}, {gbest[1]:g}): {acc(grid[gbest]):.1f}%"],
                    ["Below τ≈0.7 the old sampler collapsed — two separate causes",
                     "churn: the last step (σ 0.157→0.002) corrupted bits; deterministic: mode/length collapse"],
                    ["Fix: untempered last step (or stop at σ_min=0.1)", "churn works down to τ=0.2; invalid tokens back to 0"],
                    ["Lower γ widens the usable τ range but not accuracy", "γ=0.1–0.2 stable at τ=0.1 (32–33%); best 33.6% < 35.3%"],
                    ["Ancestral (DDPM-style) sampler: stable but weaker", "≤ 29.0% (starts at 16.6% at τ=1)"],
                    ["DeepMind's logit temperature: no real gain", f"best T={bd:g}: {acc(D[bd]):.1f}% ({fmt_d(paired(D[bd], s5[1]))} pts, n.s.)"],
                    ["σ_min=0.002 sends the sampler where the model was never trained", "no training mass below σ≈0.17"]],
                   col_widths=[78 * mm, 92 * mm]),
             Spacer(1, 4),
             P("Caveats up front: one sampling seed per cell (intervals are bootstrap over problems; differences are paired "
               "on identical problems and noise); the CoBit-M and CFG arms use the Karras σ grid while the main unguided arm "
               "uses the entropic grid, so absolute numbers are comparable only within an arm.", "small"),
             PageBreak()]

    # 1 definitions
    story += [H1("1  Three ways to apply a temperature"),
              P("CoBit diffuses bits x<sub>0</sub> ∈ {0,1} with a variance-exploding process x = x<sub>0</sub> + σε. The "
                "network gives the per-bit posterior mean D = sigmoid(learned logit + analytic matched filter), and the "
                "score is s = (D − x)/σ². The three temperatures we tested act on this in different places:"),
              table([["name", "operation", "where it acts"],
                     ["A1 “local” (our original)", "s → κ(σ) s,   κ = (v + σ²)/(τv + σ²),  v = 0.25",
                      "≈1 at high σ, 1/τ only as σ → 0: sharpens only the end of the trajectory"],
                     ["Score temperature (Sigma, Yang et al. 2026)", "s → s/τ  ⇔  D<sub>τ</sub> = x + (D − x)/τ",
                      "every σ, constant; extrapolates D beyond [0,1]"],
                     ["Logit temperature (DeepMind SDM, eq. 25)", "token logits ζ → ζ/T, D<sub>T</sub> = E<sub>softmax(ζ/T)</sub>[code]",
                      "every σ; sharpens the prediction inside [0,1]"]],
                    col_widths=[46 * mm, 62 * mm, 62 * mm]),
              Spacer(1, 4),
              P("Your note of 11 Oct pointed out that our earlier negative result was for A1 only, which leaves the high-noise "
                "part of the trajectory untempered; everything below tests the global (Sigma) operation, and at the end "
                "DeepMind's. Code: <font face='Mono'>--score_temp_mode global</font> (CoBit), τ=1 bit-identical to the old "
                "sampler (unit-tested)."),
              H1("2  Setup"),
              table([["checkpoint", "what it is", "σ grid used"],
                     ["CoBit-S unguided, 500k", "the main TinyGSM run (shared Drive last.pt, global_step 500000)", "entropic"],
                     ["CoBit-S CFG, 425k", "same recipe with 10% prompt dropout", "Karras"],
                     ["CoBit-M, 250k", "462M, unguided, half-way through a 500k cosine", "Karras"]],
                    col_widths=[40 * mm, 92 * mm, 38 * mm]),
              Spacer(1, 4),
              bullets(["All 1319 GSM8K problems, one answer each, EMA weights, 1024 steps unless stated, seed 0.",
                       "EDM churn γ=0.41 per step (“churn”) unless stated; “deterministic” = γ=0 (probability-flow ODE).",
                       "Δ = paired difference on identical problems, bootstrap 95% CI. Programs that ran = share of "
                       "generated programs that execute to a number; invalid tokens = generated ids outside the vocabulary."],
                      "small"),
              H1("3  Before this study"),
              P("Two earlier results bear on the question (from the June/September FKC work; 256 problems, so ±6 pts)."),
              table([[Pc for Pc in ("A1, 1024 steps", "", "whole-trajectory scaling (FKC drift-only, 512 steps)", "")]]
                    + [[a1[i][0] if i < len(a1) else "", a1[i][1] if i < len(a1) else "",
                        fk[i][0] if i < len(fk) else "", fk[i][1] if i < len(fk) else ""] for i in range(max(len(a1), len(fk)))],
                    col_widths=[48 * mm, 30 * mm, 62 * mm, 30 * mm], header_rows=2),
              Spacer(1, 4),
              P("A1 gave +0.8 pts at τ=0.7 then failed (0% at τ=0.3). The FKC drift-only runs were already constant scaling "
                "of the drift, but only for τ ≥ 0.77, at 512 steps: a +4 pt trend that was read at the time as a step-size "
                "effect. Neither tested global score temperature at strong sharpening.", "small"),
              PageBreak()]

    # 4 sweep on three checkpoints
    story += [H1("4  Global score temperature on three checkpoints (12 Oct)"),
              figure(bs.fig_tau(cells), "ts_tau", 160,
                     "τ alone with churn, 1024 steps. Bars: 95% CIs. Note the CoBit-S 500k curve (entropic grid) collapses "
                     "below τ=0.7 while the two Karras-grid curves hold to τ=0.2 — explained in Sections 6–8."),
              H2("CoBit-M 462M (250k)"), tau_table(m), Spacer(1, 3),
              H2("CoBit-S CFG checkpoint at w=1 (no guidance)"), tau_table(c1),
              PageBreak(), H2("CFG × τ (CFG checkpoint)"),
              figure(bs.fig_grid(cells), "ts_grid", 115, "Accuracy % per (w, τ); w=1 is the conditional model."),
              bullets([f"τ does the work: best cell w={gbest[0]}, τ={gbest[1]:g} ({acc(grid[gbest]):.1f}%). CFG alone peaks at "
                       f"w=3 ({acc(grid[(3, 1)]):.1f}%), and every w&gt;1 row at its best τ is at or below the w=1 row.",
                       "Guidance moves the breaking point: at τ=0.2 accuracy falls from "
                       f"{acc(grid[(1, 0.2)]):.1f}% (w=1) to {acc(grid[(3, 0.2)]):.1f}% (w=3) and ≈0 for w≥9.",
                       "Sigma's operating points (w=2, τ=0.1) and (1, 0.05) give 0% here: with EDM churn the drift ÷ τ makes "
                       "every step unstable below τ≈0.15 (Section 6)."], "small"),
              PageBreak()]

    # 5 experiment 1 and the checkpoint problem
    story += [H1("5  Experiment 1: the main unguided CoBit-S model"),
              H2("A checkpoint problem first"),
              bullets(["The 425k file in the shared Drive folder scores 1.7% at τ=1 (churn, 1024 steps). Its EMA weights are "
                       "complete, finite and were updating normally (an earlier note blamed a stale EMA; that was wrong). Its "
                       "running weights do better (18.3% vs 4.9% at 512 steps), consistent with your recollection, but both are "
                       "far below the 29.4% of June, which our handoff attributes to a main-run 425k that was never copied "
                       "to the Drive folder.",
                       "The Drive last.pt is the 500k checkpoint of the same run and is healthy, so Experiment 1 uses it.",
                       "A silent fallback was found and fixed on the way: without the run's entropy tables, churn switched "
                       "itself off with no warning; the launcher now refuses to run in that state."], "small"),
              H2("Which σ grid? (500k EMA, τ=1, churn)"),
              table([["σ grid", "accuracy %", "95% CI", "programs that ran %"],
                     ["entropic", f"{acc(a_ent):.1f}", ci(a_ent), f"{100 * a_ent['executable_rate']:.1f}"],
                     ["Karras", f"{acc(a_kar):.1f}", ci(a_kar), f"{100 * a_kar['executable_rate']:.1f}"]],
                    col_widths=[30 * mm, 26 * mm, 30 * mm, 36 * mm]),
              P("Both reproduce your ~29%; entropic is used for this arm.", "small"),
              H2("τ sweep with the original sampler (entropic grid)"),
              H2("churn γ=0.41, 1024 steps"), tau_table(s5), Spacer(1, 3),
              H2("deterministic (γ=0), 1024 steps"), tau_table(s5d), Spacer(1, 3),
              H2("churn γ=0.41, 512 steps (step-count control)"), tau_table(s5h),
              P(f"τ=0.7 helps ({fmt_d(b5[2])} pts with churn; the deterministic sampler doubles), and 512 vs 1024 steps "
                "give the same gain, so it is not a step-size effect. But every setting collapses at τ ≤ 0.5.", "small")]

    # 6 diagnosis
    story += [H1("6  Why it collapsed: the sampler, step by step"),
              P("Every sampler we have is a step of the form below: the state moves towards the prediction D by a gain g and "
                "fresh noise is added. Score temperature multiplies the gain by 1/τ."),
              M("x' − D = (1 − g/τ) (x − D) + noise"),
              bullets(["The step is <b>stable</b> only if g/τ &lt; 2 and <b>does not overshoot</b> only if g/τ ≤ 1. When it "
                       "overshoots, the state is reflected through D and the residual — including the noise just injected — "
                       "is amplified by |1 − g/τ|.",
                       "Gains: deterministic DDIM g = 1 − σ'/σ (≈0.01 per step on a 1024-step grid); EDM churn first raises "
                       "σ by (1+γ) on every step, so g = 1 − σ'/((1+γ)σ) ≈ 0.29 at γ=0.41 regardless of the step count — the "
                       "reason for the τ ≈ 0.15 wall on the Karras grid; the ancestral (DDPM) step g = 1 − (σ'/σ)², small.",
                       "<b>The entropic grid ends with one huge step</b>, σ = 0.157 → 0.002 (g = 0.99). Any τ &lt; 1 makes it "
                       "overshoot; with churn, the reflected residual is large enough to flip already-decided bits.",
                       "DeepMind's logit temperature cannot overshoot: it sharpens D inside [0,1] and leaves the step a "
                       "convex combination, which is why their T=0.1 works with churn."], "small"),
              H2("The two failure modes, from the real outputs (500k, entropic, 1024 steps)"),
              table(bm.evidence(), col_widths=[37 * mm, 9 * mm, 18 * mm, 21 * mm, 20 * mm, 26 * mm, 30 * mm]),
              Spacer(1, 3),
              bullets(["<b>With churn</b>: invalid token ids appear (0 → 1.4 → 40%) while program length is unchanged — the "
                       "right sentence with scrambled tokens. Bit corruption at the last step.",
                       "<b>Deterministic</b>: never an invalid token, but programs shrink to <i>return result</i>. Constant 1/τ "
                       "on an ODE is strongly mode-seeking, and ending early is the dominant mode; churn's noise counteracts it."],
                      "small")]
    taus = (0.5, 0.3, 0.2, 0.1)
    trows = [["sampler (toy)", "σ grid", "last step"] + [f"τ={t}" for t in taus]]
    for kind, gma, lab in (("churn", 0.41, "churn γ=0.41"), ("churn", 0.2, "churn γ=0.2"), ("ancestral", 0.0, "ancestral")):
        for gname, fs, flab in (("entropic", True, "sharpened"), ("entropic", False, "unsharpened"), ("karras", True, "sharpened")):
            trows.append([lab, gname, flab] + [bm.toy_flips(G[gname], t, kind, gma, fs) for t in taus])
    story += [H2("Predictions from an exact-posterior toy"),
              P("Independent bits with the exact posterior (no network), the real grids: share of bits whose readout flips "
                "the decision held one step earlier.", "small"),
              table(trows, col_widths=[26 * mm, 20 * mm, 24 * mm] + [25 * mm] * len(taus)),
              P("Prediction: leaving the last step untempered removes the flips; lower γ and ancestral steps move the "
                "divergence wall to smaller τ. Section 8 tests this on the real model.", "small"),
              PageBreak()]

    # 7 entropy profile
    pdf0, mid0, _ = be.tables(be.RUNS[0][1])
    lo = mid0[pdf0 > 0].min()
    story += [H1("7  The entropy-rate profile and σ_min"),
              figure(be.fig_pdf(), "ts_profile", 160,
                     f"Training σ distribution stored by each run. {int((pdf0 == 0).sum())} of 128 bins are empty: after the "
                     f"40k-step warm-up the model is essentially never trained below σ≈{lo:.2f}."),
              figure(be.fig_mse(), "ts_mse", 160,
                     "Training MSE per σ recovered from the same tables (scaled to 1 at σ=20); dashed: exact MMSE of an "
                     "isolated bit with no context. TensorBoard on Isambard holds only loss/LR/σ_data, so the tables are the "
                     "per-σ record."),
              bullets(["We sample to σ_min = 0.002 (EDM's default, {0,1}-bit units; not 2e-4). The grid's last step jumps from "
                       "0.157 to 0.002, into the untrained region.",
                       "No numerical blow-up: below σ≈0.13 the matched-filter logit sits at its ±30 clip, and the drift stays "
                       "O(1) in fp32. The risk is the learned logit evaluated where it was never trained.",
                       "Even an isolated bit is decided by σ ≈ 0.1, so nothing is gained by going lower. Ending at σ_min = 0.1 "
                       "(as on OWT) cuts the last step's gain from 0.99 to 0.36."], "small"),
              PageBreak()]

    # 8 fixes
    sec, _ = bs.blocks_section(cells, title="8  Fixing the collapse (13 Oct)", sec_cells="Section 5", sec_mode="Section 6")
    story += sec[1:]   # drop its leading PageBreak

    # 9 conclusions
    story += [PageBreak(), H1("9  Conclusions and recommendation"),
              bullets([f"<b>Use global score temperature with an untempered last step.</b> Recommended setting for CoBit on "
                       f"TinyGSM: EDM churn γ=0.41, τ≈0.3, last step unsharpened (or stop at σ_min=0.1). Gains of 6–8 pts on all "
                       f"three checkpoints; best {acc(A_c[best_t]):.1f}% for CoBit-S 500k.",
                       "<b>The collapse below τ=0.7 was the sampler, not the model or the readout</b>: one oversized final step "
                       "(churn arm) and ODE mode collapse (deterministic arm). The τ ≈ 0.15 wall with churn is a stability "
                       "limit of the churn step itself.",
                       "<b>CFG does not add to temperature</b> on our CFG checkpoint, whose unconditional branch comes from 10% "
                       "dropout on a 130M model; this is not a fair test of Sigma's CFG.",
                       "<b>Lower γ</b> makes very low τ usable (τ=0.1) but did not raise the best accuracy; <b>ancestral</b> "
                       "steps are stable but start too low; <b>DeepMind's logit temperature</b> gave no significant gain here.",
                       "<b>σ_min</b> should be 0.1, not 0.002: below ≈0.17 the model was never trained and nothing is left to "
                       "decide."], "small"),
              H2("Suggested next steps"),
              bullets(["Seeds 1–2 on the best cells of each arm (≈1.5 GH200-h) before quoting numbers externally.",
                       "Make σ_min = 0.1 and the untempered last step the default sampler settings for TinyGSM.",
                       "Give the training σ schedule a floor so low-σ bins are not left empty.",
                       "Carry over to rolling-CoBit: its sampler already reads out the final step untempered; its score- and "
                       "logit-temperature sweeps are ready."], "small"),
              Spacer(1, 10), H1("Appendix — runs, compute and code")]
    story += [table([["batch", "cells", "what", "GH200-h"],
                     ["12 Oct sweep", "59 + 3 gates", "τ on three checkpoints; (w, τ) grid; first Exp. 1 attempt (425k)", "27.2"],
                     ["12 Oct reruns", "14 + 2", "Exp. 1 churn arm with entropy tables; raw-weight diagnostic", "3.2"],
                     ["12 Oct Exp. 1 redo", "21 + 2 anchors", "500k checkpoint, entropic vs Karras anchor", "5.3"],
                     ["13 Oct blocks A–E", "38", "unsharpened last step, ancestral, γ × τ, logit temperature, σ_min = 0.1", "10.6"]],
                    col_widths=[30 * mm, 26 * mm, 94 * mm, 20 * mm]),
              Spacer(1, 4),
              P("Total ≈ 46 GH200-h. Results: results/scoretemp/eval_gh200, anchor500/, diag/, blocks_gh200/. Drivers: "
                "scripts/hpc/arch/scoretemp_gh200.sh and scoretemp_blocks.sh; launcher scoretemp_eval.slurm. New sampler "
                "options (uncommitted, unit-tested): <font face='Mono'>--score_temp_mode global</font>, "
                "<font face='Mono'>--score_temp_final unsharpened</font>, <font face='Mono'>--sampler_kind ancestral</font>, "
                "<font face='Mono'>--sigma_min</font>; the EM sampler now accepts score temperature. Companion notes: "
                "report_score_temperature.pdf (full tables), sampler_math.pdf, entropy_profile.pdf. Built by "
                "docs/pdf_reports/build_temperature_study.py.", "small")]

    path = build(HERE / "temperature_study_cobit.pdf", story, "Temperature scaling for CoBit · 11–13 Oct 2026",
                 "temperature study")
    print(path)


if __name__ == "__main__":
    main()
