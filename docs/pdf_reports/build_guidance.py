"""Report 1: guidance (CFG, AutoGuidance, Self-Guidance) on TinyGSM -> GSM8K. CPU only.

Accuracies come from results/guidance/all_cells.csv and confirm_summary.json;
fix/break rates from results/cpu_reanalysis/t6_fix_break.json. Confidence
intervals that exist only in the docs (churn grids, compute controls, Regime B)
are quoted from docs/findings.md, docs/open_questions.md,
docs/cross_regime_comparison.md and commit cdde4c6, and labelled as such.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdfkit_common import (  # noqa: E402
    H1, H2, HERE, INK2, MUTED, SERIES, KeepTogether, P, Spacer, build, bullets, callout, figure,
    mm, mpl_style, plt, table,
)

REPO = HERE.parents[1]
G = REPO / "results" / "guidance"


def f(x):
    return float(x) if x not in ("", None) else 0.0


def cells():
    out = defaultdict(list)
    for r in csv.DictReader(open(G / "all_cells.csv")):
        bad = r["bad_checkpoint"].split("=")[-1][:9] if r["bad_checkpoint"] else ""
        k = (r["grid"], f(r["gamma"]), int(r["steps"]), int(r["num_examples"]), f(r["guidance_scale"]),
             f(r["ag_scale"]), bad, f(r["sg_scale"]), r["sg_variant"] if f(r["sg_scale"]) else "")
        out[k].append(float(r["accuracy"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def fig_confirm(conf):
    order = sorted(conf, key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for i, r in enumerate(order):
        c = SERIES[0] if r["delta"] > 0 else "#b3261e"
        ax.plot([100 * r["ci95_low"], 100 * r["ci95_high"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(100 * r["delta"], i, "o", color=c, ms=6, mec="white", mew=1.5)
        ax.text(100 * r["ci95_high"] + 0.25, i, f"{100 * r['accuracy']:.1f}%  (NFE {int(r['nfe_per_sample'])})",
                va="center", fontsize=7, color=INK2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r["config"].replace("bad=350000", "bad=350k").replace("bad=250000", "bad=250k")
                        for r in order], fontsize=7.5)
    ax.set_xlim(-3, 12.5)
    ax.set_xlabel("change in GSM8K accuracy vs no guidance (percentage points, 95% CI)")
    ax.set_title("Deterministic sampling (γ=0), 1319 problems × 3 seeds: baseline 13.8%")
    ax.grid(axis="y", visible=False)
    return fig


def fig_churn(c):
    gam = [0.0, 0.2, 0.3]
    arms = [("no guidance", (0.0, 0.0, "", 0.0, ""), MUTED),
            ("CFG w=12", (12.0, 0.0, "", 0.0, ""), SERIES[0]),
            ("AG w=15", (0.0, 15.0, "000350000", 0.0, ""), SERIES[2]),
            ("SG-prev w=2", (0.0, 0.0, "", 2.0, "prev"), SERIES[1])]
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    labels = []
    for lab, k, col in arms:
        y = [100 * c[("stoch_confirm", g, 256, 1319) + k] for g in gam]
        ax.plot(gam, y, color=col, marker="o", ms=5, lw=2.4 if lab == "no guidance" else 2.0)
        labels.append([y[-1], f"{lab}  {y[-1]:.1f}%"])
    retune = 100 * c[("churn_anatomy", 0.3, 256, 1319, 2.0, 0.0, "", 0.0, "")]
    ax.plot([0.3], [retune], marker="D", color=SERIES[0], ms=6, mfc="white", mew=1.8)
    labels.append([retune, f"CFG re-tuned to w=2  {retune:.1f}%"])
    # Spread end labels so none overlap (min 1.6 pp apart), keeping their order.
    labels.sort(key=lambda t: t[0])
    pos = [t[0] for t in labels]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + 2.4)
    for (y, txt), yp in zip(labels, pos):
        ax.text(0.312, yp, txt, fontsize=7, color=INK2, va="center")
        if abs(yp - y) > 0.3:
            ax.plot([0.302, 0.31], [y, yp], color=MUTED, lw=0.6)
    ax.set_xticks(gam)
    ax.set_xticklabels(["γ=0\n(deterministic)", "γ=0.2", "γ=0.3"])
    ax.set_xlim(-0.02, 0.43)
    ax.set_ylim(0, 34)
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("Adding sampler noise (churn) helps more than any guidance, and reverses most of it")
    return fig


def main():
    mpl_style()
    c = cells()
    conf = json.loads((G / "confirm_summary.json").read_text())
    t6 = json.loads((REPO / "results/cpu_reanalysis/t6_fix_break.json").read_text())["configs"]
    base512 = conf[0]["baseline"]

    def acc(*k):
        return c[k]

    st = []
    st.append(P("Guidance on CoBit: classifier-free, auto- and self-guidance", "title"))
    st.append(P("TinyGSM → GSM8K, production 500k bits model. Every guidance experiment we ran: what "
                "worked, what did not, why, and what to keep. Written 2 Oct 2026 from results on disk; "
                "no GPU was used.", "subtitle"))
    st.append(callout(
        "<b>Bottom line.</b> With deterministic sampling, CFG, AutoGuidance and SG-prev all help and "
        f"SG-exact hurts. <b>CFG is best</b> ({100 * base512:.1f}% → 22.0%, +8.1 pp, at 2× compute), AutoGuidance "
        "gives +6.6 pp (2× compute plus a second model) and SG-prev +4.8 pp at no extra compute. But a "
        "<b>free sampler setting, stochastic churn, gives +12 pp on its own</b> (13.7% → 25.7%) and beats "
        "every guided deterministic run. Under churn, CFG only helps if re-tuned down to w≈2 (+1.9 pp), "
        "AutoGuidance turns harmful and SG-prev collapses. In the strongest regime nothing helps. "
        "Combining methods always hurts. Spending the extra compute on a second sample and a vote is as "
        "good or better. <b>The study is closed</b>; keep churn, voting, and CFG w≈2 only when a single "
        "sample is required."))

    # ------------------------------------------------------------------ methods
    st.append(H1("1. The three methods in one page"))
    st.append(P("All three take the model's prediction of the clean bits, D, and push it away from a "
                "\"worse\" prediction: D<sub>guided</sub> = D<sub>worse</sub> + w·(D<sub>good</sub> − D<sub>worse</sub>). "
                "They differ only in what \"worse\" means. Because the score is linear in D, guiding D is "
                "the same as guiding the score; the analytic matched-filter term cancels out of the "
                "difference, so only the learned part is amplified."))
    st.append(table([
        ["Method", "The \"worse\" prediction", "Extra cost", "Notes"],
        ["<b>CFG</b> (classifier-free)", "the same model with the question hidden (unconditional; trained "
         "by dropping the prompt 10% of the time)", "2× network evaluations", "w=1 is no guidance; w&lt;1 "
         "moves <i>towards</i> the unconditional model"],
        ["<b>AG</b> (AutoGuidance)", "an earlier, weaker checkpoint of the same run (250k, 350k or 425k "
         "steps vs the 500k model)", "2× evaluations + a second 2 GB model", "only 250k–425k checkpoints "
         "were available"],
        ["<b>SG-exact</b> (self-guidance)", "the same model evaluated at a noisier level, σ·e<super>0.5</super>",
         "2× evaluations", "the matched filter is held fixed (\"hold\" mode)"],
        ["<b>SG-prev</b>", "the prediction cached from the previous sampling step", "none", "effectively a "
         "step-to-step extrapolation (momentum) term, ~33–96× amplified"],
    ], col_widths=[30 * mm, 66 * mm, 32 * mm, 42 * mm]))
    st.append(Spacer(1, 4))
    st.append(P("<b>Setup.</b> One model: the production conditional run "
                "(<font face='Mono'>cobit_raw_binary_bits_cfg</font>, 500k steps, EMA), the only CoBit model "
                "trained with prompt dropout. The model writes a Python function; we run it and compare the "
                "returned number with the gold answer. Screens: the first 250 problems, 1 seed, 256 steps. "
                "Confirmations: all 1319 problems × 3 seeds, 512 steps, paired bootstrap over problems. "
                "All on CSD3 A100s. Three sampler regimes:"))
    st.append(bullets([
        "<b>Regime A, deterministic</b> (γ=0): DDIM on the entropy-rate schedule. Baseline 13.8% (512 steps).",
        "<b>Regime A with churn</b> (γ=0.2–0.3): fresh noise is injected at each step (EDM \"churn\"). "
        "Same cost, much higher baseline.",
        "<b>Regime B</b> (γ=0.41, 1024 steps): the collaborator's strongest setting, baseline 29.6%.",
    ]))
    st.append(P("Note: the often-quoted 0.164 baseline is the 250-problem screen only; those problems are "
                "about 4 points easier than the full set. All conclusions below use full-set numbers "
                "unless marked n=250.", "small"))

    # ------------------------------------------------------------------ deterministic
    st.append(H1("2. Deterministic sampling: guidance works"))
    st.append(H2("Screens (n=250): finding each method's operating point"))
    cfg_w = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5, 7, 8, 10, 12, 15, 20, 30]
    cfgacc = []
    for w in cfg_w:
        for grid in ("cfg_coarse", "cfg_high"):
            k = (grid, 0.0, 256, 250, float(w), 0.0, "", 0.0, "")
            if k in c:
                cfgacc.append((w, c[k]))
                break
    st.append(table([
        ["CFG scale w", *[str(w) for w, _ in cfgacc[:9]]],
        ["accuracy", *[f"{100 * a:.1f}%" for _, a in cfgacc[:9]]],
        ["CFG scale w", *[str(w) for w, _ in cfgacc[9:]], ""],
        ["accuracy", *[f"{100 * a:.1f}%" for _, a in cfgacc[9:]], ""],
    ], header_rows=1, zebra=False))
    st.append(Spacer(1, 3))
    st.append(bullets([
        "<b>CFG</b> rises from w=1 and plateaus above w≈7, with no turnover up to w=30. Below w=1 it is "
        "catastrophic (0% at w=0.25–0.5), because it moves towards an unconditional model that cannot do "
        "the problem.",
        "<b>AG</b>: the 350k \"bad\" model is best and 425k worst at every scale. A bad model too similar to "
        "the good one gives a tiny, noisy difference. Only 2 of ~30 screen cells were individually "
        "significant, roughly chance.",
        "<b>SG-prev</b> helps at w=1–2 at every step count; <b>SG-exact</b> gets worse as its scale grows, "
        "at every step count. Holding the matched filter fixed beats varying it, more so at larger scale.",
        "The <b>null prompt</b> used for the unconditional branch barely matters (22.8% / 22.8% / 21.6%).",
        "<b>Steps:</b> the baseline plateaus near 17% (n=250) whatever the step count; CFG at only 8 steps "
        "already beats the baseline at any step count. Guidance buys a higher plateau, not a faster one.",
    ]))
    st.append(H2("Confirmation on the full test set"))
    st.append(figure(fig_confirm(conf), "guid_confirm", 160,
                     "Source: results/guidance/confirm_summary.json (n=1319, 3 seeds, 512 steps). All p &lt; 1e-4. "
                     "Re-scoring on the 1069 problems never used for selection moves every number by ≤ 0.2 pp."))
    st.append(P("The CFG optimum is a plateau (w=7, 12 and 20 are within noise of each other). "
                "SG-prev's gain is not a better integrator in disguise: a second-order Heun solver at the "
                "same cost gains nothing (−0.1 pp), and SG-prev beats it by +4.9 pp [+4.0, +5.9]."))
    st.append(H2("Is it worth the compute?"))
    st.append(P("CFG and AG double the cost. The fair comparison spends that budget on a second sample "
                "instead (numbers from GUIDANCE_REPORT.md and commit cdde4c6; voting keeps only programs "
                "that run):"))
    st.append(KeepTogether([table([
        ["At γ=0, 1024 network evaluations each", "Accuracy / change"],
        ["Best of 2 samples (oracle, pass@2)", "20.1%"],
        ["Majority vote of 2 samples (deployable)", "18.0%"],
        ["CFG w=12 vs oracle pass@2", "+1.8 pp [+0.3, +3.4]"],
        ["CFG w=12 vs vote of 2", "+3.3 pp [+2.0, +4.5]"],
        ["AG w=15 vs oracle pass@2", "+0.4 pp [−1.1, +1.8] (not significant)"],
        ["AG w=15 vs vote of 2", "+1.8 pp [+0.6, +3.0]"],
    ], col_widths=[100 * mm, 70 * mm])]))
    st.append(P("So in the deterministic regime CFG is genuinely worth its compute; AG only beats the "
                "deployable vote."))

    st.append(H2("Combining methods: always worse"))
    fac = [("CFG w=12", (12.0, 0.0, "", 0.0, ""), 1024), ("AG w=15", (0.0, 15.0, "000350000", 0.0, ""), 1024),
           ("AG + SG-prev", (0.0, 15.0, "000350000", 2.0, "prev"), 1024),
           ("SG-prev w=2", (0.0, 0.0, "", 2.0, "prev"), 512), ("no guidance", (0.0, 0.0, "", 0.0, ""), 512),
           ("CFG + SG-prev", (12.0, 0.0, "", 2.0, "prev"), 1024),
           ("SG-exact w=2", (0.0, 0.0, "", 2.0, "exact"), 1024),
           ("CFG + SG-exact", (12.0, 0.0, "", 2.0, "exact"), 2048),
           ("CFG + AG", (12.0, 15.0, "000350000", 0.0, ""), 2048),
           ("CFG + AG + SG-prev", (12.0, 15.0, "000350000", 2.0, "prev"), 2048)]
    rows = [["Cell (2×2×3 factorial, n=1319, 3 seeds)", "Accuracy", "NFE"]]
    for lab, k, nfe in fac:
        rows.append([lab, f"{100 * c[('factorial_confirm', 0.0, 512, 1319) + k]:.1f}%", str(nfe)])
    st.append(table(rows, col_widths=[90 * mm, 30 * mm, 30 * mm]))
    st.append(P("Every combination is worse than its best component, and CFG+AG collapses to 0%: "
                "stacking two amplifications drives the bits to hard 0/1 and the programs stop being code."))

    # ------------------------------------------------------------------ churn
    st.append(H1("3. With churn: the picture reverses"))
    st.append(figure(fig_churn(c), "guid_churn", 160,
                     "n=1319, 3 seeds, 256 steps (so γ=0 here is 13.7%, not the 13.8% at 512 steps). "
                     "Source: results/guidance/all_cells.csv, grids stoch_confirm and churn_anatomy."))
    st.append(table([
        ["Change vs the same-γ baseline (pp, 95% CI)", "γ = 0", "γ = 0.2", "γ = 0.3"],
        ["CFG w=12", "+8.0 [+6.7, +9.4]", "+0.9 [−0.6, +2.3]", "−0.5 [−1.8, +0.9]"],
        ["CFG re-tuned w=2 / w=4", "—", "—", "<b>+1.9 [+0.7, +3.2] / +1.7 [+0.5, +3.0]</b>"],
        ["AG w=15", "+6.9 [+5.6, +8.2]", "+0.2 [−1.2, +1.6]", "−1.7 [−3.1, −0.3]"],
        ["SG-prev w=2", "+5.0 [+4.1, +5.9]", "−17.4 [−19.0, −15.8]", "−21.2 [−23.1, −19.5]"],
        ["SG-prev, smallest tried (w=0.06)", "—", "—", "−1.3 [−2.5, −0.2]"],
        ["<b>Churn itself</b> (no guidance, vs γ=0)", "—", "<b>+10.5 [+9.2, +12.0]</b>", "<b>+12.0 [+10.6, +13.5]</b>"],
    ], col_widths=[64 * mm, 34 * mm, 34 * mm, 38 * mm]))
    st.append(P("CIs from docs/findings.md (F4–F7) and docs/report/numbers.json.", "small"))
    st.append(bullets([
        "<b>Churn is the single largest effect in the study</b>, at zero extra cost: the unguided γ=0.3 "
        "baseline beats CFG w=12 at γ=0 by +4.0 pp [+2.6, +5.4].",
        "<b>CFG was mis-tuned, not made useless:</b> its best scale moves from 12 down to 2–4 and its gain "
        "shrinks from +8 to +2 pp.",
        "<b>AG</b> becomes harmful (only w=15 was tested under churn). <b>SG-prev</b> fails at every scale, "
        "and the damage grows linearly with w, which points to a biased direction rather than noise.",
        "<b>Equal compute at γ=0.3:</b> a 2-sample vote without guidance (28.8%, 512 NFE) is at least as good "
        "as CFG w=2 (27.6%, 512 NFE): −1.2 pp [−2.7, +0.2] for CFG. Voting adds about +3 pp on top of either, "
        "so the two stack.",
        "<b>Regime B</b> (γ=0.41, 1024 steps; screen of n=250, 1 seed): nothing helps. Best is CFG w=2 at "
        "+0.4 pp [−3.6, +4.4]; AG w=15 −9.2, SG-exact −14.8, SG-prev w=2 goes to 0%.",
    ]))

    # ------------------------------------------------------------------ why
    st.append(H1("4. Why: the mechanisms we found"))
    st.append(H2("Guidance fixes some answers and breaks others, at fixed rates"))
    rows = [["Method (γ=0, n=1319, 3 seeds)", "Fixes (of wrong answers)", "Breaks (of right answers)",
             "Break-even accuracy a*"]]
    for name in ("CFG w=12", "CFG w=7", "CFG w=4", "SG-prev w=2", "AG w=15 bad=350k", "AG w=6 bad=250k"):
        r = t6[name]
        rows.append([name.replace(" bad=350k", "").replace(" bad=250k", " (250k bad model)"),
                     f"{100 * r['fix_rate']:.1f}%", f"{100 * r['break_rate']:.1f}%",
                     f"<b>{100 * r['a_star']:.0f}%</b> [{100 * r['a_star_ci95_binomial'][0]:.0f}, "
                     f"{100 * r['a_star_ci95_binomial'][1]:.0f}]"])
    st.append(table(rows, col_widths=[54 * mm, 38 * mm, 38 * mm, 40 * mm]))
    st.append(P("Guidance is not a targeted correction: CFG rescues ~10–14% of wrong answers and destroys "
                "~20–25% of right ones, in every regime we checked. The net effect is arithmetic: gain = "
                "fix·(1 − a) − break·a, which is positive only while the baseline accuracy a is "
                "<b>below</b> a* = fix/(fix + break) ≈ 33–35%. At 13.8% there are many wrong answers to fix, so "
                "guidance wins; churn raises the baseline to 25–30%, close to a*, and the margin disappears. "
                "(The 9 Sep meeting report states this the wrong way round, \"only positive above ≈33%\"; "
                "the correct direction is <i>below</i>.)"))
    st.append(H2("Guidance damages code validity, not arithmetic"))
    st.append(P("In the Regime B error analysis (n=250), the number of programs that do not run rises "
                "with guidance strength (98 baseline → 102 CFG w=2 → 126 AG w=15 → 166 SG-exact → 250 SG-prev "
                "w=2), while \"runs but wrong\" falls. Guidance mostly breaks answers by making the code invalid."))
    st.append(H2("Why each method fails where it does"))
    st.append(bullets([
        "<b>SG-prev under churn:</b> the cached prediction comes from the noisier state <i>before</i> the "
        "churn kick, so the difference points back toward a noisier estimate. Its size grows ~16× with "
        "churn while the step spacing is unchanged. Not yet tested directly.",
        "<b>SG-prev at w=2 in Regime B:</b> a runaway trajectory, not a decoding failure. The state "
        "becomes anomalous by step ~88; guidance and score then inflate together (positive feedback), and "
        "the output becomes token soup with no <font face='Mono'>def</font> in 100/100 samples.",
        "<b>AG:</b> needs a bad model that is different enough (350k) but not useless; under churn its "
        "correction adds harm.",
        "<b>SG-exact:</b> harmful in every regime; the one robust negative.",
        "<b>Overall pattern:</b> guidance's value is concentrated where the sampler is weak. Once the "
        "sampler is improved (churn), most of guidance's benefit is already captured.",
    ]))

    # ------------------------------------------------------------------ keep
    st.append(H1("5. What to keep"))
    st.append(callout(
        "<b>Keep:</b> churn (γ≈0.3) as the default sampler setting; <b>majority voting over programs that "
        "run</b> (the best known configuration is no guidance, γ=0.3, vote of 2: 28.8% at 512 NFE); CFG at "
        "w≈2 with churn only when a single sample is required (27.6%); the \"hold\" matched-filter mode; and "
        "the habit of quoting every scale together with its sampler regime.<br/>"
        "<b>Drop:</b> SG-exact (any scale), SG-prev (anywhere churn is used), AutoGuidance (needs a second "
        "model and turns harmful with churn), CFG w&lt;1, and every combination of methods.", "keep"))

    st.append(H1("6. Corrections to earlier write-ups"))
    st.append(bullets([
        "<b>0.164 is not the full-set baseline.</b> It is the first 250 problems, which are easier. The "
        "full-set deterministic baseline is 13.7% (256 steps) / 13.8% (512 steps).",
        "<b>Sign of the break-even rule</b> in docs/report_meeting/report.tex:87: positive <i>below</i> ≈33%, not "
        "above (correction drafted in docs/cpu_reanalysis.md, awaiting approval).",
        "<b>GUIDANCE_REPORT.md §6–9 is superseded</b> by docs/findings.md (F2, F5–F8): the factorial, solver and "
        "compute controls did run, and its recommendations do not hold once churn is on.",
        "<b>23.4% vs ~29%</b> are different models and samplers (base run at 250k with churn γ=0.34, vs "
        "425k with FKC churn 0.41), not a discrepancy. The ~29% replicates (28.1% on all 1319 problems); "
        "the gap to our 13.8% is churn, not steps or precision.",
        "The factorial \"interaction −0.166\" in docs/report/report.tex does not match any value on disk; "
        "docs/cross_regime_comparison.md mixes 256- and 512-step numbers in its γ=0 column.",
    ]))

    st.append(H1("7. Open questions (not pursued)"))
    st.append(bullets([
        "Why does churn help so much? It is the most valuable open question and is not about guidance.",
        "Test the SG-prev-under-churn explanation by recomputing the cached prediction after the churn "
        "kick.",
        "Larger votes under churn (maj@4 vs CFG w=2 maj@2 at equal compute), ~0.15 GPU-h.",
        "Caveats: one task, one model, one sampler family; differences below ~2 pp are not resolvable at "
        "n=1319; Regime B is n=250 with one seed; the production model itself carries the multi-GPU "
        "σ<sub>data</sub> training bug (T1), whose training-time effect is untested.",
    ]))
    st.append(Spacer(1, 6))
    st.append(P("<b>Sources.</b> docs/GUIDANCE.md, GUIDANCE_REPORT.md, findings.md, open_questions.md, "
                "cross_regime_comparison.md, tiny_gsm8k_error_analysis.md, trajectory_dynamics.md, "
                "replication_audit.md, best_baselines.md, report/report.tex, report_meeting/report.tex, "
                "cpu_reanalysis.md; diffusion/continuous/guidance.py; results/guidance/all_cells.csv, "
                "confirm_summary.json, confirm_summary_holdout250.json, paired_comparisons.json; "
                "results/cpu_reanalysis/t6_fix_break.json, t4_voting.json.", "small"))

    out = HERE / "report_guidance.pdf"
    build(out, st, "Guidance (CFG, AutoGuidance, Self-Guidance) — CoBit on TinyGSM/GSM8K", "Guidance study")
    print(out)


if __name__ == "__main__":
    main()
