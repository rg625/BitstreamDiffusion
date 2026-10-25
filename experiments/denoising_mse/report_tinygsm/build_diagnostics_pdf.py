"""TinyGSM / GSM8K sampling-time diagnostics (Maj@N, SC reset / decay, confidence clamping, staggered noise, failure
taxonomy) for CoBit-S / CoBit-M: PDF report. Data-driven: rebuild after new runs (analyze_diagnostics.py, then this
script); configurations without results are marked pending.
Inputs: ../results_tinygsm/diagnostics/{summary.json, fig_*.png}. Output: CoBit_TinyGSM_diagnostics.pdf (this folder)
"""
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent

REPO = HERE.parents[2]
OUT = HERE / "CoBit_TinyGSM_diagnostics.pdf"
FONTS = Path("/usr/share/fonts/truetype/liberation2")
for name, f in [("Serif", "LiberationSerif-Regular"), ("Serif-B", "LiberationSerif-Bold"),
                ("Serif-I", "LiberationSerif-Italic"), ("Serif-BI", "LiberationSerif-BoldItalic"),
                ("Sans", "LiberationSans-Regular"), ("Sans-B", "LiberationSans-Bold"),
                ("Sans-I", "LiberationSans-Italic"), ("Sans-BI", "LiberationSans-BoldItalic"),
                ("Mono", "LiberationMono-Regular")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f"{f}.ttf")))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-BI")

INK, INK2 = colors.HexColor("#1b1b1a"), colors.HexColor("#52514e")
ACCENT, RULE, TINT = colors.HexColor("#2a78d6"), colors.HexColor("#d9d8d2"), colors.HexColor("#eef3fa")
st = {
    "title": ParagraphStyle("title", fontName="Sans-B", fontSize=16, leading=20, textColor=INK, spaceAfter=5),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=9.5, leading=13, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Sans-B", fontSize=12, leading=15, textColor=INK, spaceBefore=10,
                         spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=10.2, leading=14, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=10.2, leading=13.8, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.5, leading=11.3, textColor=INK2, spaceBefore=3,
                          spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.6, leading=10.9, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=8.6, leading=10.9, textColor=INK),
    "box": ParagraphStyle("box", fontName="Serif", fontSize=10.2, leading=13.9, textColor=INK),
    "boxh": ParagraphStyle("boxh", fontName="Sans-B", fontSize=10.5, leading=13, textColor=ACCENT, spaceAfter=4),
}
W = A4[0] - 2 * 2.1 * cm


def P(t, s="body"):
    if s == "h1":
        return KeepTogether([CondPageBreak(3.2 * cm), Paragraph(t, st[s])])
    return Paragraph(t, st[s])


def M(t):
    return f"<font face='Mono' size='8.8'>{t}</font>"


def bullets(items, style="bullet"):
    return ListFlowable([ListItem(P(t, style), leftIndent=12, value="•") for t in items], bulletType="bullet",
                        start="•", leftIndent=12, bulletFontName="Serif", bulletFontSize=10, spaceAfter=6)


def figure(png, num, caption, width=W):
    from PIL import Image as PILImage

    w, h = PILImage.open(png).size
    return KeepTogether([Image(str(png), width=width, height=width * h / w), P(f"<b>Figure {num}.</b> {caption}", "cap")])


def table(rows, widths):
    data = [[P(c, "cellb" if i == 0 else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 5),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK2)]
                          + [("LINEBELOW", (0, i), (-1, i), 0.3, RULE) for i in range(1, len(rows) - 1)]))
    return t


def box(title, items):
    t = Table([[[P(title, "boxh"), bullets(items, "box")]]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(INK2)
    canvas.drawString(2.1 * cm, 1.2 * cm, "TinyGSM sampling diagnostics: Maj@N, self-conditioning, clamping, staggered noise (CoBit-S / CoBit-M, 250k)")
    canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()






# ------------------------------------------------------------------ numbers (all from summary.json)
import datetime  # noqa: E402

EXP = HERE.parent
RD = EXP / "results_tinygsm/diagnostics"
SU = json.loads((RD / "summary.json").read_text())
V, MJ, CL, FA = SU["variants"], SU["majn"], SU["clamping"], SU["failures"]
ORDER = ["screset4.0", "screset2.5", "screset1.3", "scdecay0.5", "scdecay0.8", "scdecay0.9", "clampS0.99",
         "clampP0.99", "clampS0.99s1.0", "clampP0.99s1.0", "stagger0.0", "stagger0.05", "stagger0.1", "stagger0.2"]
LABEL = {"base": "baseline (SC carry)", "screset4.0": "SC reset at σ ≤ 4.0", "screset2.5": "SC reset at σ ≤ 2.5",
         "screset1.3": "SC reset at σ ≤ 1.3", "scdecay0.5": "SC decay α = 0.5", "scdecay0.8": "SC decay α = 0.8",
         "scdecay0.9": "SC decay α = 0.9", "clampS0.99": "clamp, scattered, τ = 0.99",
         "clampP0.99": "clamp, prefix, τ = 0.99", "clampS0.99s1.0": "clamp, scattered, τ = 0.99, σ ≤ 1",
         "clampP0.99s1.0": "clamp, prefix, τ = 0.99, σ ≤ 1", "stagger0.0": "staggered, Δt = 0 (control)",
         "stagger0.05": "staggered, Δt = 0.05", "stagger0.1": "staggered, Δt = 0.10", "stagger0.2": "staggered, Δt = 0.20"}
PEND = "<i>pending</i>"


def ok(m, v):
    e = V.get(m, {}).get(v)
    return e is not None and len(e["seeds"]) >= 2


def acc(m, v):
    return f"{100 * V[m][v]['accuracy']:.1f}" if ok(m, v) else PEND


def dl(m, v, ci=True):
    if not ok(m, v):
        return PEND
    e = V[m][v]
    return f"{100 * e['delta']:+.1f}" + (f" [{100 * e['lo']:+.1f}, {100 * e['hi']:+.1f}]" if ci else "")


def mj(m, N, key="maj_mean"):
    e = MJ.get(m, {}).get("by_N", {}).get(str(N))
    return f"{100 * e[key]:.1f}" if e else PEND


pending = [f"CoBit-{m} {LABEL[v]}" for m in ("S", "M") for v in ORDER if not ok(m, v)]
nM = MJ.get("M", {}).get("seeds", 0)
if nM < 32:
    pending.append(f"CoBit-M Maj@N seeds {nM}–31 (Maj@{'/'.join(str(N) for N in (4, 8, 16, 32) if N > nM)} pending)")
fS, fM = FA["S"], FA.get("M")
sh = lambda f, k: f"{100 * f['shares'].get(k, 0):.1f}%" if f else PEND  # noqa: E731
nk = lambda f, k: f"{100 * f['not_runnable_kinds'].get(k, 0):.1f}%" if f else PEND  # noqa: E731
unrun = lambda f: f"{100 * (1 - f['shares'].get('wrong', 0)):.0f}%" if f else PEND  # noqa: E731
hS = fS["near_copy_substitution_hamming"]


def best(m, vs):
    """Best accuracy among the finished variants vs, with its label."""
    done = [v for v in vs if ok(m, v)]
    if not done:
        return PEND
    v = max(done, key=lambda v: V[m][v]["accuracy"])
    return f"{100 * V[m][v]['accuracy']:.1f} ({LABEL[v].split(' at ')[-1].replace('SC decay ', '')})" + \
        ("" if len(done) == len(vs) else ", partial")

story = [
    P("Why does CoBit lag autoregressive models on GSM8K? Sampling-time diagnostics on TinyGSM", "title"),
    P("CoBit-S (130M) and CoBit-M (462M), TinyGSM, 250k steps, EMA, float32 RoPE. GSM8K test (1319 problems), "
      "ddim_entropic, 256 steps, γ = 0.31, CoBit-S's schedule, self-conditioning carry. Every intervention is paired: "
      "the same initial and churn noise per seed as its baseline, 2 seeds, 95% bootstrap CIs over problems. "
      f"Built {datetime.date.today().isoformat()}.", "subtitle"),
]
if pending:
    story.append(box("Status: interim report, runs still in progress",
                     ["Pending (marked <i>pending</i> below; the report is rebuilt from the same scripts when they finish): "
                      + "; ".join(pending) + "."]))
    story.append(Spacer(1, 6))
story += [
    box("Summary", [
        f"<b>Ensembling works; within-trajectory interventions mostly do not.</b> Majority voting over N samples lifts "
        f"CoBit-S from {mj('S', 1)}% to {mj('S', 8)}% (N = 8, CoBit-M single-pass level) and {mj('S', 32)}% (N = 32), "
        f"and CoBit-M from {mj('M', 1)}% to {mj('M', 32)}% (N = 32). Capacity still wins per unit of compute: CoBit-M "
        f"Maj@4 ({mj('M', 4)}%) matches or beats CoBit-S Maj@32 ({mj('S', 32)}%) at about 3/8 of the model calls' cost "
        f"(one CoBit-M pass ≈ 3 CoBit-S passes). A correct program is among 32 samples for {mj('S', 32, 'pass_mean')}% "
        f"(CoBit-S) and {mj('M', 32, 'pass_mean')}% (CoBit-M) of problems: selecting the right sample is the larger gap.",
        f"<b>Staggered (position-dependent) noise is the only within-sampler change that helps</b>, without retraining "
        f"and at the same number of model calls: CoBit-S {dl('S', 'stagger0.05')} pp at Δt = 0.05 and "
        f"{dl('S', 'stagger0.1')} at Δt = 0.10; CoBit-M {dl('M', 'stagger0.05')} and {dl('M', 'stagger0.1')} "
        f"({acc('M', 'stagger0.1')}%). Δt ≈ 0.10 works for both models; it reduces non-runnable programs, while "
        f"Δt = 0.20 starts to truncate answers.",
        "<b>Self-conditioning is not a lock-in mechanism.</b> Resetting x̂₀ at σ = 4.0 / 2.5 / 1.3 or decaying it "
        "(α = 0.5–0.9) changes accuracy by at most about 1 pp on either model, all CIs include or touch 0.",
        f"<b>Per-bit confidence is not a commitment signal at high noise.</b> At pure noise (σ ≈ 77) "
        f"{100 * CL['S']['clampS0.99']['frac_clamped_step0']:.0f}% of CoBit-S's answer tokens already have all 16 bit "
        f"probabilities above 0.99; clamping them as specified collapses accuracy (CoBit-S {acc('S', 'clampS0.99')}%, "
        f"CoBit-M {acc('M', 'clampS0.99')}%). Clamping only below σ = 1 is harmless and useless (most tokens are decided).",
        f"<b>Failures are about global program consistency, not bit flips.</b> {unrun(fS)} of CoBit-S's wrong answers "
        f"do not run: undefined variables {nk(fS, 'raise:NameError')}, no return value {nk(fS, 'returns:NoneType')}, "
        f"syntax errors {nk(fS, 'raise:SyntaxError')}. Bit-level copy errors (a question number one digit off) are "
        f"{sh(fS, 'wrong_with_near_copy_error')} (CoBit-M {sh(fM, 'wrong_with_near_copy_error')}); they are somewhat "
        "more often 1 bit apart than chance, but too rare to matter.",
    ]),
    P("1. What was run, and changes from the requested protocol", "h1"),
    table([["Phase", "Run", "Change from the request, and why"],
           ["1 Maj@N", "base sampler, seeds 0–31 per model; vote over executed answers (ints; floats at the "
                       "scorer's 1e-3 tolerance; ties → earliest seed); + pass@N",
            "pass@N added (free; the ceiling of the ensemble). Maj@8/16 use the first seeds and, where possible, the "
            "mean over disjoint seed groups"],
           ["2 SC", "x̂₀ ← 0 once at the first step with σ ≤ 4.0 / 2.5 / 1.3; x̂₀ ← α x̂₀ + (1 − α) D, α = 0.5 / 0.8 / 0.9",
            "none (zeros is the no-SC training input, so in distribution)"],
           ["3 Clamping", "a token whose 16 bits all have max(p, 1 − p) > 0.99 is fixed to its hard bits for the rest "
                          "of sampling, like the prompt",
            "token level; two modes: scattered (as requested) and prefix (only extend the clamped prefix: in "
            "distribution, since training only has a clean prefix); plus σ ≤ 1 gated versions after the pilot showed "
            "commitments at pure noise"],
           ["4 Staggered", "token at relative answer position r: σ = σ_base(clip(u + Δt·r, 0, 1)), u from 1 to −Δt "
                           "over the same 255 steps (same NFE); Δt = 0 control",
            "the network takes one σ per sequence: it gets the largest answer-bit σ (the geometric mean collapsed: "
            "0/64 correct in the pilot, trailing tokens decoded as if clean); the matched filter uses each position's σ"],
           ["5 Failures", "all wrong base answers, both models; execution outcome, exception type, copy errors",
            "digits are single tokens with ids 32–41, so any digit swap is 1–3 bits apart (73% ≤ 2 bits by chance) "
            "and ASCII digits behave the same: bit distances are reported against that chance baseline"]],
          [0.12 * W, 0.42 * W, 0.46 * W]),
    P("The diagnostic sampler (tinygsm/diag_sampler.py) re-implements DDIMSampler's conditional SC path so variants can act "
      "inside a step; with no variant it reproduces DDIMSampler and the earlier runs problem by problem (seeds 0 and 1: "
      "CoBit-S 19.1 / 21.5%, CoBit-M 33.1 / 32.9%). No AR baseline accuracy was provided, so the AR gap is not "
      "quantified here.", "cap"),
    P("2. Maj@N and pass@N", "h1"),
    figure(RD / "fig_majn.png", 1, "Majority vote (solid) and pass@N (dashed, any sample correct) against the number of "
           "sampled trajectories (mean over disjoint seed groups).", width=0.62 * W),
    table([["N", "1", "2", "4", "8", "16", "32"],
           ["CoBit-S Maj@N"] + [mj("S", N) for N in (1, 2, 4, 8, 16, 32)],
           ["CoBit-S pass@N"] + [mj("S", N, "pass_mean") for N in (1, 2, 4, 8, 16, 32)],
           ["CoBit-M Maj@N"] + [mj("M", N) for N in (1, 2, 4, 8, 16, 32)],
           ["CoBit-M pass@N"] + [mj("M", N, "pass_mean") for N in (1, 2, 4, 8, 16, 32)]],
          [0.22 * W] + [0.13 * W] * 6),
    P("GSM8K accuracy [%]. Part of the gain at small N is skipping samples that do not run (they cast no vote). The gap "
      "between pass@N and Maj@N grows with N: the correct program is often in the pool but outvoted. "
      "Cost: one CoBit-M pass takes about 3× the time of a CoBit-S pass (24.5 vs 8 min for 1319 problems).", "cap"),
    P("3. Self-conditioning reset and decay", "h1"),
    table([["Variant", "CoBit-S acc", "Δ vs base [95% CI]", "CoBit-M acc", "Δ vs base [95% CI]"]] +
          [[LABEL[v], acc("S", v), dl("S", v), acc("M", v), dl("M", v)] for v in
           ["base", "screset4.0", "screset2.5", "screset1.3", "scdecay0.5", "scdecay0.8", "scdecay0.9"]],
          [0.28 * W, 0.13 * W, 0.23 * W, 0.13 * W, 0.23 * W]),
    P("Neither clearing nor smoothing the carried estimate helps. Together with the earlier finding that extra SC "
      "refreshes per step have no effect, the carried x̂₀ is neither a lock-in channel nor a bottleneck at 256 steps.",
      "cap"),
    P("4. Confidence clamping", "h1"),
    table([["Variant", "CoBit-S acc", "Δ vs base", "CoBit-M acc", "Δ vs base", "tokens clamped at step 0 (S / M)"]] +
          [[LABEL[v], acc("S", v), dl("S", v), acc("M", v), dl("M", v),
            " / ".join(f"{100 * CL[m][v]['frac_clamped_step0']:.0f}%" if v in CL.get(m, {}) else PEND for m in ("S", "M"))]
           for v in ["clampS0.99", "clampP0.99", "clampS0.99s1.0", "clampP0.99s1.0"]],
          [0.26 * W, 0.1 * W, 0.18 * W, 0.1 * W, 0.18 * W, 0.18 * W]),
    figure(RD / "fig_clamp_commit.png", 2, "Mean step at which an answer token is committed, by relative position in "
           "the answer (255 = never clamped). Ungated, early tokens commit at pure noise (step ≈ 0); gated at σ ≤ 1 "
           "(step ≈ 180), commitment happens after most decisions are made, so it cannot protect later tokens."),
    P("The model's per-bit marginals at high noise are saturated even where the joint answer is undecided (alignment of "
      "the restated question, line structure); committing them freezes mutually inconsistent tokens. Clamping only the "
      "prefix is less harmful than scattered clamping, as expected from training (a clean prefix is in distribution), "
      "but still loses about 13 pp ungated.", "cap"),
    P("5. Staggered (position-dependent) noise", "h1"),
    table([["Variant", "CoBit-S acc", "Δ vs base [95% CI]", "CoBit-M acc", "Δ vs base [95% CI]"]] +
          [[LABEL[v], acc("S", v), dl("S", v), acc("M", v), dl("M", v)] for v in
           ["stagger0.0", "stagger0.05", "stagger0.1", "stagger0.2"]],
          [0.28 * W, 0.13 * W, 0.23 * W, 0.13 * W, 0.23 * W]),
    P("Earlier answer tokens are denoised slightly ahead of later ones (Δt is the lag of the last position in diffusion "
      "time), at the same number of model calls. The Δt = 0 control runs the same per-position code path and matches the "
      "baseline. The gain comes from fewer non-runnable programs; at Δt = 0.20 more answers end without a return value. "
      "The models were never trained with per-position noise, so this is a lower bound on what a trained "
      "staggered/causal schedule could give.", "cap"),
    figure(RD / "fig_variants.png", 3, "All interventions: paired change in GSM8K accuracy vs the baseline (same seeds), "
           "95% CIs; right: zoom."),
    CondPageBreak(9 * cm),
    P("6. Failure taxonomy (Phase 5)", "h1"),
    figure(RD / "fig_failures.png", 4, f"Wrong answers of the baseline runs by outcome (CoBit-S {fS['runs']} runs, "
           f"CoBit-M {fM['runs'] if fM else 0} runs)."),
    table([["", "CoBit-S", "CoBit-M"],
           ["wrong answers analysed", f"{fS['n_wrong']:,}", f"{fM['n_wrong']:,}" if fM else PEND],
           ["do not run (any reason)", unrun(fS), unrun(fM)],
           ["  undefined variable (NameError)", nk(fS, "raise:NameError"), nk(fM, "raise:NameError")],
           ["  function returns no value", nk(fS, "returns:NoneType"), nk(fM, "returns:NoneType")],
           ["  syntax error", nk(fS, "raise:SyntaxError"), nk(fM, "raise:SyntaxError")],
           ["  variable used before assignment", nk(fS, "raise:UnboundLocalError"), nk(fM, "raise:UnboundLocalError")],
           ["contain invalid token ids (≥ vocab)", sh(fS, "has_invalid_ids"), sh(fM, "has_invalid_ids")],
           ["run, a constant not taken from the question", sh(fS, "wrong_with_foreign_constant"),
            sh(fM, "wrong_with_foreign_constant")],
           ["run, near-copy error (question number one digit off)", sh(fS, "wrong_with_near_copy_error"),
            sh(fM, "wrong_with_near_copy_error")],
           ["near-copy digit substitutions: ≤ 1 bit apart (chance)",
            f"{100 * hS['le1_observed']:.0f}% ({100 * hS['le1_chance']:.0f}%), n = {hS['n']}",
            (f"{100 * fM['near_copy_substitution_hamming']['le1_observed']:.0f}% "
             f"({100 * hS['le1_chance']:.0f}%), n = {fM['near_copy_substitution_hamming']['n']}") if fM else PEND],
           ["near-copy digit substitutions: ≤ 2 bits apart (chance)",
            f"{100 * hS['le2_observed']:.0f}% ({100 * hS['le2_chance']:.0f}%)",
            (f"{100 * fM['near_copy_substitution_hamming']['le2_observed']:.0f}% ({100 * hS['le2_chance']:.0f}%)") if fM else PEND],
           ["predicted vs gold answer, same length and ≤ 2 ASCII bits apart",
            f"{100 * fS['ascii_hamming']['le2_same_length']:.0f}%",
            f"{100 * fM['ascii_hamming']['le2_same_length']:.0f}%" if fM else PEND]],
          [0.52 * W, 0.24 * W, 0.24 * W]),
    P("Bit-level artifacts are rare: invalid token ids appear in well under 1% of wrong answers, and near-copy errors are "
      f"about 2%. Their digit swaps are more often 1 bit apart than a random swap (CoBit-S {100 * hS['le1_observed']:.0f}%"
      + (f", CoBit-M {100 * fM['near_copy_substitution_hamming']['le1_observed']:.0f}%" if fM else "")
      + f" vs {100 * hS['le1_chance']:.0f}% by chance), so some of them are probably bit-level slips, but they explain "
      "only a small fraction of the errors either way. The ASCII "
      "Hamming criterion requested is shown for completeness but cannot separate bit flips from semantic errors (digits "
      "differ in their low ASCII bits). The dominant failure is a program that is locally plausible but globally "
      "inconsistent: names defined in one line and used differently in another, missing returns. CoBit-M makes these "
      "less often, and much of Maj@N's gain is filtering them out.", "cap"),
    P("7. Summary table (GSM8K accuracy, %)", "h1"),
    table([["Configuration", "CoBit-S", "CoBit-M"],
           ["baseline, single pass (SC carry)", acc("S", "base"), acc("M", "base")],
           ["Maj@8 / Maj@16 / Maj@32", f"{mj('S', 8)} / {mj('S', 16)} / {mj('S', 32)}",
            f"{mj('M', 8)} / {mj('M', 16)} / {mj('M', 32)}"],
           ["pass@32 (oracle)", mj("S", 32, "pass_mean"), mj("M", 32, "pass_mean")],
           ["best SC reset", best("S", ("screset4.0", "screset2.5", "screset1.3")),
            best("M", ("screset4.0", "screset2.5", "screset1.3"))],
           ["best SC decay", best("S", ("scdecay0.5", "scdecay0.8", "scdecay0.9")),
            best("M", ("scdecay0.5", "scdecay0.8", "scdecay0.9"))],
           ["clamping τ = 0.99 (scattered / prefix)", f"{acc('S', 'clampS0.99')} / {acc('S', 'clampP0.99')}",
            f"{acc('M', 'clampS0.99')} / {acc('M', 'clampP0.99')}"],
           ["staggered noise Δt = 0.05 / 0.10 / 0.20",
            f"{acc('S', 'stagger0.05')} / {acc('S', 'stagger0.1')} / {acc('S', 'stagger0.2')}",
            f"{acc('M', 'stagger0.05')} / {acc('M', 'stagger0.1')} / {acc('M', 'stagger0.2')}"]],
          [0.5 * W, 0.25 * W, 0.25 * W]),
    P("8. Takeaways for the diffusion-time MoE router", "h1"),
    bullets([
        "<b>Spend extra compute across trajectories, not within one.</b> Re-processing the same trajectory (SC "
        "refreshes, SC resets) buys nothing, while independent samples help a lot (Maj@8 of CoBit-S matches CoBit-M "
        "single-pass), and a larger model helps most per unit of compute (CoBit-M Maj@4 vs CoBit-S Maj@32). "
        "A router that adds capacity should change what is computed (a larger expert, as in the switching study), not "
        "repeat the same small computation.",
        "<b>Position matters, and noise level alone does not capture it.</b> A small per-position offset in noise level "
        "helps without retraining. A router keyed only on diffusion time misses this; giving it (or the schedule) the "
        "position within the answer is a cheap extension, and training with per-position noise should do better.",
        "<b>Do not route on the model's own per-bit confidence at high noise</b>: it is saturated where the answer is "
        "still undecided. Confidence becomes informative only around σ ≲ 1, after most decisions are made.",
        "<b>The target is global consistency of the program</b> (variable names, return, structure), which a "
        "larger expert improves and which bit-level corrections would not. Verification or reranking of samples "
        "(the pass@N − Maj@N gap) is the largest remaining lever at test time.",
    ]),
    P("9. Caveats", "h1"),
    bullets([
        "Two seeds per intervention; effects of 1–2 pp are near the resolution (CI half-width ≈ 0.5–1.5 pp, paired).",
        "One sampler setting (256 steps, γ = 0.31, CoBit-S schedule). Interventions may behave differently at fewer steps.",
        "Staggered noise and clamping feed the networks inputs they were not trained on; the results bound what the "
        "trained versions could do rather than measure them.",
        "Failure categories come from executing the generated code in the scoring sandbox; the copy-error heuristic only "
        "covers constants assigned directly (x = 45) and numbers of at least two digits.",
    ]),
    P("Files: experiments/denoising_mse/tinygsm/{diag_sampler.py, diagnostics.py, run_diagnostics.sh, "
      "analyze_diagnostics.py}; results in experiments/denoising_mse/results_tinygsm/diagnostics/ (summary.json, "
      "variants.csv, majn.csv, raw/ per-run JSON with per-problem answers and texts). Rebuild: analyze_diagnostics.py, "
      "then report_tinygsm/build_diagnostics_pdf.py.", "cap"),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="TinyGSM sampling diagnostics", author="BitstreamDiffusion experiments")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
