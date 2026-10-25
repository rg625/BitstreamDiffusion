"""TinyGSM report built to mirror the OWT study exactly: the same analysis scripts (analyze.py, analyze_levels.py,
plot_sqrt_rate.py, analyze_traj.py, analyze_gt.py, run with DMSE_TASK=tinygsm) produce the same figures and the same
summary keys for both tasks, and this report sets them side by side.

Inputs: ../results/{main,levels,traj,gt}/ (OWT) and ../results_tinygsm/{static,levels,traj,gt}/ (TinyGSM), plus
runs/denoising_mse/{traj/main/scores.json, tinygsm/gsm8k/*.json}. Requires reportlab + numpy + pillow (throwaway venv).
Output: CoBit_TinyGSM_report.pdf (this folder)
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
RES = HERE.parent / "results_tinygsm" / "S"
REPO = HERE.parents[2]
OUT = HERE / "CoBit_TinyGSM_report.pdf"
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
    canvas.drawString(2.1 * cm, 1.2 * cm, "CoBit-S vs CoBit-M on TinyGSM (unguided, 250k)")
    canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


# ------------------------------------------------------------------ numbers (both tasks, identical keys)
EXP = HERE.parent
J = lambda p: json.loads(Path(p).read_text())  # noqa: E731
O = dict(main=J(EXP / "results/main/summary.json"), lv=J(EXP / "results/levels/schedule_match.json"),
         sq=J(EXP / "results/levels/rate_vs_sqrt.json"), tr=J(EXP / "results/traj/traj_summary.json"),
         gt=J(EXP / "results/gt/gt_summary.json"))
T = dict(main=J(EXP / "results_tinygsm/static/summary.json"), lv=J(EXP / "results_tinygsm/levels/schedule_match.json"),
         sq=J(EXP / "results_tinygsm/levels/rate_vs_sqrt.json"), tr=J(EXP / "results_tinygsm/traj/traj_summary.json"),
         gt=J(EXP / "results_tinygsm/gt/gt_summary.json"))
SM = J(EXP / "results_tinygsm/SM/summary.json")
S1 = J(EXP / "results_tinygsm/S/summary.json")
GS = {}
for f in sorted((REPO / "runs/denoising_mse/tinygsm/gsm8k").glob("*.json")):
    d = J(f)
    if d["n"] == 1319 and d["rope"] == "fp32":
        GS[(d["model"], d.get("sched", d["model"]))] = d
own = lambda r: next(v for k, v in r["rates"].items() if k.startswith(("same-input disagreement rate", "same-input rate, CoBit-S")))  # noqa: E731
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
F = {"static": EXP / "results_tinygsm/static", "lv": EXP / "results_tinygsm/levels", "tr": EXP / "results_tinygsm/traj",
     "gt": EXP / "results_tinygsm/gt"}


def r_(d, key):
    return next(v for k, v in d["rates"].items() if k.startswith(key))


def schedrow(d, m):
    return d["lv"]["schedule_vs_surplus"]["per_schedule"][m]


def first20(d, m):
    b = d["lv"]["surplus_per_step_bin"][m]
    sh, tb = np.array(b["share_per_bin_mean1"]), np.array(b["t_bin_edges"])
    return float(np.interp(0.2, tb[1:], np.cumsum(sh) / sh.sum()))


pa = SM["paired"]
cmp_rows = [
    ["Quantity (same definition, same code)", "OWT", "TinyGSM"],
    ["Static gain S − M: peak (per-bit MSE, loss region)",
     f"{O['main']['train']['max_diff']:.2e} at σ {O['main']['train']['sigma_at_max_diff']:.2f}",
     f"{T['main']['train']['max_diff']:.2e} at σ {T['main']['train']['sigma_at_max_diff']:.2f}"],
    ["noise levels where M is significantly worse",
     f"σ {min(x for x in O['main']['train']['sigma_where_M_worse_significant'] if x > 0.08):.1f}–"
     f"{max(O['main']['train']['sigma_where_M_worse_significant']):.0f}",
     ("none" if not [x for x in T['main']['train']['sigma_where_M_worse_significant'] if x > 0.1] else
      f"σ {min(x for x in T['main']['train']['sigma_where_M_worse_significant'] if x > 0.1):.1f}–"
      f"{max(T['main']['train']['sigma_where_M_worse_significant']):.0f}")],
    ["CoBit-S entropy rate MSE/σ²: peak / median σ",
     f"{O['sq']['rate_S']['sigma_peak']:.2f} / {O['sq']['rate_S']['sigma_median']:.2f}",
     f"{T['sq']['rate_S']['sigma_peak']:.2f} / {T['sq']['rate_S']['sigma_median']:.2f}"],
    ["surplus rate (MSE_S − MSE_M)/σ²: peak / median σ",
     f"{O['lv']['schedule_vs_surplus']['sigma_peak_surplus_rate']:.2f} / {O['lv']['schedule_vs_surplus']['sigma_median_surplus_rate']:.2f}",
     f"{T['lv']['schedule_vs_surplus']['sigma_peak_surplus_rate']:.2f} / {T['lv']['schedule_vs_surplus']['sigma_median_surplus_rate']:.2f}"],
    ["schedule rule on measured error vs stored table (TV), S / M",
     f"{schedrow(O, 'S')['schedule_vs_rule_on_measured_mse_TV']:.3f} / {schedrow(O, 'M')['schedule_vs_rule_on_measured_mse_TV']:.3f}",
     f"{schedrow(T, 'S')['schedule_vs_rule_on_measured_mse_TV']:.3f} / {schedrow(T, 'M')['schedule_vs_rule_on_measured_mse_TV']:.3f}"],
    ["overlap surplus / schedule, S / M",
     f"{schedrow(O, 'S')['overlap_coefficient']:.2f} / {schedrow(O, 'M')['overlap_coefficient']:.2f}",
     f"{schedrow(T, 'S')['overlap_coefficient']:.2f} / {schedrow(T, 'M')['overlap_coefficient']:.2f}"],
    ["surplus in the central 50% of steps, S / M schedule",
     f"{100 * schedrow(O, 'S')['surplus_fraction_in_schedule_25_75']:.0f}% / {100 * schedrow(O, 'M')['surplus_fraction_in_schedule_25_75']:.0f}%",
     f"{100 * schedrow(T, 'S')['surplus_fraction_in_schedule_25_75']:.0f}% / {100 * schedrow(T, 'M')['surplus_fraction_in_schedule_25_75']:.0f}%"],
    ["surplus in the 20% of steps nearest the clean end, S / M schedule",
     f"{100 * first20(O, 'S'):.0f}% / {100 * first20(O, 'M'):.0f}%", f"{100 * first20(T, 'S'):.0f}% / {100 * first20(T, 'M'):.0f}%"],
    ["quality on shared seeds (paired, CoBit-S schedule)",
     f"GenPPL S {O['tr']['scores']['S']['genppl']:.1f} vs M {O['tr']['scores']['M']['genppl']:.1f}",
     f"GSM8K S {pct(pa['acc_S'])} vs M {pct(pa['acc_M'])}"],
    ["final token disagreement of the paired samples",
     pct(O['tr']['final_token_disagreement']), pct(T['tr']['final_token_disagreement'])],
    ["token divergence of paired generation: peak / median σ",
     f"{r_(O['tr'], 'divergence density: growth of token')['sigma_peak']:.2f} / {r_(O['tr'], 'divergence density: growth of token')['sigma_median']:.2f}",
     f"{r_(T['tr'], 'divergence density: growth of token')['sigma_peak']:.2f} / {r_(T['tr'], 'divergence density: growth of token')['sigma_median']:.2f}"],
    ["squared state-gap growth: peak / median σ",
     f"{r_(O['tr'], 'divergence density: growth of squared')['sigma_peak']:.2f} / {r_(O['tr'], 'divergence density: growth of squared')['sigma_median']:.2f}",
     f"{r_(T['tr'], 'divergence density: growth of squared')['sigma_peak']:.2f} / {r_(T['tr'], 'divergence density: growth of squared')['sigma_median']:.2f}"],
    ["same-input disagreement rate (S / M trajectories): median σ",
     f"{r_(O['gt'], 'same-input rate, CoBit-S')['sigma_median']:.2f} / {r_(O['gt'], 'same-input rate, CoBit-M')['sigma_median']:.2f}",
     f"{r_(T['gt'], 'same-input rate, CoBit-S')['sigma_median']:.2f} / {r_(T['gt'], 'same-input rate, CoBit-M')['sigma_median']:.2f}"],
    ["same-input token disagreement peak (S trajectories), at t",
     f"{pct(O['gt']['tokdis_peak_Straj']['value'])} at {O['gt']['tokdis_peak_Straj']['t']:.2f}",
     f"{pct(T['gt']['tokdis_peak_Straj']['value'])} at {T['gt']['tokdis_peak_Straj']['t']:.2f}"],
    ["grey curve area: static / zero SC / one-step / carry",
     " / ".join(f"{v['integral_nats_per_bit']:.4f}" for v in O['gt']['gray_curve'].values()),
     " / ".join(f"{v['integral_nats_per_bit']:.4f}" for v in T['gt']['gray_curve'].values())],
    ["true chains, token disagreement peak: zero / carry SC",
     f"{pct(O['gt']['tokdis_peak_true_zeros']['value'])} / {pct(O['gt']['tokdis_peak_true_carry']['value'])}",
     f"{pct(T['gt']['tokdis_peak_true_zeros']['value'])} / {pct(T['gt']['tokdis_peak_true_carry']['value'])}"],
]

g = {k: v for k, v in GS.items()}
acc = lambda m, s: g[(m, s)]["accuracy"]  # noqa: E731
lvT = T["lv"]
seqT = lvT["sequence_level"]
tokT = lvT["token_level"]
sqT = T["sq"]
grT = list(T["gt"]["gray_curve"].values())
trT = T["tr"]

story = [
    P("CoBit-S vs CoBit-M on TinyGSM: the OWT study, repeated exactly", "title"),
    P("Unguided CoBit-S (130M) and CoBit-M (462M), step 250k (EMA) · TinyGSM → GSM8K · experiments/denoising_mse · "
      "7 October 2026 · every figure and summary number below is produced by the same analysis code as the OWT reports",
      "subtitle"),
    box("Summary", [
        f"<b>CoBit-M is much better on GSM8K</b>: {pct(pa['acc_S'])} vs {pct(pa['acc_M'])} on shared seeds and schedule "
        f"(+{100 * pa['diff_M_minus_S']:.1f} pp, CI +{100 * pa['diff_ci95'][0]:.1f} to +{100 * pa['diff_ci95'][1]:.1f}); the "
        f"schedule choice hardly matters (2 × 2 table, section 1).",
        f"<b>Statically, CoBit-M is better at every noise level</b> (OWT: worse for σ ≈ 1.3–20), with the gain peaking at "
        f"σ ≈ {T['main']['train']['sigma_at_max_diff']:.1f}, and TinyGSM's information is decided at higher noise "
        f"(entropy-rate peak σ ≈ {sqT['rate_S']['sigma_peak']:.1f} vs {O['sq']['rate_S']['sigma_peak']:.2f}).",
        f"<b>The schedules match the gain far better than on OWT</b>: no slice of steps carries more than about "
        f"{max(lvT['surplus_per_step_bin']['S']['max_share'], lvT['surplus_per_step_bin']['M']['max_share']):.1f}× its "
        f"share (OWT {max(O['lv']['surplus_per_step_bin']['S']['max_share'], O['lv']['surplus_per_step_bin']['M']['max_share']):.1f}×), "
        f"and the 20% of steps nearest the clean end hold {100 * first20(T, 'S'):.0f}% of the gain (OWT {100 * first20(O, 'S'):.0f}%).",
        f"<b>The divergence picture holds</b>: the generations split at higher noise than where the models locally disagree, "
        f"through amplification. But on TinyGSM the models already disagree on {pct(pa['tokdis_first_step'])} of answer "
        f"tokens at pure noise, and the paired samples end only {pct(trT['final_token_disagreement'])} apart (OWT 71%).",
        f"<b>One-step self-conditioning leaves the grey curve unchanged</b> on both tasks (TinyGSM area {grT[2]['integral_nats_per_bit']:.4f} "
        f"vs {grT[1]['integral_nats_per_bit']:.4f} with zero SC); carrying the full history inflates it.",
    ]),

    P("OWT and TinyGSM side by side", "h1"),
    P("Every row is computed by the same code from the same summary keys for both tasks (σ cut-offs: 0.08 for OWT, the "
      "sampler's terminal σ; 0.1 for TinyGSM, whose sampler runs to 0.002 and whose CoBit-S has a tiny error spike near "
      "σ ≈ 0.03)."),
    table(cmp_rows, [6.6 * cm, 4.9 * cm, W - 11.5 * cm]),

    P("1. Setup, checks and GSM8K accuracy", "h1"),
    P("TinyGSM examples are [BOS] question \\n code [EOS] (SmolLM-135M, 512 tokens, 16 bits/token); the question is clamped "
      "clean and errors are measured over the answer region. The validation split is rebuilt as in training "
      "(118,215 examples); generation uses the 1,319 GSM8K test prompts. Both checkpoints use σ_data = 0.3998 and float32 "
      "rotary positions (the fp32 fix leaves GSM8K accuracy unchanged on this task, see the CoBit-S report). Paired runs "
      "share the sampler (γ = 0.31, 256 NFE, CoBit-S's schedule) and every random number; CoBit-S against itself is "
      "bit-identical."),
    table([
        ["GSM8K accuracy (γ = 0.31, 256 NFE)", "CoBit-S schedule", "CoBit-M schedule"],
        ["CoBit-S", f"{pct(acc('S', 'S'))}", f"{pct(acc('S', 'M'))}"],
        ["CoBit-M", f"{pct(pa['acc_M'])} (paired run)", f"{pct(acc('M', 'M'))}"],
    ], [6.0 * cm, 4.4 * cm, W - 10.4 * cm]),
    Spacer(1, 4),
    P(f"On shared seeds: {pa['both']} problems solved by both, {pa['only_M']} only by CoBit-M, {pa['only_S']} only by "
      "CoBit-S (OWT's counterpart: CoBit-M's sample had lower GPT-2 NLL on 65% of examples)."),

    P("2. Static denoising error", "h1"),
    figure(F["static"] / "fig_mse_sigma.png", 1,
           "Per-bit MSE over the training loss region (answer + EOS + padding), S − M with ±2 SE, and relative difference.",
           width=0.62 * W),
    figure(F["static"] / "fig_variants.png", 2,
           "S − M by self-conditioning input; top: loss region, bottom: answer tokens only.", width=W),
    figure(F["static"] / "fig_mse_time_own.png", 3,
           "As Figure 1 against diffusion time on each model's own schedule (S's and M's schedule versions are in "
           "results_tinygsm/static/).", width=0.62 * W),

    P("3. Token and sequence level", "h1"),
    figure(F["lv"] / "fig_token_level.png", 4, "Answer-token error rate, NLL and bit error rate, and S − M, against t.", width=0.88 * W),
    figure(F["lv"] / "fig_global_time.png", 5, "Per-sequence totals: wrong answer tokens and answer NLL per sequence.", width=0.88 * W),
    figure(F["lv"] / "fig_sequence_level.png", 6,
           "Per-sequence MSE trajectories with quantile bands, distribution of the paired per-sequence difference, and the "
           "fraction of sequences on which CoBit-M has the lower MSE.", width=0.88 * W),
    figure(F["lv"] / "fig_token_position_rel.png", 7,
           "Answer-token error rate against position from the answer start (the OWT-style absolute-position figure is in "
           "results_tinygsm/levels/).", width=W),
    P(f"Peak answer-token gain: {100 * tokT['max_terC_diff']:.1f} pp at t ≈ {tokT['t_at_max_terC_diff']:.2f}, about "
      f"{tokT['max_wrongtok_per_seq_diff']:.1f} fewer wrong tokens per sequence (answers average "
      f"{tokT['mean_content_tokens_per_seq']:.0f} tokens). Error grows along the answer: later tokens, which depend on "
      "earlier ones, are both harder and where CoBit-M gains most."),

    P("4. Schedule versus the gain from the extra parameters", "h1"),
    figure(F["lv"] / "fig_schedule_overlay.png", 8,
           "Schedules vs the surplus rate; surplus covered per 2% slice of sampler steps; cumulative surplus vs steps.",
           width=W),
    figure(F["lv"] / "fig_rate_vs_sqrt.png", 9,
           "Entropy rate, sqrt-rate, schedule rule on measured error, and stored schedule, for both models, with the surplus.",
           width=0.9 * W),
    P(f"The schedule rule applied to the measured error reproduces both stored TinyGSM schedules (TV "
      f"{schedrow(T, 'S')['schedule_vs_rule_on_measured_mse_TV']:.3f} and {schedrow(T, 'M')['schedule_vs_rule_on_measured_mse_TV']:.3f}). "
      f"The surplus sits below the schedules' centre, as on OWT, but much less so: the per-slice share peaks at about "
      f"{lvT['surplus_per_step_bin']['S']['max_share']:.1f}× near t ≈ {lvT['surplus_per_step_bin']['S']['t_of_max']:.2f} and the "
      "cumulative curve stays close to the diagonal."),

    P("5. Paired generation and divergence vs MSE/σ²", "h1"),
    figure(F["tr"] / "fig_traj_distance.png", 10,
           "Paired trajectories on the GSM8K prompts: state gap, relative gap, prediction entropy, denoiser gap and token "
           "disagreement (own trajectories and same input along S's), and the same-input disagreement rate.", width=W),
    figure(F["tr"] / "fig_traj_vs_rate.png", 11,
           "Divergence densities of the paired generations against the same-input rate and the static rates.", width=0.66 * W),

    P("6. Same-input disagreement, ground-truth chains and the grey curve", "h1"),
    figure(F["gt"] / "fig_same_input.png", 12, "Disagreement on identical inputs along CoBit-S, CoBit-M and true trajectories.",
           width=0.9 * W),
    figure(F["gt"] / "fig_truechain.png", 13,
           "Answer-token MSE along true chains with carried, one-step and zero self-conditioning; effect of self-conditioning; "
           "CoBit-M's advantage along true chains.", width=W),
    figure(F["gt"] / "fig_gray_curve.png", 14, "The surplus rate (grey curve) static and on true chains with the three "
           "self-conditioning inputs, over answer tokens.", width=0.9 * W),
    figure(F["gt"] / "fig_rates_all.png", 15, "All densities together.", width=0.7 * W),
    P(f"Same-input disagreement is largest at pure noise on TinyGSM ({pct(T['gt']['tokdis_peak_Straj']['value'])} of answer "
      f"tokens) and falls as the answer appears, whereas on OWT it peaked mid-way ({pct(O['gt']['tokdis_peak_Straj']['value'])} "
      f"at t ≈ {O['gt']['tokdis_peak_Straj']['t']:.2f}). The one-step self-conditioning input leaves the grey curve where the "
      "static measurement puts it on both tasks; the full carried history inflates it. Carrying the history does not raise "
      f"the models' disagreement on true TinyGSM states ({pct(T['gt']['tokdis_peak_true_zeros']['value'])} with zero SC, "
      f"{pct(T['gt']['tokdis_peak_true_carry']['value'])} carried), whereas on OWT it doubled it "
      f"({pct(O['gt']['tokdis_peak_true_zeros']['value'])} → {pct(O['gt']['tokdis_peak_true_carry']['value'])})."),

    P("7. Caveats and files", "h1"),
    bullets([
        "Shared sampler for paired runs: OWT γ = 0.13, TinyGSM γ = 0.31 (each CoBit-S's best 256-NFE setting); CoBit-M's "
        "optimum may differ. Quality metrics differ by task (GenPPL vs GSM8K accuracy).",
        f"Analysis scripts take DMSE_TASK=tinygsm ({M('analyze.py')}, {M('analyze_levels.py')}, {M('plot_sqrt_rate.py')}, "
        f"{M('analyze_traj.py')}, {M('analyze_gt.py')}); the OWT outputs were re-generated and are unchanged. TinyGSM results: "
        f"{M('results_tinygsm/{static,levels,traj,gt}/')}; runs: {M('runs/denoising_mse/tinygsm/')}.",
    ]),
    P("Appendix: additional figures", "h1"),
    figure(EXP / "report_tinygsm/fig2_owt_vs_tinygsm.png", "A1",
           "CoBit-S entropy rate MSE/σ² and stored schedules, OWT vs TinyGSM (unit area on σ ≥ 0.1): TinyGSM's information "
           "is decided at about three times higher noise.", width=0.72 * W),
    figure(F["static"] / "fig_mse_time_S.png", "A2",
           "Static error against diffusion time on CoBit-S's schedule for both models (the common axis used throughout).",
           width=0.6 * W),
    figure(F["static"] / "fig_mse_time_M.png", "A3", "As A2 on CoBit-M's schedule.", width=0.6 * W),
    figure(EXP / "results_tinygsm/SM/fig_static.png", "A4",
           "Answer tokens only: answer-token MSE for both models and S − M over answer tokens and over the loss region "
           "(±2 SE).", width=W),
    figure(EXP / "results_tinygsm/SM/fig_token.png", "A5",
           "Answer-token error-rate and NLL differences S − M against CoBit-S's diffusion time (±2 SE).", width=0.78 * W),
    figure(F["lv"] / "fig_token_position.png", "A6",
           "Answer-token error rate against absolute position in the 512-token sequence (the OWT-style version of Figure 7; "
           "the answer starts after a variable-length prompt, about 51 tokens on average).", width=W),

]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="CoBit-S vs CoBit-M on TinyGSM: the OWT study repeated",
                        author="BitstreamDiffusion experiments")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
