"""Full TinyGSM report: CoBit-S (130M) vs CoBit-M (462M), unguided, step 250k. Supersedes the CoBit-S-only
report (CoBit_TinyGSM_S_report.pdf) and mirrors the OWT reports (static error, schedule, paired generation,
same-input decompositions, ground-truth chains with zero / one-step / carried self-conditioning).

Numbers are read at build time from ../results_tinygsm/{S,SM}/summary.json and runs/denoising_mse/tinygsm/gsm8k/;
figures from ../results_tinygsm/{S,SM}/ and make_figs.py. Requires reportlab + numpy + pillow (throwaway venv).
Output: CoBit_TinyGSM_report.pdf (this folder)
"""
import json
import shutil
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


# ------------------------------------------------------------------ numbers
SM = HERE.parent / "results_tinygsm" / "SM"
S1 = json.loads((RES / "summary.json").read_text())
D = json.loads((SM / "summary.json").read_text())
for f in ("fig_static.png", "fig_token.png", "fig_schedule.png", "fig_traj.png", "fig_same_input.png",
          "fig_truechain.png", "fig_gray_curve.png", "fig_rates_all.png"):
    shutil.copy(SM / f, HERE / f"sm_{f}")
G = D["gsm8k_own_schedule"]
gS, gM = G["S_g0.31_s256"], G["M_g0.31_s256"]
pa, stt, si, tc, Rt = D["paired"], D["static"], D["same_input"], D["truechain"], D["rates"]
gr = tc["gray"]
rope = S1["gsm8k"]
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
rk = {"surplus": "static surplus rate (MSE_S − MSE_M)/σ²", "entS": "static entropy rate MSE_S/σ²",
      "entM": "static entropy rate MSE_M/σ²", "div": "token divergence, paired generation",
      "divD": "denoiser-gap growth, paired generation", "locS": "same-input rate, S trajectories",
      "locM": "same-input rate, M trajectories", "true": "same-input rate, true chains (carry SC)"}
OWT = dict(entS_peak=0.56, surplus_peak=0.40, div_peak=2.5, loc_peak=0.49, overlap_S=0.42, central_S=0.16, final_tokdis=0.71,
           paired_nll_frac=0.65)

story = [
    P("CoBit-S vs CoBit-M on TinyGSM: where the extra parameters matter", "title"),
    P("Unguided CoBit-S (130M) and CoBit-M (462M), both step 250k (EMA) · TinyGSM → GSM8K · "
      "experiments/denoising_mse/tinygsm · 6 October 2026 · the TinyGSM counterpart of the OWT reports", "subtitle"),
    box("Summary", [
        f"<b>CoBit-M solves many more GSM8K problems.</b> With the same sampler, schedule and random numbers, accuracy "
        f"is {pct(pa['acc_S'])} (S) vs {pct(pa['acc_M'])} (M), +{100 * pa['diff_M_minus_S']:.1f} pp (95% CI "
        f"+{100 * pa['diff_ci95'][0]:.1f} to +{100 * pa['diff_ci95'][1]:.1f}); {pa['only_M']} problems are solved only by M, "
        f"{pa['only_S']} only by S. On their own schedules: {pct(gS['accuracy'])} vs {pct(gM['accuracy'])}.",
        f"<b>Statically, CoBit-M is better at every noise level</b>, unlike OWT where it was slightly worse at high noise. The "
        f"gain peaks at σ ≈ {stt['peak_diff_loss_region']['sigma']:.1f} (t ≈ {stt['peak_diff_loss_region']['t']:.2f}): "
        f"{100 * stt['terC_diff_peak']['value']:.1f} pp fewer wrong answer tokens and {stt['tnll_diff_peak_bits']['value']:.2f} "
        "bits/token less NLL.",
        f"<b>The schedule matches the gain better than on OWT.</b> The surplus rate peaks at σ ≈ "
        f"{stt['surplus_rate']['sigma_peak']:.1f} vs the schedules' medians ≈ "
        f"{stt['schedules']['S']['schedule']['sigma_median']:.1f}–{stt['schedules']['M']['schedule']['sigma_median']:.1f}; "
        f"overlap {stt['schedules']['S']['overlap']:.2f} (OWT {OWT['overlap_S']:.2f}), and the central half of the steps holds "
        f"{100 * stt['schedules']['S']['surplus_in_central_50pct_steps']:.0f}% of it (OWT {100 * OWT['central_S']:.0f}%).",
        f"<b>The two models already disagree before any denoising.</b> At pure noise, with only the question visible, they "
        f"predict {pct(pa['tokdis_first_step'])} of answer tokens differently; along their own trajectories this grows only to "
        f"{pct(pa['tokdis_last_step'])}, while on identical inputs it falls toward 0. The trajectory split (denoiser-gap growth, "
        f"peak σ ≈ {Rt[rk['divD']]['sigma_peak']:.1f}) again sits above the local disagreement (σ ≈ "
        f"{Rt[rk['locS']]['sigma_median']:.1f}), by about 2.5× in σ instead of 5× on OWT.",
        f"<b>The one-step self-conditioning test leaves the grey curve unchanged</b> (surplus area {gr['onestep']['area']:.5f} vs "
        f"{gr['zeros']['area']:.5f} with zero SC); carrying the full history raises it by "
        f"{100 * (gr['carry']['area'] / gr['zeros']['area'] - 1):.0f}% and shifts it to higher noise.",
    ]),

    P("1. Setup and checks", "h1"),
    P("TinyGSM examples are [BOS] question \\n code [EOS] in SmolLM-135M tokens, padded to 512 (16 bits/token); the "
      "question is clamped clean, the loss covers answer, EOS and padding. CoBit-M (24 × 1024) shares CoBit-S's data, "
      "conditioning, diffusion, loss and compile settings (only trunk and optimiser differ) and the same σ_data = 0.3998. "
      "All runs use float32 rotary positions (the compiled training numerics)."),
    table([
        ["Check", "Result"],
        ["CoBit-M upload", "SHA256 of checkpoint and 7 run files match the manifest; 462.1M params, EMA, step 250k"],
        ["Schedule rule on measured error vs stored table", f"TV {S1['static']['rule_vs_table_TV']:.3f} (S), {stt['rate_M_rule_vs_table_TV']:.3f} (M)"],
        ["GSM8K reproduction (S, eager bf16, as the branch)", f"{pct(rope['bf16_eager_g0.31_s256']['accuracy'])} vs reported 21.6%"],
        ["Rotary positions: fp32 vs eager bf16 (S, 1,319 problems)",
         f"{100 * rope['fp32_minus_bf16']['diff']:+.1f} pp (CI {100 * rope['fp32_minus_bf16']['ci95'][0]:+.1f} to "
         f"{100 * rope['fp32_minus_bf16']['ci95'][1]:+.1f}); answers end before position 256"],
        ["Paired sampling S vs S; trajectory replays", "bit-identical; replays reproduce inputs exactly (both directions)"],
    ], [6.4 * cm, W - 6.4 * cm]),

    P("2. GSM8K accuracy", "h1"),
    table([
        ["Setting (γ = 0.31, 256 NFE, 1,319 problems)", "CoBit-S", "CoBit-M", "M − S"],
        ["each model on its own schedule", pct(gS["accuracy"]), pct(gM["accuracy"]), f"{100 * (gM['accuracy'] - gS['accuracy']):+.1f} pp"],
        ["paired: CoBit-S's schedule, identical random numbers", pct(pa["acc_S"]), pct(pa["acc_M"]),
         f"{100 * pa['diff_M_minus_S']:+.1f} pp [{100 * pa['diff_ci95'][0]:+.1f}, {100 * pa['diff_ci95'][1]:+.1f}]"],
    ], [7.6 * cm, 2.4 * cm, 2.4 * cm, W - 12.4 * cm]),
    Spacer(1, 4),
    P(f"Per problem (paired): {pa['both']} solved by both, {pa['only_M']} only by CoBit-M, {pa['only_S']} only by CoBit-S. "
      "γ = 0.31 is CoBit-S's best 256-NFE setting; CoBit-M's own optimum may differ, so these numbers may understate it."),

    P("3. Static denoising error", "h1"),
    figure(HERE / "sm_fig_static.png", 1,
           f"Static error on {stt['n']:,} validation sequences × 129 noise levels, question clean. (a) Answer-token per-bit MSE. "
           "(b, c) Paired difference S − M over answer tokens and over the training loss region, ±2 SE.", width=0.96 * W),
    figure(HERE / "sm_fig_token.png", 2,
           "Answer-token error rate and NLL differences S − M against CoBit-S's diffusion time, ±2 SE.", width=0.72 * W),
    bullets([
        f"CoBit-M's error is lower at every noise level; there is no high-noise region where it is worse (on OWT it was "
        f"worse for σ ≈ 1.3–20). The gain over the loss region peaks at σ ≈ {stt['peak_diff_loss_region']['sigma']:.2f} "
        f"({stt['peak_diff_loss_region']['value']:.2e} ± {stt['peak_diff_loss_region']['se']:.0e}); over answer tokens at "
        f"σ ≈ {stt['peak_diff_answer']['sigma']:.2f} ({stt['peak_diff_answer']['value']:.2e}).",
        f"At t ≈ {stt['terC_diff_peak']['t']:.2f} the answer-token error falls from {pct(stt['terC_diff_peak']['S'])} to "
        f"{pct(stt['terC_diff_peak']['M'])}. Even at pure noise CoBit-M keeps a small edge (about 0.001 in answer MSE): its "
        "reading of the question alone is better.",
    ]),

    P("4. Schedule versus the gain from the extra parameters", "h1"),
    figure(HERE / "sm_fig_schedule.png", 3,
           "Schedule densities of both models and the surplus rate (MSE<sub>S</sub> − MSE<sub>M</sub>)/σ², unit area on σ ≥ 0.1.",
           width=0.72 * W),
    table([["", "TinyGSM", "OWT (4 Oct report)"],
           ["surplus rate peak σ", f"{stt['surplus_rate']['sigma_peak']:.2f}", f"{OWT['surplus_peak']:.2f}"],
           ["CoBit-S entropy-rate peak σ", f"{Rt[rk['entS']]['sigma_peak']:.2f}", f"{OWT['entS_peak']:.2f}"],
           ["overlap surplus / CoBit-S schedule", f"{stt['schedules']['S']['overlap']:.2f}", f"{OWT['overlap_S']:.2f}"],
           ["surplus in the central 50% of steps (S schedule)", f"{100 * stt['schedules']['S']['surplus_in_central_50pct_steps']:.0f}%",
            f"{100 * OWT['central_S']:.0f}%"],
           ["surplus in the 20% of steps nearest the clean end", f"{100 * stt['schedules']['S']['surplus_in_first_20pct_steps']:.0f}%", "57%"]],
          [7.4 * cm, 3.4 * cm, W - 10.8 * cm]),
    Spacer(1, 4),
    P("The gain sits below the schedule's centre on both tasks, but on TinyGSM the mismatch is much smaller: the gain lies "
      "where most of the information is decided, and the schedule places a good share of its steps there."),

    P("5. Paired generation", "h1"),
    figure(HERE / "sm_fig_traj.png", 4,
           f"Paired generation on the GSM8K prompts (N = {pa['n']}): relative state gap, denoiser gap, fraction of answer tokens "
           "predicted differently, and each model's mean bit entropy, over the answer region. Central 68% / 95% bands.", width=W),
    bullets([
        f"The models start apart: {pct(pa['tokdis_first_step'])} of answer tokens differ at the first step, from the question "
        f"alone, and {pct(pa['tokdis_last_step'])} at the end (OWT: from 0 to {100 * OWT['final_tokdis']:.0f}%).",
        f"The denoiser gap rises from t ≈ {pa['t_dD2_10_50_90'][0]:.2f} (σ ≈ {pa['sigma_dD2_10_50_90'][0]:.1f}), is half-way at "
        f"t ≈ {pa['t_dD2_10_50_90'][1]:.2f} (σ ≈ {pa['sigma_dD2_10_50_90'][1]:.1f}) and 90% complete at t ≈ "
        f"{pa['t_dD2_10_50_90'][2]:.2f} (σ ≈ {pa['sigma_dD2_10_50_90'][2]:.1f}); CoBit-M commits slightly earlier (right panel).",
    ]),

    P("6. Disagreement on identical inputs", "h1"),
    figure(HERE / "sm_fig_same_input.png", 5,
           "Denoiser gap (log) and answer-token disagreement on identical inputs along CoBit-S's and CoBit-M's trajectories "
           "(512 prompts each), with the paired own-trajectory curves.", width=0.92 * W),
    P(f"On identical inputs the two models disagree most at pure noise ({pct(si['S']['tokdis_peak'])} of answer tokens) "
      "and progressively less as the answer becomes visible, along either model's trajectory. The own-trajectory gap passes "
      f"2× the same-input gap at t ≈ {si['own_over_same_input_2x_10x_50x_t'][0]:.2f}, 10× at "
      f"{si['own_over_same_input_2x_10x_50x_t'][1]:.2f} and 50× at {si['own_over_same_input_2x_10x_50x_t'][2]:.2f}: the later "
      "separation is again amplification of earlier differences, starting from the models' different reading of the question."),

    P("7. Ground-truth chains and the grey curve", "h1"),
    figure(HERE / "sm_fig_truechain.png", 6,
           f"Answer-token MSE against the true x<sub>0</sub> along ground-truth chains (N = {tc['n']}) with zero, one-step and "
           "carried self-conditioning, and the change relative to zero SC.", width=0.92 * W),
    figure(HERE / "sm_fig_gray_curve.png", 7,
           "The surplus rate (MSE<sub>S</sub> − MSE<sub>M</sub>)/σ² over answer tokens: static, and on true chains with zero, "
           "one-step and carried self-conditioning. (left) absolute, (right) unit area on σ ≥ 0.1.", width=0.92 * W),
    table([["Surplus rate (answer tokens)", "Area", "Peak σ", "Median σ"]]
          + [[lab, f"{gr[k]['area']:.5f}", f"{gr[k]['sigma_peak']:.2f}", f"{gr[k]['sigma_median']:.2f}"]
             for lab, k in (("static (training mix)", "static_answer"), ("true chains, zero SC", "zeros"),
                            ("true chains, one-step SC", "onestep"), ("true chains, carried SC", "carry"))],
          [7.4 * cm, 2.8 * cm, 2.6 * cm, W - 12.8 * cm]),
    Spacer(1, 4),
    bullets([
        f"One-step self-conditioning leaves the grey curve unchanged in place and size; it moves each model's error by at "
        f"most {max(tc['onestep_minus_zero']['S']['max'], tc['onestep_minus_zero']['M']['max']):.4f}.",
        f"Carrying the full history raises CoBit-S's error by up to +{tc['carry_minus_zero']['S']['max']:.4f} (t ≈ "
        f"{tc['carry_minus_zero']['S']['t']:.2f}) and CoBit-M's by +{tc['carry_minus_zero']['M']['max']:.4f} (t ≈ "
        f"{tc['carry_minus_zero']['M']['t']:.2f}); unlike OWT, CoBit-M is hit less, so its advantage on true chains grows.",
        f"On true chains the models disagree on at most {pct(max(tc['tokdis_peak'].values()))} of answer tokens with any "
        "self-conditioning input (OWT: 62% with carried SC).",
    ]),

    P("8. All curves together", "h1"),
    figure(HERE / "sm_fig_rates_all.png", 8,
           "Densities over log σ (unit area on σ ≥ 0.1): static entropy and surplus rates, same-input disagreement rates on the "
           "three kinds of states, and the divergence of the paired generations (token disagreement is noisy; the "
           "denoiser-gap growth is the cleaner measure).", width=0.8 * W),
    table([["Curve", "Peak σ", "Median σ", "10–90% range of σ"]]
          + [[lab, f"{Rt[rk[k]]['sigma_peak']:.2f}", f"{Rt[rk[k]]['sigma_median']:.2f}",
              f"{Rt[rk[k]]['sigma_q10']:.2f}–{Rt[rk[k]]['sigma_q90']:.2f}"]
             for lab, k in (("same input, CoBit-S trajectories", "locS"), ("same input, CoBit-M trajectories", "locM"),
                            ("static surplus (grey)", "surplus"), ("same input, true chains (carried SC)", "true"),
                            ("static entropy rate, CoBit-S / CoBit-M", "entS"), ("denoiser-gap growth, paired generation", "divD"))],
          [6.8 * cm, 2.2 * cm, 2.2 * cm, W - 11.2 * cm]),
    Spacer(1, 4),
    P("The ordering is the same as on OWT: local model differences (same-input disagreement, static surplus) sit lowest in "
      "noise, the entropy rate in between, and the split of the generated trajectories highest. On TinyGSM the curves are "
      "closer together, because the task's information is decided at higher noise and the models already differ at the "
      "start. The model-switching experiment (one model above a switch point t*, the other below, GSM8K accuracy against t*) "
      "would show directly which region produces CoBit-M's +14 pp; with GSM8K's per-problem correctness it is cheap to run."),

    P("9. Caveats and files", "h1"),
    bullets([
        "One shared sampler (γ = 0.31, CoBit-S's schedule, 256 NFE) for the paired and decomposition runs; CoBit-M's own "
        "optimum may differ. Static and true-chain results use the TinyGSM validation split rebuilt as in training; "
        "generation uses the GSM8K test prompts. CoBit-M's config is loaded from its flattened config.json (its "
        "original_config.py composes files that were not uploaded).",
        f"Code: {M('experiments/denoising_mse/tinygsm/')}; results {M('results_tinygsm/{S,SM}/')}; raw runs "
        f"{M('runs/denoising_mse/tinygsm/')}. The CoBit-S-only report (CoBit_TinyGSM_S_report.pdf) has the RoPE check, "
        "the TinyGSM vs OWT entropy-rate comparison and the CoBit-S schedule analysis in more detail.",
    ]),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="CoBit-S vs CoBit-M on TinyGSM", author="BitstreamDiffusion experiments")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
