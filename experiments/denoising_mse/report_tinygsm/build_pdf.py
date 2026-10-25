"""Build the PDF report for CoBit-S (unguided, 250k) on TinyGSM: setup checks, GSM8K / RoPE, static denoising
error, where information is decided (vs OWT), schedule, self-conditioning on ground-truth chains. The S-vs-M parts
follow when the CoBit-M TinyGSM checkpoint is available.

Numbers are read at build time from ../results_tinygsm/S/{summary.json,curves.csv} and
runs/denoising_mse/tinygsm/gsm8k/*.json; figures from make_figs.py. Requires reportlab + numpy + pillow
(throwaway venv). Output: CoBit_TinyGSM_S_report.pdf (this folder)
"""
import csv
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
OUT = HERE / "CoBit_TinyGSM_S_report.pdf"
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
    canvas.drawString(2.1 * cm, 1.2 * cm, "CoBit-S on TinyGSM (unguided, 250k): denoising error, schedule, self-conditioning")
    canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


# ------------------------------------------------------------------ numbers
S = json.loads((RES / "summary.json").read_text())
st_, tc = S["static"], S["truechain"]
gk = S["gsm8k"]
bf, fp, df = gk["bf16_eager_g0.31_s256"], gk["fp32_g0.31_s256"], gk["fp32_minus_bf16"]
ref = json.loads((REPO / "runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval/"
                  "gsm8k_results_stochastic_g0.31_s256_sd0.3998_ema1.json").read_text())
vm = json.loads((REPO / "datasets/tinygsm/val_only/meta.json").read_text())
rows = list(csv.DictReader(open(RES / "curves.csv")))
Cc = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
sg = Cc["sigma"]
OWT = dict(rate_peak=0.557, rate_median=0.672)  # results/levels/rate_vs_sqrt.json (OWT CoBit-S)


def at(key, s):
    return float(np.interp(np.log(s), np.log(sg), Cc[key]))


pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731

story = [
    P("CoBit-S on TinyGSM: denoising error, schedule and self-conditioning", "title"),
    P("Unguided CoBit-S, step 250k (EMA) · TinyGSM → GSM8K · experiments/denoising_mse/tinygsm · "
      "single-model part; the CoBit-S vs CoBit-M comparisons follow when CoBit-M is available", "subtitle"),
    box("Summary", [
        f"<b>The fp32 rotary-position fix does not change GSM8K accuracy here</b>: {pct(bf['accuracy'])} with the "
        f"original eager bf16 evaluation vs {pct(fp['accuracy'])} with fp32 positions (paired difference "
        f"{100 * df['diff']:+.1f} pp, 95% CI {100 * df['ci95'][0]:+.1f} to {100 * df['ci95'][1]:+.1f}). Prompt plus "
        f"answer average {vm['mean_prompt_len'] + vm['mean_answer_len']:.0f} tokens, so positions above 256 are almost "
        f"all padding. The reported {pct(ref['accuracy'])} stands.",
        f"<b>The measurement reproduces the training schedule</b>: applying the trainer's schedule rule to the measured "
        f"error gives the stored TinyGSM schedule to total-variation distance {st_['rule_vs_table_TV']:.3f}.",
        f"<b>TinyGSM's information is decided at much higher noise than OWT's.</b> The entropy rate MSE/σ² peaks at "
        f"σ ≈ {st_['rate']['sigma_peak']:.1f} (median {st_['rate']['sigma_median']:.1f}) vs σ ≈ {OWT['rate_peak']:.2f} "
        f"on OWT. With the question visible, even pure noise leaves only {pct(st_['at_max_noise']['terC'])} of answer "
        f"tokens wrong ({st_['at_max_noise']['tnllC_nats'] / np.log(2):.1f} bits/token).",
        f"<b>Self-conditioning on ground-truth chains behaves as on OWT</b>: carrying the full history raises CoBit-S's "
        f"answer error by up to +{tc['carry_minus_zero_max']:.3f} (t ≈ {tc['t_at']:.2f}, σ ≈ {tc['sigma_at']:.1f}); the "
        f"one-step variant moves it by at most {max(abs(tc['onestep_minus_zero_min']), tc['onestep_minus_zero_max']):.3f}.",
    ]),

    P("1. Setup and checks", "h1"),
    P("TinyGSM is a conditional task: each example is [BOS] question \\n code [EOS] in SmolLM-135M tokens, padded to "
      "512 tokens (16 bits each). The question is kept clean and the loss covers answer, EOS and padding. CoBit-S has "
      "exactly the OWT CoBit-S architecture; the run estimated σ_data = 0.3998 during training, which the model's "
      "input scaling uses, and trained under torch.compile (float32 rotary positions)."),
    table([
        ["Check", "Result"],
        ["Run files (entropy tables, σ_data, config)", "from the shared Drive folder; σ_data = 0.39984"],
        ["TinyGSM validation split, rebuilt as the training code does",
         f"{vm['n']:,} examples (seeded 1% of {vm['N']:,}; {vm['dropped_too_long']} dropped as too long)"],
        ["GSM8K test file rebuilt from the HF test split", "matches all 100 reference prompts by index"],
        ["Schedule rule on measured error vs stored table", f"TV distance {st_['rule_vs_table_TV']:.3f}"],
        ["GSM8K accuracy on this repo's sampler (eager bf16, as the branch)",
         f"{pct(bf['accuracy'])} vs reported {pct(ref['accuracy'])} (CIs overlap)"],
        ["Paired sampling of CoBit-S against itself; trajectory replay", "bit-identical; replay reproduces inputs exactly"],
    ], [7.2 * cm, W - 7.2 * cm]),

    P("2. GSM8K accuracy and rotary positions", "h1"),
    P("On OWT, running the model eagerly under bf16 autocast gives 511 of the 1,024 positions their neighbour's rotary "
      "encoding, and fixing it to float32 (the compiled training numerics) was essential. The TinyGSM evaluation also "
      "runs eagerly, so both settings were evaluated on all 1,319 GSM8K problems at the branch's 256-NFE operating "
      "point (γ = 0.31, same seeds):"),
    table([
        ["Rotary positions", "Accuracy", "95% CI", "Correct"],
        ["eager bf16 (branch evaluation)", pct(bf["accuracy"]), f"{pct(bf['ci95'][0])}–{pct(bf['ci95'][1])}", str(bf["num_correct"])],
        ["float32 (training numerics)", pct(fp["accuracy"]), f"{pct(fp['ci95'][0])}–{pct(fp['ci95'][1])}", str(fp["num_correct"])],
        ["reported (branch, 19 June)", pct(ref["accuracy"]), f"{pct(ref['ci95_low'])}–{pct(ref['ci95_high'])}", str(ref["num_correct"])],
    ], [5.6 * cm, 2.6 * cm, 3.6 * cm, W - 11.8 * cm]),
    Spacer(1, 4),
    P(f"The difference is noise: {df['only_fp32_correct']} problems are solved only with fp32 positions and "
      f"{df['only_bf16_correct']} only with bf16. Prompts average {vm['mean_prompt_len']:.0f} tokens and answers "
      f"{vm['mean_answer_len']:.0f}, so the collapsed positions above 256 are almost always padding. All results below "
      "use float32 positions. One fp32 generation contained a null byte; it is scored as wrong, as upstream."),

    P("3. Denoising error", "h1"),
    P(f"Per-bit MSE of the denoiser against the clean bits on {st_['n_sequences']:,} validation sequences × 129 noise "
      "levels, question clean, the noise draw shared across noise levels; training objective = average of the "
      "no-self-conditioning and self-conditioned inputs. The median standard error is "
      f"{100 * st_['se_rel_median']:.1f}% of the value."),
    figure(HERE / "fig1_static.png", 1,
           "(a) Per-bit MSE over the training loss region (answer + EOS + padding) and over answer tokens only. "
           "(b) Answer-token error rate and (c) answer-token NLL against TinyGSM's diffusion time t = F(σ).",
           width=0.9 * W),
    table([["σ", "t", "MSE (loss region)", "MSE (answer)", "Answer token error", "Answer NLL [bits/token]"]]
          + [[f"{s:g}", f"{at('t', s):.2f}", f"{at('mse_train', s):.5f}", f"{at('mseC_train', s):.5f}",
              pct(at('terC_train', s)), f"{at('tnllC_train', s) / np.log(2):.2f}"] for s in (0.5, 1.0, 2.0, 4.0, 10.0, 80.0)],
          [1.6 * cm, 1.6 * cm, 3.2 * cm, 2.8 * cm, 3.4 * cm, W - 12.6 * cm]),
    Spacer(1, 4),
    P(f"Two side notes. The error has an isolated spike near σ ≈ {st_['small_sigma_spike']['sigma']:.2f} (answer MSE "
      f"{st_['small_sigma_spike']['mseC']:.1e}), like CoBit-M's tiny-noise anomaly on OWT; divided by σ² it is large, but "
      "the schedule's g(σ) factor suppresses it. And the self-conditioning bootstrap pass (input None) is far worse "
      f"than the trained inputs (answer token error {pct(at('terC_none', 2.0))} vs {pct(at('terC_train', 2.0))} at σ = 2): "
      "in training the self-conditioning channel always carries the clean question, which None removes. The sampler "
      "never uses that pass."),

    P("4. Where the information is decided", "h1"),
    figure(HERE / "fig2_owt_vs_tinygsm.png", 2,
           "Entropy rate MSE/σ² of CoBit-S (training objective, unit area on σ ≥ 0.1) and the stored schedules, OWT vs "
           "TinyGSM.", width=0.74 * W),
    P(f"The TinyGSM rate sits about three times higher in σ than OWT's (peak {st_['rate']['sigma_peak']:.2f} vs "
      f"{OWT['rate_peak']:.2f}; median {st_['rate']['sigma_median']:.2f} vs {OWT['rate_median']:.2f}). With the "
      "question clean, the program's structure is largely determined by the prompt, and the remaining decisions are "
      "made while the answer is still very noisy. The training schedule tracks this, through the same rule as on OWT."),
    figure(HERE / "fig3_rate_vs_sqrt.png", 3,
           "Entropy rate, its square root, the schedule rule g(σ)·(MSE/σ²)^½ applied to the measured error, and the stored "
           "TinyGSM schedule (unit area). The narrow spike near σ ≈ 0.03 is the tiny-noise anomaly discussed in "
           "section 3.", width=0.7 * W),
    table([["Curve (σ ≥ 0.1)", "Peak σ", "Median σ", "5–95% range of σ"]]
          + [[lab, f"{v['sigma_peak']:.2f}", f"{v['sigma_median']:.2f}", f"{v['sigma_q05']:.2f}–{v['sigma_q95']:.1f}"]
             for lab, v in (("entropy rate MSE/σ²", st_["rate"]), ("sqrt-rate", st_["sqrt_rate"]),
                            ("schedule rule on measured error", st_["rule"]), ("stored schedule", st_["table"]))],
          [6.6 * cm, 2.4 * cm, 2.4 * cm, W - 11.4 * cm]),
    Spacer(1, 4),
    P(f"As on OWT, the square root keeps the peak but widens the schedule toward high noise: the 95th percentile moves "
      f"from σ ≈ {st_['rate']['sigma_q95']:.1f} (rate) to {st_['sqrt_rate']['sigma_q95']:.1f} (sqrt-rate)."),

    P("5. Self-conditioning on ground-truth chains", "h1"),
    P(f"{tc['n']} validation sequences are corrupted along the VE forward chain on the sampler's evaluation noise levels "
      "(answer region only, question clean) and walked from high to low noise. At each step CoBit-S denoises the true "
      "state with zero self-conditioning, with the one-step input D(x<sub>t</sub>, σ<sub>t</sub>, D(x<sub>t+1</sub>, "
      "σ<sub>t+1</sub>, 0)), or with its full carried history as in sampling."),
    figure(HERE / "fig4_truechain.png", 4,
           "(left) Answer-token MSE against the true x<sub>0</sub> along true chains for the three self-conditioning "
           "inputs. (right) Change relative to zero self-conditioning.", width=0.86 * W),
    bullets([
        f"Carrying the full history raises the error by up to +{tc['carry_minus_zero_max']:.3f} at t ≈ {tc['t_at']:.2f} "
        f"(σ ≈ {tc['sigma_at']:.1f}), the same commitment effect as on OWT: the true state keeps pointing at the real "
        "answer, so a committed earlier guess keeps costing error.",
        f"The one-step input is close to zero self-conditioning: at most +{tc['onestep_minus_zero_max']:.3f} at high "
        f"noise and {tc['onestep_minus_zero_min']:+.3f} at t ≈ {tc['onestep_t_at_min']:.2f}.",
    ]),

    P("6. Next, and files", "h1"),
    P("With the CoBit-M TinyGSM checkpoint and its run directory: the static S − M surplus and its match to the schedule, "
      "paired generation on the 1,319 GSM8K prompts with per-problem paired accuracy, the same-input decomposition along "
      f"both models' trajectories, and the S-vs-M comparison on ground-truth chains. Code in "
      f"{M('experiments/denoising_mse/tinygsm/')}, results in {M('results_tinygsm/S/')}, raw runs in "
      f"{M('runs/denoising_mse/tinygsm/')}."),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="CoBit-S on TinyGSM: denoising error, schedule, self-conditioning",
                        author="BitstreamDiffusion experiments")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
