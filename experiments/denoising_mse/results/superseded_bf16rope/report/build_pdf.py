"""Build the short PDF report for the denoising-MSE study (CoBit-S vs CoBit-M on OpenWebText).

Numbers are taken from ../results/main/summary.json, ../results/levels/{schedule_match.json,*.csv}
(see the extraction notes in the session log); figures come from make_figs.py. Requires reportlab,
which is not in the `sedd` env; build with a throwaway virtualenv:
    python -m venv /tmp/pdfenv && /tmp/pdfenv/bin/pip install reportlab pillow
    /tmp/pdfenv/bin/python experiments/denoising_mse/report/build_pdf.py
Output: CoBit_denoising_mse_report.pdf (this folder)
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
OUT = HERE / "CoBit_denoising_mse_report.pdf"
FONTS = Path("/usr/share/fonts/truetype/liberation2")
for name, f in [("Serif", "LiberationSerif-Regular"), ("Serif-B", "LiberationSerif-Bold"),
                ("Serif-I", "LiberationSerif-Italic"), ("Serif-BI", "LiberationSerif-BoldItalic"),
                ("Sans", "LiberationSans-Regular"), ("Sans-B", "LiberationSans-Bold"),
                ("Sans-I", "LiberationSans-Italic"), ("Sans-BI", "LiberationSans-BoldItalic"),
                ("Mono", "LiberationMono-Regular")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f"{f}.ttf")))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-BI")

INK = colors.HexColor("#1b1b1a")
INK2 = colors.HexColor("#52514e")
ACCENT = colors.HexColor("#2a78d6")
RULE = colors.HexColor("#d9d8d2")
TINT = colors.HexColor("#eef3fa")

st = {
    "title": ParagraphStyle("title", fontName="Sans-B", fontSize=17, leading=21, textColor=INK, spaceAfter=5),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=9.5, leading=13, textColor=INK2, spaceAfter=12),
    "h1": ParagraphStyle("h1", fontName="Sans-B", fontSize=12, leading=15, textColor=INK, spaceBefore=11, keepWithNext=1,
                         spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=10.3, leading=14.2, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=10.3, leading=14, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.6, leading=11.4, textColor=INK2, spaceBefore=3,
                          spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.7, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=8.7, leading=11, textColor=INK),
    "box": ParagraphStyle("box", fontName="Serif", fontSize=10.4, leading=14.2, textColor=INK),
    "boxh": ParagraphStyle("boxh", fontName="Sans-B", fontSize=10.5, leading=13, textColor=ACCENT, spaceAfter=4),
}
TEXT_W = A4[0] - 2 * 2.1 * cm


def P(t, s="body"):
    return Paragraph(t, st[s])


def bullets(items, style="bullet"):
    return ListFlowable([ListItem(P(t, style), leftIndent=12, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontName="Serif",
                        bulletFontSize=10, spaceBefore=0, spaceAfter=6)


def figure(png, num, caption, width=TEXT_W):
    from PIL import Image as PILImage

    w, h = PILImage.open(png).size
    img = Image(str(png), width=width, height=width * h / w)
    return KeepTogether([img, P(f"<b>Figure {num}.</b> {caption}", "cap")])


def table(rows, widths):
    data = [[P(c, "cellb" if i == 0 else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK2),
    ] + [("LINEBELOW", (0, i), (-1, i), 0.3, RULE) for i in range(1, len(rows) - 1)]))
    return t


def box(title, items):
    inner = [P(title, "boxh"), bullets(items, "box")]
    t = Table([[inner]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(INK2)
    canvas.drawString(2.1 * cm, 1.2 * cm, "CoBit-S vs CoBit-M: denoising error across diffusion time")
    canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


story = [
    P("Where do the extra parameters help? Denoising error of CoBit-S (130M) and CoBit-M (462M) "
      "across diffusion time", "title"),
    P("OpenWebText validation split · experiments/denoising_mse · 3 October 2026", "subtitle"),
    box("Summary", [
        "CoBit-M has lower denoising error than CoBit-S at low-to-mid noise (σ ≈ 0.15–2). In absolute terms "
        "the gap peaks at σ ≈ 0.47; in relative terms it is largest at the low-noise end, "
        "46% at σ = 0.2 and 3% at σ = 1.",
        "At the token level the gain peaks at <b>8.8 percentage points of token error</b> (t ≈ 0.19) and "
        "<b>0.60 bits/token of NLL</b> (t ≈ 0.24), about 64 fewer wrong tokens and 0.44 kbit less per sequence. "
        "M wins on essentially every sequence up to t ≈ 0.45.",
        "At high noise (σ ≈ 2.5–20, t above about 0.8) M is slightly <i>worse</i>, and it has an isolated error spike at "
        "σ ≈ 0.003–0.01, below where sampling stops.",
        "The entropy-rate schedules are poorly matched to where the extra parameters help: the gain peaks at "
        "σ ≈ 0.37, the schedules at σ ≈ 0.56–0.61. The 20% of sampler steps nearest the clean end hold "
        "55–63% of the gain.",
        "Both models lose position information past token 256: RoPE is computed in bf16 under autocast, so "
        "511 of the 1,024 positions share their neighbour's encoding. Token error steps up at 256 and 512.",
    ]),
    Spacer(1, 4),

    P("1. Setup", "h1"),
    P("Both released EMA checkpoints (step 750k) were evaluated with their own eval configs, in bf16 as in "
      "the paper. The data are the configs' validation split, <font face='Mono' size='9'>train[-100000:]</font> "
      "of OpenWebText (110,520 sequences of 1,024 code tokens = 16,384 bits), rebuilt with the repo's own cache "
      "builder."),
    bullets([
        "<b>Metric.</b> The per-bit squared error of the denoiser output sigmoid(logits) against the clean bits, "
        "i.e. the unweighted training loss. The headline curve is the training objective, the average of the "
        "no-self-conditioning and self-conditioned passes (self-conditioning probability 0.5 in training). "
        "Token-level metrics are computed on non-padding tokens: token error rate (any bit wrong after "
        "rounding) and token NLL under the factorised bit posterior.",
        "<b>Sampling.</b> 4,096 sequences × 129 log-spaced σ in [0.002, 80] for the bit-level curves; "
        "2,048 sequences × 97 σ for the token-level ones. Each sequence has one fixed noise draw, shared across "
        "all σ and both models, so curves are smooth and S − M is a paired estimate. Standard errors are across "
        "sequences; in the informative range the SE is about 0.1% of the MSE, and every bit-level point "
        "averages 67M bits. Total cost: 7.5 GPU-hours on one RTX 4090.",
        "<b>Diffusion time.</b> t = F(σ), the CDF of the entropy-rate schedule (t = 0 clean, t = 1 noise); the "
        "sampler steps uniformly in t. The two models' schedules differ by about 7% in σ at a given t, so "
        "plots against t use CoBit-S's schedule for both models, making any gap a model difference.",
    ]),

    P("2. Bit level", "h1"),
    P("The gap peaks at σ = 0.47 (t ≈ 0.26), where MSE falls from 0.0584 (S) to 0.0525 (M), a difference of "
      "5.85 × 10<super>-3</super> ± 3 × 10<super>-5</super>. Relative to M's error the gap is 46% at σ = 0.2, "
      "9.5% at σ = 0.5, 2.7% at σ = 1 and 0.2% at σ = 2. Two regions go the other way. Above σ ≈ 2.5 M is "
      "worse by up to 6.2 × 10<super>-4</super> ± 6 × 10<super>-5</super> (σ ≈ 3.4); near the prior both "
      "models predict the bit marginals and the curves coincide. At σ ≈ 0.003–0.01 M's error spikes to "
      "3.4 × 10<super>-3</super> (6 × 10<super>-3</super> without self-conditioning) while S's is ≈ 0. The "
      "sampler never goes below σ = 0.08, but a likelihood bound integrated down to σ = 0.002 would pick it up. "
      "The same picture holds for each self-conditioning input separately and when padding tokens are excluded."),
    figure(HERE / "fig1_mse_sigma.png", 1,
           "(a) Expected per-bit MSE against noise level for both models (log scale). (b) Paired difference "
           "MSE<sub>S</sub> − MSE<sub>M</sub>, positive where CoBit-M is better; the ±2 SE band is thinner "
           "than the line. Dotted line: the sampler's terminal σ = 0.08.", width=0.8 * TEXT_W),

    P("3. Token and sequence level", "h1"),
    figure(HERE / "fig2_token_time.png", 2,
           "Against diffusion time (CoBit-S schedule for both models): (a) token error rate difference, "
           "(b) token NLL difference, (c) fraction of sequences on which CoBit-M has the lower MSE. "
           "Differences are S − M with ±2 SE.", width=0.88 * TEXT_W),
    table([
        ["Quantity", "S − M at peak (± 1 SE)", "Where", "S → M at that point"],
        ["Per-bit MSE", "5.85e-3 ± 3e-5", "σ = 0.47, t ≈ 0.26", "0.0584 → 0.0525"],
        ["Token error rate", "8.8 ± 0.05 pp", "t = 0.19", "51.9% → 43.0%"],
        ["Token NLL", "0.60 ± 0.004 bits/token", "t = 0.24", "5.46 → 4.86"],
        ["Wrong tokens per sequence", "64.2 ± 0.4", "t = 0.19", "of ≈ 731 content tokens"],
        ["NLL per sequence", "0.437 ± 0.003 kbit", "t = 0.24", ""],
        ["M worse: token NLL", "−0.12 ± 0.017 bits/token", "t ≈ 0.89", "crosses zero at t ≈ 0.80"],
    ], [3.9 * cm, 4.3 * cm, 3.5 * cm, TEXT_W - 11.7 * cm]),
    Spacer(1, 6),
    P("The improvement is not a few easy sequences moving: CoBit-M has the lower MSE on 100% of sequences at "
      "t = 0.1, 99.9% at t = 0.3, 96% at t = 0.5, 85% at t = 0.75, and only 51% at t = 0.9. The spread of "
      "per-sequence differences widens with t: the 10–90% band runs from 0.7–1.3× the mean at t = 0.1–0.2 to "
      "0.3–1.7× at t = 0.5."),

    P("4. Schedule versus the parameter surplus", "h1"),
    P("The trainer builds each schedule from the measured denoising error: per log-σ bin, the density is proportional to "
      "g(σ)·(MSE/σ<super>2</super>)<super>1/2</super>, where MSE/σ<super>2</super> is the I-MMSE information "
      "rate and g = σ<super>3</super>/(σ<super>3</super> + 0.1<super>3</super>). Applying that rule to our "
      "measured MSE reproduces the stored tables (total-variation distance 0.05 for S, 0.07 for M). The "
      "matching measure of what the extra parameters buy is therefore the rate surplus "
      "(MSE<sub>S</sub> − MSE<sub>M</sub>)/σ<super>2</super>, the density over log σ of the reduction in the "
      "denoising NLL bound."),
    bullets([
        "The surplus peaks at σ = 0.37 (median 0.39); the schedules peak at σ = 0.56 (S) and 0.61 (M), with "
        "medians 0.77 and 0.82. The overlap coefficient is 0.52 (S) and 0.49 (M).",
        "The 20% of steps nearest the clean end hold 55% (S schedule) and 63% (M schedule) of the surplus; the "
        "central 50% of steps hold only 25% and 20%. Slices near t ≈ 0.05–0.07 carry 3.1–3.9× their share, "
        "and slices beyond t ≈ 0.35 carry less than their share.",
        "The high-noise region where M is worse on raw MSE is negligible in rate terms (its net contribution "
        "is −4 × 10<super>-5</super> against a total of 0.031).",
    ]),
    P("So the extra capacity pays off mostly in the last stretch of sampling, at lower noise than where the "
      "schedule concentrates its steps. Whether shifting steps toward that region helps CoBit-M's samples is "
      "an open question; this analysis measures denoising error, not sample quality."),
    figure(HERE / "fig3_schedule.png", 3,
           "(left) Schedule densities of both models and the rate surplus (solid) and ΔMSE (dashed), each "
           "scaled to unit area over σ ≥ 0.08. (middle) Surplus covered by each 2% slice of sampler steps; 1 "
           "would mean the schedule is matched to the surplus. (right) Cumulative surplus against the fraction "
           "of steps counted from the clean end; the diagonal is a perfect match.", width=0.96 * TEXT_W),

    P("5. A position-precision bug in both models", "h1"),
    P("Token error jumps at position 256 for both models, and again near 512, although the token statistics "
      "are the same on either side. The cause is in <font face='Mono' size='9'>models/sdt.py</font>: "
      "<font face='Mono' size='9'>RotaryEmbedding</font> forms the angles with "
      "<font face='Mono' size='9'>torch.einsum</font>, which autocast runs in bf16, and bf16 cannot represent "
      "integers above 256 exactly. Under autocast, 511 of the 1,024 positions receive exactly the same cos/sin "
      "as the previous position: pairs collapse from 256 and groups of four from 512. Training used bf16 "
      "autocast too, so both checkpoints learned with the collapsed positions; the S-vs-M comparison is fair, "
      "but both models are likely held back past position 256. Computing the angles in float32 is a "
      "one-line fix for future training; evaluating these checkpoints with exact positions would be a "
      "train/test mismatch and has not been tested."),
    figure(HERE / "fig4_position.png", 4,
           "Token error rate against position in the sequence at two diffusion times (positions with at "
           "least 200 content tokens). Dotted lines mark positions 256 and 512.", width=0.72 * TEXT_W),

    P("6. Caveats and files", "h1"),
    bullets([
        "The validation split was rebuilt from the final parquet shard of Skylion007/openwebtext, the source "
        "<font face='Mono' size='9'>load_dataset(\"openwebtext\")</font> resolves to; row order is assumed "
        "to match the original split.",
        "These are one-step denoising errors at fixed noise levels on clean data, not errors along sampling "
        "trajectories, and not sample quality.",
        "Code: <font face='Mono' size='9'>experiments/denoising_mse/</font> (mse_curves.py, analyze.py, "
        "analyze_levels.py). Curves and SEs: <font face='Mono' size='9'>results/main/</font> and "
        "<font face='Mono' size='9'>results/levels/</font>. Raw per-sequence values: "
        "<font face='Mono' size='9'>runs/denoising_mse/</font>.",
    ]),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="CoBit-S vs CoBit-M: denoising error across diffusion time",
                        author="BitstreamDiffusion experiments")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
