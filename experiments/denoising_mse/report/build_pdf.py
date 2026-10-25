"""Build the PDF report: static denoising error and paired generation trajectories, CoBit-S vs CoBit-M (OWT).

Supersedes results/superseded_bf16rope/report (computed with eager bf16 RoPE, see section 1). Numbers come
from ../results/{main,levels,traj}/*.json|csv and runs/denoising_mse/traj/main/scores.json; figures from
make_figs.py (fig1, fig2, fig4) and ../results/levels/fig_schedule_overlay.png (fig3),
../results/traj/fig_traj_vs_rate.png (fig5). Requires reportlab (not in `sedd`; use a throwaway venv):
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
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

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
WARN = colors.HexColor("#b4441c")
RULE = colors.HexColor("#d9d8d2")
TINT = colors.HexColor("#eef3fa")
TINT_W = colors.HexColor("#fbf0ea")

st = {
    "title": ParagraphStyle("title", fontName="Sans-B", fontSize=16.5, leading=20.5, textColor=INK, spaceAfter=5),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=9.5, leading=13, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Sans-B", fontSize=12, leading=15, textColor=INK, spaceBefore=11,
                         spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=10.2, leading=14, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=10.2, leading=13.8, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.5, leading=11.3, textColor=INK2, spaceBefore=3,
                          spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.6, leading=10.9, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=8.6, leading=10.9, textColor=INK),
    "box": ParagraphStyle("box", fontName="Serif", fontSize=10.2, leading=13.9, textColor=INK),
    "boxh": ParagraphStyle("boxh", fontName="Sans-B", fontSize=10.5, leading=13, textColor=ACCENT, spaceAfter=4),
    "boxhw": ParagraphStyle("boxhw", fontName="Sans-B", fontSize=10.5, leading=13, textColor=WARN, spaceAfter=4),
}
TEXT_W = A4[0] - 2 * 2.1 * cm


def P(t, s="body"):
    if s == "h1":  # never leave a heading at the foot of a page
        return KeepTogether([CondPageBreak(3.2 * cm), Paragraph(t, st[s])])
    return Paragraph(t, st[s])


def M(t):
    return f"<font face='Mono' size='8.8'>{t}</font>"


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
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK2),
    ] + [("LINEBELOW", (0, i), (-1, i), 0.3, RULE) for i in range(1, len(rows) - 1)]))
    return t


def box(title, items, warn=False):
    inner = [P(title, "boxhw" if warn else "boxh"), bullets(items, "box")]
    t = Table([[inner]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT_W if warn else TINT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(INK2)
    canvas.drawString(2.1 * cm, 1.2 * cm, "CoBit-S vs CoBit-M: static denoising error and paired generation")
    canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


W = TEXT_W
story = [
    P("Where do CoBit-M's extra parameters matter? Static denoising error and paired generation "
      "trajectories", "title"),
    P("CoBit-S (130M) vs CoBit-M (462M), OpenWebText · experiments/denoising_mse · 4 October 2026 · "
      "supersedes the 3 October report", "subtitle"),
    box("Correction to the 3 October report", [
        "The earlier results ran the models eagerly under bf16 autocast, where the rotary position embedding is "
        "computed in bf16 and 511 of the 1,024 positions get their neighbour's encoding. Training and the paper's "
        "evaluation run under torch.compile, which computes it in float32, so those runs were a train/test "
        "mismatch. The old low-noise errors were inflated by up to 4× (CoBit-S at σ = 0.2: 9.9e-4 vs 2.3e-4), "
        "and the reported \"position bug\" at tokens 256/512 was an artefact of that setup. All results below are "
        "recomputed with float32 RoPE and reproduce the paper's GenPPL.",
    ], warn=True),
    Spacer(1, 5),
    box("Summary", [
        "<b>Statically, CoBit-M is better only at low noise.</b> Its denoising error is lower for σ ≈ 0.15–1 "
        "(t below about 0.6), peaking at σ ≈ 0.43 (t ≈ 0.23): 11.5 percentage points fewer wrong tokens and "
        "0.76 bits/token less NLL. For σ ≈ 1.3–20 (t ≈ 0.7–0.99) it is slightly <i>worse</i> (up to 1.2% "
        "higher MSE, 0.33 bits/token higher NLL).",
        "<b>The schedules do not target that region.</b> The rate surplus (MSE<sub>S</sub> − "
        "MSE<sub>M</sub>)/σ<super>2</super> peaks at σ ≈ 0.40; the entropy-rate schedules have medians σ ≈ "
        "0.77 (S) and 0.82 (M). The 20% of sampler steps nearest the clean end hold 57–68% of the surplus.",
        "<b>Paired generation</b> (same sampler, schedule and random numbers; GenPPL S 27.2, M 18.7): the two "
        "trajectories start to separate at <b>t ≈ 0.91 (σ ≈ 4)</b> and are half separated by t ≈ 0.83 "
        "(σ ≈ 2.4), i.e. in the high-noise region where CoBit-M is statically no better.",
        "<b>The separation is amplification, not local disagreement.</b> Fed identical inputs, the two denoisers "
        "disagree 10–100× less than along their own trajectories, and that same-input disagreement peaks at "
        "σ ≈ 0.49, close to the static curves.",
        "<b>So the divergence curve is not the MSE/σ<super>2</super> curve.</b> Token divergence peaks at "
        "σ ≈ 2.5; the static entropy rate at σ ≈ 0.56–0.61; the static surplus at σ ≈ 0.40. Divergence marks "
        "where the sampler commits to content, which is sensitive to model differences, not where the larger "
        "model is locally more accurate. A model-switching experiment (section 6) would tell which region "
        "produces the GenPPL gap.",
    ]),

    P("1. What was fixed, and the checks", "h1"),
    P(f"{M('models/sdt.py')} {M('RotaryEmbedding')} builds its angles with {M('torch.einsum')}, which autocast "
      "runs in bf16 in eager mode. Under torch.compile (torch 2.5.1, checked here; training used 2.4) the "
      "compiled output is bitwise equal to the float32 computation. torch.compile itself crashes on this "
      f"machine's torch 2.0.1, so the experiment code patches the angle computation to float32 "
      f"({M('experiments/denoising_mse/rope_fix.py')}); the model file is unchanged."),
    table([
        ["Check", "Result", "Reference"],
        ["GenPPL / entropy, 1,024 samples, γ = 0.13, CoBit-S schedule", "S 27.22 / 5.254;  M 18.69 / 5.384",
         "paper: S 27.06 / 5.26;  M 18.47 / 5.378 (own schedule)"],
        ["Same, 128 samples, CoBit-M's schedule", "S 27.43 / 5.274;  M 19.08 / 5.376", ""],
        ["Real validation text through the same scorer", "14.88 / 5.46", "paper: 15.06 / 5.44"],
        ["Schedule rule applied to the measured MSE vs stored tables", "TV distance 0.012 (S and M)",
         "0.05–0.07 before the fix"],
        ["CoBit-S run twice with identical seeds", "bit-identical trajectories", ""],
    ], [5.6 * cm, 5.0 * cm, W - 10.6 * cm]),
    Spacer(1, 4),

    P("2. Setup", "h1"),
    bullets([
        "<b>Static error.</b> For held-out OWT sequences x<sub>0</sub> (the configs' validation split, 16,384 "
        "bits each), x = x<sub>0</sub> + σε and per-bit MSE of the denoiser D = sigmoid(logits) against "
        "x<sub>0</sub>, averaged over the no-self-conditioning and self-conditioned passes as in training. "
        "4,096 sequences × 129 log-spaced σ in [0.002, 80] (token metrics: 2,048 × 97). Each sequence has one "
        "noise draw shared across σ and both models, so S − M is paired; in the informative range the SE is "
        "0.1% of the MSE.",
        "<b>Paired generation.</b> Both models run the paper's sampler: entropic DDIM, 256 NFE (254 churned "
        "Euler steps + final denoise), self-conditioning in carry mode, EDM churn with γ = 0.13 (CoBit-S's "
        "Table 1 setting and a reported CoBit-M 256-NFE point), terminal σ = 0.08, on <b>CoBit-S's schedule</b> "
        "for both (identical σ grid, asserted). For each of 1,024 examples the initial noise and the churn "
        "noise of every step are the same for both models. At every step we record the state gap x<sub>S</sub> − "
        "x<sub>M</sub>, the denoiser gap D<sub>S</sub> − D<sub>M</sub> and the fraction of tokens predicted "
        "differently.",
        "<b>Decomposition.</b> For 512 of the examples, CoBit-M is also evaluated at every step on exactly "
        "CoBit-S's input (its churned state and self-conditioning input), giving the models' disagreement on "
        "identical inputs.",
        "<b>Time axis.</b> t = F<sub>S</sub>(σ), the CDF of CoBit-S's entropy-rate schedule (t = 1 noise, "
        "t = 0 clean); the sampler steps uniformly in t. Total GPU time 8.9 h on one RTX 4090.",
    ]),

    P("3. Static denoising error", "h1"),
    P("CoBit-M's advantage is confined to low noise. The gap peaks at σ = 0.43 (t = 0.23), where MSE falls "
      "from 0.0337 to 0.0265 (S − M = 7.2e-3 ± 3e-5). Relative to M's error the gap is 118% at σ = 0.2, 73% "
      "at 0.31, 37% at 0.4, 12% at 0.5 and 1.7% at 0.7, and it changes sign at σ ≈ 1. From σ ≈ 1.3 to 20 "
      "CoBit-M is significantly worse, by up to 1.9e-3 ± 1e-4 (1.2%) at σ ≈ 2.7 (t ≈ 0.85). Integrated over "
      "σ ≥ 0.08, MSE/σ<super>2</super> gives 9.11 (S) vs 8.28 (M) bits per non-padding token, a rough "
      "denoising-bound view of the 0.8 bit/token gap. Separately, CoBit-M has an isolated error spike at "
      "σ ≈ 0.007 (3.3e-3, mostly without self-conditioning) that the sampler never visits."),
    figure(HERE / "fig1_mse_sigma.png", 1,
           "(a) Expected per-bit MSE against noise level (log scale), training objective. (b) Paired difference "
           "MSE<sub>S</sub> − MSE<sub>M</sub>, positive where CoBit-M is better (±2 SE band is thinner than the "
           "line). Dotted: the sampler's terminal σ = 0.08.", width=0.72 * W),
    figure(HERE / "fig2_token_time.png", 2,
           "Against diffusion time (CoBit-S schedule for both): (a) token error rate difference, (b) token NLL "
           "difference, on non-padding tokens, S − M with ±2 SE; (c) fraction of sequences on which CoBit-M "
           "has the lower MSE.", width=0.84 * W),
    table([
        ["Quantity", "S − M (± 1 SE)", "Where", "S → M"],
        ["Token error rate, peak", "11.5 ± 0.05 pp", "t = 0.24", "t = 0.2: 35.6% → 24.9%"],
        ["Token NLL, peak", "0.76 ± 0.004 bits/token", "t = 0.23", "t = 0.2: 3.01 → 2.30"],
        ["Wrong tokens / NLL per sequence, peak", "84 ± 0.4 tokens;  0.55 kbit", "t ≈ 0.24", "of ≈ 731 tokens"],
        ["Token NLL, M worse", "−0.33 ± 0.03 bits/token", "t ≈ 0.88", "sign change at t ≈ 0.63"],
        ["Sequences where M has lower MSE", "100% · 66% · 33% · 27% · 45%", "t = 0.3 · 0.5 · 0.7 · 0.8 · 0.9", ""],
    ], [4.6 * cm, 4.4 * cm, 3.9 * cm, W - 12.9 * cm]),

    P("4. Schedule versus the parameter surplus", "h1"),
    P("Each schedule is built from the measured error: per log-σ bin the density is proportional to "
      "g(σ)·(MSE/σ<super>2</super>)<super>1/2</super>, where MSE/σ<super>2</super> is the I-MMSE information "
      "rate. The matching measure of what the extra parameters buy is the rate surplus "
      "(MSE<sub>S</sub> − MSE<sub>M</sub>)/σ<super>2</super>."),
    figure(HERE / "fig3_schedule.png", 3,
           "(left) Schedule densities and the surplus (solid) and ΔMSE (dashed), each with unit area on "
           "σ ≥ 0.08. (middle) Surplus covered by each 2% slice of sampler steps; 1 = matched. (right) "
           "Cumulative surplus against the fraction of steps counted from the clean end."),
    bullets([
        "The surplus is narrow: peak and median σ ≈ 0.40, 90% of it in σ ≈ 0.25–0.62. The schedules peak at "
        "σ = 0.56 (S) and 0.61 (M) with medians 0.77 and 0.82; overlap coefficient 0.42 (S) and 0.38 (M).",
        "The 20% of steps nearest the clean end hold 57% (S schedule) and 68% (M schedule) of the surplus; "
        "the central 50% of steps hold only 16% and 10%. Slices near t ≈ 0.11–0.13 carry 3.7–4.1× their share.",
        "The square root in the \"sqrt-rate\" rule widens the schedule toward high noise: the plain entropy rate "
        "of CoBit-S has median σ 0.67 and 95th percentile 2.4; its square root, 0.81 and 7.9.",
    ]),

    P("5. Paired generation trajectories", "h1"),
    figure(HERE / "fig4_traj.png", 4,
           "Paired trajectories, 1,024 examples (decomposition: 512), mean with the central 68% and 95% of "
           "examples shaded. (a) State gap relative to the remaining noise. (b) Denoiser gap along each model's "
           "own trajectory (black) and on identical inputs (purple), log scale. (c) Fraction of tokens predicted "
           "differently. Sampling runs from left to right.", width=0.98 * W),
    bullets([
        "<b>Until t ≈ 0.94 the runs are almost identical.</b> The own-trajectory denoiser gap passes twice the "
        "same-input gap at t = 0.94 (σ = 6.2), 10× at t = 0.91 (σ = 4.2) and 50× at t = 0.78 (σ = 1.8).",
        "<b>The generations separate at high noise.</b> Token disagreement reaches half its final value at "
        "t = 0.83 (σ = 2.4) and 90% at t = 0.54 (σ = 0.84). The state gap equals the remaining noise at "
        "t = 0.33; at the end 71% of tokens differ.",
        "<b>On identical inputs the models agree closely.</b> Same-input token disagreement stays below 20% "
        "(peak 19% at t = 0.47) and the same-input denoiser gap is 10–100× smaller than along the own "
        "trajectories. CoBit-M also commits to confident predictions earlier than CoBit-S at the same t.",
        "On shared seeds CoBit-M's sample has lower GPT-2 NLL on 65% of examples (0.38 ± 0.03 nats/token).",
    ]),

    P("6. Divergence versus the MSE/σ² curve", "h1"),
    P("The comparable quantities are densities over log σ. With shared noise the state gap obeys "
      "δx<sub>i+1</sub> = r<sub>i</sub>δx<sub>i</sub> + (1 − r<sub>i</sub>)δD<sub>i</sub>, "
      "r<sub>i</sub> = σ<sub>i+1</sub> / (churned σ<sub>i</sub>), so δD<super>2</super>/σ<super>2</super> is the squared gap "
      "injected per unit log σ while the runs are close; on identical inputs it is the direct analogue of "
      "MSE/σ<super>2</super>. Divergence is measured as the growth of token disagreement and of the squared "
      "state gap per unit log σ."),
    figure(HERE / "fig5_traj_vs_rate.png", 5,
           "Densities over log σ, each with unit area on σ ≥ 0.08: growth of token disagreement and of the "
           "squared state gap along the paired trajectories, the same-input disagreement rate, and the static "
           "entropy rates and surplus rate.", width=0.68 * W),
    table([
        ["Curve", "Peak σ", "Median σ", "10–90% range of σ"],
        ["Token divergence (paired trajectories)", "2.51", "2.30", "0.83–5.10"],
        ["Squared state-gap growth", "0.59", "0.79", "0.40–2.28"],
        ["Same-input disagreement rate", "0.49", "0.49", "0.27–0.88"],
        ["Static entropy rate, CoBit-S / CoBit-M", "0.56 / 0.61", "0.67 / 0.72", "0.38–1.69 / 0.41–1.79"],
        ["Static surplus rate (S − M)", "0.40", "0.40", "0.28–0.56"],
    ], [6.4 * cm, 2.6 * cm, 2.6 * cm, W - 11.6 * cm]),
    Spacer(1, 5),
    P("The curves split into two groups. The quantities that measure <i>local</i> model differences (same-input "
      "disagreement, static surplus) sit at σ ≈ 0.4–0.5, and the squared state gap grows like the static entropy "
      "rate, as both track when bits are decided. Token divergence sits four to six times higher in σ, where the "
      "sampler commits to which content to write: small denoiser differences there (the models agree on 90% "
      "of tokens given the same input) are amplified into different samples."),
    P("This means the onset of divergence is not where CoBit-M is more accurate: at σ ≈ 2–4 it is statically "
      "slightly worse. It is where outcomes are most sensitive to the model. Whether CoBit-M's better samples "
      "come from its high-noise decisions or its low-noise refinement is not identified by divergence alone. A "
      "direct test: generate with CoBit-M for t above a switch point t* and CoBit-S below it (and the reverse), "
      "on the same seeds, and score GenPPL as a function of t*. With the sampler hooks already in place this is "
      "a few GPU-hours."),

    P("7. Caveats and files", "h1"),
    bullets([
        "Validation split rebuilt from the last parquet shard of Skylion007/openwebtext (row order assumed to "
        "match). The float32-RoPE patch reproduces torch 2.5.1's compiled numerics; training used torch 2.4, "
        "assumed to compile the same way (the GenPPL checks agree with the paper). Static results are one-step "
        "errors on clean data and the trajectories use one shared sampler; neither alone measures sample quality.",
        f"Code in {M('experiments/denoising_mse/')}, results in {M('results/{main,levels,traj}/')}, raw runs in "
        f"{M('runs/denoising_mse/')}; superseded eager-bf16 results in {M('results/superseded_bf16rope/')}.",
    ]),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="CoBit-S vs CoBit-M: static denoising error and paired generation",
                        author="BitstreamDiffusion experiments")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
