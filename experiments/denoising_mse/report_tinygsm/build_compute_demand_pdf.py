"""TinyGSM compute demand by difficulty and model switching (CoBit-S / CoBit-M): PDF report.
Inputs: ../results_tinygsm/compute_demand/ (analyze_compute_demand.py) and runs/denoising_mse/tinygsm/compute_demand/.
Output: CoBit_TinyGSM_compute_demand.pdf (this folder)
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
OUT = HERE / "CoBit_TinyGSM_compute_demand.pdf"
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
    canvas.drawString(2.1 * cm, 1.2 * cm, "TinyGSM compute demand by difficulty and model switching (CoBit-S / CoBit-M, 250k)")
    canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()




# ------------------------------------------------------------------ numbers
import re  # noqa: E402

EXP = HERE.parent
RD = EXP / "results_tinygsm/compute_demand"
SU = json.loads((RD / "summary.json").read_text())
ST, G = SU["static"]["strata"], SU["gsm8k"]
RUNS = REPO / "runs/denoising_mse/tinygsm/compute_demand"
BY, GOLD = {}, None
for f in RUNS.glob("*_seed[0-9].json"):
    d = json.loads(f.read_text())
    BY.setdefault(d["config"], {})[d["seed"]] = np.array(d["per_problem_correct"], float)
    GOLD = np.array(d["gold_steps"])
SEEDS = sorted(BY["base"])
STR = [("all", lambda g: np.ones_like(g, bool)), ("≤2 steps", lambda g: g <= 2), ("3–4 steps", lambda g: (g >= 3) & (g <= 4)),
       ("5+ steps", lambda g: g >= 5)]
RNG = np.random.default_rng(1)


def pp(c):
    return np.mean([BY[c][s] for s in SEEDS], 0)


def dci(a, b, m):
    d = (pp(b) - pp(a))[m]
    bs = d[RNG.integers(0, len(d), (4000, len(d)))].mean(1)
    return f"{100 * d.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]"


ACC = G["accuracy"]
a = lambda c, n="all": f"{100 * ACC[c][n]:.1f}"  # noqa: E731
seedacc = lambda c: " / ".join(f"{100 * BY[c][s].mean():.1f}" for s in SEEDS)  # noqa: E731
ratio = 24.5 / 8.0  # measured wall time of a full CoBit-M vs CoBit-S pass (same sampler, same GPU)
cost = lambda m_steps: (m_steps * ratio + (256 - m_steps)) / (256 * ratio)  # noqa: E731
SEGSIG = ["76.8–7.9", "7.9–4.1", "4.1–2.6", "2.6–1.8", "1.8–1.3", "1.3–0.83", "0.83–0.48", "0.48–0.002"]
BA, BF = G["B"], G["B_fraction"]
AA = G["A"]
seg = lambda d, k: f"{100 * d['delta'][k]:+.1f} [{100 * d['lo'][k]:+.1f}, {100 * d['hi'][k]:+.1f}]"  # noqa: E731

story = [
    P("Where does extra compute and capacity pay off on TinyGSM? Compute demand by problem difficulty, and model "
      "switching between CoBit-S and CoBit-M", "title"),
    P(f"CoBit-S (130M) and CoBit-M (462M), TinyGSM, 250k steps, EMA, float32 RoPE. GSM8K test (1319 problems), "
      f"γ = 0.31, 256 steps, CoBit-S's ddim_entropic schedule, self-conditioning carry; {len(BY)} configurations × "
      f"{len(SEEDS)} seeds = {len(BY) * len(SEEDS)} full GSM8K evaluations. These experiments are inputs to the design of "
      "a diffusion-time expert-choice MoE.", "subtitle"),
    box("Summary", [
        f"<b>Capacity matters, extra iterations do not.</b> Two extra self-conditioning refreshes per step (+25% NFE "
        f"inside a segment) change overall GSM8K accuracy by at most ±0.4 pp in any of the 8 segments (all 95% CIs include 0; "
        f"no stratum shows a gain). "
        f"Replacing CoBit-S by CoBit-M for the same 32 steps gives up to +{100 * BA['all']['delta'][4]:.1f} pp.",
        f"<b>The first quarter of the trajectory (σ > 4) does not need the large model.</b> CoBit-S for steps 0–63 "
        f"followed by CoBit-M reaches {a('SthenM64')}% vs {a('M')}% for CoBit-M throughout (paired Δ {dci('M', 'SthenM64', STR[0][1](GOLD))} pp), "
        f"at about {100 * cost(192):.0f}% of CoBit-M's compute. CoBit-M only for those first 64 steps gives nothing "
        f"(Δ vs CoBit-S {dci('base', 'MthenS64', STR[0][1](GOLD))} pp).",
        f"<b>Capacity pays off most at σ ≈ 1.3–2.6 (steps 96–159)</b>, the peak of the static surplus rate. CoBit-M in "
        f"one of these two 32-step segments recovers about 20% of the full gain each (+{100 * BA['all']['delta'][3]:.1f} "
        f"and +{100 * BA['all']['delta'][4]:.1f} pp of +{100 * BF['all']['full_gain']:.1f} pp), against 12.5% of the steps.",
        f"<b>Single-segment gains are sub-additive.</b> The 8 single-segment gains sum to {100 * BF['all']['sum_fraction']:.0f}% "
        f"of the full gain, while contiguous suffixes recover more than the sum of their segments (e.g. CoBit-M on "
        f"steps 64–255: +{100 * (ACC['SthenM64']['all'] - ACC['base']['all']):.1f} pp vs Σ segments 2–7 = "
        f"+{100 * sum(BA['all']['delta'][2:]):.1f} pp). The large model's mid-trajectory decisions pay off fully only "
        "if it also carries them through the low-noise end.",
        f"<b>Harder problems need more surplus, over a longer stretch, at lower noise.</b> On TinyGSM validation, the "
        f"S − M surplus per answer bit grows {ST['≤2 steps']['surplus_integral_per_bit'] / 1:.4f} → "
        f"{ST['3–4 steps']['surplus_integral_per_bit']:.4f} → {ST['5+ steps']['surplus_integral_per_bit']:.4f} "
        f"(≤2 / 3–4 / 5+ computation lines), and per problem 4× (answers are longer), and its peak moves from σ ≈ "
        f"{ST['≤2 steps']['surplus_peak_sigma']:.2f} to {ST['5+ steps']['surplus_peak_sigma']:.2f}. On GSM8K, easy "
        f"problems already gain +{100 * BA['≤2 steps']['delta'][3]:.1f} pp from CoBit-M in one segment; on 5+ step "
        f"problems (n = 225, wide CIs) no single segment helps detectably and the gain appears only with CoBit-M on at least the last 128 steps "
        f"({a('base', '5+ steps')}% → {a('SthenM128', '5+ steps')}% → {a('SthenM64', '5+ steps')}% for the last 128 / 192 steps; "
        f"{a('M', '5+ steps')}% everywhere).",
    ]),
    P("1. Setup", "h1"),
    P("Both experiments use the paired sampler of the earlier trajectory study: for a given seed, every configuration "
      "starts from the same initial noise per problem and draws the same churn noise per batch and step, so accuracy "
      "differences are paired per problem. Seed 0 reproduces the earlier paired S-vs-M run problem by problem "
      f"(CoBit-S {seedacc('base').split(' / ')[0]}%, CoBit-M {seedacc('M').split(' / ')[0]}%); seed 1 uses fresh initial "
      f"and churn noise (CoBit-S {seedacc('base').split(' / ')[1]}%, CoBit-M {seedacc('M').split(' / ')[1]}%). The seed-to-seed "
      "spread of about 2 pp on CoBit-S is why all results below average each problem over both seeds before "
      "bootstrapping over problems (95% CIs, 4000 resamples)."),
    P("The 256 steps are split into 8 segments of 32 steps (segment 0 = pure noise; segment 7 includes the final "
      "denoise). Difficulty: GSM8K problems by the number of calculation annotations in the gold solution "
      f"(≤2: n = {G['strata_n']['≤2 steps']}, 3–4: {G['strata_n']['3–4 steps']}, 5+: {G['strata_n']['5+ steps']}); "
      "TinyGSM validation problems by the number of computation lines in the solution code (an assignment with an "
      "arithmetic operator on its right-hand side)."),
    table([["Experiment", "Configurations", "What changes"],
           ["A  compute demand", "base; sc2_seg0 … sc2_seg7",
            "CoBit-S everywhere; +2 self-conditioning refreshes per step inside one segment, x̂₀ ← D(x, σ_next, x̂₀) "
            "(+64 NFE, the compute unit of the LM1B adaptive-compute study)"],
           ["B  model switching", "M; Mseg0 … Mseg7; MthenS64/128/192; SthenM64/128/192",
            "each denoiser call is routed to CoBit-S or CoBit-M by its noise level (= its step); CoBit-M in one segment, "
            "in a prefix (first c steps) or in a suffix (from step c on)"],
           ["Static strata", "4096 TinyGSM val problems × 129 σ (existing runs)",
            "entropy rate MSE/σ² and surplus (MSE_S − MSE_M)/σ² per difficulty stratum; training objective, answer tokens"]],
          [0.2 * W, 0.3 * W, 0.5 * W]),
    Spacer(1, 6),
    table([["Segment", "0", "1", "2", "3", "4", "5", "6", "7"],
           ["steps"] + [f"{32 * k}–{32 * k + 31}" for k in range(8)],
           ["σ range"] + SEGSIG],
          [0.12 * W] + [0.11 * W] * 8),
    P("2. Static compute-demand curves by difficulty", "h1"),
    figure(RD / "fig_static_strata.png", 1,
           "TinyGSM validation, per difficulty stratum: entropy rate of each model, the S − M surplus rate (95% bootstrap "
           "bands over problems) and its shape (unit area; dotted lines mark the sampler's segment boundaries). Harder "
           "problems carry more information per answer bit, CoBit-M's advantage on them is larger, and it is concentrated "
           "at lower noise."),
    table([["Stratum (val)", "n", "answer tokens", "rate ∫ S / M (per bit)", "surplus ∫ per bit", "surplus ∫ per problem",
            "surplus peak σ", "surplus share in seg 3–4 / 5–7"]] +
          [[n, str(v["n"]), f"{v['mean_answer_tokens']:.0f}", f"{v['rate_integral_S_per_bit']:.4f} / {v['rate_integral_M_per_bit']:.4f}",
            f"{v['surplus_integral_per_bit']:.4f}", f"{v['surplus_integral_per_problem']:.1f}", f"{v['surplus_peak_sigma']:.2f}",
            f"{100 * sum(v['surplus_segment_mass'][3:5]):.0f}% / {100 * sum(v['surplus_segment_mass'][5:]):.0f}%"]
           for n, v in ST.items()],
          [0.12 * W, 0.07 * W, 0.1 * W, 0.17 * W, 0.12 * W, 0.13 * W, 0.11 * W, 0.18 * W]),
    P("Integrals are over log σ on σ ≥ 0.1 (excludes CoBit-M's σ ≈ 0.03 error spike, as in the main report). The "
      "validation set is easier than GSM8K: 68% of its problems have at most two computation lines, so the 5+ stratum "
      "(n = 197) has wide bands. The answer region includes the restated question in the docstring (mostly copying), which "
      "dilutes the per-bit numbers equally in every stratum.", "cap"),
    P("3. GSM8K: extra iterations vs extra capacity, per segment", "h1"),
    figure(RD / "fig_gsm8k_segments.png", 2,
           "Paired accuracy change over CoBit-S, per segment and difficulty (both seeds; 95% bootstrap CIs over problems). "
           "Left: +2 self-conditioning refreshes per step in the segment. Right: CoBit-M in the segment."),
    table([["Segment (σ)", "A: +2 SC refreshes, all", "B: CoBit-M, all", "B: ≤2 steps", "B: 3–4 steps", "B: 5+ steps"]] +
          [[f"{k} ({SEGSIG[k]})", seg(AA['all'], k), seg(BA['all'], k), seg(BA['≤2 steps'], k), seg(BA['3–4 steps'], k),
            seg(BA['5+ steps'], k)] for k in range(8)],
          [0.16 * W, 0.18 * W, 0.17 * W, 0.17 * W, 0.16 * W, 0.16 * W]),
    P("Δ accuracy over CoBit-S in pp [95% CI]. Experiment A has no detectable effect anywhere: at this step count, "
      "additional refinement of the self-conditioning estimate by the same small network does not buy reasoning "
      "accuracy. Experiment B is positive from segment 2 on and peaks in segments 3–4.", "cap"),
    figure(RD / "fig_gsm8k_switch_vs_surplus.png", 3,
           "Share of the full CoBit-M gain recovered by CoBit-M in one segment (blue) vs the share of the static surplus "
           "(MSE_S − MSE_M)/σ² falling in that segment (grey, TinyGSM validation, same stratum). The location of the "
           "peak (segments 3–4) agrees; the static surplus over-predicts the low-noise segments 5–7 as single-segment "
           "interventions. The 5+ panel is noise: shares of a +4.7 pp gain on 225 problems, with every single-segment "
           "Δ inside its CI around 0."),
    P("4. GSM8K: prefix and suffix switching", "h1"),
    figure(RD / "fig_gsm8k_prefix_suffix.png", 4,
           "Accuracy against the number of steps run by CoBit-M, with CoBit-M either first (high noise, blue) or last "
           "(low noise, orange). Running CoBit-M on the low-noise end is always better for the same number of large-model "
           "steps; the first 64 steps can be left to CoBit-S at no cost."),
    table([["CoBit-M steps", "M first (MthenS)", "M last (SthenM)", "relative compute*", "M last, ≤2", "M last, 3–4", "M last, 5+"]] +
          [["0 (CoBit-S)", a("base"), a("base"), f"{100 * cost(0):.0f}%", a("base", "≤2 steps"), a("base", "3–4 steps"), a("base", "5+ steps")]] +
          [[str(c), a(f"MthenS{c}"), a(f"SthenM{256 - c}"), f"{100 * cost(c):.0f}%", a(f"SthenM{256 - c}", "≤2 steps"),
            a(f"SthenM{256 - c}", "3–4 steps"), a(f"SthenM{256 - c}", "5+ steps")] for c in (64, 128, 192)] +
          [["256 (CoBit-M)", a("M"), a("M"), "100%", a("M", "≤2 steps"), a("M", "3–4 steps"), a("M", "5+ steps")]],
          [0.15 * W, 0.15 * W, 0.15 * W, 0.15 * W, 0.13 * W, 0.13 * W, 0.14 * W]),
    P("GSM8K accuracy [%], both seeds. *Compute relative to CoBit-M everywhere, from the measured wall-time ratio of a "
      f"full CoBit-M to a full CoBit-S pass ({ratio:.2f}×). Paired: M last on 192 steps vs CoBit-M everywhere "
      f"{dci('M', 'SthenM64', STR[0][1](GOLD))} pp; M first on 192 steps vs CoBit-M everywhere "
      f"{dci('M', 'MthenS192', STR[0][1](GOLD))} pp.", "cap"),
    P("5. Implications for a diffusion-time expert-choice MoE", "h1"),
    bullets([
        "<b>Spend parameters, not iterations.</b> Extra passes of the small model (self-conditioning refreshes) are "
        "worth nothing on GSM8K at 256 steps in any segment; the same budget spent on a larger expert is worth up to "
        "+2.7 pp per 32 steps. Router capacity should go to width/parameters per step rather than to repeated "
        "evaluations.",
        "<b>Little capacity at σ > 4 (first ~25% of steps on this schedule).</b> Neither GSM8K (Δ ≈ 0 for segments 0–1, "
        "SthenM64 ≈ full CoBit-M) nor the static surplus (≤ 3% of its mass) shows demand there. A small expert suffices.",
        "<b>Peak demand at σ ≈ 1.3–2.6, and continued demand down to the end.</b> The single-segment peak is at segments 3–4, "
        "but suffix runs show the large model must also carry the trajectory to σ_min to realise those gains "
        "(sub-additivity). An expert-choice router conditioned on diffusion time should therefore assign the large expert "
        "over a contiguous low-noise range starting around σ ≈ 4, with the highest density around σ ≈ 1.3–2.6.",
        "<b>Difficulty shifts and widens the demand.</b> Hard problems have about 2× the surplus per bit (4× per problem), "
        "peaking at lower σ, and on GSM8K need the large model over at least the last half of the trajectory. A router "
        "that also sees an input-dependent signal (e.g. the model's own uncertainty or answer length) could allocate the "
        "large expert to hard problems over a longer range, and save it on easy ones.",
        "<b>The static surplus rate is a usable prior for where in σ to place capacity</b> (it locates the peak correctly "
        "and predicts the empty high-noise region), but single-segment interventions under-realise its low-noise mass; "
        "routing decisions should be validated end to end, as here.",
    ]),
    P("6. Caveats", "h1"),
    bullets([
        "Two seeds; per-segment effects of 1–3 pp are near the resolution of 1319 problems (CI half-width ≈ 1–1.3 pp "
        "overall, 2–3 pp per stratum). The 5+ stratum (225 problems, 4.7% → 9.3%) is too small for per-segment conclusions.",
        "One schedule (CoBit-S's, γ = 0.31, 256 steps). With fewer steps, extra iterations may matter where they do not here.",
        "Switching models mid-trajectory carries the self-conditioning estimate of one model into the other; the two "
        "models were trained independently, so this measures a hand-off, not a jointly trained MoE.",
        "Difficulty proxies differ between the static (computation lines in TinyGSM code) and GSM8K (gold calculation "
        "annotations) parts; both count arithmetic steps.",
    ]),
    P("Files: experiments/denoising_mse/tinygsm/compute_demand.py (runs), analyze_compute_demand.py (analysis), "
      "run_compute_demand.sh; results in experiments/denoising_mse/results_tinygsm/compute_demand/ "
      "(summary.json, gsm8k_accuracy.csv, static_strata.csv); per-problem correctness in "
      "runs/denoising_mse/tinygsm/compute_demand/.", "cap"),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="TinyGSM compute demand and model switching",
                        author="BitstreamDiffusion experiments")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
