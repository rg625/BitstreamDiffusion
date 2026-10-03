"""Build the short PDF research report from the adaptive-compute study.

Every number is taken from ../REPORT.md / ../../results/analysis/*.json; figures come from make_figs.py
(same data and plotting code as ../figs.py). Requires reportlab (not in the `sedd` env; built with a
separate virtualenv). Output: ../CoBit_adaptive_compute_report.pdf
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "CoBit_adaptive_compute_report.pdf"
FONTS = Path("/usr/share/fonts/truetype/liberation2")
for name, f in [("Serif", "LiberationSerif-Regular"), ("Serif-B", "LiberationSerif-Bold"),
                ("Serif-I", "LiberationSerif-Italic"), ("Serif-BI", "LiberationSerif-BoldItalic"),
                ("Sans", "LiberationSans-Regular"), ("Sans-B", "LiberationSans-Bold"),
                ("Sans-I", "LiberationSans-Italic"), ("Sans-BI", "LiberationSans-BoldItalic")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f"{f}.ttf")))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-BI")

INK = colors.HexColor("#1b1b1a")
INK2 = colors.HexColor("#52514e")
ACCENT = colors.HexColor("#2a78d6")
RULE = colors.HexColor("#d9d8d2")
TINT = colors.HexColor("#eef3fa")
TINT2 = colors.HexColor("#f6f5f1")

st = {
    "title": ParagraphStyle("title", fontName="Sans-B", fontSize=17.5, leading=21.5, textColor=INK, spaceAfter=5),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=9.5, leading=13, textColor=INK2, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Sans-B", fontSize=12.2, leading=15, textColor=INK, spaceBefore=13,
                         spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=10.4, leading=14.4, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=10.4, leading=14.2, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.7, leading=11.6, textColor=INK2, spaceBefore=3,
                          spaceAfter=12),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.8, leading=11.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=8.8, leading=11.2, textColor=INK),
    "box": ParagraphStyle("box", fontName="Serif", fontSize=10.6, leading=14.6, textColor=INK),
    "boxh": ParagraphStyle("boxh", fontName="Sans-B", fontSize=10.5, leading=13, textColor=ACCENT, spaceAfter=4),
}
TEXT_W = A4[0] - 2 * 2.1 * cm


def P(t, s="body"):
    return Paragraph(t, st[s])


def bullets(items):
    return ListFlowable([ListItem(P(t, "bullet"), leftIndent=12, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontName="Serif",
                        bulletFontSize=10, spaceBefore=0, spaceAfter=6)


def figure(png, num, caption, width=TEXT_W):
    from PIL import Image as PILImage

    w, h = PILImage.open(png).size
    img = Image(str(png), width=width, height=width * h / w)
    return KeepTogether([img, P(f"<b>Figure {num}.</b> {caption}", "cap")])


def table(rows, widths, header=True, shade_rows=()):
    data = [[P(c, "cellb" if (header and i == 0) else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE)]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK2)]
    for r in shade_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), TINT2))
    t.setStyle(TableStyle(style))
    return t


def callout(head, paras, bg=TINT):
    inner = [P(head, "boxh")] + [P(p, "box") for p in paras]
    t = Table([[inner]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                           ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    return t


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(INK2)
    canvas.drawString(2.1 * cm, 1.25 * cm, "CoBit-S adaptive compute: timestep vs. state")
    canvas.drawRightString(A4[0] - 2.1 * cm, 1.25 * cm, f"{doc.page}")
    canvas.restoreState()


story = []
story += [
    P("Where should extra compute go in a diffusion language model? Timestep versus state in CoBit-S", "title"),
    P("Short research report · CoBit-S (130M) on LM1B · counterfactual adaptive-compute study · October 2026<br/>"
      "Summarises <font face='Sans-I'>experiments/compute_allocation/report/REPORT.md</font> (Study 1) and "
      "<font face='Sans-I'>experiments/adaptive_compute/report/REPORT.md</font> (Study 2); all numbers are from "
      "those studies’ result files.", "subtitle"),
]

# 1 ---------------------------------------------------------------------------------------------
story += [P("1. Executive summary", "h1"),
          P("This report covers two linked studies of CoBit-S. Study 1 asked which part of the denoising trajectory "
            "limits quality under the fixed production budget, and found that the value of compute changes strongly "
            "along the trajectory. That raised the question this report focuses on (Study 2): would it help to spend "
            "inference compute <i>adaptively</i>, giving extra denoiser evaluations to the individual states that need "
            "them most rather than following a fixed schedule? Using counterfactual branches from saved denoising "
            "states, we measured how much one more unit of compute changes the final sample’s quality, at different "
            "points in the trajectory and for different samples."),
          bullets([
              "<b>Compute value depends strongly on the timestep.</b> The same extra compute lowers GPT-2 NLL by up to "
              "0.39 nats/token when spent in the middle of denoising (steps 64–127), but does almost nothing early and "
              "slightly hurts late (after step 160).",
              "<b>Compute value depends only weakly on the individual state.</b> Within a timestep, samples differ "
              "measurably at only one of eight checkpoints, the differences are hard to predict from cheap state "
              "features, and no state-aware allocation beats a timestep-only schedule.",
              "<b>The NLL gains are confounded with entropy.</b> Every large gain comes with lower sample entropy "
              "(up to −0.17 nats), so we cannot yet call the mid-trajectory gains quality improvements.",
          ]),
          table([["", "What the experiments show", "Verdict"],
                 ["(a) <b>Timestep-dependent</b> allocation (fixed schedule over denoising steps)",
                  "Large, consistent differences between timesteps; a timestep-only schedule captures all the "
                  "allocation gain we could find.", "<b>Supported</b> (gains entropy-confounded)"],
                 ["(b) <b>State-dependent or token-level</b> allocation (routing by sample, state or token)",
                  "Small per-state signal at one checkpoint; within-timestep rank correlation 0.03–0.06; "
                  "no gain over (a). Token-level allocation not tested.", "<b>Not justified</b> by current data"]],
                [4.6 * cm, 8.4 * cm, 3.8 * cm], shade_rows=(1,)),
          ]

# 2 ---------------------------------------------------------------------------------------------
story += [P("2. Background: why we ran these experiments", "h1"),
          P("<b>Study 1: where does the fixed budget limit quality?</b> The production sampler spends its 255 model "
            "evaluations on a step schedule that gives every stretch of the denoising trajectory the same share of "
            "compute. We tested that assumption directly. The 254 steps were split into 8 segments of 32, and from saved "
            "states of one reference run (1,024 samples; GenPPL 60.2 at entropy 4.31, consistent with the published 59.8) "
            "we changed one segment at a time: making its steps 4× finer (+96 NFE), or switching off its stochastic "
            "error-correcting noise (“churn”, no extra NFE). Comparing each branch with the same samples’ reference "
            "isolates the effect of that one segment."),
          P("<b>What Study 1 found.</b>"),
          bullets([
              "<b>Compute is not equally useful everywhere.</b> Only 5.5% of final token decisions are made in segments "
              "0–2, which receive 37.5% of the compute; most decisions are made in segments 3–6.",
              "<b>The middle responds strongly to finer steps.</b> Finer steps in segments 1–4 lowered GPT-2 NLL by 0.05 "
              "to 0.35 nats/token (segment 4: −0.087 [−0.111, −0.063]). In segments 1–3 this came with entropy drops "
              "beyond 0.02, so it could not be read as a quality gain.",
              "<b>Late segments depend on error correction, not resolution.</b> Removing churn in segments 4–7 cost "
              "+0.06 to +0.11 nats/token, while finer steps there did not help and in segments 5–6 made samples worse "
              "(+0.029, +0.038).",
              "<b>The planned fixed-budget confirmation could not be run:</b> changing γ from 0.185 to 0.215 moved "
              "entropy by at most 0.013, within seed noise, so entropy could not be held fixed.",
          ]),
          P("<b>Why a second study.</b> Study 1 showed that the value of compute changes along the trajectory, but it "
            "treated every sample the same. It left open whether part of that variation is sample-specific, that is, "
            "whether a controller reading the current state could spend compute better than a fixed schedule. Study 2 "
            "asks exactly that. Two lessons from Study 1 shaped its design. Finer steps change the trajectory and its "
            "entropy, so Study 2 uses a cleaner unit of extra compute that keeps the noise path fixed. And a single "
            "continuation per sample is dominated by randomness: in segment 0, finer steps changed per-sample NLL with "
            "a standard deviation of 0.25 nats/token while the mean effect was only −0.008, and removing churn, which "
            "adds no compute, produced the same spread. Study 2 therefore continues every state four times."),
          ]

# 3 ---------------------------------------------------------------------------------------------
story += [P("3. Study 2 setup", "h1"),
          P("<b>Model and sampler.</b> CoBit-S on LM1B with the production stochastic sampler (255 model evaluations, "
            "NFE, per sample). We took 1,000 of Study 1’s reference samples and used their saved states at 8 checkpoints "
            "(the start of steps 0, 32, …, 224; noise level σ from 77 down to 0.34). Checkpoint s is the start of "
            "Study 1’s segment s."),
          P("<b>Unit of extra compute.</b> “+c” adds c extra self-conditioning passes at every step of the 32-step "
            "window after a checkpoint (+32c NFE, i.e. +12.5% of baseline compute per unit). The noise schedule, the "
            "update rule and the random noise are unchanged, so the branch differs only in compute. We tested +1, "
            "+2 and +4."),
          P("<b>Quality and noise control.</b> Quality Q is the negative GPT-2-large NLL per token of the final "
            "sample (higher is better). Because a single continuation is dominated by trajectory randomness, each "
            "state and compute level was continued 4 times with shared random numbers across compute levels. Two "
            "independent halves of these continuations let us separate real differences between states from noise; "
            "predictors and allocation policies were chosen on one half and scored on the other. Total cost: "
            "3.3 GPU-hours on one RTX 4090."),
          ]

# 4 ---------------------------------------------------------------------------------------------
story += [P("4. Main finding: compute value varies substantially by timestep", "h1"),
          P("The average benefit of extra compute changes sharply along the trajectory (Figure 1). With +4 "
            "(+128 NFE), mean Q improves by <b>0.393</b> nats/token [95% CI 0.368, 0.418] at checkpoint 2 and "
            "<b>0.278</b> [0.260, 0.297] at checkpoint 3, by 0.068 at checkpoint 4 and by only 0.002–0.048 at "
            "checkpoints 0–1. From checkpoint 5 on, extra compute slightly <i>lowers</i> Q (−0.007 to −0.037). "
            "Returns diminish: at checkpoint 3 the first unit adds 0.162, the second 0.068."),
          figure(HERE / "pdf_fig1.png", 1,
                 "Mean gain in Q from extra self-conditioning compute applied in the 32 steps after each checkpoint "
                 "(1,000 states, 95% CIs). The benefit is concentrated in the middle of denoising (shaded, steps "
                 "64–127) and is near zero or negative elsewhere. Gains here are partly entropy reductions (Section 6).", width=TEXT_W * 0.86),
          P("End-to-end runs over all 1,024 samples confirm the pattern at matched compute. Spending +4 only in "
            "checkpoints 2–3 (511 NFE) lowered GenPPL from 60.2 to <b>36.4</b> (ΔQ 0.510 [0.464, 0.556]); spreading "
            "the same budget as +1 everywhere (509 NFE) reached 45.8 (ΔQ 0.276 [0.236, 0.314]). Gains from different "
            "checkpoints do not fully add up: the combined runs delivered 76% and 83% of the sum of the per-checkpoint "
            "effects."),
          ]

# 5 ---------------------------------------------------------------------------------------------
story += [P("5. State-dependent compute: evidence is weak and not practically exploitable", "h1"),
          P("For state-dependent allocation to help, three things must hold: states at the same timestep must "
            "differ in how much they gain, those differences must be predictable, and acting on the prediction must "
            "beat a simpler schedule. The evidence is weak on the first point and negative on the other two."),
          bullets([
              "<b>Opportunity is small.</b> The spread of gains between states (τ, after removing noise) is "
              "detectable only at checkpoint 3: τ = 0.120 [0.071, 0.153] for +4, against a mean gain of 0.278. At "
              "the other seven checkpoints the confidence intervals include zero (Figure 2). Two apparent effects in "
              "the 200-sample pilot did not replicate at 1,000 samples. Even at checkpoint 3, measurement noise "
              "(0.26) is twice τ.",
              "<b>Predictability is weak.</b> Ridge predictors on 15 cheap state and trajectory features rank states "
              "within a timestep with Spearman ρ = 0.03–0.06 (CIs exclude zero, but the signal is tiny). The timestep "
              "alone gives a pooled ρ of 0.37 for +4, and state features add nothing to it. Trained without the "
              "timestep and tested on unseen checkpoints, state features transfer poorly (ρ = 0.12, negative R²).",
              "<b>No allocation gain over timestep.</b> At equal compute, a learned state-plus-timestep allocator "
              "matches the best timestep-only schedule (difference +0.003 [−0.005, +0.011] nats/token at +100% compute; "
              "−0.008 to +0.004 across all budgets). Giving whole samples more compute is no better than random "
              "(+0.000 [−0.029, +0.031]), and picking individual states from noisy measurements is clearly worse "
              "(Figure 3). Token- and bit-level allocation could not be tested: the sampler has no mechanism to "
              "spend compute on part of a sequence.",
          ]),
          figure(HERE / "pdf_fig2.png", 2,
                 "Between-state spread of the gain (τ, noise removed by split-half covariance; 95% CIs) versus the noise "
                 "in a 4-continuation estimate. Only checkpoint 3 shows a spread clearly above zero, and it is "
                 "well below the measurement noise."),
          figure(HERE / "pdf_fig3.png", 3,
                 "Quality gain per sample at a fixed extra-compute budget (+4 unit). Policies are chosen on half the "
                 "continuations and scored on the other half. The learned state-aware allocator (dashed) lies on top "
                 "of the timestep-only schedule (orange); per-sample and per-state selection add nothing. The star is "
                 "a full end-to-end run, which achieves 76% of the curve’s additive estimate.", width=TEXT_W * 0.93),
          ]

# 6 ---------------------------------------------------------------------------------------------
story += [KeepTogether([P("6. The entropy confound and what remains unresolved", "h1"),
          P("GPT-2 NLL falls when samples become less diverse, so an NLL gain is only a quality gain if diversity is "
            "held fixed. Here, every large gain comes with lower unigram entropy of the samples:"),
          table([["Sampler (all 1,024 samples, paired)", "NFE", "ΔQ (nats/token)", "GenPPL", "Δ entropy (nats)"],
                 ["Production baseline", "255", "–", "60.2", "– (4.313)"],
                 ["+1 at every checkpoint", "509", "+0.276 [0.236, 0.314]", "45.8", "−0.069 [−0.086, −0.052]"],
                 ["+4 at checkpoints 2–3 only", "511", "+0.510 [0.464, 0.556]", "36.4", "−0.172 [−0.193, −0.150]"]],
                [5.3 * cm, 1.3 * cm, 3.7 * cm, 1.7 * cm, 4.8 * cm])]),
          Spacer(1, 8),
          P("Per checkpoint, the +4 branch lowers entropy by 0.146 at checkpoint 2 and 0.027 at checkpoint 3, and "
            "across states the size of the gain correlates with the size of the entropy drop (r from −0.17 to "
            "−0.66, negative at every checkpoint). By the rule from Study 1, an entropy change "
            "beyond 0.02 means an NLL change cannot be read as quality on its own; every high-gain policy here "
            "exceeds it. A GenPPL of 36.4 at entropy 4.14 is therefore <i>not</i> directly comparable with 60.2 at "
            "4.31, and we do not claim that mid-trajectory compute makes samples better."),
          P("<b>What survives.</b> The timestep pattern itself is robust, and the small between-state spread at "
            "checkpoint 3 persists after a linear entropy adjustment (τ 0.120 → 0.111). <b>What is unresolved.</b> "
            "Whether the mid-trajectory gain is a genuine quality improvement or mostly a move along the "
            "quality–diversity trade-off. The study used one model, one sampler setting, one kind of extra compute "
            "and a single quality metric (GPT-2 NLL)."),
          ]

# 7 ---------------------------------------------------------------------------------------------
story += [P("7. Conclusion and recommended next experiment", "h1"),
          P("The value of extra inference compute in CoBit-S is mainly a function of <i>when</i> it is spent along "
            "the denoising trajectory, not of <i>which</i> sample or state receives it. This supports, at most, a "
            "fixed timestep-dependent compute schedule. It does not support building state-dependent routing, "
            "mixture-of-experts or token-level adaptive compute: the opportunity is small, poorly predictable and "
            "adds nothing over the schedule. The two studies, using different kinds of extra compute (finer steps and "
            "extra self-conditioning passes), point the same way: compute in the middle of the trajectory gives large "
            "NLL gains that coincide with lower entropy, and compute after about step 160 slightly hurts."),
          P("<b>Recommended next experiment.</b> Test the timestep schedule at matched entropy before adopting it: "
            "compare extra self-conditioning passes in steps 64–127 (and none after step 160) against the baseline "
            "sampler with an entropy lever adjusted so that both produce the same entropy. In Study 1 the "
            "churn parameter γ barely moved entropy, so another lever such as the churn noise scale or a decoding "
            "temperature is needed. Report an entropy-insensitive metric such as MAUVE alongside GPT-2 NLL. If the "
            "gain survives, adopt a fixed per-timestep schedule; if it disappears, the mid-trajectory effect was an "
            "entropy trade-off."),
          Spacer(1, 6),
          callout("Takeaway", [
              "Extra compute in CoBit-S matters mostly by <i>when</i> it is spent: concentrating it in the middle of "
              "denoising lowers GPT-2 NLL far more than spreading it evenly. Those gains coincide with lower sample "
              "entropy, so they are not yet established quality gains. State-dependent or token-level adaptive "
              "compute is not justified by these data: the per-state signal is small, hard to predict, and adds "
              "nothing over a simple timestep schedule.",
          ]),
          ]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.9 * cm,
                        bottomMargin=2.0 * cm, title="Where should extra compute go in a diffusion language model?",
                        author="CoBit adaptive-compute study", subject="CoBit-S adaptive compute: timestep vs. state")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
