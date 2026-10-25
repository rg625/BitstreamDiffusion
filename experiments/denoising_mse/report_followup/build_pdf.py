"""Build the follow-up PDF report: model disagreement along CoBit-M and ground-truth trajectories, and
self-conditioning on ground-truth states (experiments/denoising_mse, 4-5 October 2026).

All numbers are read from ../results/gt/{gt_summary.json, gt_curves.csv} and ../results/main/curves.npz at
build time; figures are ../results/gt/fig_*.png (from analyze_gt.py). Requires reportlab and numpy (use a
throwaway venv, e.g. python -m venv --system-site-packages /tmp/pdfenv && /tmp/pdfenv/bin/pip install reportlab).
Output: CoBit_followup_report.pdf (this folder)
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
RES = HERE.parent / "results"
OUT = HERE / "CoBit_followup_report.pdf"
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
    canvas.drawString(2.1 * cm, 1.2 * cm, "CoBit follow-up: disagreement on true and generated trajectories, self-conditioning")
    canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


# ------------------------------------------------------------------ numbers
S = json.loads((RES / "gt/gt_summary.json").read_text())
rows = list(csv.DictReader(open(RES / "gt/gt_curves.csv")))
G = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
t, sig = G["t"], G["sigma"]
N = S["n"]


def at_t(key, tt):
    return float(np.interp(-tt, -t, G[key]))


def argext(arr, fn=np.argmax, mask=None):
    a = np.where(mask, arr, np.nan) if mask is not None else arr
    i = int(np.nanargmax(a) if fn is np.argmax else np.nanargmin(a))
    return float(arr[i]), float(t[i]), float(sig[i])


gc = S["gray_curve"]
gk = list(gc)  # static, zeros, onestep, carry
g_static, g_zero, g_one, g_carry = (gc[k] for k in gk)
cz = {m: G[f"mse_{m}_carry_mean"] - G[f"mse_{m}_zeros_mean"] for m in "SM"}
oz = {m: G[f"mse_{m}_onestep_mean"] - G[f"mse_{m}_zeros_mean"] for m in "SM"}
czmax = {m: argext(cz[m]) for m in "SM"}
oz_ext = {m: max(abs(oz[m].max()), abs(oz[m].min())) for m in "SM"}
oz_lo = {m: argext(oz[m], np.argmin) for m in "SM"}
dc = G["dmse_true_carry_mean"]
dc_min, dc_max = argext(dc, np.argmin), argext(dc)
mse_peak_M_carry = argext(G["mse_M_carry_mean"], mask=t < 0.95)
tp = {k: S[f"tokdis_peak_{k}"] for k in ("Straj", "Mtraj", "true_carry", "true_zeros")}
R = S["rates"]
rk = {"S": "same-input rate, CoBit-S trajectories", "M": "same-input rate, CoBit-M trajectories",
      "true": "same-input rate, true chains (carry SC)", "div": "token divergence, paired generation",
      "static": "static surplus rate (MSE_S − MSE_M)/σ²", "ent": "static entropy rate MSE_S/σ²"}
pct = lambda a, b: 100 * (a / b - 1)  # noqa: E731

# ------------------------------------------------------------------ story
story = [
    P("Follow-up: how differently CoBit-S and CoBit-M think along generated and ground-truth trajectories, "
      "and what self-conditioning does on true states", "title"),
    P(f"OpenWebText · experiments/denoising_mse · 5 October 2026 · follows the 4 October report "
      f"(report/CoBit_denoising_mse_report.pdf)", "subtitle"),
    box("Summary", [
        f"<b>Along CoBit-M's own trajectories</b> (S fed M's exact inputs) the two models disagree more at "
        f"high noise and less at low noise than along CoBit-S's trajectories, but after the 1/σ² weighting the "
        f"same-input disagreement rate still sits at σ ≈ {R[rk['M']]['sigma_median']:.2f} (S trajectories: "
        f"{R[rk['S']]['sigma_median']:.2f}), far below where generations split (σ ≈ {R[rk['div']]['sigma_median']:.1f}).",
        f"<b>Your one-step experiment leaves the grey curve essentially unchanged.</b> Denoising x<sub>t</sub> "
        f"with the model's own zero-SC estimate from x<sub>t+1</sub> (one true forward step noisier) as the "
        f"self-conditioning input gives a surplus rate (MSE<sub>S</sub> − MSE<sub>M</sub>)/σ² in the same place "
        f"(median σ {g_one['sigma_median']:.2f} vs {g_static['sigma_median']:.2f}) with "
        f"{pct(g_one['integral_nats_per_bit'], g_static['integral_nats_per_bit']):+.0f}% area. Each model's error "
        f"moves by at most {max(oz_ext.values()):.4f}.",
        f"<b>Carrying the full self-conditioning history along a true chain is different.</b> Each model's error "
        f"rises well above its zero-SC error, CoBit-M early (up to +{czmax['M'][0]:.3f} at t ≈ {czmax['M'][1]:.2f}) "
        f"and CoBit-S later (+{czmax['S'][0]:.3f} at t ≈ {czmax['S'][1]:.2f}). The surplus becomes negative at "
        f"high noise and "
        f"{g_carry['integral_nats_per_bit'] / g_static['integral_nats_per_bit']:.1f}× larger at low noise. On a true "
        f"chain the state follows the real x<sub>0</sub>, not the model's guesses, so this measures how strongly "
        f"each model sticks to its earlier commitments; CoBit-M commits earlier.",
        f"<b>The self-conditioning history makes the models disagree even on identical true states</b>: token "
        f"disagreement peaks at {100 * tp['true_carry']['value']:.0f}% with carried SC vs "
        f"{100 * tp['true_zeros']['value']:.0f}% without.",
        "Ground truth check: with zero self-conditioning, the true chains reproduce the static curves, as they must "
        "(each step's state has the static marginal).",
    ]),

    P("1. Why these experiments", "h1"),
    P("The 4 October report fed CoBit-M exactly CoBit-S's inputs along S's generated trajectories and found the "
      "two denoisers disagree little on identical inputs, with the disagreement rate peaking near the static "
      "curves (σ ≈ 0.5), while the generations themselves split much earlier (σ ≈ 2.5). Two questions followed: "
      "does that hold along CoBit-M's trajectories (closer to the data), and ideally along ground-truth "
      "trajectories? On true trajectories each step's state has exactly the static marginal, so per-step averages "
      "can only differ from the static curves through the self-conditioning input, which in sampling is the model's "
      "own earlier prediction. The experiments below test that directly."),

    P("2. Setup", "h1"),
    bullets([
        f"<b>Along CoBit-M trajectories</b> ({N['Mtraj']} examples). CoBit-M generates with the paper's sampler "
        "(γ = 0.13, 256 NFE, carry self-conditioning, CoBit-S's schedule, same seeds as before). At every step CoBit-S "
        "gets M's exact input: the churned state and M's self-conditioning input. A check re-evaluating M on the "
        "rebuilt input reproduces its outputs exactly.",
        f"<b>Ground-truth chains</b> ({N['true']} validation sequences). On the sampler's fixed grid of 255 "
        "evaluation noise levels, each clean sequence x<sub>0</sub> is corrupted along the VE forward Markov chain "
        "(x<sub>t</sub> directly from x<sub>0</sub>, then x<sub>t+1</sub> = x<sub>t</sub> + √(σ<sub>t+1</sub>² − "
        "σ<sub>t</sub>²)·ε to the next noisier level), and the chain is walked from high to low noise. At every "
        "step both models denoise the same true state with three self-conditioning inputs:",
        "<b>zero</b>: x̂<sub>0</sub> = 0, the no-self-conditioning training input (reproduces the static numbers); "
        "<b>one-step</b> (the proposed experiment): the model's own zero-SC estimate at x<sub>t+1</sub>, i.e. "
        "D(x<sub>t</sub>, σ<sub>t</sub>, D(x<sub>t+1</sub>, σ<sub>t+1</sub>, 0)); <b>carry</b>: the model's own "
        "previous prediction with its full history, as in sampling.",
        "Scaled MSE: (MSE vs x<sub>0</sub>)/σ², per bit; the surplus rate (MSE<sub>S</sub> − MSE<sub>M</sub>)/σ² "
        "is the grey curve of the previous report. Diffusion time t = F<sub>S</sub>(σ) as before (t = 1 noise, "
        "t = 0 clean). GPU time ≈ 1.4 h on one RTX 4090.",
    ]),

    P("3. Disagreement on identical inputs along three kinds of states", "h1"),
    figure(RES / "gt/fig_same_input.png", 1,
           "Disagreement of the two denoisers on identical inputs: mean squared gap (log scale, left) and fraction "
           "of tokens predicted differently (right), along CoBit-S trajectories, CoBit-M trajectories, and true "
           "chains with carried or zero self-conditioning. Mean with central 68% / 95% of examples."),
    table([
        ["t (σ)", "S trajectories: gap / tokens", "M trajectories: gap / tokens", "true chains, zero SC",
         "true chains, carry SC"],
    ] + [[f"{tt:.2f} ({float(np.interp(-tt, -t, sig)):.2f})",
          f"{at_t('ldD2_Straj_mean', tt):.4f} / {100 * at_t('tokdis_Straj_mean', tt):.0f}%",
          f"{at_t('ldD2_Mtraj_mean', tt):.4f} / {100 * at_t('tokdis_Mtraj_mean', tt):.0f}%",
          f"{at_t('dD2_true_zeros_mean', tt):.4f} / {100 * at_t('tokdis_true_zeros_mean', tt):.0f}%",
          f"{at_t('dD2_true_carry_mean', tt):.4f} / {100 * at_t('tokdis_true_carry_mean', tt):.0f}%"]
         for tt in (0.86, 0.79, 0.65, 0.5, 0.35, 0.2)],
          [2.2 * cm, 3.6 * cm, 3.6 * cm, 3.4 * cm, W - 12.8 * cm]),
    Spacer(1, 5),
    bullets([
        f"Along CoBit-M's trajectories the models disagree about twice as much at high noise as along CoBit-S's "
        f"(M commits to content earlier and S does not follow), and less at low noise, where M's trajectory has "
        f"settled. Token disagreement peaks at t ≈ {tp['Mtraj']['t']:.2f} ({100 * tp['Mtraj']['value']:.0f}%) vs "
        f"t ≈ {tp['Straj']['t']:.2f} ({100 * tp['Straj']['value']:.0f}%) along S's.",
        f"On true chains with zero self-conditioning the disagreement is largest at low noise (peak "
        f"{100 * tp['true_zeros']['value']:.0f}% of tokens at t ≈ {tp['true_zeros']['t']:.2f}): real text is harder "
        f"to agree on than either model's own samples.",
        f"With carried self-conditioning the disagreement is an order of magnitude larger "
        f"(peak {100 * tp['true_carry']['value']:.0f}% at t ≈ {tp['true_carry']['t']:.2f}): each model's own "
        "history in the self-conditioning channel pulls its predictions apart even when the noisy states are "
        "identical.",
    ]),

    P("4. Error along ground-truth chains and the effect of self-conditioning", "h1"),
    figure(RES / "gt/fig_truechain.png", 2,
           "(a) MSE against the true x<sub>0</sub> along true chains, for each model with carried, one-step and "
           "zero self-conditioning. (b) Change in MSE caused by self-conditioning relative to zero SC. (c) CoBit-M's "
           "advantage MSE<sub>S</sub> − MSE<sub>M</sub> along true chains, with the static curve."),
    bullets([
        f"<b>Validation.</b> Zero self-conditioning on the chains reproduces the static curves (surplus area "
        f"{g_zero['integral_nats_per_bit']:.4f} vs {g_static['integral_nats_per_bit']:.4f} nats/bit static; "
        f"peak σ {g_zero['sigma_peak']:.2f} vs {g_static['sigma_peak']:.2f}).",
        f"<b>One-step self-conditioning</b> changes each model's error by at most {oz_ext['S']:.4f} (S) and "
        f"{oz_ext['M']:.4f} (M). It helps slightly at low noise (M: {oz_lo['M'][0]:+.4f} at t ≈ {oz_lo['M'][1]:.2f}; "
        f"S: {oz_lo['S'][0]:+.4f} at t ≈ {oz_lo['S'][1]:.2f}) and costs CoBit-M a little at high noise.",
        f"<b>Carried self-conditioning</b> raises CoBit-M's error early (largest at t ≈ {czmax['M'][1]:.2f}, "
        f"σ ≈ {czmax['M'][2]:.2f}: +{czmax['M'][0]:.3f}, reaching {mse_peak_M_carry[0]:.3f}, above the prior-level "
        f"error ≈ 0.17) and CoBit-S's later (t ≈ {czmax['S'][1]:.2f}, σ ≈ {czmax['S'][2]:.2f}: +{czmax['S'][0]:.3f}). "
        f"CoBit-M's advantage turns into a deficit at high noise ({dc_min[0]:+.3f} at t ≈ {dc_min[1]:.2f}) and grows "
        f"at low noise ({dc_max[0]:+.3f} at t ≈ {dc_max[1]:.2f}).",
        "Interpretation: in sampling, the state moves toward the model's own guesses, so carried predictions stay "
        "consistent with the state. On a true chain the state keeps pointing at the real x<sub>0</sub>, so a model "
        "that has committed to a different guess keeps paying for it. The timing of the error bump is therefore a "
        "read-out of when each model commits: CoBit-M around t ≈ 0.7, CoBit-S around t ≈ 0.5.",
    ]),

    P("5. Does the grey curve change?", "h1"),
    figure(RES / "gt/fig_gray_curve.png", 3,
           "The surplus rate (MSE<sub>S</sub> − MSE<sub>M</sub>)/σ² (the grey curve) recomputed on true chains with "
           "zero, one-step and carried self-conditioning. (a) Absolute, with ±2 SE for the one-step variant; the "
           "area is the surplus in the denoising bound. (b) Each curve scaled to unit area on σ ≥ 0.08."),
    table([
        ["Surplus rate", "Area (nats/bit)", "Peak σ", "Median σ", "10–90% range of σ"],
    ] + [[lab, f"{v['integral_nats_per_bit']:.4f}", f"{v['sigma_peak']:.2f}", f"{v['sigma_median']:.2f}",
          f"{v['sigma_q10']:.2f}–{v['sigma_q90']:.2f}"]
         for lab, v in (("static (training mix), the grey curve", g_static), ("true chains, zero SC", g_zero),
                        ("true chains, one-step SC", g_one), ("true chains, carry SC", g_carry))],
          [5.8 * cm, 2.6 * cm, 2.0 * cm, 2.2 * cm, W - 12.6 * cm]),
    Spacer(1, 5),
    P(f"The one-step construction is closer to generation than the static test, and it leaves the grey curve "
      f"where it was: the place where CoBit-M's extra parameters reduce the denoising bound is robust to giving the "
      f"model the previous step's estimate. Only the full self-conditioning history changes the picture, and that "
      f"change ({g_carry['integral_nats_per_bit'] / g_static['integral_nats_per_bit']:.1f}× the area, shifted to "
      f"σ ≈ {g_carry['sigma_median']:.2f}, with a negative lobe at σ ≈ 1–3) mostly reflects commitment dynamics "
      f"that only make sense on the models' own trajectories."),

    KeepTogether([Paragraph("6. All curves together", st["h1"]), figure(RES / "gt/fig_rates_all.png", 4,
           "Densities over log σ (unit area on σ ≥ 0.08): same-input disagreement rates on the three kinds of states, "
           "surplus rates on true chains (one-step and carry SC), the static surplus and entropy rates, and the "
           "token divergence of the paired generations.", width=0.82 * W)]),
    table([["Curve", "Peak σ", "Median σ", "10–90% range of σ"]]
          + [[lab, f"{R[k]['sigma_peak']:.2f}", f"{R[k]['sigma_median']:.2f}",
              f"{R[k]['sigma_q10']:.2f}–{R[k]['sigma_q90']:.2f}"]
             for lab, k in (("Same input, CoBit-S trajectories", rk["S"]), ("Same input, CoBit-M trajectories", rk["M"]),
                            ("Same input, true chains (carry SC)", rk["true"]), ("Static surplus (grey)", rk["static"]),
                            ("Static entropy rate, CoBit-S", rk["ent"]), ("Token divergence, paired generation", rk["div"]))],
          [6.4 * cm, 2.4 * cm, 2.4 * cm, W - 11.2 * cm]),
    Spacer(1, 5),
    P("Every measure of local model difference, on S's states, M's states or true states, sits at σ ≈ 0.4–0.6. "
      "The split of the generated trajectories at σ ≈ 2.5 stands apart: it is where small differences are "
      "amplified into different samples. The model-switching experiment proposed in the previous report (generate "
      "with one model above a switch point t* and the other below, same seeds, GenPPL against t*) remains the "
      "direct test of which region produces CoBit-M's better samples."),

    P("7. Caveats and files", "h1"),
    bullets([
        "One shared sampler (γ = 0.13, CoBit-S's schedule); the M-trajectory and true-chain results use 512 "
        "examples each, so per-step means have small but visible noise. Same RoPE fix as the 4 October report.",
        f"Code: {M('experiments/denoising_mse/trajectories.py')} (modes {M('--local --ref M')}, {M('--truechain')}), "
        f"{M('analyze_gt.py')}. Results: {M('results/gt/')}; raw runs {M('runs/denoising_mse/traj/{localM,truechain}')}.",
    ]),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="CoBit follow-up: disagreement and self-conditioning on true trajectories",
                        author="BitstreamDiffusion experiments")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
