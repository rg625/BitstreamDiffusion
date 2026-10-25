"""Report 5: the maths of CoBit's sampler, and what breaks it under score temperature. CPU only.

Evidence: result JSONs in results/scoretemp/eval_gh200 (500k unguided, entropic, 1319 problems), the measured
entropic / Karras grids from SigmaSchedule, and an exact-posterior toy (independent bits) computed here.

    python docs/pdf_reports/build_sampler_math.py   # -> docs/pdf_reports/sampler_math.pdf
"""
from __future__ import annotations

import glob
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdfkit_common import (  # noqa: E402
    H1, H2, HERE, P, PageBreak, Spacer, build, bullets, callout, mm, table,
)

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "scoretemp" / "eval_gh200"


def M(t):
    """An equation line."""
    return P(f"<font face='Mono' size='8.6'>{t}</font>", "callout")


def grids():
    import torch
    spec = importlib.util.spec_from_file_location("c", REPO / "configs/tasks/tinygsm_bits.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = m.get_config()
    cfg.device = "cpu"
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import SigmaSchedule
    ss = SigmaSchedule(ContinuousForwardProcess(cfg), cfg, torch.device("cpu"))
    run = str(REPO / "runs/tasks/tinygsm/cobit_raw_binary_bits")
    return {g: ss.prepare(schedule=g, num_steps=1024, entropy_run_dir=run).cpu().numpy().astype(float)
            for g in ("entropic", "karras")}


def toy_flips(grid, tau, kind, gamma=0.0, final_sharp=True, M_=60_000, seed=0):
    """Share of bits whose readout flips the decision held one step earlier; exact posterior, independent bits."""
    rng = np.random.default_rng(seed)

    def D(x, s):
        return 1.0 / (1.0 + np.exp(-np.clip((x - 0.5) / s ** 2, -60, 60)))
    s = grid
    x = 0.5 + s[0] * rng.standard_normal(M_)
    pre = None
    for i in range(len(s) - 1):
        sc, sn = s[i], s[i + 1]
        if kind == "churn" and gamma > 0:
            sh = sc * (1 + gamma)
            x = x + np.sqrt(sh ** 2 - sc ** 2) * rng.standard_normal(M_)
            sc = sh
        d = D(x, sc)
        last = i == len(s) - 2
        if last:
            pre = d >= 0.5
        t = 1.0 if (last and not final_sharp) else tau
        Dt = x + (d - x) / t
        r = sn / sc
        if kind == "ancestral":
            x = Dt + r * r * (x - Dt) + sn * np.sqrt(max(1 - r * r, 0.0)) * rng.standard_normal(M_)
        else:
            x = Dt + r * (x - Dt)
        if not np.isfinite(x).all() or np.abs(x).max() > 1e6:
            return "diverges"
    v = float(((D(x, s[-1]) >= 0.5) != pre).mean())
    return "0" if v == 0 else f"{100 * v:.2g}%"


def evidence():
    rows = [["sampler (500k EMA, entropic, 1024 steps)", "τ", "accuracy %", "programs run %", "invalid tokens %",
             "median program length", "body = 'return result' only"]]
    for g, lab in (("0.41", "churn γ=0.41"), ("0", "deterministic")):
        for t in ("1", "0.7", "0.5", "0.3"):
            f = glob.glob(str(OUT / f"gsm8k_results_*__s500_w0_t{t}_g{g}_n1024*.json"))
            if not f:
                continue
            r = json.load(open(f[0]))
            recs = r.get("sample_records") or []
            bodies = [x["response"].split("'''")[-1].strip() for x in recs]
            med = st.median(len(b) for b in bodies) if bodies else float("nan")
            short = 100 * sum(b in ("return result", "") for b in bodies) / max(1, len(bodies))
            rows.append([lab, t, f"{100 * r['accuracy']:.1f}", f"{100 * r['executable_rate']:.1f}",
                         f"{100 * r['invalid_token_rate']:.2f}", f"{med:.0f} chars", f"{short:.0f}%"])
    return rows


def main():
    G = grids()
    ent, kar = G["entropic"], G["karras"]
    r_ent = 1 - ent[1:] / ent[:-1]
    r_kar = 1 - kar[1:] / kar[:-1]
    gam = 0.41
    g_churn_kar = 1 - (kar[1:] / kar[:-1]) / (1 + gam)

    story = [P("How CoBit's sampler works, and why score temperature breaks it", "title"),
             P("The maths of the continuous sampler in one notation, the four samplers we have (DDIM, EDM churn, "
               "the Euler–Maruyama SDE of Appendix D, ancestral) as members of one family, what each kind of "
               "temperature does to it, and what collapses the 500k CoBit-S results below τ = 0.7 · 13 Oct 2026 · "
               "CPU-only (no new GPU runs)", "subtitle"),
             callout(
                 "<b>Two different failures, not one.</b> Below τ=0.7 the 500k model fails differently with and "
                 "without churn. <b>With churn</b> the readout is corrupted bit by bit: invalid token ids go 0% → 1.4% → "
                 "40% and the text becomes the right sentence with scrambled tokens. Cause: the entropic grid's last "
                 "step (σ 0.157 → 0.002) is so large that a sharpened drift reflects the state through the "
                 "prediction and amplifies the noise churn has just injected. <b>Deterministically</b> no token is ever "
                 "invalid, but programs shrink (median 158 → 74 → 13 characters; 60% are just <i>return result</i> at "
                 "τ=0.3): constant 1/τ on an ODE is strongly mode-seeking, and the dominant mode is ending early. "
                 "<b>“CFG collapses later” is the σ schedule</b>: that curve used Karras, which has no large final step."),
             callout(
                 "<b>One rule covers every sampler.</b> Each step moves the state towards the prediction D by a gain g; "
                 "score temperature multiplies the gain by 1/τ, and a step is stable only while g/τ &lt; 2 (and does not "
                 "overshoot only while g/τ ≤ 1). EDM churn has g ≈ 0.29 on <i>every</i> step, whatever the step count; "
                 "the ancestral step has g ≈ 2Δσ/σ ≈ 0.02. That is why ancestral sampling, or a smaller γ, can reach "
                 "τ ≈ 0.1. DeepMind's logit temperature never overshoots at all: it sharpens D inside [0, 1] instead of "
                 "extrapolating past it.", "keep")]

    story += [H1("1  The process and the denoiser"),
              P("CoBit diffuses bits x<sub>0</sub> ∈ {0,1}<super>S</super> (16 per token, 512 tokens) with a "
                "variance-exploding Gaussian process. The network predicts the per-bit posterior mean, a sigmoid of a "
                "learned logit plus an analytic matched-filter term, and Tweedie's formula turns it into the score:"),
              M("x<sub>σ</sub> = x<sub>0</sub> + σ ε,   ε ~ N(0, I)"),
              M("D(x, σ) = E[x<sub>0</sub> | x<sub>σ</sub> = x] = sigmoid( l<sub>θ</sub>(x, σ) + mf(x, σ) ) ∈ (0,1)<super>S</super>"),
              M("s(x, σ) = ∇ log p<sub>σ</sub>(x) = (D − x) / σ²"),
              P("All samplers below integrate from σ<sub>max</sub> ≈ 77–80 down to σ<sub>min</sub> = 0.002 on a grid "
                "σ<sub>0</sub> &gt; … &gt; σ<sub>N−1</sub> and read out by thresholding D at σ<sub>min</sub>. We write "
                "ρ = σ<sub>i+1</sub>/σ<sub>i</sub> for the step ratio and Δ = σ<sub>i</sub> − σ<sub>i+1</sub>."),
              H1("2  One family of samplers"),
              P("Georgios's entropy-gated SDE (paper, Appendix D) contains every sampler we use, indexed by the "
                "Langevin strength λ(σ) (r runs with decreasing σ):"),
              M("dx = (1 + λ) σ s(x, σ) dr + sqrt(2 λ σ) dW<sub>r</sub>"),
              P("λ = 0 is the probability-flow ODE; λ = 1 is the standard reverse-time SDE (DDPM's); λ(σ) ∝ the "
                "entropy rate is CoBit's stochastic sampler. The four discretisations in the code are:"),
              table([["sampler (code)", "one step", "noise injected"],
                     ["DDIM / Euler, γ=0 (DDIMSampler)", "x' = D + ρ (x − D)", "none"],
                     ["EDM churn (DDIMSampler + γ)", "σ̂ = (1+γ)σ,  x̂ = x + σ sqrt((1+γ)² − 1) ξ,  then  x' = D̂ + (σ'/σ̂)(x̂ − D̂)",
                      "before the step, ∝ σ, every step"],
                     ["Euler–Maruyama (EulerMaruyamaSampler)", "x' = x − (1+λ)(Δ/σ)(x − D) + sqrt(2λσΔ) z",
                      "after the step, ∝ sqrt(Δ)"],
                     ["ancestral, λ=1 (AncestralSampler, new)", "x' = D + ρ² (x − D) + σ' sqrt(1 − ρ²) z",
                      "after the step, exact Gaussian bridge"]],
                    col_widths=[48 * mm, 82 * mm, 40 * mm]),
              Spacer(1, 4),
              P("All four are the same update written around the prediction: the residual to D is multiplied by "
                "(1 − g) and fresh noise is added. Only the <b>gain g</b> and the noise differ:"),
              M("x' − D = (1 − g) (x − D) + noise"),
              table([["sampler", "gain g per step", "1024 Karras steps", "entropic last step (σ 0.157 → 0.002)"],
                     ["DDIM", "1 − ρ", f"median {np.median(r_kar):.4f}, max {r_kar.max():.3f}", f"{r_ent[-1]:.3f}"],
                     ["EDM churn γ", "1 − ρ/(1+γ)", f"median {np.median(g_churn_kar):.3f} (γ=0.41)",
                      f"{1 - (ent[-1] / ent[-2]) / (1 + gam):.3f}"],
                     ["Euler–Maruyama", "(1+λ)(1 − ρ)", "≈ churn at the same λ<sub>0</sub> = γ(N−1)", "capped (em_step_gamma_cap)"],
                     ["ancestral", "1 − ρ²", f"median {np.median(1 - (1 - r_kar) ** 2):.4f}", f"{1 - (ent[-1] / ent[-2]) ** 2:.3f}"]],
                    col_widths=[30 * mm, 30 * mm, 55 * mm, 55 * mm]),
              Spacer(1, 4),
              P("The key contrast: EDM churn inflates σ by (1+γ) on <i>every</i> step, so its gain stays near "
                "γ/(1+γ) ≈ 0.29 however fine the grid is, whereas the ancestral and DDIM gains shrink with the step. "
                "Churn is a large step taken N times; the ancestral sampler takes N small ones."),
              PageBreak()]

    story += [H1("3  What the three temperatures do to the step"),
              H2("Score temperature (Sigma; what we ran)"),
              M("D<sub>τ</sub> = x + (D − x)/τ    ⇒    x' − D<sub>τ</sub> = (1 − g)(x − D<sub>τ</sub>)  ⇔  x' − D = (1 − g/τ)(x − D)"),
              P("The gain becomes g/τ. The step is <b>stable only if g/τ &lt; 2</b> (otherwise the residual grows "
                "geometrically and the state diverges) and <b>does not overshoot only if g/τ ≤ 1</b> (otherwise x jumps "
                "past D and the residual — including the noise just injected — is reflected and amplified by "
                "|1 − g/τ|). This gives the observed limits: EDM churn on Karras needs τ &gt; 0.31/2 ≈ 0.15; any sampler "
                "on the entropic grid's last step (g ≈ 0.99) overshoots for τ &lt; 0.99."),
              H2("Our earlier A1 temperature"),
              P("κ(σ) = (v + σ²)/(τv + σ²) multiplies the gain by κ ∈ [1, 1/τ], near 1 at high σ: the same effect, "
                "confined to the end of the trajectory."),
              H2("Logit temperature (DeepMind, Simplex Diffusion Models, App. I.1 eq. 25)"),
              M("D<sub>T</sub> = E<sub>c ~ softmax(ζ/T)</sub>[c] ∈ [0,1]<super>16</super>  per token,   ζ(c) = token logit of code c"),
              P("The prediction is sharpened <i>inside</i> [0, 1] and then used in the unchanged step, so x' = D<sub>T</sub> "
                "+ ρ(x − D<sub>T</sub>) is a convex combination: the gain is unchanged and the step can never overshoot "
                "or diverge, at any T. This is why their T = 0.1 works with churn κ = 1, and why score temperature does "
                "not. CoBit already implements it (<font face='Mono'>--posterior_temp</font> with "
                "<font face='Mono'>--posterior_temp_space token</font>, target “full”); the rolling sampler now has "
                "it as <font face='Mono'>logit_temp</font>."),
              H2("Why a deterministic sampler collapses even when it is stable"),
              P("On a Gaussian toy the PF-ODE with drift ÷ τ ends with variance v<super>1/τ</super>"
                "(v + σ<sub>max</sub>²)<super>1−1/τ</super> instead of the tempered τv: the whole trajectory contracts "
                "onto the dominant mode. Noise injection (churn, λ &gt; 0) keeps re-spreading the state and counteracts "
                "this. In TinyGSM the dominant mode is the end of the program (most of the 512 positions are EOS/PAD)."),
              H1("4  What collapses the 500k model below τ = 0.7"),
              table(evidence(), col_widths=[37 * mm, 9 * mm, 18 * mm, 21 * mm, 20 * mm, 26 * mm, 30 * mm]),
              Spacer(1, 4),
              bullets([
                  "<b>Churn arm — readout corruption.</b> Invalid ids appear at τ=0.5 and dominate at τ=0.3, while "
                  "program length is unchanged: the decoder emits the right sentence with flipped bits. Mechanism: on "
                  "the entropic grid the last step goes from σ̂ = 1.41 × 0.157 = 0.22 (after churn) to 0.002 with "
                  "g ≈ 0.99, so at τ ≤ 0.5 it reflects the freshly churned residual through D with gain ≥ 1 and flips "
                  "bits that were already decided.",
                  "<b>Deterministic arm — mode collapse.</b> No invalid ids; programs shorten until most are "
                  "<i>return result</i>. This is the contraction of Section 3, not a step-size problem; no change to "
                  "the last step will fix it.",
                  "<b>“With CFG it collapses much later.”</b> The CFG checkpoint and CoBit-M were run on the Karras "
                  "grid, whose largest steps are tiny (DDIM max g = 0.024; churn g ≈ 0.31, hence their wall at τ≈0.15). "
                  "The 500k arm used the entropic grid. The difference is the schedule, not guidance.",
              ], "small")]

    taus = (0.5, 0.3, 0.2, 0.1, 0.05)
    trows = [["sampler (toy)", "grid", "final step"] + [f"τ={t}" for t in taus]]
    for kind, gma, lab in (("churn", 0.41, "churn γ=0.41"), ("churn", 0.2, "churn γ=0.2"),
                           ("churn", 0.1, "churn γ=0.1"), ("ancestral", 0.0, "ancestral")):
        for gname in ("entropic", "karras"):
            for fs, flab in ((True, "sharpened"), (False, "unsharpened")):
                if gname == "karras" and not fs:
                    continue
                trows.append([lab, gname, flab] + [toy_flips(G[gname], t, kind, gma, fs) for t in taus])
    story += [PageBreak(), H1("5  Fixes, and what a toy predicts for them"),
              P("Toy: independent bits with the exact posterior (no network, no context), 1024 steps, the real grids. "
                "Entry = share of bits whose readout flips the decision held one step earlier (0 is ideal), or "
                "“diverges”. It isolates the integrator; it says nothing about mode collapse or model error."),
              table(trows, col_widths=[24 * mm, 18 * mm, 22 * mm] + [21 * mm] * len(taus)),
              Spacer(1, 4),
              bullets([
                  "<b>Leaving the last step unsharpened</b> (new <font face='Mono'>--score_temp_final unsharpened</font>) "
                  "removes every final-step flip, for every sampler and γ. It does not move the divergence wall.",
                  "<b>Lower γ</b> moves the wall: on the Karras grid it follows τ &gt; (1 − 1/(1+γ))/2 (γ=0.41 → 0.15, "
                  "γ=0.2 → 0.08, γ=0.1 → 0.045); the entropic grid has a few larger steps, so its wall sits a little "
                  "higher (γ=0.1 still diverges at τ=0.05). Georgios's hunch is right on stability; the cost is less noise, i.e. more "
                  "of the deterministic arm's mode collapse, so the best γ is an empirical question.",
                  "<b>Ancestral</b> (new <font face='Mono'>--sampler_kind ancestral</font>) never diverges down to "
                  "τ=0.05 and, on Karras or with the unsharpened last step, never flips. Its noise (σ' sqrt(1−ρ²) per step) "
                  "is far below churn's, so whether it avoids mode collapse must be measured.",
                  "<b>Logit temperature</b> avoids both integrator problems by construction (Section 3) and is the "
                  "closest match to DeepMind's T = 0.1.",
              ], "small"),
              H1("6  Proposed runs (nothing submitted)"),
              P("All on the 500k EMA checkpoint, entropic grid, 1024 steps, 1319 problems, paired with the existing τ=1 "
                "cells; ~0.27 GH200-h per cell (measured), run in parallel (~20 min wall). Ordered so the cheap "
                "hypothesis test comes first; each later block is only worth running if the one before says so."),
              table([["block", "cells", "question", "GH200-h"],
                     ["A  unsharpened last step, churn γ=0.41 and γ=0, τ ∈ {0.5, 0.3, 0.2}, with per-step diagnostics",
                      "6", "Is the churn collapse the final step (predicted: invalid ids → 0, accuracy recovers)? Does "
                      "the deterministic collapse persist (predicted: yes)?", "1.6"],
                     ["B  ancestral, unsharpened last step, τ ∈ {1, 0.7, 0.5, 0.3, 0.2, 0.1}", "6",
                      "Can a low-noise exact sampler reach τ≈0.1 without mode collapse?", "1.6"],
                     ["C  γ ∈ {0.1, 0.2, 0.3} × τ ∈ {0.5, 0.3, 0.2, 0.1}, unsharpened last step", "12",
                      "Does lower γ play better with temperature (stability vs mode collapse)?", "3.2"],
                     ["D  logit temperature (token space, full logit), churn γ=0.41, T ∈ {0.5, 0.3, 0.1}", "3",
                      "DeepMind's operation on CoBit; cost of the codeword softmax to be checked on a 64-problem smoke first",
                      "≥ 0.8"]],
                    col_widths=[62 * mm, 12 * mm, 78 * mm, 18 * mm]),
              Spacer(1, 4),
              P("Total ≈ 7–8 GH200-h. The rolling sampler needs no new mechanics: its η = 1 step is exactly the "
                "ancestral bridge above (ratio<super>(1+η²)/2</super> = ρ² at η = 1), and its score temperature already "
                "reads out the finalising step unsharpened; its own sweeps are the ones in TINYGSM.md.", "small"),
              Spacer(1, 4),
              P("Code (uncommitted): <font face='Mono'>DDIMSampler(score_temp_final=…)</font>, "
                "<font face='Mono'>AncestralSampler</font> in diffusion/continuous/samplers.py; eval flags "
                "<font face='Mono'>--score_temp_final</font>, <font face='Mono'>--sampler_kind ancestral</font>; "
                "score temperature now also reaches the EM sampler; tests in tests/test_sampler_fixes.py (6 new, 247 "
                "existing pass). Built by docs/pdf_reports/build_sampler_math.py.", "small")]

    path = build(HERE / "sampler_math.pdf", story, "CoBit sampler maths · 13 Oct 2026", "sampler maths")
    print(path)


if __name__ == "__main__":
    main()
