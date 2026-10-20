"""Report 6: the entropy-rate profile, MSE vs sigma, and where the sampler should stop. CPU only.

The schedule controller (utils/schedule_controller.py) stores pdf(sigma) proportional to g(sigma) * rate(sigma)^p with
rate = mean training error / sigma^2 per log-sigma bin, g(sigma) = s^3/(s^3 + c^3), c = 0.1, p = 1/2 (sqrt-rate,
configs/tasks/tinygsm_bits.py). Inverting gives the training MSE per bin up to one constant:
MSE(sigma) proportional to sigma^2 * (pdf / g)^(1/p). The TensorBoard logs on Isambard (CoBit-M only) hold only
loss / LR / sigma_data, so the tables are the per-sigma record.

    python docs/pdf_reports/build_entropy_profile.py   # -> docs/pdf_reports/entropy_profile.pdf
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdfkit_common import (  # noqa: E402
    H1, HERE, INK2, MUTED, SERIES, P, Spacer, build, bullets, callout, figure, mm, mpl_style, plt, table,
)

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
RUNS = [("CoBit-S unguided (main run)", "runs/tasks/tinygsm/cobit_raw_binary_bits"),
        ("CoBit-S CFG", "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg"),
        ("CoBit-M 462M", "runs/tasks/tinygsm/cobit_m_bits_s42")]
C, N_REG, POWER = 0.1, 3.0, 0.5
MARKS = [(0.002, "σ_min = 0.002\n(our end point)"), (0.1, "σ = 0.1\n(OWT end point)"),
         (0.157, "")]


def tables(run):
    import torch
    ld = lambda n: torch.load(REPO / run / f"{n}.pt", weights_only=False).numpy().astype(float)
    return ld("entropy_pdf"), ld("entropy_sigmas"), ld("entropy_edges")


def implied_mse(pdf, mid):
    g = (mid / C) ** N_REG / (1 + (mid / C) ** N_REG)
    mse = mid ** 2 * (pdf / np.maximum(g, 1e-12)) ** (1 / POWER)
    ref = mse[np.argmin(abs(mid - 20))]
    out = mse / ref
    out[pdf == 0] = np.nan                                  # empty bin: not measured, not zero
    return out


def indep_mmse(sigmas, n=200_000, seed=0):
    """Exact MMSE of an equiprobable {0,1} bit from x = b + sigma*eps (no context), / prior variance 0.25."""
    rng = np.random.default_rng(seed)
    out = []
    for s in sigmas:
        b = rng.integers(0, 2, n)
        x = b + s * rng.standard_normal(n)
        D = 1 / (1 + np.exp(-np.clip((x - 0.5) / s ** 2, -60, 60)))
        out.append(np.mean((D - b) ** 2) / 0.25)
    return np.array(out)


def grid(smin=None, schedule="entropic"):
    import torch
    spec = importlib.util.spec_from_file_location("c", REPO / "configs/tasks/tinygsm_bits.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = m.get_config()
    cfg.device = "cpu"
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import SigmaSchedule
    ss = SigmaSchedule(ContinuousForwardProcess(cfg), cfg, torch.device("cpu"))
    return ss.prepare(schedule=schedule, num_steps=1024, entropy_run_dir=str(REPO / RUNS[0][1]),
                      sigma_min_override=smin).numpy().astype(float)


def _marks(ax, top):
    for s, lab in MARKS:
        ax.axvline(s, color=MUTED, lw=0.9, ls=(0, (3, 3)))
        if lab:
            ax.text(s * 1.12, top, lab, fontsize=6.5, color=INK2, va="top")


def fig_pdf():
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for (name, run), col in zip(RUNS, SERIES):
        pdf, mid, edges = tables(run)
        dens = pdf / np.diff(np.log(edges))                  # density per unit log-sigma
        ax.plot(mid, dens, color=col, label=name, lw=1.8)
    pdf0, mid0, _ = tables(RUNS[0][1])
    lo = mid0[pdf0 > 0].min()
    ax.axvspan(0.002, lo, color="#f4f3ef", zorder=0)
    ax.text(0.0045, ax.get_ylim()[1] * 0.55, f"empty bins: no training\nsamples below σ≈{lo:.2f}",
            fontsize=7, color=INK2)
    _marks(ax, ax.get_ylim()[1] * 0.98)
    ax.set_xscale("log")
    ax.set_xlim(0.0018, 90)
    ax.set_xlabel("σ ({0,1}-bit units)")
    ax.set_ylabel("training density per log σ")
    ax.set_title("Entropy-rate profile = the training σ distribution (TinyGSM)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.08), fontsize=7)
    return fig


def fig_mse():
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for (name, run), col in zip(RUNS, SERIES):
        pdf, mid, _ = tables(run)
        ax.plot(mid, implied_mse(pdf, mid), color=col, label=name, lw=1.8)
    s = np.logspace(np.log10(0.05), np.log10(80), 40)
    ax.plot(s, indep_mmse(s), color=INK2, lw=1.2, ls=(0, (4, 2)), label="no-context reference (exact, independent bits)")
    _marks(ax, 3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0018, 90)
    ax.set_ylim(1e-5, 4)
    ax.set_xlabel("σ ({0,1}-bit units)")
    ax.set_ylabel("MSE relative to σ = 20")
    ax.set_title("Training MSE vs σ, recovered from the entropy tables (gaps = not measured)")
    ax.legend(loc="lower right", fontsize=7)
    return fig


def fig_steps():
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    series = [("entropic, σ_min = 0.002 (current)", grid(None), SERIES[0]),
              ("entropic, σ_min = 0.1 (proposed)", grid(0.1), SERIES[1]),
              ("Karras, σ_min = 0.002", grid(None, "karras"), SERIES[2])]
    for lab, s, col in series:
        g = 1 - s[1:] / s[:-1]
        ax.plot(s[:-1], g, color=col, lw=1.6, label=lab)
        ax.plot(s[-2], g[-1], "o", color=col, ms=6, mec="white", mew=1.2)
        ax.annotate(f"last step {s[-2]:.3g}→{s[-1]:.3g}: g={g[-1]:.2f}", (s[-2], g[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=6.5, color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0015, 90)
    ax.invert_xaxis()
    ax.set_xlabel("σ at the start of the step  (sampling runs right to left →)")
    ax.set_ylabel("deterministic gain g = 1 − σ_next/σ")
    ax.set_title("Step gains along the 1024-step trajectory")
    ax.legend(loc="upper left", fontsize=7)
    return fig


def main():
    mpl_style()
    pdf0, mid0, _ = tables(RUNS[0][1])
    lo = mid0[pdf0 > 0].min()
    empty = int((pdf0 == 0).sum())
    mse0 = implied_mse(pdf0, mid0)
    pick = lambda s: mse0[np.argmin(abs(mid0 - s))]
    g_cur, g_01, g_005 = (1 - (lambda s: s[-1] / s[-2])(grid(x)) for x in (None, 0.1, 0.05))

    story = [P("Entropy-rate profile and where the sampler should stop", "title"),
             P("TinyGSM CoBit runs · the stored entropy tables, the training MSE they imply, and the step grid · "
               "13 Oct 2026 · CPU only", "subtitle"),
             callout(
                 f"<b>The model is essentially never trained below σ ≈ {lo:.2f}, yet we sample down to σ = 0.002.</b> "
                 f"{empty} of the 128 bins of the learned training distribution are empty, all at low σ. After the "
                 "40k-step warm-up, training draws σ from this profile, so the network sees almost no σ below "
                 f"≈{lo:.2f} for "
                 "the remaining ~460k steps (and the emptiness is self-sustaining: the controller only measures error "
                 "where it samples). The sampler's last step then jumps from σ = 0.157 to 0.002 in one go "
                 f"(gain {g_cur:.2f}) into that region. Ending at σ = 0.1, as on OWT, keeps the trajectory where the "
                 f"model was trained and cuts the last step's gain to {g_01:.2f}."),
             callout(
                 "<b>Our σ_min is 0.002, not 2e-4</b> (configs/tasks/tinygsm_bits.py, EDM's default, in {0,1}-bit "
                 "units; 0.004 in ±1 units). Below σ ≈ 0.13 the matched-filter logit (x − ½)/σ² exceeds its clip of "
                 "±30, so the prediction there is close to the analytic hard decision, which is why this has not "
                 "produced visible numerical blow-ups (and the drift (x − D)/σ stays O(1) in fp32). The risk is the "
                 "learned logit, which is evaluated there almost untrained.", "warn")]

    story += [H1("1  The entropy-rate profile"),
              figure(fig_pdf(), "entropy_profile_pdf", 165,
                     "Stored training σ distribution of each run (128 log-spaced bins on [0.002, 80]). Dashed lines: our "
                     "end point 0.002, the OWT end point 0.1, and 0.157, the last point the entropic grid places before "
                     "jumping to σ_min. All three runs agree; the peak is at σ ≈ 1.9–2.0."),
              H1("2  MSE vs σ"),
              figure(fig_mse(), "entropy_profile_mse", 165,
                     "Training MSE per bin implied by the tables, MSE ∝ σ² (pdf/g)², scaled to 1 at σ=20; dashed: the "
                     "exact MMSE of an isolated bit with no context, scaled to its prior variance. The gap between "
                     "them is what context buys. Empty bins are left blank: they were not measured, which is not the "
                     "same as zero error."),
              bullets([
                  f"Relative MSE is {pick(0.3):.1e} at σ = 0.3, {pick(1):.3f} at σ = 1, {pick(3):.2f} at σ = 3 and ≈1 above "
                  f"σ ≈ 10. Even an isolated bit, with no context at all, is decided from x alone by σ ≈ 0.1 (relative "
                  f"MMSE {indep_mmse([0.1])[0]:.0e}; dashed line), so nothing is gained by integrating below it.",
                  "The informative band — where the model's error falls fastest and the schedule spends its steps — is "
                  "σ ≈ 0.3–10. Nothing in the profile argues for steps below σ ≈ 0.1.",
              ], "small"),
              H1("3  The step grid, and the proposed end point"),
              figure(fig_steps(), "entropy_profile_steps", 165,
                     "Per-step deterministic gain along the 1024-step grids (EDM churn adds ≈ γ/(1+γ) to every step). "
                     "The current entropic grid has one isolated step of gain 0.99 at the very end; ending at σ = 0.1 "
                     "removes it."),
              table([["end point σ_min", "last step", "deterministic gain", "with churn γ=0.41",
                      "no-overshoot needs τ ≥", "stable needs τ >"],
                     ["0.002 (current)", "0.157 → 0.002", f"{g_cur:.2f}", f"{1 - (1 - g_cur) / 1.41:.2f}",
                      f"{1 - (1 - g_cur) / 1.41:.2f}", f"{(1 - (1 - g_cur) / 1.41) / 2:.2f}"],
                     ["0.05", "0.157 → 0.05", f"{g_005:.2f}", f"{1 - (1 - g_005) / 1.41:.2f}",
                      f"{1 - (1 - g_005) / 1.41:.2f}", f"{(1 - (1 - g_005) / 1.41) / 2:.2f}"],
                     ["0.1 (as on OWT)", "0.157 → 0.1", f"{g_01:.2f}", f"{1 - (1 - g_01) / 1.41:.2f}",
                      f"{1 - (1 - g_01) / 1.41:.2f}", f"{(1 - (1 - g_01) / 1.41) / 2:.2f}"]],
                    col_widths=[26 * mm, 24 * mm, 26 * mm, 28 * mm, 33 * mm, 30 * mm]),
              Spacer(1, 4),
              P("Ending at 0.1 does not make a sharpened last step safe on its own with churn (it still overshoots below "
                "τ ≈ 0.55), so it combines naturally with leaving the last step unsharpened; it does remove the "
                "untrained-σ exposure and the one-step jump. Readout at σ = 0.1 is effectively analytic: the "
                "matched-filter term is already at its clip there.", "small"),
              H1("4  Updated run plan (nothing submitted)"),
              table([["block", "what", "cells", "GH200-h"],
                     ["A", "unsharpened last step; churn γ=0.41 and γ=0; τ ∈ {0.5, 0.3, 0.2}; trajectory diagnostics", "6", "1.6"],
                     ["B", "ancestral sampler, unsharpened last step, τ ∈ {1, 0.7, 0.5, 0.3, 0.2, 0.1}", "6", "1.6"],
                     ["C", "γ ∈ {0.1, 0.2, 0.3} × τ ∈ {0.5, 0.3, 0.2, 0.1}, unsharpened last step", "12", "3.2"],
                     ["D", "DeepMind logit temperature on CoBit, T ∈ {0.5, 0.3, 0.1}", "3", "≥ 0.8"],
                     ["E (new)", "end at σ_min = 0.1, churn γ=0.41: τ ∈ {1, 0.7, 0.5, 0.3, 0.2} with the last step "
                      "sharpened, plus τ ∈ {0.3, 0.2} unsharpened", "7", "1.9"]],
                    col_widths=[16 * mm, 118 * mm, 14 * mm, 20 * mm]),
              Spacer(1, 4),
              P("All on the 500k EMA checkpoint, entropic grid, 1024 steps, 1319 problems; ≈ 9–10 GH200-h in total, "
                "≈ 20 min wall per block. If E confirms the end point, the cleanest follow-up is to move A–C to "
                "σ_min = 0.1 rather than add more cells. Data: the entropy tables of the three runs; the step grids "
                "from SigmaSchedule. Built by docs/pdf_reports/build_entropy_profile.py.", "small")]

    path = build(HERE / "entropy_profile.pdf", story, "Entropy-rate profile · 13 Oct 2026", "entropy profile")
    print(path)


if __name__ == "__main__":
    main()
