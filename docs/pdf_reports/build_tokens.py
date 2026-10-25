"""Report 2: replacing raw per-bit logits with token logits (V-way softmax). CPU only.

Accuracy tables come from results/tokens/*.json and results/cpu_reanalysis/*.json;
validation losses were exported once from the runs' TensorBoard logs into
val_loss_tokens_vs_bits.json (this directory).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdfkit_common import (  # noqa: E402
    H1, H2, HERE, INK2, MUTED, SERIES, KeepTogether, P, Spacer, build, bullets, callout, figure, mm,
    mpl_style, plt, table,
)

REPO = HERE.parents[1]


def mean_acc(pattern):
    accs = [json.loads(Path(p).read_text())["accuracy"] for p in sorted(glob.glob(str(REPO / pattern)))]
    return accs


def b128_delta():
    """Pooled paired bootstrap, token - bits, CSD3 batch-128 pair at 500k (per-problem records)."""
    def per_prob(pattern):
        ds = [json.loads(Path(p).read_text())["per_problem"] for p in sorted(glob.glob(str(REPO / pattern)))]
        idx = sorted(set.intersection(*[set(d["idx"]) for d in ds]))
        m = [dict(zip(d["idx"], d["correct"])) for d in ds]
        return idx, {i: np.mean([mm_[i] for mm_ in m]) for i in idx}
    it, t = per_prob("results/tokens/eval/*token_500k_s*.json")
    ib, b = per_prob("results/tokens/eval/*bits_500k_s*.json")
    idx = sorted(set(it) & set(ib))
    diff = np.array([t[i] - b[i] for i in idx])
    rng = np.random.default_rng(0)
    boots = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"<b>{100 * diff.mean():+.2f} pp</b> [{100 * lo:+.2f}, {100 * hi:+.2f}]"


def fig_curve(tab):
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for run, lab, c in (("ord_gh500k_none_s42", "bits head", SERIES[0]),
                        ("tok_gh500k_token_ce_s42", "token head (V-way softmax)", SERIES[1])):
        rows = sorted((r for r in tab if r["run"] == run), key=lambda r: r["step"])
        x = [r["step"] / 1000 for r in rows]
        y = [100 * r["mean"] for r in rows]
        lo = [100 * min(r["acc"]) for r in rows]
        hi = [100 * max(r["acc"]) for r in rows]
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(x, y, color=c, marker="o", ms=4)
        ax.text(x[-1] + 8, y[-1], f"{lab}  {y[-1]:.1f}%", color=INK2, fontsize=7, va="center")
    ax.axhline(13.14, color=MUTED, ls="--", lw=1.2)
    ax.text(105, 13.5, "production bits model, 500k (different recipe: lr 3e-4)", color=MUTED, fontsize=7)
    ax.set_xlim(80, 480)
    ax.set_ylim(0, 15)
    ax.set_xticks([100, 200, 300, 400, 500])
    ax.set_xlabel("training steps (thousands)")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("Matched runs on Isambard: the token head learns slower at every evaluated step")
    return fig


def fig_loss():
    v = json.loads((HERE / "val_loss_tokens_vs_bits.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for ax, run, lab, c in ((axs[0], "tok_gh500k_token_ce_s42", "token head: cross-entropy (nats/token)", SERIES[1]),
                            (axs[1], "ord_gh500k_none_s42", "bits head: score loss (per bit)", SERIES[0])):
        x = [s / 1000 for s, _ in v[run]]
        y = [val for _, val in v[run]]
        ax.plot(x, y, color=c, marker="o", ms=3)
        ax.axvspan(40, 50, color="#ecebe7", lw=0)
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel("steps (thousands)")
    axs[0].set_ylabel("validation loss")
    axs[0].text(120, 1.15, "grey band: noise schedule\nswitches (40–50k)", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    return fig


def main():
    mpl_style()
    s = json.loads((REPO / "results/tokens/summary_gh200.json").read_text())
    tab, paired = s["table"], {p["step"]: p for p in s["paired"]}
    t3 = json.loads((REPO / "results/cpu_reanalysis/t3_taxonomy.json").read_text())["summary"]
    t4 = {(r["run"], r["step"]): r for r in json.loads((REPO / "results/cpu_reanalysis/t4_voting.json").read_text())["rows"]}
    t5 = json.loads((REPO / "results/cpu_reanalysis/t5_selection.json").read_text())
    b128_tok = mean_acc("results/tokens/eval/*token_500k_s*.json")
    b128_bit = mean_acc("results/tokens/eval/*bits_500k_s*.json")
    get = lambda run, step: next(r for r in tab if r["run"] == run and r["step"] == step)  # noqa: E731

    st = []
    st.append(P("Replacing raw bits with token logits", "title"))
    st.append(P("TinyGSM → GSM8K. How the V-way token-softmax head was built, how the Isambard run "
                "behaved, and whether it beat the per-bit head. Written 2 Oct 2026 from results on "
                "disk; no GPU was used.", "subtitle"))
    tk1, bt1 = get("tok_gh500k_token_ce_s42", 100000)["mean"], get("ord_gh500k_none_s42", 100000)["mean"]
    tk2, bt2 = get("tok_gh500k_token_ce_s42", 200000)["mean"], get("ord_gh500k_none_s42", 200000)["mean"]
    st.append(callout(
        f"<b>Bottom line: worse, at every point we measured.</b> With everything else matched, the token "
        f"head reached about <b>0.55× the bits accuracy</b>: {100 * tk1:.1f}% vs {100 * bt1:.1f}% at 100k "
        f"steps and {100 * tk2:.1f}% vs {100 * bt2:.1f}% at 200k (significant on every seed). An independent "
        f"pair on CSD3 at 500k steps agrees ({100 * np.mean(b128_tok):.1f}% vs {100 * np.mean(b128_bit):.1f}%). "
        "Training was stable, but the run was stopped at 414k steps for budget and its later checkpoints "
        "were never evaluated. The failure is execution: <b>93–97% of token programs do not run</b>, "
        "although the ones that do run are right more often than bits programs on the same problems.", "warn"))

    st.append(H1("1. What we changed"))
    st.append(P("CoBit normally writes each token as a 16-bit codeword and predicts each bit on its "
                "own. The idea was to predict the whole token at once: one probability over the "
                "49,153-word vocabulary per position, so the model can express which <i>tokens</i> are "
                "likely instead of 16 independent coin flips (which can only describe a product "
                "distribution over codewords)."))
    st.append(table([
        ["", "Bits head (production)", "Token head (this study)"],
        ["State per position", "16 bits in {0,1} (8,192 numbers per sequence)", "one-hot vector over V = 49,153 tokens (512 positions)"],
        ["Output", "1 logit per bit → sigmoid", "V logits per token → softmax"],
        ["Matched filter", "added per bit: (x<sub>t</sub> − ½)/σ²", "same term per vocabulary entry (the exact Gaussian likelihood for a one-hot code)"],
        ["Loss", "score loss on bits (binary_sm)", "cross-entropy on the token (token_ce), same EDM weight"],
        ["Decode", "threshold bits → codeword → token", "argmax over the vocabulary"],
        ["Parameters", "133.8M", "212.3M (the V-way input and output layers)"],
    ], col_widths=[30 * mm, 64 * mm, 76 * mm]))
    st.append(Spacer(1, 4))
    st.append(P("<b>Held the same:</b> the 12-layer, 768-wide transformer, the Gaussian noise "
                "process and σ range, the learned noise schedule, self-conditioning, CFG dropout "
                "(p=0.1), EMA, AdamW at a constant lr 1e-4 with 2.5k warmup, effective batch 512, "
                "seed 42, 500k-step target, the same data and 512-token canvas. The bits control "
                "is <font face='Mono'>ord_gh500k_none_s42</font>, trained in the same week on the same cluster."))
    st.append(P("<b>Not held the same</b> (worth knowing when reading the result): the loss family "
                "changes with the head (cross-entropy vs score loss), the token logits come out of a "
                "narrow 128-wide head layer (a wider <font face='Mono'>token_full</font> head exists but was not "
                "used), and the token run used 8 GPUs (2 nodes) vs 4 for bits, because the V-way "
                "logits need much more memory."))

    st.append(H1("2. How the Isambard run behaved"))
    st.append(table([
        ["", "Token run (tok_gh500k_token_ce_s42)", "Bits control (ord_gh500k_none_s42)"],
        ["Hardware", "8 × GH200 on 2 nodes (first 5k steps on 1 node)", "4 × GH200 on 1 node"],
        ["Speed", "2.66 optimiser steps/s", "≈ 3.95 steps/s"],
        ["Steps reached", "<b>414,000 of 500,000</b>, stopped 26 Sep at the budget line", "500,000 (finished 25 Sep)"],
        ["Stability", "no NaN/Inf, divergence guard never fired", "no divergence"],
        ["Interruptions", "3 NCCL hangs at epoch boundaries (5-epoch entropy-plot callback, audit T2); "
                          "each resumed from the epoch checkpoint, no steps lost", "none"],
        ["σ<sub>data</sub> (rank-0 estimate)", "0.4575", "0.3998"],
        ["Approx. GPU-h (from log times)", "≈ 355", "≈ 141"],
    ], col_widths=[34 * mm, 72 * mm, 64 * mm]))
    st.append(Spacer(1, 4))
    st.append(figure(fig_loss(), "tok_val_loss", 160,
                     "Validation loss per epoch, from the runs' TensorBoard logs. The two losses are "
                     "different quantities and are not comparable across panels. Both jump when the noise "
                     "schedule switches to the learned distribution (40–50k), then improve slowly and "
                     "steadily: no instability."))

    st.append(H1("3. Results"))
    st.append(P("Every cell: full GSM8K test set (1319 problems), 3 sampling seeds, Karras/DDIM 256 "
                "steps, deterministic, EMA weights, one GH200. Δ is token − bits, paired over problems."))
    rows = [["Step", "Token head: seeds → mean", "Bits head: seeds → mean", "Δ token − bits, pooled [95% CI]",
             "Ratio"]]
    for step in (100000, 200000):
        tk, bt = get("tok_gh500k_token_ce_s42", step), get("ord_gh500k_none_s42", step)
        pp = paired[step]["pooled"]
        rows.append([f"{step // 1000}k",
                     " / ".join(f"{100 * a:.1f}" for a in tk["acc"]) + f" → <b>{100 * tk['mean']:.2f}%</b>",
                     " / ".join(f"{100 * a:.1f}" for a in bt["acc"]) + f" → <b>{100 * bt['mean']:.2f}%</b>",
                     f"<b>{100 * pp['delta']:+.2f} pp</b> [{100 * pp['ci95'][0]:+.2f}, {100 * pp['ci95'][1]:+.2f}]",
                     f"{tk['mean'] / bt['mean']:.2f}×"])
    for step in (300000, 400000):
        bt = get("ord_gh500k_none_s42", step)
        rows.append([f"{step // 1000}k", "not evaluated (jobs cancelled at the stop)",
                     " / ".join(f"{100 * a:.1f}" for a in bt["acc"]) + f" → {100 * bt['mean']:.2f}%", "—", "—"])
    rows.append(["500k (CSD3, batch 128)", " / ".join(f"{100 * a:.1f}" for a in b128_tok) +
                 f" → <b>{100 * np.mean(b128_tok):.2f}%</b>",
                 " / ".join(f"{100 * a:.1f}" for a in b128_bit) + f" → <b>{100 * np.mean(b128_bit):.2f}%</b>",
                 b128_delta(), f"{np.mean(b128_tok) / np.mean(b128_bit):.2f}×"])
    st.append(table(rows, col_widths=[24 * mm, 42 * mm, 42 * mm, 40 * mm, 18 * mm]))
    st.append(Spacer(1, 4))
    st.append(figure(fig_curve(tab), "tok_curve", 160,
                     "Mean of 3 sampling seeds; the band spans the seeds. One training seed per arm. "
                     "Source: results/tokens/summary_gh200.json."))

    why_heading = H2("Why it is worse: the programs do not run")
    ex_t = t4[("tok_gh500k_token_ce_s42", 200000)]["exec_rate"]
    ex_b = t4[("ord_gh500k_none_s42", 200000)]["exec_rate"]
    both = t5["both_arms_execute"]
    st.append(KeepTogether([why_heading, table([
        ["At 200k steps", "Token head", "Bits head"],
        ["Programs that execute at all", f"{100 * ex_t:.1f}%", f"{100 * ex_b:.1f}%"],
        ["Correct, among executing programs (all problems)",
         f"{100 * t5['all_problems']['token']['acc_given_exec_pooled'][0]:.0f}%",
         f"{100 * t5['all_problems']['bits']['acc_given_exec_pooled'][0]:.0f}%"],
        [f"Same, on the {both['n_problems']} problems where both arms execute",
         f"{100 * both['token']['acc_given_exec_problem_weighted'][0]:.0f}%",
         f"{100 * both['bits']['acc_given_exec_problem_weighted'][0]:.0f}%"],
        ["Typical failure (programs short enough to inspect)",
         "NameError: a variable is defined under one name and used under a corrupted one "
         "(<font face='Mono'>meters_per_sprint</font> → <font face='Mono'>meters_per_printprint</font>)",
         "stops early, before <font face='Mono'>return</font>"],
    ], col_widths=[66 * mm, 54 * mm, 50 * mm])]))
    st.append(Spacer(1, 4))
    d = both["token_minus_bits_problem_weighted"]
    st.append(P(f"So the token head's deficit is in <b>writing code that runs</b>, not in the arithmetic. "
                f"The raw \"57% vs 31% correct when it runs\" is partly a selection effect (token programs "
                f"only run on short, easy problems), but on matched problems the token head is still "
                f"{100 * d['delta']:+.0f} pp [{100 * d['ci95'][0]:+.0f}, {100 * d['ci95'][1]:+.0f}] more "
                f"often right. Its failures look like losing track of identifiers across the program, a "
                f"long-range consistency problem that per-bit prediction does not show as strongly."))
    v_t = t4[("tok_gh500k_token_ce_s42", 200000)]
    v_b = t4[("ord_gh500k_none_s42", 200000)]
    st.append(P(f"Voting over the 3 samples among programs that run helps both arms (token "
                f"{100 * v_t['pass@1']:.1f}% → {100 * v_t['maj_exec']:.1f}%, bits {100 * v_b['pass@1']:.1f}% → "
                f"{100 * v_b['maj_exec']:.1f}%) and does not change the ranking."))

    st.append(H1("4. Verdict and how much to trust it"))
    st.append(callout(
        "<b>Verdict: worse, not inconclusive</b>, for this head, this loss and this budget. Two "
        "independent matched pairs (Isambard batch 512 at 100k and 200k; CSD3 batch 128 at 500k) agree "
        "on a large, significant gap of the same size (~0.5–0.55×). What is <b>not</b> settled is "
        "whether a token head <i>could</i> win with a different loss, a wider head or a longer run.", "key"))
    st.append(bullets([
        "<b>One training seed per arm.</b> The 3 seeds are sampling seeds; the intervals cover problem "
        "and sampling noise, not training-run variance.",
        "<b>Only 100k and 200k are compared on Isambard.</b> Token 300k/400k and bits 500k were never "
        "evaluated, and the token run stopped at 414k.",
        "<b>Head and loss change together</b> (token cross-entropy vs bit score loss), and the token "
        "logits pass through a 128-wide head layer.",
        "<b>Training bugs found afterwards</b> (fixed 1 Oct, not re-run): GPUs other than rank 0 trained with "
        "σ<sub>data</sub> 0.5 instead of the estimate (T1); every GPU drew the same noise (T3), which "
        "costs the 8-GPU token run more than the 4-GPU bits run (≈64 vs ≈128 distinct noise draws per "
        "step). Neither is likely to explain a 2× accuracy gap, but neither is ruled out.",
        "<b>The learning rate (1e-4) is untuned for both heads.</b> The production bits model used 3e-4, "
        "which diverged in every from-scratch attempt here, so it is not a matched reference.",
    ]))

    st.append(H1("5. What to keep and what to do next"))
    st.append(callout(
        "<b>Keep the bits head</b> as the default. <b>Keep the token-head code path</b> (it works, is "
        "stable, and is the right tool for later tests) and the finding that its weakness is code "
        "validity, not arithmetic. <b>Do not</b> spend more training budget on this exact recipe.", "keep"))
    st.append(P("If the question is reopened (GPU estimates; nothing has been run):"))
    st.append(bullets([
        "Evaluate the existing token 300k/400k and bits 500k checkpoints (~4 GPU-h): does the ratio hold "
        "or the gap close?",
        "Separate head from loss: a token head with the score loss, or a bits head with cross-entropy; "
        "and try the wide <font face='Mono'>token_full</font> head.",
        "Confirm the T1/T3 fixes first (12k-step test, ~20 GPU-h for 3 seeds), then a second training "
        "seed for any re-run.",
    ]))
    st.append(Spacer(1, 6))
    st.append(P("<b>Sources.</b> configs/tasks/tinygsm_tokens.py; scripts/hpc/arch/logits_gh200.sh; "
                "diffusion/continuous/losses.py, logit_postprocess.py; runs/tasks/tinygsm/{tok_gh500k_token_ce_s42, "
                "ord_gh500k_none_s42}/config.json, sigma_data.json, training_logs; logs/arch/cobit_gh_tok_*.log; "
                "results/tokens/summary_gh200.json, results/tokens/eval/*.json; results/cpu_reanalysis/"
                "t3_taxonomy.json, t4_voting.json, t5_selection.json; docs/experiments_ordering_objective.md "
                "(token sections); docs/cpu_reanalysis.md; CODEBASE.md (T1–T4, R4).", "small"))

    out = HERE / "report_token_logits.pdf"
    build(out, st, "Replacing raw bits with token logits — CoBit on TinyGSM/GSM8K", "Token-head vs bits-head study")
    print(out)


if __name__ == "__main__":
    main()
