# Findings

Running record. Each entry: what was found, the evidence, how much to trust it,
what else could explain it, what would settle it, and what it changes.

Confidence vocabulary:
**established** = full test set, ≥3 seeds, paired intervals ·
**screening** = 250 problems, 1 seed, ±0.04 ·
**mechanistic** = diagnostic evidence for *why*, not just *that*.

---

## F1 — The study reproduces bit-for-bit after the eval-loop change

**Finding.** Re-running four headline cells with current code gives per-problem
outcome vectors *identical* to the originals, element by element.

| cell | PDF | repro | Δ | per-problem |
|---|---|---|---|---|
| baseline s42 | 0.1334 | 0.1334 | +0.0000 | identical |
| CFG w=12 s42 | 0.2108 | 0.2108 | +0.0000 | identical |
| SG-prev w=2 s42 | 0.1744 | 0.1744 | +0.0000 | identical |
| AG w=15 s42 | 0.1986 | 0.1986 | +0.0000 | identical |

**Evidence.** `runs/guidance/repro/`, grid `repro`, 512 steps, 1319 problems.
**Confidence.** Established.
**Why it was needed.** `evaluate_samples(text, gold)` was replaced by
`predict_answer` + `_numbers_equal` to record executed answers for maj@k. That
is the same composition, but a silent grading change would have invalidated
every comparison in the study.
**Implication.** The PDF's numbers stand and remain comparable to new runs.

---

## F2 — SG-prev is **not** a second-order integration effect

**Finding.** At matched forward passes a genuine second-order solver buys
nothing, while SG-prev buys +0.048.

| arm | forwards | exact match | vs DDIM-512 |
|---|---|---|---|
| DDIM-512 | 512 | 0.1385 | — |
| Heun-256 | 511 | 0.1375 | −0.0010 [−0.0030,+0.0010] n.s. |
| **SG-prev-512** | 512 | **0.1868** | **+0.0483 [+0.0392,+0.0576]** |
| DDIM-1024 | 1024 | 0.1390 | +0.0005 n.s. |
| Heun-512 | 1023 | 0.1390 | +0.0000 n.s. |

SG-prev beats Heun by +0.0493 [+0.0402,+0.0586] at identical compute.
**Evidence.** `runs/guidance/solver_control/`, 1319 problems, 3 seeds.
**Confidence.** Established.
**Alternatives considered.** "Finer grid" is separately excluded — DDIM-1024
equals DDIM-512. Heun's forward count was *measured* (2N−1), not assumed, so
the comparison is genuinely compute-matched.
**Remaining test.** SG-prev **+** Heun, to see whether the effects are
independent. Blocked: AG/SG are DDIM-only by an explicit guard.
**Implication.** The PDF's leading hypothesis for SG-prev is rejected. H3 out,
H4 supported. SG-prev's mechanism is now genuinely open.

---

## F3 — NFE was mis-recorded for every Heun run *(implementation artifact)*

**Finding.** `nfe_per_sample` was `steps × guidance_branches`, blind to solver
order, so Heun under-reported ~2×.
**Evidence.** Measured by wrapping `model.forward` on a real checkpoint: DDIM
exactly 1.00/step; Heun 7/15/31 at 4/8/16 steps = 2N−1.
**Confidence.** Established; fixed and pinned in `tests/test_nfe_accounting.py`.
**Implication.** F2's grid was specified in forward passes to begin with, so its
conclusion is unaffected — but any quality-vs-compute plot built from the old
column would have been wrong. Recorded as a reporting artifact, not a silent fix.

---

## F4 — Stochastic sampling alone beats every guidance method, for free

**Finding.** Churn takes the baseline from 0.164 to 0.276 at **zero** additional
cost — larger than any guidance effect measured in this study.

| γ | 0.0 | 0.1 | 0.2 | 0.3 | 0.41 |
|---|---|---|---|---|---|
| baseline | 0.1640 | 0.2440 | 0.2560 | **0.2760** | 0.2720 |
| CFG w=12 | 0.2280 | 0.2920 | 0.2800 | 0.2680 | 0.2840 |
| AG w=15 | 0.2200 | 0.2800 | 0.2960 | 0.2600 | 0.2640 |

Cost is measured, not inferred: every γ runs at 256 NFE, ~94 s, 2.31 GB.
**The baseline at γ=0.3 (0.2760, 256 NFE, 94 s) beats CFG w=12 at γ=0 (0.2280,
512 NFE, 177 s) on half the compute.**
**Evidence.** `runs/guidance/stoch_screen/`, 250 problems, 1 seed.
**Confidence.** **Established.** `stoch_confirm`, 1319 problems, 3 seeds:
baseline γ=0.2 **+0.1054 [+0.0915,+0.1203]**, γ=0.3 **+0.1200 [+0.1056,+0.1350]**.
**Implication.** The whole study optimised guidance inside γ=0, which is the
*worst* operating point available. Every headline recommendation in the PDF is
conditional on a sampler setting that should not have been fixed.

---

## F5 — Guidance's margin shrinks as stochasticity rises

**Finding.** Gain over the *same-γ* baseline:

| method | γ=0 | γ=0.1 | γ=0.2 | γ=0.3 | γ=0.41 |
|---|---|---|---|---|---|
| CFG w=12 | +0.0640 | +0.0480 | +0.0240 | −0.0080 | +0.0120 |
| AG w=15 | +0.0560 | +0.0360 | +0.0400 | −0.0160 | −0.0080 |

**Confidence.** **Established** at full size — guidance's advantage does not
merely shrink, it disappears:

| vs same-γ baseline | γ=0 | γ=0.2 | γ=0.3 |
|---|---|---|---|
| CFG w=12 | +0.0801 [+0.0665,+0.0938] wins | +0.0088 n.s. | −0.0045 n.s. |
| AG w=15 | +0.0690 [+0.0561,+0.0819] wins | +0.0018 n.s. | −0.0169 **loses** |

**Alternatives.** w=12 was tuned at γ=0, so "guidance stops helping" and
"guidance is mis-tuned for this regime" remain confounded.
**Remaining test.** `churn_anatomy` arm 1 — CFG scale swept *at* γ=0.3.
**Implication.** Consistent with guidance partly substituting for stochasticity
rather than adding to it (H13/H14). Does not yet establish it.

---

## F6 — SG-prev inverts under churn, and the reason is noise amplification

**Finding.** SG-prev goes from +0.028 at γ=0 to **−0.244** at γ=0.41,
monotonically. Diversity collapses with it (distinct-2 0.700 → 0.553).

**Mechanism.** SG-prev's direction is
`(δ_ref / realised log-σ spacing) × (D_cur − D_prev)`.

* The **denominator is not the cause.** `sg_delta_used` is essentially
  unchanged across γ (min 0.01045, median 0.0150 at every γ), and log-σ stays
  monotone under churn — 0 % increasing steps at every γ. *This refutes my
  initial hypothesis that churn broke the spacing normaliser.*
* The **numerator is.** `sg_dir_rms` grows 0.188 → 1.337 → 2.029 → 2.553 →
  3.113, a ~16× increase driven purely by γ.

Under deterministic sampling `D_cur − D_prev` is the trajectory derivative —
signal. Under churn it is signal plus injected noise, and the noise dominates;
the ~33× amplification then turns a noise estimate into a large random kick.
SG fired on all 255 steps at every γ (`sg_applied=255, sg_skipped=0`), so this
is not the known "SG skipped under churn" failure mode.

**Evidence.** Per-step traces in `runs/guidance/stoch_screen/`.
**Confidence.** **Established** for the effect (−0.1736 [−0.1898,−0.1577] at
γ=0.2; −0.2120 [−0.2305,−0.1946] at γ=0.3); mechanistic for the cause.
**Remaining test.** `churn_anatomy` arm 2 — if only the *magnitude* is wrong,
`sg_scale ≈ 2/16 = 0.125` should restore it. A flat null across 0.06/0.125/0.25
would falsify the magnitude explanation and implicate the direction itself.
**Implication.** SG-prev is a *deterministic-sampler* method as implemented.
The PDF's "SG-prev for fixed compute" recommendation is void once churn is on.

---

---

## F7 — Plain stochastic sampling beats **every** guided configuration, at half the compute

**Finding.** The best cell in the entire 36-cell confirmation is the *unguided*
baseline at γ=0.3.

| configuration | exact match | NFE |
|---|---|---|
| **baseline, γ=0.3** | **0.2568** | **256** |
| CFG w=12, γ=0.3 | 0.2522 | 512 |
| CFG w=12, γ=0.2 | 0.2509 | 512 |
| baseline, γ=0.2 | 0.2421 | 256 |
| CFG w=12, γ=0 | 0.2168 | 512 |
| AG w=15, γ=0 | 0.2057 | 512 |
| SG-prev w=2, γ=0 | 0.1865 | 256 |
| baseline, γ=0 | 0.1367 | 256 |

Against the guided γ=0 configurations the PDF recommends:

| | Δ | 95% CI |
|---|---|---|
| baseline γ=0.3 − CFG w=12 γ=0 | **+0.0399** | [+0.0263,+0.0538] |
| baseline γ=0.3 − AG w=15 γ=0 | **+0.0510** | [+0.0374,+0.0647] |
| baseline γ=0.3 − SG-prev w=2 γ=0 | **+0.0703** | [+0.0564,+0.0844] |

**Confidence.** Established, 1319 problems, 3 seeds, paired.
**Evidence.** `runs/guidance/stoch_confirm/`; cost measured, not inferred —
every γ runs at 256 NFE / ~94 s / 2.31 GB, so the baseline wins on accuracy
*and* uses half CFG's forward passes.
**Alternatives.** Confined to 256 steps and this checkpoint/task. CFG might pay
at a different scale under churn (`churn_anatomy`).
**Implication.** On this model and task the PDF's entire recommendation table is
superseded. Guidance is not merely smaller than reported — it is dominated by a
sampler setting that costs nothing and was held fixed throughout the study.

---

## Standing implication for the programme

The PDF's headline claims are all true *at γ=0*, and γ=0 is dominated. Guidance
anatomy (Stages 5–7) must be re-centred on the stochastic operating point or it
will describe a regime nobody should deploy.

The scientific question also changes shape. It is no longer "which guidance
method is best" but **"why does injected noise do, for free, what guidance was
being paid 2× compute to do — and are they the same mechanism?"** Both raise
accuracy while reshaping the bit-entropy/saturation profile, which is the
concrete thing to compare next.
