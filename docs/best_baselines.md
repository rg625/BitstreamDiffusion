# Best baselines — before any guidance

Selection criterion is stated, not implied, and **two notions of "best" are
reported because they disagree**. Every row carries its regime; regimes are
never pooled.

---

## 1. The table

All at 1319 problems unless noted. "fwd" = measured denoiser forward passes per
sample (solver order × guidance branches).

| # | Regime | Checkpoint | Sampler | Stochasticity | Steps | fwd | Accuracy | Wall (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | B | base 425k | **DDIM** | γ=0.41 | 1024 | 1024 | **0.2957** | 10192 *(fp32)* |
| 2 | A | CFG 500k | DDIM | γ=0.41 | 1024 | 1024 | **0.2957** | — *(bf16)* |
| 3 | A | CFG 500k | DDIM | γ=0.41 | 512 | 512 | 0.2911 | — |
| 4 | B | base 425k | DDIM | γ=0.41 | 1024 | 1024 | 0.2835 | 1963 *(bf16)* |
| 5 | B | base 425k | **FKC-EM** | churn_γ=0.41 | 1024 | 1024 | 0.2813 | — *(canonical)* |
| 6 | A | CFG 500k | DDIM | γ=0.41 | 256 | 256 | **0.2661** | — |
| 7 | A | CFG 500k | DDIM | γ=0.3 | 256 | 256 | 0.2568 | — |
| 8 | A | CFG 500k | DDIM | γ=0.2 | 256 | 256 | 0.2421 | — |
| 9 | A | CFG 500k | DDIM | γ=0 | 1024 | 1024 | 0.1390 | — |
| 10 | A | CFG 500k | DDIM | γ=0 | 512 | 512 | 0.1385 | — |
| 11 | B | base 425k | DDIM | γ=0 | 512 | 512 | 0.1259 | — |

### Two notions of best, and they disagree

* **Best accuracy:** row 1/2 — γ=0.41 at 1024 steps, **0.2957**.
* **Best accuracy per unit compute:** row 6 — γ=0.41 at **256** steps, 0.2661
  for a quarter of the forward passes. Going 256 → 1024 buys +0.0296 for 4×
  the compute.

Reported separately rather than collapsed, because which one is "best" depends
entirely on whether the budget is per-sample or total.

---

## 2. Two results that decide how the rest of the project runs

### The two churn implementations are equivalent

`regime_b_control`, matched on base 425k / 1024 steps / seed 42 / entropic /
1319 problems:

| Arm | Accuracy |
|---|---|
| FKC-EM, `churn_gamma=0.41`, fp32 *(canonical)* | 0.2813 |
| DDIM, `gamma=0.41`, fp32 *(control)* | 0.2957 |

**DDIM − FKC = +0.0144 [−0.0038, +0.0326]** — the interval spans zero.

This matters because the canonical sampler **cannot host the guidance study**:
`FeynmanKacEulerMaruyamaSampler.sample_particles()` takes no `bad_model` and no
`sg_scale`, and its `guidance_scale` is a geometric-average exponent, not linear
CFG. Running Regime B guidance on DDIM+churn is therefore an *evidenced*
substitution, anchored to the canonical FKC baseline — not the silent kind.

### fp32 costs 5.2× wall clock for nothing measurable

| Precision | Accuracy | Wall (s) |
|---|---|---|
| fp32 | 0.2957 | 10192 |
| bf16 | 0.2835 | **1963** |

**bf16 − fp32 = −0.0121 [−0.0311, +0.0061]**, not significant — and the audit
found the same null at γ=0 (+0.0023). Screening runs use bf16; a 5× multiplier
on every cell is not affordable for an unresolvable difference. Flagged
honestly: the point estimate is negative and larger than at γ=0, so if a later
headline result rests on a ~1-point margin it should be re-checked in fp32.

---

## 3. The constraint that sets the guidance operating point

**CFG is undefined on the canonical checkpoint.** The base run trained with
`cond.p_uncond = 0.0` — it has no unconditional branch. Only the CFG run
(`p_uncond = 0.1`) can be guided.

So Regime B guidance keeps the canonical **sampler** (DDIM, γ=0.41, 1024 steps)
and takes the CFG **checkpoint**. The swap was measured: **−0.0076
[−0.0243, +0.0091]**, not significant.

**Regime B guidance baseline: 0.2957** (row 2). Every Regime B guidance result
is reported as a delta against *this*, never against Regime A's 0.1385 and never
as raw accuracy.

---

## 4. What has not been established

* γ has only been swept finely in Regime A at 256 steps. γ=0.41 may not be the
  optimum at 1024 steps; the NFE × γ grid is not yet run.
* Row 6 vs row 1 is a 4× compute difference at one γ. The full NFE × γ surface
  is needed before "best per compute" is more than a two-point comparison.
* Every stochastic baseline here is a single seed. Seed variance at γ>0 is
  unmeasured, and stochastic sampling plausibly has more of it than
  deterministic. Headline claims will need ≥3 seeds.
