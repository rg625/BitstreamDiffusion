# Tiny GSM8K error analysis — Regime B canonical screen

**Zero additional GPU.** Everything below comes from the 11 stored result JSONs
of `rb_guidance_screen` (DDIM γ=0.41, 1024 steps, CFG-500k, 250 problems,
seed 42) plus existing Regime A cells.

## 0. What is stored, and what is not

| Field | Coverage | Usable for |
|---|---|---|
| `per_problem.{idx,correct,answer}` | **all 250** | transitions, error-set algebra, answer collapse |
| `sample_records.{idx,prompt,response,correct}` | **100 of 250** | text-level taxonomy |
| `guidance_diagnostics.per_step` | 1023 steps | trajectory — **batch-aggregated, not per problem** |
| `diversity` | aggregate | length, distinct-n |

**Missing:** per-problem trajectories, guidance cosines, per-sample entropy. So
trajectory claims below are population-level; they cannot be joined to
individual problems.

---

## 1. Error taxonomy (grounded in observed outputs, not invented)

Only three categories were needed to account for what actually appears:

| # | Category | Signature | Notes |
|---|---|---|---|
| **T1** | **Non-executing program** | `predict_answer` → `None` | dominant failure everywhere, incl. baseline (98/250) |
| **T2** | **Executes, wrong number** | answer parses, ≠ gold | the "reasoning/arithmetic error" class |
| **T3** | **Token-level degeneration** | no `def `, repeated rare tokens, never terminates | appears **only** under strong guidance |

Categories such as "premature termination" and "malformed answer" collapse into
T1 at this granularity — the executed-answer grader cannot separate them without
per-problem trajectory data. Not fabricating that distinction.

---

## 2. Paired transitions (all 250 problems)

| Method | acc | fixed (of 171 errors) | broke (of 79 correct) | net |
|---|---|---|---|---|
| baseline | 0.3160 | — | — | — |
| CFG w=1 | 0.3120 | 1 (0.6 %) | 2 (2.5 %) | −1 |
| **CFG w=2** | 0.3200 | 14 (8.2 %) | 13 (16.5 %) | **+1** |
| CFG w=4 | 0.3040 | 15 (8.8 %) | 18 (22.8 %) | −3 |
| CFG w=8 | 0.2760 | 16 (9.4 %) | 26 (32.9 %) | −10 |
| CFG w=0.5 | 0.0840 | 3 (1.8 %) | 61 (77.2 %) | −58 |
| AG w=4 | 0.2760 | 12 (7.0 %) | 22 (27.8 %) | −10 |
| AG w=15 | 0.2240 | 12 (7.0 %) | 35 (44.3 %) | −23 |
| SG-exact w=1 | 0.1680 | 5 (2.9 %) | 42 (53.2 %) | −37 |
| SG-prev w=0.125 | 0.1560 | 6 (3.5 %) | 46 (58.2 %) | −40 |
| SG-prev w=2 | 0.0000 | 0 (0 %) | **79 (100 %)** | −79 |

CFG w=2's "+0.0040" is **14 fixed against 13 broken** — churn, not a targeted
correction. It is not a small consistent effect; it is a large bidirectional one
that nearly cancels.

---

## 3. The headroom hypothesis — CONFIRMED, quantitatively

CFG's **per-problem behaviour barely changes across regimes**. What changes is
the base rate it is applied to.

| Regime | baseline acc | CFG | fix % of errors | break % of correct | fixed/broke | net |
|---|---|---|---|---|---|---|
| A γ=0 | 0.133 | w=12 | 12.6 % | 23.9 % | **3.43** | +102 |
| A γ=0 | 0.133 | w=20 | 13.4 % | 26.1 % | 3.33 | +107 |
| A γ=0.3 | 0.252 | w=2 | 10.7 % | 23.8 % | **1.34** | +27 |
| A γ=0.3 | 0.252 | w=4 | 11.3 % | 24.7 % | 1.37 | +30 |
| B γ=0.41 | 0.316 | w=2 | 8.2 % | 16.5 % | **1.08** | +1 |
| B γ=0.41 | 0.316 | w=8 | 9.4 % | 32.9 % | 0.62 | −10 |

**CFG is a fixed-probability bidirectional operation**: it rescues ~10 % of
errors and destroys ~20–25 % of correct answers, *in every regime tested*. Its
net value is therefore governed by arithmetic, not by any change in what it does:

```
net > 0   ⟺   fix% · E  >  break% · C   ⟺   E/C  >  break%/fix%  ≈  2
```

E/C is 6.5 at γ=0, 3.0 at γ=0.3, 2.2 at γ=0.41 — tracking the observed
3.43 → 1.34 → 1.08 ratio.

**Falsifiable prediction:** CFG turns net-harmful once baseline accuracy exceeds
≈33 %. Any future baseline improvement should push CFG below break-even, and
this can be checked on data we already have to collect anyway.

---

## 4. SG-prev w=2 = 0.0000 — trajectory failure, not decoding failure

**Output evidence** (100 stored records):

| | baseline | SG-prev 0.125 | **SG-prev 2** |
|---|---|---|---|
| contains `def ` | 100/100 | 97/100 | **0/100** |
| chars | 392 avg (300–400) | 396 | **400 exactly, every sample** |
| max repeated 12-char block (median) | 2 | 3 | **13** |
| parsed answers (of 250) | 152 | 78 | **0** |
| unique answers | 109 | 59 | **0** |

Verbatim: `quantityPricequantity Passivequantity educators SVquantityPrice…` —
repeated rare tokens, no Python syntax, never terminating.

**Trajectory evidence** (population-level, binned by log₁₀σ):

| Method | bit entropy (low σ) | saturation p<0.01 | **guidance/score** | SG dir rms |
|---|---|---|---|---|
| baseline | 0.0019 | 0.801 | 0.000 | — |
| CFG w=2 | 0.0016 | 0.799 | 0.077 | — |
| AG w=15 | 0.0017 | 0.791 | 0.211 | — |
| SG-prev 0.125 | 0.0049 | 0.770 | 0.466 | 9.5 |
| **SG-prev 2** | **0.0338** | **0.520** | **2.375** | **27.1** |

This is **§11 type C→A**: divergence begins at **step 62 of 1023** (σ≈21, the
high-noise region), the guidance term grows to **2.4× the score itself**, and the
trajectory then never converges — bit entropy stays 18× baseline and saturation
never rises, so no bit ever commits. The 400-character token soup is the
*consequence*, not the cause.

SG-prev w=0.125 is the same failure, weaker and later: divergence at step 242,
ratio 0.47. So this is a **graded slide, not a clean phase transition** — but see
§5.

---

## 5. A single quantity orders every method

`guidance/score` — the guidance term's magnitude relative to the score it
modifies — predicts the damage across *all* mechanisms:

| guidance/score | method | accuracy |
|---|---|---|
| 0.000 | baseline | 0.3160 |
| 0.077 | CFG w=2 | 0.3200 |
| 0.211 | AG w=15 | 0.2240 |
| 0.466 | SG-prev 0.125 | 0.1560 |
| 2.375 | SG-prev 2 | 0.0000 |

Monotone, across three different mechanisms. **Hypothesis (not established):**
the canonical stochastic sampler tolerates guidance only while the guidance term
stays small relative to the score, with damage setting in well below ratio 1 and
collapse near/above it. This is correlational — five points, one seed, and the
ratio is a batch mean — but it is a concrete, testable stability criterion and
it explains why *every* mechanism reverses at this operating point rather than
just SG.

---

## 6. Does stochasticity already solve what guidance solved?

Partially answerable. Baseline T1 (non-executing) is 98/250 = 39 % of problems —
still the dominant failure mode *after* stochasticity. So churn has **not**
eliminated the error class; it has raised accuracy from 0.137 to 0.316 while
leaving a large T1 population.

CFG does not preferentially attack T1: it rescues 8 % of errors and breaks 17 %
of correct answers, and its parsed-answer count (102 None) is close to
baseline's (98). **CFG is not fixing a distinct error class** — it perturbs
outcomes roughly uniformly.

That is the sharper version of the headroom result: guidance's problem here is
not that stochasticity solved its target errors, but that guidance was never
targeted at all.

---

## 7. Missing measurements (needed before the next experiment)

1. **Per-problem trajectories** — every trajectory statement above is a batch
   mean and cannot be joined to individual outcomes.
2. **Guidance cosines / projections** — only RMS norms are logged, so
   direction-vs-magnitude is untestable.
3. **Intermediate decoded answers** — commitment timing is unmeasurable.
4. **Seed variance under churn** — every Regime B cell is one seed.
5. **Confidence at the answer tokens** — only whole-sequence bit entropy exists,
   so "confident-wrong" cannot be separated from "uncertain-wrong".
