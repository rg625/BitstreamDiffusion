# Experiment notes: temporal ordering & objective replacement

Commit at time of running: see each entry. Checkpoint, seeds, precision,
schedule and split are recorded per experiment so nothing has to be
reconstructed later.

---

## Framing: what this environment can and cannot answer

**Hard constraint.** Every training run in this environment diverges before
~12k steps — 7/7 across two code versions, including the production-era commit
`036a2b5` itself (break at 11,278). The production model needed **500k steps**
to reach 13.85%. Therefore:

- A **training-time** arm (ordering-as-training, or a converged CE model) cannot
  be evaluated here: the model would be broken long before it was useful.
- A **sampling-time** intervention on the healthy 500k production checkpoint is
  unaffected, because it involves no training at all.

So ordering is run as a **decoding** intervention, and the objective branch is
run at the **largest matched budget the environment permits (5,000 steps)**,
with that limitation stated rather than hidden.

---

## Experiment A1 — temporal ordering, screening

**Hypothesis.** Denoising the generated suffix in an order — left-to-right or a
random permutation — rather than all positions simultaneously improves GSM8K
accuracy.

**Mathematical definition.** Per-token time
`t_j(t) = clip(t(1+w) − w·u_j, 0, 1)`, with `u_j ∈ [0,1]` a **denoising
priority** defined over the generated **suffix only** (`u=1` denoises first).
`w=0` gives every token the global time, i.e. today's model. Per-token time is
mapped back through the schedule to a per-token sigma, expanded blockwise to
bits. Prompt positions keep the **global** sigma.

**Control.** `order_w = 0`, `OrderedSampler`, entropic schedule, 256 steps.
Verified by test to be a plain uniform-sigma deterministic Euler run and to be
invariant to the ranks themselves.

**Intervention.** Identical in every respect except `order_w`:
l2r at w ∈ {0.1, 0.25, 0.5, 1.0}; random at w ∈ {0.25, 0.5}; plus a
control/​w=0.25 pair at 512 steps to check that too short a trajectory is not
hiding an effect.

**Exact config difference.** One CLI number, `--order_w` (plus `--order_mode`).
Same checkpoint, schedule, steps, seed, EMA, batch, split.

**Checkpoint.** `tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/last.pt`
(500k, EMA=1) — the healthy production model.

**Budget.** 250 problems × 9 cells, 1 seed, ~1.35 samples/s on one A100 ⇒
~0.6 GPU-h.

**Causal-activity check (done, pre-GPU, on the real model).** Per-position sigma
at the midpoint of the trajectory:

| w | mode | suffix sigma range | spread | prompt sigma |
|---|---|---|---|---|
| 0.0 | — | 0.7237 – 0.7237 | 1.00× | 0.7237 |
| 0.5 | l2r | 0.0703 – 13.4858 | 192× | 0.7237 |
| 1.0 | l2r | 0.0068 – 76.7561 | 11247× | 0.7237 |
| 1.0 | random | 0.0068 – 76.7561 (scrambled) | 11247× | 0.7237 |

The intervention is active, directional (first suffix token cleanest under l2r),
and leaves the prompt alone.

**Known risk, stated in advance.** The model was **trained with uniform sigma**.
Feeding it per-position sigma is out-of-distribution, and large `w` is a long
way out. A null or negative result is a plausible and legitimate outcome.

**Result.** _pending_

---

## Experiment B1 — binary_sm vs binary_ce, task performance at matched budget

**Hypothesis.** Replacing the score-matching objective with cross-entropy
improves GSM8K task performance at an identical training budget.

**Control.** `binary_sm`, production-matched (`p_uncond=0.1`, clamp **OFF**,
`loss_weighting=edm`), seed 42, step 5,000.

**Intervention.** `binary_ce`, same run family, same seed, same step.

**Exact config difference.** `train.loss_type` only (verified by test that the
two arms' configs differ in exactly that key plus output paths).

**Why 5,000 steps.** It is the largest checkpoint both arms share and it
precedes **both** breaks (sm at 6,356; ce at 11,444), so neither model is
contaminated by its divergence.

**sigma-weighting.** The derivation
(`docs/ce_weighting_derivation.md`) concluded that a **bespoke CE weighting is
not justified**: matching the Bayes-risk scale gives a factor spanning only
0.250–0.361 across four decades of sigma (0.82×–1.18× anchored at sigma_data),
while gradient-scale matching is self-defeating because it reintroduces the
`D(1−D)` factor under test. Both arms therefore use the same EDM weighting, and
that is a derived conclusion rather than an inherited default.

**Evaluation.** Karras schedule (analytic, identical for both arms) rather than
entropic, which is fitted per run and would confound the objective with its own
sigma schedule. DDIM, gamma=0, 256 steps, EMA=1, seed 0, same problems.

**Budget.** 2 cells × 250 problems ⇒ ~0.1 GPU-h.

**Expectation, stated in advance.** Production needed 500k steps for 13.85%. At
5k neither arm is close to converged and both may score ≈0. That would be a
legitimate result — the task-performance question is then **unanswerable at the
budget this environment permits** — and will be reported as such, not rescued.

**Result — the pre-registered null. Both arms score exactly 0.**

| arm | n | accuracy | correct | T1 non-executing | T2 wrong number | invalid-token rate | samples/s |
|---|---|---|---|---|---|---|---|
| binary_sm @5k | 250 | **0.0000** | 0/250 | **250** | 0 | 0.2602 | 2.66 |
| binary_ce @5k | 250 | **0.0000** | 0/250 | **250** | 0 | 0.2613 | 2.72 |

Every single generation is a **non-executing program** (T1) in both arms. There
is no T2 population at all, so there is not even a wrong-arithmetic class to
compare. The invalid-token rates differ by 0.001 — noise. The two arms are
undifferentiated on every measured axis.

**Verdict: CE does not improve task performance at this budget — and neither
does SM, because at 5,000 steps neither objective produces a model that can
emit runnable code.** This is 1% of the 500k steps production needed. The
comparison is uninformative about the objectives, not evidence against CE.

**Why no more seeds or a longer budget.** More seeds cannot separate 0 from 0.
A longer matched budget does not exist: `binary_sm` broke at step 6,356, so 5k
is the last checkpoint both arms share. Extending would require fixing the
environment first.

**Distinguishing the two questions the brief asks to keep apart:**
- *better optimization/stability* — CE lasts **1.80× longer** before diverging
  (median 11,444 vs 6,356, perfect rank separation, p=0.05). Measured, real,
  but environment-limited and with an effect size comparable to run-to-run
  spread across code versions.
- *better final task performance* — **no evidence either way.** Not measurable
  in this environment.

**Environment-limited.** Marked as such: Python 3.9 here versus the
collaborator's ≥3.10, and every training run diverges before ~12k steps.

---

## Experiment E1 — environment validation (Python 3.10). **PREDICTION CONFIRMED: NOT FIXED.**

**Hypothesis under test (from the brief).** Python >= 3.10 is the environmental
difference responsible for the early divergences; the corrected environment
should let the production recipe survive past 20k.

**My pre-registered counter-prediction.** It would diverge anyway, because the
interpreter is not a plausible mechanism: CPU gradients are **bit-identical**
between the two environments (`gradhash 026d7bd7a5d3...` in both) and every
bundled CUDA library is the **same version** (cuBLAS 12.8.4.1, cuDNN 9.10.2.21,
NCCL 2.27.3, Triton 3.4.0). Same compute stack, different interpreter wrapper.

**Setup.** `sedd310` (Python 3.10.21, torch 2.8.0+cu128), production recipe,
`binary_sm`, `p_uncond=0.1`, clamp OFF, seed 42, 20k budget, guard at factor 10.
One variable: the interpreter.

**Result.** **Diverged at step 5,346**, EMA 0.3287 vs best 0.0247 (13.3x).

| environment | seed | break step |
|---|---|---|
| Python 3.9 | 42 | 6,356 |
| **Python 3.10** | 42 | **5,346** |

**Verdict.** The interpreter is **exonerated**. The environment hypothesis, as
stated, is falsified. My earlier reporting overstated the case: what was
actually established is that production's environment *differed* (proved by the
module-scope `mauve` import and by PEP 604 in a signature, unimportable on 3.9);
inferring that the interpreter was *causally responsible* did not follow, and is
now shown to be wrong.

**What this leaves.** The divergence cause remains open. Remaining candidates,
in order of plausibility:
1. the collaborator's **working tree** (their directory is permission-denied,
   so uncommitted differences cannot be excluded);
2. a different torch **build**, driver or GPU generation than we pin;
3. the recipe being genuinely marginal, with production lucky over 500k --
   argued against by 8/8 divergences here but not excluded.

**Consequence for the objective branch.** Item 2 of the brief made the corrected
environment a precondition for interpreting objective results. That precondition
is **not met**, so no strong objective-training claim can be made in either
environment, and the CE-vs-SM stability difference stays environment-limited.

---

## Experiment A3 — training-time ordering, smoke (400 steps, 4 arms)

All four arms initialised from the production 500k checkpoint and completed.

| arm | loss @start | loss @400 | throughput |
|---|---|---|---|
| control (none, w=0) | 0.0082 | 0.0399 | 5.09 it/s |
| l2r w=0.25 | 0.1615 | **0.0106** | 3.59 it/s |
| r2l w=0.25 | 0.1929 | **0.2479** | 3.61 it/s |
| random w=0.25 | 0.1659 | 0.1582 | 3.55 it/s |

Two things worth noting before the full runs:

- **The ordering arms start ~20x above the control.** Expected: the model has
  never seen per-position sigma, so the ordered arms begin out-of-distribution.
  l2r adapts quickly (0.16 -> 0.011), r2l gets *worse*, random is flat.
- **Losses are NOT comparable across arms.** Each arm's sigma field differs, so
  the loss is computed under a different noise distribution. Only the GSM8K
  accuracy comparison is meaningful, and that is the declared endpoint.
- Ordering costs ~30% throughput (3.6 vs 5.1 it/s), recorded as compute.

**Confound recorded:** the control starts at loss 0.0082 while production's own
`iter_train` at 500k was ~0.103. This is the entropy schedule: a fresh run uses
the base log-normal sigma draw until `entropy_warmup_steps=40000`, whereas
production had switched to its entropy-adapted draw. It is **matched across all
four arms**, so the comparison holds, but the arms are not directly comparable
to production's own loss curve.
