# Branch 1 — prediction representation: logits vs score matching

**Assessment only. No code changed, no GPU spent.** Guidance results untouched.

---

## 1. What the network actually predicts

| Stage | Object | Shape |
|---|---|---|
| SDT backbone | `out_dim = 1`, `patch_size = 16` | per-bit scalar |
| `canonicalize_continuous_logits` | **canonical public logits** `ell_raw` | `[B, S]`, S = 512×16 = 8192 |
| `apply_continuous_logit_postprocessing` | `ell = ell_raw + mf` (`continuous_logit_scaling = matched_filter_residual`) | `[B, S]` |
| `logits_to_x0_hat` | `D = sigmoid(ell)` = posterior mean estimate of the clean bit | `[B, S]` |
| sampler | `score = (D − x)/σ²` (Tweedie) | `[B, S]` |

**The network already emits logits.** The sigmoid is applied downstream, in
`logits_to_x0_hat`, not inside the model.

## 2. Can logits be exposed exactly without retraining?

**Yes — they already are, and it is exact and lossless.** `_model_logits_continuous`
returns `ell`; `D = sigmoid(ell)` is a bijection, so nothing is lost either way.

This is option **A** in the brief, and it is a **no-op**: there is no work to do
and no experiment to run. It must not be confused with option **B**.

## 3. What the real question turns out to be

The head is not the variable — **the training objective is.** Both losses already
exist in `diffusion/continuous/losses.py::binary_score_interpolation_loss`:

| `cfg.train.loss_type` | Loss | Status |
|---|---|---|
| **`binary_sm`** | `w(σ) · ‖sigmoid(ell) − x0‖²` | **what every CoBit checkpoint was trained with** |
| `binary_ce` | `w(σ) · BCE_with_logits(ell, x0)` | implemented, never used, and the *code default* |

Same head, same output shape, same σ-weighting, same sampler, same matched
filter. **The two differ by one config line.**

### They have the same optimum — so this is not a capacity question

Both MSE-on-probabilities and BCE-on-logits are **proper scoring rules for the
Bernoulli mean**, so both are minimised at the same target
`D* = E[x0 | x_t]`. Changing the loss cannot change what the model is trying to
represent. What changes is the gradient geometry:

```
∂L_ce/∂ell = w(σ) · (D − x0)
∂L_sm/∂ell = w(σ) · (D − x0) · D(1−D)
```

The score-matching gradient carries an extra `D(1−D)` factor that **vanishes as
D → 0 or 1**. A saturated-but-wrong bit receives almost no gradient under
`binary_sm`, and a full-strength one under `binary_ce`.

**This is not speculative for CoBit specifically.** The trajectory logs measure
`frac_p_lt_0.01 ≈ 0.80` at low σ — roughly 80 % of bits sit in exactly the
region where the `binary_sm` gradient is suppressed.

**Falsifiable prediction:** `binary_ce` should show faster loss reduction at low
σ, better-calibrated bit confidence, and — if the saturated-wrong bits are the
ones carrying answer errors — better accuracy. If accuracy is unchanged while
calibration improves, that is still a useful result for future guidance/RL work,
which needs meaningful confidences.

### The genuinely different representation, for completeness

A **per-token categorical** formulation also exists: `is_cont_tokens` mode, logits
`[B, S, V]`, `softmax`, `token_score_interpolation_loss`. This is what "softmax"
properly means, as opposed to the per-bit Bernoulli sigmoid. It changes
`out_dim` from 1 to V (~49k), the data pipeline, and memory — a far larger
change than the loss switch, and **not** the minimal experiment. It should only
be considered if the loss-switch pilot shows the representation matters.

## 4. Smallest valid pilot

**Two training runs differing in exactly one config line**, everything else
frozen: same data, model, optimizer, LR schedule, σ schedule, batch size, seed,
step count, and evaluation.

| Arm | `loss_type` |
|---|---|
| A (control) | `binary_sm` — reproduces the existing recipe |
| B | `binary_ce` |

Evaluate both at the canonical operating point (DDIM γ=0.41, 1024 steps) *and*
deterministically, since the guidance study showed conclusions can invert
between regimes. Primary metrics: validation loss, **bit-level calibration**
(reliability curve / ECE — nothing in the repo measures this yet), saturation
profile, and GSM8K accuracy.

The control arm is not optional: it re-derives the baseline under the pilot's
reduced step budget, so arm B is compared against a matched short run rather
than against the 500k-step production checkpoint.

## 5. MEASURED: the gradient suppression is real and severe

Rather than infer the mechanism from a retraining pilot, it can be measured
directly at the production checkpoint. The two losses differ by exactly the
factor `D(1−D)` per bit, so that factor *is* the effect. Measured on real
TinyGSM data with the EMA weights of `cobit_raw_binary_bits_cfg/last.pt`
(free/suffix bits only, prompt excluded):

| σ | mean D(1−D) | median | frac < 0.01 | frac < 0.001 | **‖grad‖ sm ÷ ce** |
|---|---|---|---|---|---|
| 0.05 | 0.00000 | 0.00000 | 1.000 | 1.000 | **0.000** |
| 0.20 | 0.00016 | 0.00000 | 0.998 | 0.994 | 0.135 |
| 1.00 | 0.02446 | 0.00000 | 0.801 | 0.757 | 0.138 |
| 5.00 | 0.01472 | 0.00003 | 0.803 | 0.713 | 0.075 |
| 20.0 | 0.00612 | 0.00008 | 0.886 | 0.704 | 0.031 |
| 80.0 | 0.00697 | 0.00007 | 0.873 | 0.727 | 0.035 |

`D(1−D)` peaks at 0.25 and vanishes as bits saturate. **The median bit has a
suppression factor of essentially zero at every noise level**, 80–100 % of bits
sit below 0.01, and in aggregate `binary_sm` delivers **3–14 % of the gradient
magnitude** that `binary_ce` would — and **none at all** at σ=0.05.

This is the mechanism, measured rather than assumed, for **zero GPU-hours**.

**What it does and does not establish.** It establishes that the suppression is
real and large *at the operating point the current recipe reaches*. It does
**not** establish that `binary_ce` trains to a better model: a model trained
under CE would occupy different `D` values, so this cannot be extrapolated to
its trajectory. It also plausibly describes a trap — a saturated-but-wrong bit
receives almost no corrective gradient under `binary_sm`, and the trajectory
logs independently measure ~80 % saturation — but "trap" is an interpretation,
not a measurement.

## 6. Cost — measured, and it constrains the design

Throughput probe, 4×A100, 200 steps, capped cache:

| Arm | wall (200 steps) | steps/s | GPU-h per 1k steps |
|---|---|---|---|
| `binary_sm` | 350 s | 0.571 | **1.94** |
| `binary_ce` | 329 s | 0.608 | **1.83** |

The two are within ~6 % — **the loss switch is cost-neutral**, so Branch 1 is not
paying for its own experiment. (The gap is likely node/startup variance, not a
real speed difference; it should not be quoted as one.)

Projected arm cost, and the constraint it imposes:

| steps | GPU-h per arm | two arms |
|---|---|---|
| 25k | ~49 | ~97 |
| 50k | ~97 | ~194 |
| 100k | ~194 | ~389 |
| **500k (production)** | **~972** | **~1944** |

**Reproducing production scale is impossible**: one 500k arm exceeds the entire
~889 GPU-h remaining. Any Branch 1 pilot is necessarily a short-run comparison,
and must be reported as such rather than as a comparison of converged models.

**These figures include startup and `torch.compile` warmup** and therefore
*overstate* the true cost. A 400-step probe differenced against the 200-step one
separates the fixed offset, costs ~12 GPU-minutes, and could cut these estimates
substantially. It should precede any pilot-size decision.
