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

## 5. Cost — needs one measurement first

Production training was 500k steps on **4×A100** with 12 h SLURM jobs. I have
**not** measured throughput, so I will not quote a firm figure. The first action
should be a ~10-minute timing probe to get steps/sec, from which the pilot cost
follows exactly.

Order-of-magnitude, assuming a 50k-step pilot (10 % of production) per arm:
**~15–35 GPU-h per arm, ~30–70 GPU-h for the pair.** That range is wide because
it rests on an unmeasured throughput; treat it as a planning placeholder, not an
estimate.
