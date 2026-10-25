# Guidance for CoBit: CFG, AutoGuidance and Self-Guidance

How three diffusion-guidance mechanisms map onto CoBit's continuous bitstream
formulation, what had to change because CoBit is not an image diffusion model,
and how to reproduce the experiments.

- Implementation: [`diffusion/continuous/guidance.py`](../diffusion/continuous/guidance.py)
- Tests: [`tests/test_guidance.py`](../tests/test_guidance.py),
  [`tests/test_guidance_integration.py`](../tests/test_guidance_integration.py),
  [`tests/test_guidance_regression.py`](../tests/test_guidance_regression.py)
- Experiment grids: [`experiments/guidance/grids.py`](../experiments/guidance/grids.py)
- HPC jobs: [`scripts/hpc/guidance/`](../scripts/hpc/guidance/)

---

## 1. The CoBit quantities guidance acts on

CoBit's continuous binary model predicts a per-bit clean probability. Writing
`ell_raw` for the network output and `mf` for the analytic matched filter:

```
mf(x, sigma)  = matched_filter_scale * (x - center) / sigma^2      (analytic)
ell(x, sigma) = ell_raw(x, sigma) + mf(x, sigma)                   (postprocessed logit)
D(x, sigma)   = sigmoid(ell(x, sigma))                             (posterior mean, x0_hat)
score(x, sigma) = (D(x, sigma) - x) / sigma^2                      (Tweedie)
drift         = -sigma * score
```

(`diffusion/continuous/logit_postprocess.py`,
`diffusion/continuous/samplers.py::_score_from_probs`.)

Conditioning is **inpainting-style**, not a learned conditioning input: the
prompt enters by clamping the prompt coordinates of `x_t` to the clean prefix
bits (`prefix_full`), and the unconditional branch clamps the same coordinates
to a null prefix (`null_full`, `null_strategy="half"` → 0.5). The prompt
coordinates are also excluded from the drift (`_zero_mask_`), so only the free
suffix coordinates move.

Training already supports the dropout CFG needs: `cfg.cond.p_uncond` replaces
the prefix with the null prefix on a fraction of examples
(`trainers/trainer.py`), and `configs/tasks/tinygsm_bits_cfg.py` sets it to 0.1.
**No separate unconditional model is trained.**

### 1.1 Guidance is applied in D-space, and that is exactly score guidance

`score` is affine in `D` with coefficients `(x, sigma)` shared by every branch
being combined, so for weights summing to one:

```
sum_k a_k * score_k = ( sum_k a_k * D_k - x ) / sigma^2
```

Combining posterior means and combining scores are therefore the *same
operation*. The repository already exploited this for CFG
(`probs_g = probs_u + w*(probs_c - probs_u)`), and we keep the convention: one
combinator serves all three mechanisms, probabilities stay directly
interpretable for the bit-level diagnostics, and the refactor is numerically
identical to what shipped before. Pinned by
`test_b1_guiding_D_equals_guiding_the_score`.

The equivalence needs the branches to share `x` and `sigma`. CFG and AG satisfy
this; self-guidance deliberately does not, which is why it is treated separately
in §4.

---

## 2. The matched filter, and where the task brief had to be adapted

The brief asked for

```
final_score = matched_filter_score + guided_learned_score
```

**That decomposition does not exist in CoBit.** The matched filter is added to
the *logit*, inside the sigmoid, not to the score:

```
score = (sigmoid(ell_raw + mf) - x) / sigma^2
```

There is no additive split of `score` into a learned and an analytic part, so
"guide only the learned component" cannot be implemented by separating score
terms. The concern behind the instruction is still right, and is addressed
per-mechanism instead:

| Mechanism | Do the two branches share `(x, sigma)`? | Is the difference purely learned? |
|---|---|---|
| CFG | Same `x` at every **free** coordinate, same `sigma` | **Yes**, automatically |
| AutoGuidance | Identical `x` and `sigma` | **Yes**, automatically |
| Self-Guidance | Same `x`, **different `sigma`** | **No** — needs explicit handling |

For CFG the branches differ only on clamped prompt coordinates, which carry no
drift; on the free coordinates `x_c == x_u`, so `mf_c == mf_u` and it cancels
from `D_c - D_u` exactly. For AG both models see the same input, so `mf` is
identical. Gate `test_e3` confirms this on a model whose learned logit is
constant: CFG's direction is exactly zero.

Self-guidance is the real risk. Since `mf ∝ 1/sigma^2`, evaluating at
`sigma' > sigma` changes the analytic term substantially, and a naive difference
amplifies *data consistency* rather than correcting model error. The default
`sg_mf_mode="hold"` therefore evaluates the network at the shifted level but
re-attaches the matched filter at the true level:

```
D_bad_sg = sigmoid( ell_raw(x, sigma') + mf(x, sigma) )
```

implemented by stripping `mf(x, sigma')` off the postprocessed logit and adding
`mf(x, sigma)` back (exactly, using the same helper the postprocessing uses, and
accounting for how posterior temperature divides the term).

The decisive test is `test_e1` / `test_e2`: on a network whose learned logit is
independent of sigma, `hold` produces **exactly zero** guidance while `vary`
produces a large spurious correction that is entirely matched filter.
`sg_mf_mode="vary"` is retained as an ablation (grid `sg_mf`), because whether
this matters *empirically* is a question worth answering, not just asserting.

**Limitation.** `hold` is implemented for the binary representation only, and is
refused when `posterior_temp_space="token"` (codeword sharpening folds the
matched filter through a softmax over valid codewords, so it cannot be stripped
exactly). Both raise rather than silently degrading.

---

## 3. Classifier-free guidance

```
D_cfg = D_u + w_cfg * (D_c - D_u)
```

with `c` = the true prompt clamped into `x_t`, `u` = the null prefix. This
matches the reference implementation on `tasks/fkc-temperature` exactly; the
regression suite drives the pinned pre-refactor sampler and asserts bit-identical
output across `w ∈ {None, 0, 1, 2, 5}` × self-conditioning × refresh mode.

**Scale convention (inherited, and worth stating because it is unusual).**
`w = 0` *disables* CFG and takes the single-branch conditional path — it does
**not** give the unconditional model. `w = 1` is the algebraic no-op and also
returns the conditional prediction, via two branches. Guidance proper is `w > 1`.
Gates `test_a2` and `test_gate9_cfg_w1_reduces_to_plain_conditional` pin both.

Conditional and unconditional branches keep **separate self-conditioning
states** (`SelfCondState`), each clamped to its own prefix. Mixing them would
feed the conditional trajectory's belief into the unconditional branch and
destroy the meaning of `D_c - D_u`; `test_f3`/`test_f5` pin the separation and
that the stored state is never mutated by the clamping.

---

## 4. AutoGuidance

Following Karras et al., the guiding model is a *bad version of the same model*:

```
D_ag = D_bad + w_ag * (D_good - D_bad)
```

Both models are built from the **same config**, so only the weights differ and
the guidance direction isolates training quality rather than an architectural
difference (`_task_common.load_bad_model`). Supported badness axes:

- **an earlier checkpoint of the same run** (primary — no architectural confound);
- **raw vs EMA weights** at the same step (`--bad_ema 0`, a mild, free axis);
- a separately trained smaller model is supported by the same code path (pass any
  checkpoint whose config matches), but is not required.

Combined with CFG, the nested form from the experiment plan:

```
good_cfg = D_good_u + w_cfg * (D_good_c - D_good_u)
bad_cfg  = D_bad_u  + w_cfg * (D_bad_c  - D_bad_u)
D        = bad_cfg  + w_ag  * (good_cfg - bad_cfg)
```

With CFG off this degenerates to Karras et al.'s conditional form
`D_bad_c + w_ag*(D_good_c - D_bad_c)`. Gate `test_a8` pins the nesting;
`test_a7` pins that a bad model equal to the good one leaves the prediction
untouched at any scale, and `test_autoguidance_with_an_identical_bad_model_
reproduces_the_baseline` pins the same end to end.

Interactions:
- **Self-conditioning** — good and bad each keep their own state, so the bad
  model is bad in the same way it was during its own sampling.
- **Matched filter** — identical for both models; nothing to correct (§2).
- **FKC / temperature** — not combined. The FKC sampler keeps its own
  CFG+Feynman-Kac derivation (Prop 3.1) whose weight term is specific to the
  two-model geometric average; grafting AG onto it would need a new derivation,
  so `FeynmanKacEulerMaruyamaSampler` is out of scope here. Likewise the
  predictor/corrector-asymmetric CFG in `PredictorCorrectorSampler`. Both now
  **refuse** AG/SG rather than ignoring the request.

---

## 5. Self-guidance

The guiding model is the same good model evaluated at a **higher noise level**.

### 5.1 The coordinate is log-sigma

Two repo-specific reasons, not a convention borrowed from image diffusion:

1. **The network conditions on log-sigma directly.** `SigmaEmbedding.forward`
   computes `phases = sigma.log()[:, None] * freq` (`models/sdt.py`). A fixed
   step in log-sigma is a fixed step in the model's own time coordinate.
2. **CoBit's headline schedule is entropy-uniform.** `schedule="entropic"` draws
   sigmas from the inverse CDF of the entropy-rate distribution
   (`SigmaSchedule.prepare`), so the grid is uniform in *entropy* — neither in
   sigma nor in log-sigma. Anything keyed to raw grid spacing would make the
   guidance strength silently depend on NFE.

### 5.2 Spacing-normalised direction

With `g(u) = D(x, e^u)`, `u = log sigma`, and a bad evaluation at `u + delta`
(`delta > 0`, i.e. noisier):

```
Delta_sg = delta_ref * [ g(u) - g(u + delta) ] / delta
D_sg     = D_base + w_sg * Delta_sg
```

`Delta_sg` is `delta_ref` times a finite-difference estimate of `-dD/d log sigma`.
Dividing by the realised spacing and rescaling by the fixed `delta_ref` is what
keeps the strength comparable across NFE: the raw difference shrinks as the grid
refines, the normalised one does not (`test_d2`).

### 5.3 Two variants

| | extra model evaluations | spacing | state |
|---|---|---|---|
| **SG-exact** | 1 per branch per step (2× total) | `delta = delta_ref = sg_delta`, exact by construction | evaluated at the current `x` |
| **SG-prev** | **none** | `delta_i = log(sigma_prev) - log(sigma_cur)`, varies with the grid → removed by the normalisation | reuses the *previous* step's prediction, taken at `x_prev ≠ x_cur` |

SG-prev's residual approximation — the cached prediction was made at a different
state — is irreducible and is the "cheap" in cheap self-guidance. Whether it
matters is an empirical question the `sg` grid answers by running both at
matched NFE. SG-prev is inert on the first step (no cache yet, `test_d6`) and
costs exactly the unguided sampler's NFE (`test_sg_prev_costs_exactly_what_the_
unguided_sampler_costs`).

The cache stores **matched-filter-stripped logits**, not just `D`, so `hold` can
re-attach `mf` at the current level (§2).

### 5.4 Composition

Self-guidance is applied to the **same combined prediction the sampler is about
to use** — i.e. the CFG/AG mixture, not the bare conditional. That is the
coherent choice (it is a correction to the score actually being integrated) and
it is free for SG-prev, which caches the branch predictions it already computed.
With SG-exact it doubles the branch count, which the NFE accounting reports
honestly.

### 5.5 Self-guidance is not self-conditioning

Self-conditioning: previous x0 prediction → **model input**.
Self-guidance: prediction differences → **sampling correction**.
They are separate objects in separate places (`SelfCondState` vs `_SGCache`) and
are tested separately.

---

## 6. Architecture

```
GuidanceConfig      declarative policy (cfg_scale, ag_scale, sg_scale,
                    sg_variant, sg_delta, sg_mf_mode); knows which branches
                    a policy requires
GuidedDenoiser      evaluates those branches (batching everything belonging to
                    one network into a single forward pass) and combines them
GuidedPrediction    D (guided), D_branches (per-branch, for SC carry / decode),
                    diagnostics
SelfCondState       one self-conditioning tensor per branch, never shared
_SGCache            previous-step logits + log-sigma for SG-prev
```

The model is entirely unaware of guidance; no new architectures were added.
`DDIMSampler` (the headline `ddim_entropic` path, and the base class for the
Euler-Maruyama sampler) consumes `GuidedDenoiser`; `HeunSampler` and
`PredictorCorrectorSampler` keep their existing inline CFG and refuse AG/SG.

**Branch counts per sampler step** (`= NFE multiplier`):

| policy | branches |
|---|---|
| baseline | 1 |
| CFG *or* AG *or* SG-exact | 2 |
| CFG+AG, CFG+SG-exact, AG+SG-exact | 4 |
| CFG+AG+SG-exact | 8 |
| any `+ SG-prev` | unchanged |

`model_evaluations` counts **rows / B**, not forward passes: batching branches
saves kernel launches, not FLOPs, and a pass-count would have made every
compute comparison wrong.

> **Per-row sigma is a hard requirement.** SG-exact packs two noise levels into
> one batch, so the network must condition on sigma per row. CoBit's SDT does.
> A model that collapsed sigma to a scalar would silently return the current
> level twice and make self-guidance a no-op with no error —
> `test_d9_sg_exact_requires_per_row_sigma` makes that failure visible.

---

## 7. Reproducing the experiments

### Environment (CSD3 / Wilkes3)

```bash
export COBIT_PYTHON=/home/rg625/miniforge3/envs/sedd/bin/python   # torch 2.8.0+cu128
export PYTHONPATH=/rds/user/rg625/hpc-work/BitstreamDiffusion
```

`scripts/hpc/guidance/env.sh` sets this up inside a job. Note it does **not**
reuse the older `scripts/tasks/*.slurm` environment, which points at
`/home/gb511/miniconda3` and `/rds/project/rds-LlrDsbHU5UM/...` — neither is
readable by this account.

### Tests

```bash
PYTHONPATH=. $COBIT_PYTHON -m pytest tests/ -q
```

### Review a grid before submitting anything

```bash
$COBIT_PYTHON -m experiments.guidance.grids list
$COBIT_PYTHON -m experiments.guidance.grids show cfg_coarse
$COBIT_PYTHON -m experiments.guidance.grids size cfg_coarse
```

### Smoke test first — always

```bash
sbatch scripts/hpc/guidance/smoke.slurm
```

It runs baseline / CFG / AG / SG-prev / SG-exact / the 3-way combination on 16
problems at 32 steps, then validates the outputs
(`experiments.guidance.aggregate --check-smoke`) and **fails the job** if any
cell is missing, non-finite, missing provenance, or indistinguishable from
baseline. Its accuracies are meaningless and must not be reported.

### Sweeps

```bash
sbatch --array=0-$(($($COBIT_PYTHON -m experiments.guidance.grids size cfg_coarse)-1)) \
       --export=ALL,GRID=cfg_coarse scripts/hpc/guidance/array.slurm
```

Cells are idempotent (a cell whose result exists is skipped), so a partially
completed array can be resubmitted as a whole.

### Aggregate and analyse

```bash
$COBIT_PYTHON -m experiments.guidance.aggregate runs/guidance --out results/guidance
$COBIT_PYTHON -m experiments.guidance.analyse results/guidance --out results/guidance/figures
```

Every table and figure is regenerated from `all_cells.csv` / `per_step.csv`;
nothing is hand-edited.

---

## 8. Known limitations

- **Samplers.** Only `DDIMSampler` (and its `EulerMaruyamaSampler` subclass)
  supports AG/SG. Heun and predictor-corrector keep their own inline CFG; the
  FKC sampler keeps its own CFG+Feynman-Kac derivation. All refuse AG/SG loudly.
- **`sg_mf_mode="hold"`** is binary-representation only and incompatible with
  `posterior_temp_space="token"`.
- **Testbed.** TinyGSM→GSM8K is the only CoBit run that is both conditional and
  trained with `p_uncond > 0`, so it is the only place CFG is defined. The
  released LM1B/OWT checkpoints are unconditional (`p_uncond = 1.0`) and cannot
  be used for CFG or for a conditional AG study.
- **Pre-existing test failure**, unrelated to guidance:
  `tests/test_posterior_temp.py::test_codeword_lowT_full_codebook_is_perbit_map`
  fails on a near-tied logit (`-0.0056`) at `T=1e-3`, an ill-conditioned
  tolerance in the test rather than a defect in the code under test. It failed
  identically before any change in this work.
