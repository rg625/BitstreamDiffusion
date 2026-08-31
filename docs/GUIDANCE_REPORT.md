# Guidance for CoBit: CFG, AutoGuidance and Self-Guidance — experiment report

Final report for the guidance study on `tasks/fkc-temperature`.
Companion documents: [`docs/GUIDANCE.md`](GUIDANCE.md) (mathematical formulation
and implementation choices) and
[`scripts/hpc/guidance/RUNBOOK.md`](../scripts/hpc/guidance/RUNBOOK.md)
(exact HPC commands).

**Status discipline used throughout.** Every claim below is tagged as
*implementation* (code + tests), *exploratory* (n=250, one seed, no confirmation)
or *confirmed* (n=1319, 3 seeds, paired bootstrap). Two planned experiments did
not run; they are listed in §9 rather than estimated.

---

## 1. Testbed and why it is the only one

CFG requires a model trained with conditioning dropout. Exactly one CoBit run
qualifies: `configs/tasks/tinygsm_bits_cfg.py` (`cond.enabled`, `p_uncond=0.1`),
trained on TinyGSM to 500k steps and evaluated zero-shot on the GSM8K test set.
The base CoBit checkpoints (`cobit_s_lm1b`, `cobit_s_owt`, `cobit_m_owt`) are
**unconditional**, so CFG is undefined on them — this is why the study is a
GSM8K study and not an LM1B/OWT perplexity study.

| | |
|---|---|
| Model | CoBit raw-binary-bits, SDT backbone, ~2.1 GB checkpoint |
| Good model | `.../checkpoints/last.pt`, global step 500000, EMA |
| Bad models (AG) | steps 250000 / 350000 / 425000, same architecture |
| Task | TinyGSM → GSM8K, zero-shot, answer graded by sandboxed execution |
| Primary metric | exact match on the executed answer |
| Sampler | DDIM, entropic schedule, deterministic (γ=0) |
| Exploration | 250-problem **prefix** of the test set (a prefix, not a random subset, so every method sees identical problems) |
| Confirmation | all 1319 problems, seeds 42/43/44, 512 steps |

Metrics come from `evaluation/guidance_metrics.py`: quality (exact match),
diversity (unique fraction, distinct-1..4, token entropy, self-repetition-4),
bit-level diagnostics (bit entropy, saturation fractions, p min/mean/max),
guidance diagnostics (‖CFG/AG/SG direction‖, score norm, guidance/score ratio,
each resolved against σ) and efficiency (NFE, model evaluations, wall clock,
samples/sec, peak GPU). GenPPL and MAUVE were **not** computed: the repository's
`evaluation/mauve.py` and `external_perplexity.py` target the unconditional
LM1B/OWT generation path, not the task path, and a task with a graded answer has
a better primary metric than a proxy fluency score.

---

## 2. Mathematical formulation actually used

Full derivation in [`docs/GUIDANCE.md`](GUIDANCE.md); the operative points:

**Guidance acts on `D`, and that *is* score guidance.** CoBit's postprocessed
logit is `ell = ell_raw + mf`, with `mf` the analytic matched filter; the
posterior mean is `D = sigmoid(ell)` and the score is Tweedie,
`score = (D - x)/sigma^2`. Because the score is affine in `D` with coefficients
`(x, sigma)` shared by every branch being combined, for weights summing to one

```
sum_k a_k * score_k  ==  ( sum_k a_k * D_k - x ) / sigma^2
```

so combining posterior means and combining scores are the same operation.
Pinned by `test_b1_guiding_D_equals_guiding_the_score`.

**The matched filter is not amplified.** `mf` is analytic and identical across
branches, so it cancels out of every difference (`D_c - D_u`, `D_good - D_bad`).
The guidance interpolation therefore acts only on the learned component, which
is the property the brief demanded. This is asserted directly rather than
assumed — see §4.

**Conditioning is inpainting-style, not a learned input.** The prompt enters by
clamping the prompt coordinates of `x_t` to the clean prefix bits; the
unconditional branch clamps the same coordinates to a null prefix
(`null_strategy="half"` → 0.5). Prompt coordinates are excluded from the drift,
so only the free suffix moves.

```
CFG :  D_cfg = D_u + w_cfg * (D_c - D_u)
AG  :  D_ag  = D_bad + w_ag * (D_good - D_bad)
CFG+AG: good and bad are each CFG-combined first, then AG-combined
SG  :  D_sg  = D + w_sg * (D(x, sigma) - D(x, sigma*(1+delta)))
```

> **Scale convention.** `w=1` means *pure conditional* (= baseline); `w=0` is a
> sentinel meaning "guidance off". Weights **below 1 interpolate toward the
> unconditional model** and are actively harmful — see §5.1, where w=0.25 and
> w=0.5 score exactly 0.0000. This is the standard Ho–Salimans convention, but
> it makes the low end of a naive "0 … 1" sweep a *negative*-guidance region.

**Self-Guidance finite difference.** The natural coordinate is `log sigma`:
CoBit's entropic schedule is close to geometric in σ, so a fixed relative offset
`delta` gives a step that is uniform in `log sigma` and therefore independent of
the σ-grid spacing — an absolute-σ offset would not be. SG-exact evaluates the
model at `sigma*(1+delta)`, **clamped to the trained σ range** (commit
`309cae7`; without the clamp the shifted level leaves the support the model was
trained on at the top of the schedule). SG-prev reuses the previous step's
prediction instead, costing no extra evaluation, and is skipped on any step
where the previous σ is not usable (tracked and reported as `sg_skipped`).

Self-Guidance and self-conditioning are separate mechanisms and separate state:
`SelfCondState` keeps one self-conditioning tensor **per branch**, so the
conditional and unconditional trajectories never share it.

---

## 3. Implementation architecture

Single module, `diffusion/continuous/guidance.py` (876 lines). The model is
unaware of guidance; no new architectures were introduced.

| Component | Role |
|---|---|
| `GuidanceConfig` | declarative policy (cfg/ag/sg scales, variant, δ, mf mode); `branches()` derives which forward passes are needed |
| `GuidedDenoiser` | evaluates the required branches, batched, and returns a `GuidedPrediction` |
| `GuidedPrediction` | `D_cond`, `.score(x, sigma)`, auxiliary diagnostics |
| `SelfCondState` | per-branch self-conditioning, no cross-branch leakage |
| `_SGCache` | previous-step prediction for SG-prev, detached |
| `lerp_guidance` / `combine_cfg_ag` / `sg_direction` / `apply_sg` | the four algebraic primitives, each separately tested |

Branches are concatenated into one batched forward pass where possible;
`test_batched_equals_separate` asserts numerical equivalence with separate calls.
`model_evaluations()` reports the honest count, so NFE accounting is not inferred
from the step count (commit `eb09e1d`).

**Backward compatibility.** With guidance disabled the sampler follows its
original code path. `_resolve_guidance_config` promotes the legacy scalar
`guidance_scale` to a CFG-only policy, and `_guard_legacy_guidance` makes the
older samplers (which keep their original inline CFG block) *fail loudly* rather
than silently ignore AG/SG — AutoGuidance and Self-Guidance are available on
`DDIMSampler` only, and asking for them elsewhere raises.

---

## 4. Tests

`266 passed` (full suite, `pytest tests/ -q`). Guidance-specific:
`tests/test_guidance.py` (673 lines), `test_guidance_integration.py` (274),
`test_guidance_regression.py` (127).

| Area | What is asserted |
|---|---|
| CFG | exact algebra; w=0 → unconditional; w=1 → conditional (= baseline); monotone interpolation; shape/device/dtype |
| AutoGuidance | exact algebra; w=0 → bad; w=1 → good; **good==bad ⇒ exactly zero guidance**; batched == separate |
| Self-Guidance | finite-difference correctness; `log sigma` normalisation; behaviour under σ-grid refinement; exact vs prev; equal predictions ⇒ zero direction; σ-range clamping |
| Matched filter | guidance does not amplify `mf`; decomposed implementation == direct score formulation |
| Conditioning | only prefix positions are replaced by the null condition; suffix untouched; null conditioning correct for bitstreams |
| Self-conditioning | conditional and unconditional states stay separate; no leakage; cached prediction detached |
| Stability | no NaN/Inf; probabilities in [0,1]; low-σ stability; large-scale behaviour |
| Regression | guidance disabled reproduces the pre-change sampler within tolerance (`tests/_legacy_reference.py`) |
| Integration | all 12 factorial conditions run end-to-end on a tiny model: shape, determinism, no NaN, conditioning intact, sampler terminates, metrics and checkpoints load |

**Phase 5 regression against the pre-refactor implementation.** The
`tasks/fkc-temperature` branch *is* the working branch; the pre-existing CFG
implementation it carried is captured in `tests/_legacy_reference.py` and the
refactor is asserted equal to it with guidance disabled and under CFG-only
settings. This is a same-repository behavioural pin, not a cross-branch
checkpoint diff.

One **pre-existing** test was flaky and is now fixed (commit `f15f16f`):
`test_codeword_lowT_full_codebook_is_perbit_map` drew unseeded logits and
asserted a T→0 limit to `atol=1e-3`. The limit only holds when `temp` is small
relative to the logit margin — residual mass is exactly `1/(1+exp(|ell|/T))` —
so a draw landing a logit near zero failed by up to 0.5 with the implementation
correct. It failed on **26 of 300 seeds**. The fix seeds the draw, holds the
margin at 0.05 ≫ temp, and adds a test pinning the margin law itself. Unrelated
to guidance (introduced by `fc401f4`, before this work).

---

## 5. Results

### 5.1 CFG scale sweep

*Exploratory* (n=250, 256 steps, seed 42). Coarse then refined:

| w | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2 | 3 | 4 | 5 | 7 | 8 | 10 | 12 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EM | .000 | .000 | .128 | .164 | .168 | .188 | .196 | .216 | .212 | .224 | .220 | .224 | .228 | .244 | .252 | .244 |

Baseline 0.164. Two things this shows, neither of them "larger is better":

* **w < 1 is catastrophic.** w=0.25 and w=0.5 produce *zero* correct answers —
  paired delta −0.1640 [−0.2120, −0.1200]. Under this convention those weights
  interpolate toward the unconditional model, and an unconditional CoBit cannot
  do arithmetic. The coarse grid the brief suggested (starting at 0) spends its
  first four points in this dead region.
* **The coarse sweep never turned over**, which is why `cfg_high` was added
  (commit `2565b45`) out to w=30. It flattens from w≈7 and the apparent peak at
  w=20 is within noise of w=8..30 at n=250.

### 5.2 AutoGuidance: badness × scale

*Exploratory* (n=250, 256 steps). Exact-match, `bad` checkpoint × AG scale:

| bad \ w | 6 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| step 250000 | **.208** | .188 | .192 | .192 | .204 |
| step 350000 | .188 | .200 | .188 | **.220** | .196 |
| step 425000 | .148 | .152 | .184 | .164 | .168 |

Baseline 0.164. **There is an optimal badness, and it is not the earliest
checkpoint.** Step 425000 — the checkpoint closest to the good model — is worst
at every scale: when good and bad are too similar the difference is small and
dominated by noise, so it supplies a weak and erratic direction. The surface is
noisy at n=250 and the 250k/350k rows are not cleanly separated; the two best
cells were promoted to confirmation.

### 5.3 Self-Guidance: prev vs exact

*Exploratory* (n=250), exact-match by NFE:

| variant \ w | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| SG-prev (256 steps) | .172 | .172 | **.196** | .192 |
| SG-exact (256 steps) | .172 | .168 | .152 | .148 |

Baseline 0.164 at 256 steps. SG-prev peaks around w=1–2; **SG-exact degrades
monotonically** with scale. The pattern held across 64/128/256/512 steps.

### 5.4 Confirmation (Phase 16)

**GSM8K test set, all 1319 problems, seeds 42/43/44, 512 steps, DDIM
deterministic.** Per-problem outcomes averaged over seeds, then paired bootstrap
over problems (20 000 resamples). Problems — not the 3×1319 rows — are the
independent unit; treating rows as independent would shrink every interval by
about √3. Reproduce with
`python -m experiments.guidance.confirm_summary results/guidance`.

| Config | EM | baseline | Δ | 95% CI | p | NFE/sample |
|---|---|---|---|---|---|---|
| **CFG w=12** | 0.2196 | 0.1385 | **+0.0811** | [+0.0675, +0.0953] | <1e-4 | 1024 |
| CFG w=20 | 0.2163 | 0.1385 | +0.0778 | [+0.0637, +0.0922] | <1e-4 | 1024 |
| CFG w=7 | 0.2148 | 0.1385 | +0.0763 | [+0.0634, +0.0900] | <1e-4 | 1024 |
| **AG w=15, bad=350k** | 0.2047 | 0.1385 | **+0.0662** | [+0.0533, +0.0794] | <1e-4 | 1024 |
| CFG w=4 | 0.2014 | 0.1385 | +0.0629 | [+0.0508, +0.0751] | <1e-4 | 1024 |
| **SG-prev w=2** | 0.1868 | 0.1385 | **+0.0483** | [+0.0392, +0.0576] | <1e-4 | **512** |
| AG w=6, bad=250k | 0.1792 | 0.1385 | +0.0407 | [+0.0296, +0.0518] | <1e-4 | 1024 |
| SG-prev w=1 | 0.1736 | 0.1385 | +0.0351 | [+0.0278, +0.0425] | <1e-4 | 512 |
| **SG-exact w=1** | 0.1233 | 0.1385 | **−0.0152** | [−0.0220, −0.0083] | <1e-4 | 1024 |

All nine effects are individually significant. Note what the confirmation
*changed* relative to exploration: the CFG optimum did **not** replicate at
w=20. CFG w=7, 12 and 20 have heavily overlapping intervals — **the CFG response
is a plateau above w≈7, not a peak.** The n=250 sweep's apparent maximum at
w=20 was noise, which is precisely the failure mode the confirmation stage
exists to catch.

### 5.4a Selection contamination, and why it does not bite

The operating points confirmed above (w=12, ag=15/bad=350k, sg=2) were **chosen
on the 250-problem exploration prefix, which is a subset of the 1319-problem
confirmation set**. 19% of the confirmation set therefore played a part in
selecting what was confirmed on it — a winner's-curse channel that inflates
every effect by an unknown amount.

Re-estimating on the disjoint 1069 problems that took no part in the selection
(`confirm_summary.py --holdout-from 250`):

| Config | Δ (all 1319) | Δ (held-out 1069) | shift |
|---|---|---|---|
| CFG w=12 | +0.0811 | +0.0804 [+0.0655,+0.0957] | −0.0007 |
| CFG w=20 | +0.0778 | +0.0761 [+0.0605,+0.0920] | −0.0018 |
| CFG w=7 | +0.0763 | +0.0755 [+0.0608,+0.0904] | −0.0009 |
| AG w=15 bad=350k | +0.0662 | +0.0649 [+0.0505,+0.0795] | −0.0014 |
| CFG w=4 | +0.0629 | +0.0608 [+0.0477,+0.0745] | −0.0021 |
| SG-prev w=2 | +0.0483 | +0.0496 [+0.0393,+0.0599] | +0.0013 |
| AG w=6 bad=250k | +0.0407 | +0.0387 [+0.0259,+0.0511] | −0.0020 |
| SG-prev w=1 | +0.0351 | +0.0362 [+0.0278,+0.0446] | +0.0010 |
| SG-exact w=1 | −0.0152 | −0.0156 [−0.0231,−0.0081] | −0.0004 |

Every shift is ≤0.002 — an order of magnitude inside the confidence intervals,
and not consistently signed (SG-prev moves *up* on the holdout). **The
contamination is real but immaterial**, because the response surfaces are flat
near their optima: selection had little to select. The ranking is unchanged.

### 5.5 Diversity, saturation and cost

Confirmation runs, averaged over seeds:

| Config | EM | distinct-2 | token H | bit H | frac p<0.01 | frac p>0.99 | guid/score | NFE | samples/s | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | .1385 | .6159 | 7.223 | .0477 | .7234 | .1601 | 0.000 | 512 | 1.349 | 2.31 |
| SG-prev w=2 | .1868 | .5902 | 7.097 | .0366 | .7388 | .1712 | 0.248 | **512** | **1.338** | 2.31 |
| AG w=15 | .2047 | .5966 | 7.071 | .0332 | .7469 | .1712 | 0.135 | 1024 | 0.679 | 2.81 |
| CFG w=12 | .2196 | .5978 | 7.078 | .0272 | .7556 | .1780 | 0.199 | 1024 | 0.706 | 3.81 |
| SG-exact w=1 | .1233 | .6166 | 7.237 | .0484 | .7206 | .1614 | 0.022 | 1024 | 0.707 | 3.81 |

* **Every method that gains accuracy pays in diversity — but not in proportion
  to the gain.** Relative to baseline, distinct-2 falls by 0.026 for SG-prev
  (+4.8 pts), 0.019 for AG (+6.6 pts) and 0.018 for CFG (+8.1 pts). The ordering
  is *inverted*: the method that gains most costs the least diversity.
  **CFG dominates the quality–diversity trade-off here**, and SG-prev is the
  least efficient converter of diversity into accuracy. Unique fraction stays
  1.0000 throughout — the models sharpen, they do not collapse to a single
  output.
* **Guidance sharpens the bit posterior.** Mean bit entropy roughly halves
  (.0477 → .0272 under CFG) and saturation rises (p<0.01: .7234 → .7556). This
  is the bitstream-specific signature of guidance.
* **No pathologies anywhere.** The invalid-token rate stays in the
  3e-6 … 5e-5 range across every confirmation cell — the 0.0000 in the table is
  4-decimal rounding, not an exact zero — and does not trend with guidance
  strength. No NaN or Inf; self-repetition-4 is flat (.0044 → .0045).
* **SG-prev is free.** 512 NFE and 1.338 samples/s against the baseline's 512
  and 1.349 — within 1% — while gaining +4.8 points.

### 5.6 Guidance as a function of σ

Guidance/score ratio, binned by log₁₀σ (confirmation runs, 512 steps):

| log₁₀σ | [−3,−2) | [−1,0) | [0,1) | [1,2) |
|---|---|---|---|---|
| CFG w=12 | 0.002 | 0.081 | **0.268** | 0.110 |
| AG w=15 | 0.002 | 0.107 | **0.165** | 0.034 |
| SG-prev w=2 | 0.000 | **0.279** | **0.277** | 0.008 |
| SG-exact w=1 | 0.000 | 0.037 | 0.018 | 0.003 |

* All methods concentrate their correction in the **mid-σ band** and decay to
  ~0 below log₁₀σ = −2. **Self-Guidance is stable at low σ** — the answer to an
  explicit question in the brief — because the correction vanishes there rather
  than being clipped.
* **SG-exact and SG-prev are not the same mechanism.** At the scales tested,
  SG-prev applies a correction an order of magnitude larger (0.28 vs 0.02) and
  in a different σ profile. SG-prev is therefore *not* a cheap approximation to
  SG-exact in this model; it behaves like a step-to-step extrapolation
  (momentum) term. The finding that the cheap variant beats the exact one is
  real, but it should not be read as "the approximation is good enough".

Figures (regenerated from the CSVs by `experiments/guidance/analyse.py`, never
hand-edited): `results/guidance/figures/` — `cfg_scale.png`, `ag_scale.png`,
`ag_heatmap.png`, `sg_scale.png`, `quality_vs_compute.png`,
`diversity_vs_quality.png`, `diagnostics_vs_sigma.png`.

---

## 6. Answers to the research questions

**CFG.** Yes — the largest confirmed gain, +8.1 points (0.139 → 0.220). Useful
from w≈4, plateauing above w≈7 with no degradation observed out to w=30 at
n=250. Diversity cost is real but modest (distinct-2 −0.018). The instability
region is *below* w=1, not above it. Costs 2× NFE.

**AutoGuidance.** Yes — a deliberately weaker CoBit checkpoint supplies a useful
direction: +6.6 points, confirmed. The best bad model is an **intermediate**
checkpoint (350k of 500k), not the earliest; the checkpoint closest to the good
model (425k) is worst at every scale. AG does **not** beat CFG here (+0.066 vs
+0.081, intervals barely overlapping) and it costs the same 2× NFE *plus* a
second 2.1 GB model resident. It does **not** preserve diversity better than CFG, contrary
to the usual argument for AutoGuidance: it is lower on both distinct-2 (.5966
vs .5978) and token entropy (7.071 vs 7.078) while also less accurate, so on
this testbed CFG dominates it on every axis measured. The "earlier checkpoint vs
smaller model" comparison was **not run** (§9).

**Self-Guidance.** Yes for SG-prev, no for SG-exact. SG-prev gains +4.8 points
at **zero additional model evaluations** — the only method here that improves
quality at fixed compute. SG-exact *significantly degrades* quality (−1.5
points) while costing 2× NFE. SG-prev is stable at low σ. Its benefit was
present at every NFE tested in exploration (64–512), so it is not a high-NFE-only
effect; the systematic NFE study was not run (§9).

**Compute.** At fixed NFE, SG-prev is the only confirmed winner: it is the sole
method whose gain is not bought with a second forward pass. At fixed *wall
clock* the ranking is the same (1.338 vs 0.706 samples/s for CFG). Whether CFG
still wins when given the same compute budget as a longer baseline trajectory is
**exploratory only**: CFG at 256 steps (=512 NFE) scored .228–.252 at n=250
against a 512-step baseline's .160, which suggests CFG survives a matched-NFE
comparison — but the grid built to settle this did not run (§9).

**Bitstream-specific.** Guidance halves mean bit entropy and pushes ~3 points of
probability mass into the saturated tails, without producing invalid sequences.
The learned and analytic components are affected differently by construction:
the matched filter is identical across branches and cancels from every guidance
difference, so all three mechanisms move only the learned score.

---

## 7. Recommended defaults

| Situation | Recommendation |
|---|---|
| Best quality, compute available | **CFG w=12** (anything in 7–20 is equivalent; 12 is mid-plateau) |
| **Fixed compute / latency** | **SG-prev w=2** — +4.8 points for free |
| Model trained **without** conditioning dropout (CFG undefined) | **AG w=15, bad = 70% checkpoint** — the only option, not a preference |
| Combining | not yet supported by evidence — see §9 |
| Avoid | any CFG w<1; SG-exact at any scale tested |

---

## 8. Known limitations

1. **One task, one model.** Every result is TinyGSM→GSM8K on the single
   conditional CoBit run. Nothing here is established for LM1B/OWT, and CFG is
   not even definable on those checkpoints without retraining with dropout.
2. **One sampler regime.** All confirmation ran DDIM deterministic (γ=0). The
   `cfg_stochastic` grid exists but did not run; guidance under EDM churn is
   unmeasured, and churn is known from the FKC work to recover some of the same
   headroom, so the two may not be additive.
3. **Accuracy is a low-resolution metric.** 0.139 → 0.220 on 1319 binary
   outcomes; effects smaller than ~2 points are not resolvable at this scale.
4. **AG's badness axis is coarse** — three checkpoints (250k/350k/425k). The
   run's earlier checkpoints (100k–225k) were not on disk, so the weak end of
   the badness axis is unexplored.
5. **The SG δ choice is only lightly ablated** (`sg_delta` grid, n=250, 10
   cells); δ=0.5 was carried into confirmation without a confirmed optimum.
6. **No interaction evidence.** All confirmed results are single-mechanism.

---

## 9. What did not run

Stated plainly, because these gaps bound the conclusions above.

| Phase | Grid | Status |
|---|---|---|
| 13 — factorial | `factorial_confirm`, 36 cells, ready and reviewed | **not submitted** |
| 14 — NFE / compute | `nfe`, 28 cells, ready and reviewed | **not submitted** |
| 15 — temperature/FKC interaction | no grid written | not attempted |
| 11 — smaller bad model | `train_bad_model.slurm` written | not run (needs the TinyGSM token cache, absent from this checkout) |

The two ready grids were blocked at the submission step in this session; the
exact `sbatch` lines are in
[`RUNBOOK.md` §6](../scripts/hpc/guidance/RUNBOOK.md). Until they run:

* **no claim is made about whether CFG+AG, CFG+SG or the three-way combination
  helps.** The single 12-condition exploratory factorial cell that exists
  (`CFG+AG+SG-prev` at 32 steps, n=16) is far too small to read — a Wilson
  interval of [0.011, 0.283] around a point estimate of 0.0625. `analyse.py` refuses to estimate interaction effects
  until the grid is complete, and prints `factorial grid incomplete` instead.
* **the quality-vs-compute Pareto front is not established**, only suggested by
  the exploratory numbers in §6.

Consequently the "recommended defaults" table offers no combined configuration.
Everything needed to close both gaps is committed; they are ~21 GPU-hours.

---

## 10. Reproduction

```bash
cd /rds/user/rg625/hpc-work/BitstreamDiffusion
export COBIT_PYTHON=/home/rg625/miniforge3/envs/sedd/bin/python
export PYTHONPATH=$PWD

$COBIT_PYTHON -m pytest tests/ -q                      # 266 passed

# enumerate any grid before submitting it
$COBIT_PYTHON -m experiments.guidance.grids show cfg_confirm

# aggregate + analyse whatever has completed
$COBIT_PYTHON experiments/guidance/aggregate.py runs/guidance --out results/guidance
$COBIT_PYTHON experiments/guidance/analyse.py results/guidance
$COBIT_PYTHON -m experiments.guidance.confirm_summary results/guidance
```

Every result JSON records git commit, branch, dirty flag, config, checkpoint,
sampler, schedule, NFE, all guidance scales, seed, batch size, GPU name,
hostname, SLURM job and array IDs, wall clock, peak GPU memory and the honest
model-evaluation count. Nothing in `results/` is hand-written; all figures
regenerate from the CSVs.
