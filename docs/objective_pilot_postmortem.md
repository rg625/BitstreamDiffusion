# binary_SM vs binary_CE pilot — post-mortem

**Verdict: the pilot is void as a comparison. Both arms diverged and then froze.
The primary endpoint was never logged, because of a bug I introduced.**

Nothing below reinterprets or replaces any earlier result. The guidance study
(Regimes A and B) is untouched.

---

## 1. What was supposed to happen

Two arms, 100k steps each, differing only in `train.loss_type`. Primary endpoint:
the **gradient-survival trajectory**

```
grad_survival = sum |D - x0| * D(1-D) / sum |D - x0|
```

logged every 500 steps by `Trainer._log_objective_probe`.

## 2. What actually happened

### 2.1 The probe never ran (my bug)

The call site was placed in `Trainer._step_discrete`. Every bitstream task runs
`framework == "continuous_score"` and is dispatched to `_step_continuous`
(`trainers/trainer.py:2025`). **The probe was dead code.** Both arms completed
with zero `objective/*` scalars in TensorBoard.

The five tests that covered the probe all called `_log_objective_probe`
directly on a stub, so they passed against a dead call site. That is the gap:
they tested the arithmetic, never the wiring.

Fixed: the call now lives in `_step_continuous`, guarded to binary
representation. Three regression tests added, all three verified to **fail**
when the call is removed:

- `test_probe_is_reachable_from_the_dispatched_continuous_step`
- `test_probe_measures_only_free_bits_under_prefix_conditioning`
- `test_dispatch_selects_the_step_function_the_probe_lives_in`

315 tests pass.

### 2.2 Both arms diverged

Loss (`loss/iter_train`), median of first 100 steps vs. later:

| arm | median @ start | @15k | @20k | @40k | @60k | divergence onset |
|---|---|---|---|---|---|---|
| binary_sm | 0.056 | 0.024 | 1.11 | 34.4 | 53.4 | **step 19,540** |
| binary_ce | 0.177 | 1.3e6 | 3.8e3 | 2.5e5 | 4.1e5 | **step 6,580** |

Onsets differ by 3x, so this is not a scheduled event (LR was constant at 3e-4
throughout in both arms — verified; no decay, no scheduler artefact).

For scale: the **production** run reaches `loss ≈ 0.10` at 500k steps. The SM arm
was at 0.024 at 15k — training normally — and then blew up.

### 2.3 Both arms then froze permanently

Relative L2 weight change between consecutive checkpoints:

| transition | binary_sm | binary_ce |
|---|---|---|
| 10k -> 20k | 0.906 | 1.022 |
| 20k -> 30k | 0.228 | 0.332 |
| 30k -> 40k | **0.000000** | 0.071 |
| 40k -> 50k | **0.000000** | **0.000000** |
| ... -> 85k | **0.000000** | **0.000000** |

Not "small" — **bit-identical**. `max|delta| = 0.0` across every tensor.

Confirmed independently by the optimiser state: Adam's `exp_avg_sq` maximum
decays from 8.9e-6 (30k) to **1.1e-29** (85k). At `beta2 = 0.999` that decay
requires ~55,000 consecutive steps of *exactly zero* gradient — which matches the
frozen interval exactly.

**~55 of the 98 GPU-h computed nothing.**

### 2.4 Why the freeze is absorbing

At the frozen checkpoints the logits have blown up to `|ell| ~ 1e3`
(mean 1021, max 2673 at 30k; finite, not NaN):

| arm | sigma | loss | gradient norm |
|---|---|---|---|
| binary_sm | 0.4 | 1.89 | **exactly 0** |
| binary_sm | 20 | 0.74 | 0.045 |
| binary_ce | 0.4 | 1.1e4 | 9.3e5 |

`sigmoid(1000)` is exactly `1.0` in any float format, so `D(1-D)` is **exactly
zero**, and `dL_sm/d_ell = w(D-x0)*D(1-D)` is exactly zero. EDM samples sigma
log-normally around ~0.3, which is precisely where the SM gradient vanishes.

This is the sharpest possible form of the original hypothesis: the `D(1-D)`
factor is not merely a 3-14% attenuation — **once logits saturate it is an
absorbing state from which SM cannot recover, because the gradient is
identically zero rather than small.**

That claim is now *demonstrated* for binary_sm. It is **not** demonstrated as a
*differential* advantage for CE, because CE froze too.

## 3. Confounds — what I checked and ruled out

| Hypothesis | Test | Result |
|---|---|---|
| Per-position-sigma refactor changed `[B]`-path numerics | forward+loss+backward at HEAD vs `872674f`, fixed seed | **identical** — same gradient md5 |
| Checkpoints failed to load (`strict=False`) | missing/unexpected key count | 0 / 0 |
| Corrupt corpus cache | prompt-length distribution over 11.7M rows | clean: min 7, max 222, no degenerate rows |
| LR schedule decayed to 0 | TB `learning_rate` | constant 3e-4 throughout |
| Optimiser state went inf/NaN | scan of `exp_avg_sq`, `exp_avg` | all finite |

**Not ruled out**, and the leading suspects:

1. **`cond.p_uncond = 0.0` vs production's `0.1`.** This is the *only*
   substantive config difference from the run that trained healthily to 500k
   (full key-by-key diff against the production `config.json`: everything else —
   lr, grad_clip, EDM weighting, sigma_data, batch size, bf16, entropy schedule —
   is identical). My SM arm was therefore never a validated replication of the
   known-good baseline. That was my error in constructing the control.

2. **The EDM sigma-weighting applied to CE is not principled.** `w(sigma) =
   (sigma^2+sigma_d^2)/(sigma^2 sigma_d^2)` is derived to make
   `w * ||D - x0||^2` roughly sigma-invariant. BCE has a different sigma-scaling,
   so `w * BCE` reaches `2.9e8` at sigma=0.002 with gradient norm `2.4e10`.
   Holding the weighting fixed while changing the objective is a **confound I
   built into the design**, and it plausibly explains why CE diverged 3x earlier.

## 4. What is salvageable, at zero GPU cost

`scripts/analysis/offline_objective_probe.py` reconstructs the endpoint from the
saved checkpoints. It is strictly *better* than the in-training probe would have
been: every checkpoint of every arm sees the **same** validation examples, the
**same** Gaussian noise and an explicit **sigma grid**, all fixed-seed. The
measurement is fully paired, so arm and step differences carry no batch-sampling
noise, and the suppression is resolved against sigma instead of being averaged
over a log-normal draw.

It runs on CPU in minutes and is the right way to measure the production
checkpoints (250k/350k/425k/500k), which trained healthily.

## 5. Recommended re-run design

Do **not** simply resubmit.

1. Match production exactly for the SM control (`cond.p_uncond = 0.1`), so the
   control arm provably reproduces a known-good trajectory.
2. Give CE a sigma-weighting appropriate to CE, or run both arms unweighted, so
   the objective is the only variable that matters.
3. Add a divergence guard: abort on a sustained loss increase, and log gradient
   norm and `frac(D(1-D) < 1e-3)` every 500 steps. The run should stop at 20k,
   not burn 55 GPU-h frozen.
4. Validate at 5k steps (~2.5 GPU-h) before committing 98.
