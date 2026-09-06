# binary_SM vs binary_CE pilot — post-mortem

**Verdict: the pilot is void as a comparison. Both arms diverged and then froze.
The primary endpoint was never logged, because of a bug I introduced.**

Nothing below reinterprets or replaces any earlier result. The guidance study
(Regimes A and B) is untouched.

## 0. Framing correction — read this first

An earlier draft of this document, and my earlier reporting, framed the finding
as **global SM gradient starvation**. *That framing is not supported and is
withdrawn.*

On the healthy production run, aggregate gradient survival is **stable at
0.13-0.19 from 250k to 500k**. SM does not progressively strangle itself across
the sigma range.

What IS demonstrated is a **localised low-sigma saturation**:

| | sigma = 0.05 | sigma >= 0.2 |
|---|---|---|
| survival @250k | 0.131 | 0.089 - 0.188 |
| survival @350k | **0.000** | 0.098 - 0.189 |
| survival @500k | **0.000** | 0.134 - 0.189 |

The collapse is total, it worsens with training, and it is confined to the
sigma band where the final denoising steps fix the emitted bits. Everywhere
else the gradient is healthy and stays healthy.

Two further corrections of my own claims:

- **Saturation fraction is not a proxy for gradient survival.** At sigma=0.2,
  99.4% of free bits are saturated yet survival is ~0.13: the unsaturated
  minority carries essentially the whole signal. The two must be reported
  separately, and this document now does.
- **The pilot's divergence is not an objective-specific effect.** Both arms
  diverged. See `docs/ce_weighting_derivation.md` section 2: the EDM
  sigma-weighting puts 91% of its mass on the 18% of draws below sigma=0.1,
  where the Bayes risk is ~2e-7. That is a latent 2.5e5 amplifier that applies
  to both arms.

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

So for a model whose logits have already blown up, the SM gradient is
identically zero rather than merely small, and the state is absorbing.

**Scope of that claim — deliberately narrow.** It is demonstrated *only* for a
model that has already diverged. It is **not** evidence that SM reaches this
state in normal training: the production run sat at survival 0.13-0.19 for
250k steps without approaching it, except at low sigma (section 0). And it is
**not** a differential advantage for CE, which froze too.

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

---

## 6. The one valid measurement: the production run

Ran the offline probe against the **healthy** production checkpoints
(250k/350k/425k/500k, the run that trained normally). 16 fixed validation
examples, ~120k free bits per sigma cell, identical noise across checkpoints.

`grad_survival` — the fraction of CE's gradient magnitude that SM retains:

| step | s=0.05 | s=0.2 | s=0.4 | s=1 | s=3 | s=10 | s=40 |
|---|---|---|---|---|---|---|---|
| 250k | 0.131 | 0.089 | 0.121 | 0.164 | 0.188 | 0.186 | 0.186 |
| 350k | **0.000** | 0.098 | 0.133 | 0.173 | 0.185 | 0.184 | 0.189 |
| 425k | **0.000** | 0.149 | 0.155 | 0.167 | 0.174 | 0.187 | 0.188 |
| 500k | **0.000** | 0.134 | 0.153 | 0.186 | 0.185 | 0.188 | 0.189 |

Fraction of free bits with `D(1-D) < 0.001` (saturated):

| step | s=0.05 | s=0.2 | s=0.4 | s=1 | s=3 | s=10 | s=40 |
|---|---|---|---|---|---|---|---|
| 250k | 0.999 | 0.994 | 0.882 | 0.664 | 0.642 | 0.491 | 0.434 |
| 500k | 1.000 | 0.996 | 0.923 | 0.650 | 0.650 | 0.478 | 0.403 |

Three things follow, and they sharpen the hypothesis rather than confirming the
version of it I started with:

1. **Aggregate survival is stable, not collapsing.** ~0.13-0.19 from 250k to
   500k. There is no progressive global death in a healthy run, so "SM slowly
   strangles itself everywhere" is *not* supported.

2. **At low sigma it collapses completely, and it gets worse with training.**
   At `sigma = 0.05`, survival goes 0.131 (250k) -> **exactly 0.000** (350k
   onward), with 100% of free bits saturated. This is the strongest evidence for
   the CE hypothesis, and it is localised: low sigma is exactly where the final
   denoising steps run, i.e. the steps that fix the emitted bits.

3. **A saturated majority does not imply a dead gradient.** At `sigma = 0.2`,
   99.4% of bits are saturated yet survival is still ~0.13 — the small
   unsaturated minority carries essentially the entire learning signal. So
   saturation fraction and gradient survival must be reported separately; one
   does not stand in for the other.

Measured in fp32. Training runs bf16, which saturates `sigmoid` at a smaller
`|ell|`, so true training-time survival is a **lower** bound on these numbers.

**Revised hypothesis for the re-run:** CE's advantage, if any, should appear as
retained gradient at *low sigma* specifically. Any re-run must resolve the
endpoint by sigma; a scalar aggregate would have hidden the only real effect
here, since it barely moves while the low-sigma cell goes to zero.

Caveat: n=16 examples, one noise seed. Cheap to widen and worth widening before
this is used to justify GPU spend.
