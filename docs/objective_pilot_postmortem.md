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

---

## 7. The 5k smoke test — my clamp hypothesis is refuted

Two arms, 5k steps, ~5 GPU-h total. Both ran to completion.

### 7.1 What passed

- **The probe fix works.** 50 clean training points per arm of `objective/*`.
  This was the bug that voided the first pilot; it is closed.
- **The arms were configured as intended**: `p_uncond=0.1` (production's value),
  `loss_weight_max=100`, guard armed, `loss_type` the only difference.

### 7.2 What failed — the prediction I made

I wrote: *"If the amplifier story is right, the clamp alone should let both arms
train past 20k without diverging."*

| arm | loss trajectory | outcome |
|---|---|---|
| binary_ce | 2.14 -> 0.10, flat 0.08-0.15 to 5k | **stable** |
| binary_sm | 0.70 -> 0.024 by 3.8k, then **0.43 at step 3,900**, 1.76 spike at 4,160, climbing to 1.04 at 5k | **diverged** |

**The SM arm broke at step 3,900 — earlier than the unclamped run's 19,540.**
The clamp did not prevent divergence, and may have brought it forward.

This matters more than it first looks: that SM arm differed from the healthy
500k production run **by the clamp key alone**. So the evidence now points *at*
the clamp, not at the low-sigma amplifier it was meant to bound. The clamp has
been made **opt-in and off by default**; with it off, the SM control reproduces
production exactly.

### 7.3 The causal direction is the opposite of what I assumed

`grad_survival`, SM vs CE, around the break:

| step | SM loss | SM survival | CE survival |
|---|---|---|---|
| 3400 | 0.051 | 0.1717 | 0.1791 |
| 3700 | 0.024 | 0.1788 | 0.1762 |
| 3800 | 0.036 | 0.1810 | 0.1845 |
| **3900** | **0.426** | 0.1398 | 0.1751 |
| 4200 | 0.756 | 0.0535 | 0.1825 |
| 4900 | 0.549 | 0.0750 | 0.1840 |

**SM's gradient survival was flat at ~0.18 right up to the break, statistically
indistinguishable from CE's.** It only degrades *after* the loss breaks.

So within 5k steps, saturation is a **consequence** of divergence, not its
cause. I had been treating the implication as running the other way. On this
evidence the difference between the objectives is one of **stability**, not of
gradient survival — the endpoint the pilot was built around does not separate
the arms before the event.

Also note both arms saturate similarly (`frac D(1-D)<0.001` reaches ~0.94 in
each) while only SM loses gradient, which is the section-6 point again:
saturation fraction and survival are different quantities.

### 7.4 Honest limits

- **One seed per arm.** A single divergence event in one run is not evidence
  that SM is systematically less stable. It could be the seed.
- 5k steps is a floor, not a certificate: the previous SM arm survived to 19.5k.
- The guard did **not** fire, because `factor=20` was too permissive. Replayed
  on the real losses, SM's EMA peaked at **53.7x** its best and CE's at
  **1.8x**, so the default is now **10**, which fires on SM at step 4,320 and
  never on CE. Pinned by tests against those measured numbers.

### 7.5 The completed 2x2 — the clamp is exonerated, the objective is not

All four cells, 5k steps each, ~10 GPU-h total:

| | clamp off | clamp = 100 |
|---|---|---|
| **binary_sm** | **breaks @4,720** (0.030 -> 0.428, settles ~0.16) | **breaks @3,900** (0.026 -> 0.426, settles ~0.5 and climbing) |
| **binary_ce** | stable (0.07-0.19 throughout) | stable (0.08-0.15 throughout) |

**Both SM arms break. Neither CE arm does.** So the clamp is *not* the cause --
my section 7.2 suspicion was also wrong. It shifted the break by ~800 steps,
which is well within what one seed can tell us. The clamp remains off by
default (it buys nothing and the unclamped arm reproduces production exactly).

The break signature is near-identical in both SM runs: an abrupt jump from
~0.03 to **0.428** inside a single 20-step logging interval, then a partial
recovery to a plateau several times above the old floor. The coincidence of the
peak value across two independent runs suggests a shared mechanism with a
characteristic scale, not a random spike.

### 7.6 The pilot's primary endpoint does not separate the arms

`grad_survival`, all four arms:

| arm | 3.0-3.8k (pre-break) | >= 4.6k (post-break) |
|---|---|---|
| sm, clamped | 0.1785 | 0.1119 |
| sm, no clamp | 0.1810 | **0.1765** |
| ce, clamped | 0.1832 | 0.1820 |
| ce, no clamp | 0.1816 | 0.1803 |

All four sit at **~0.18 before any event**. The unclamped SM arm breaks at 4,720
and its survival is *still* 0.1765 afterwards — it breaks **without** a survival
collapse, in either direction.

**This is the central negative result.** The pilot was designed around the
hypothesis that CE's advantage would appear as retained gradient survival. At 5k
that endpoint does not discriminate the arms at all. What discriminates them is
**stability**: SM breaks, CE does not, under both weightings.

Anyone reading the earlier sections should carry that forward: the low-sigma
survival collapse measured on the production run (section 6) is real, but it is
not the mechanism behind these divergences, and it is not what separates the
objectives here.

### 7.7 Where this leaves the 98 GPU-h pilot

The pilot as specified would measure a primary endpoint that has now been shown
not to separate the arms. Before spending it, the design should change:

- **Endpoint**: time-to-divergence / stability, not gradient survival.
- **Seeds**: 3+ per arm. The whole result currently rests on 2 SM runs breaking
  and 2 CE runs not, one seed each -- suggestive, not conclusive.
- **Length**: long enough to see whether SM's post-break plateau recovers or
  compounds. Both smoke runs ended within 300 steps of the break.
- The divergence guard (now factor 4) makes a longer run cheap to attempt: a
  diverged arm aborts in ~400 steps instead of burning the remaining budget.

---

## 8. The multi-seed stability result (20k, clamp off, production-matched)

Six runs: 2 arms x 3 seeds, `cond.p_uncond=0.1`, clamp OFF, 20k step budget,
divergence guard armed. **All six diverged.** Every one aborted; none reached
20k. Actual cost ~19 GPU-h against a 58.7 GPU-h worst case, because the guard
stopped each run at the break.

| arm | seed 42 | seed 43 | seed 44 | median |
|---|---|---|---|---|
| binary_sm | 6,356 | 9,872 | 4,481 | **6,356** |
| binary_ce | 11,444 | 18,909 | 9,911 | **11,444** |

**CE survives 1.80x longer, with perfect rank separation**: max(SM)=9,872 <
min(CE)=9,911. Exact one-sided Wilcoxon rank-sum, n=3 vs 3: the SM rank sum is
6, the minimum attainable, giving **p = 1/20 = 0.05** — the smallest p this
design can produce.

### 8.1 CE is not stable, it is slower to fail

This overturns section 7. The 5k smoke's "CE stable, SM breaks" was an artefact
of stopping at 5k: **all three CE breaks fall after step 9,900.** A 5k budget
could not have seen them.

### 8.2 My guard threshold would have killed the production run

Merging *all* of production's event files (I had previously read only the last,
which covers 495k-500k) gives its full 500k history:

- loss falls to **0.0233 at ~41,680**
- rises to **~0.103 by ~83,000** and stays there for the remaining 420k steps
- lifetime max EMA/best = **6.29**, sitting at a stable ~5.4x plateau

The rise begins exactly at `entropy_warmup_steps=40000` and ramps over
`entropy_transition_steps=10000`. It is the **sigma-schedule handover changing
the loss scale** — the floor of 0.023 is measured under the initial log-normal
sigma draw and 0.103 under the entropy-adapted one — not a divergence.

So `factor=4` **would have aborted production at ~step 48,000**. That default
was wrong, and it is now **10**, bracketed by two measured populations:

| | ratio |
|---|---|
| production, 500k, trained successfully | max **6.29** |
| our six 20k runs, all diverging | **17.2** to **2942** |

10 sits between with ~1.6x margin below and ~1.7x above, and is pinned by a
regression test that replays production's measured ratio.

Consequences for what is already reported:

- **The six 20k breaks are unaffected.** Every one was at 17.2x-2942x when the
  guard stopped it, far past production's benign 6.29x. They are real
  divergences; `factor=4` merely happened to be what tripped first.
- **One earlier call was wrong.** The 5k `sm_smoke_noclamp` "break" peaked at
  only 4.75x — inside production's benign band. I over-called it. At factor=10
  only the *clamped* SM smoke arm (21.8x) diverged at 5k. The clamp nevertheless
  stays exonerated: all six 20k runs had it OFF and all six diverged.

### 8.3 Two measurement bugs found and fixed

- **The offline detector and the live guard disagreed 6/6.** An aborting run
  truncates its event file; for seed 44 TensorBoard ended **481 steps before**
  the guard fired, leaving no offline evidence at all. The summary now takes the
  live guard's message as authoritative and uses the replay only for runs that
  completed, reporting both so a disagreement can never pass silently. The
  trainer now also flushes TensorBoard before aborting.

### 8.4 The open question this exposes

**Production trained to 500k. Our runs, whose SM arm differs from it by zero
config keys, diverge before 19k.** Ruled out so far:

| hypothesis | check | result |
|---|---|---|
| corpus differs from production's | `sigma_data` fingerprint | 0.39984477 vs 0.39984480 — matches to ~8 s.f. |
| data config drift | full key diff | identical on all 22 keys |
| per-position-sigma refactor | fixed-seed fwd+loss+bwd vs `872674f` | bit-identical gradients |

Remaining candidates: code changes since production was trained (Aug 2026) other
than the sigma refactor, and the software/hardware stack. **Until this is
resolved, the SM-vs-CE comparison is internally valid — matched conditions, one
variable — but its external validity is unclear: it may be measuring which
objective better tolerates a defect that production did not have.**

### 8.5 Is a longer CE-vs-SM run justified?

**No — not yet.** Reasons:

1. A longer run measures nothing new. Both arms diverge; extending the budget
   only moves the breaks later. All six runs already ended before 19k with a 20k
   budget available.
2. The 1.80x CE advantage is at p=0.05, the floor of a 3v3 design. More seeds
   would sharpen it, but sharpening an effect measured inside a setup that
   itself contradicts the reference run is premature.
3. Section 8.4 is the blocking question. If our stack has a defect production
   lacked, "CE tolerates it 1.8x longer" is a statement about the defect.

The cheap next step is diagnostic, not another arm: bisect what changed between
the production-era code and now, by re-running the production recipe at the
production-era commit for ~10k steps (~5 GPU-h). If it is stable there and
diverges at HEAD, the cause is in our changes and is findable. If it diverges
there too, the recipe was always marginal, production got a lucky seed, and the
CE result becomes considerably more interesting.
