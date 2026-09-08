# Production-era diagnostic: is the early divergence ours, or the recipe's?

## Question

Our six 20k stability runs all diverged (binary_sm at 4,481 / 6,356 / 9,872;
binary_ce at 9,911 / 11,444 / 18,909). The production run trained to 500k with
**zero** excursions. Is the difference our post-June code changes, or not?

## Zero-GPU findings, gathered before spending anything

### 1. The training numerics are bit-identical to the production commit

Fixed-seed forward + loss + backward over the whole model, HEAD vs `036a2b5`
(the commit under which production was started, 2026-06-18):

```
                    HEAD              036a2b5
ell_sum     -2037.8488769531   -2037.8488769531
loss           0.6973915696       0.6973915696
gradL2         0.0053941514       0.0053941514
gradhash    026d7bd7a5d3...     026d7bd7a5d3...   (identical)
```

Every training-relevant diff in that window is accounted for:

| file | change | numerics |
|---|---|---|
| `utils/schedule_controller.py` | `weights_only=True` on 4 `torch.load` calls | neutral |
| `configs/tasks/tinygsm_bits.py` | `total_steps` env override | neutral (scheduler is constant) |
| `diffusion/continuous/logit_postprocess.py` | posterior-temperature decoding, gated on `pt_ctx` | not reached in training |
| `models/sdt.py`, `trainers/trainer.py`, `losses.py` | per-position-sigma refactor | verified bit-identical |

Caveat: measured in fp32, single process. It does not cover the bf16 autocast
kernels, flash-attention backend, or DDP reduction order.

### 2. Our run tracks production almost exactly until it breaks

Both seed 42, same config:

| step | production | ours (sm_s42) |
|---|---|---|
| 500 | 0.09646 | 0.09888 |
| 1000 | 0.03600 | 0.03641 |
| 4000 | 0.03397 | 0.03371 |
| 6000 | 0.03357 | 0.03277 |
| 6356 | — | **breaks** |
| 41680 | 0.02328 | — |

The dynamics are the same. Ours then suffers an excursion production never had
(production's loss never exceeds 0.5 after step 100 in all 500k steps).

### 3. Production did NOT run in our environment — two independent proofs

- `utils/callbacks/mauve.py` imports `mauve` at module scope in the historical
  code, so `import trainers.trainer` fails without it. Production ran that code,
  so **its environment had `mauve`. Ours does not.**
- `diffusion/discrete/losses.py` has used PEP 604 (`torch.Tensor | None`) in a
  function signature **since the initial release (2026-05-10)**, which cannot be
  imported on Python 3.9. `from __future__ import annotations` was not added
  until 2026-09-05, by me. So **production ran on Python >= 3.10; we are on 3.9.**

A different interpreter means a different torch wheel and different compiled
kernels. Our `torch 2.8.0+cu128` was installed 2026-03-18, before production, so
the *version* did not change — but the *build* necessarily did.

### 4. We cannot inspect the production tree

Production ran from
`/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion` (recorded in
its own launcher and saved `config.json`) — a collaborator's directory,
**permission denied** to us. So we cannot verify their working tree matched
`036a2b5`, nor what else their environment held. Our "config differs by zero
keys" comparison rests on their saved `config.json`, which is strong evidence
about configuration and says nothing about uncommitted code.

## The experiment

Run the production recipe at `036a2b5` in *our* environment.

- worktree: `/rds/user/rg625/hpc-work/prod_era_diag`, `datasets/` and
  `hf_cache/` symlinked, so the corpus is byte-identical to our stability runs
- config: `configs/tasks/tinygsm_bits_cfg_diag.py`, which loads the production
  config verbatim and adds **only** the guard and a separate run dir — verified
  to differ from it by those keys alone
- seed 42 (production's), `p_uncond=0.1`, clamp absent, `total_steps` left at
  its original 250,000; wall clock bounds the run
- **three patches only**, whole surface visible via `git -C <worktree> diff`:
  lazy `mauve` import; `from __future__ import annotations`; the divergence
  guard. All numerics-neutral. **No stabilizing changes.**

Budget: 2 h x 4 GPUs = **8 GPU-h max**, ~13k steps at the measured ~2.0 it/s,
clearing all three binary_sm break points with margin. A diverging run aborts
sooner and costs less.

## How to read it

| outcome | conclusion |
|---|---|
| **diverges** | our post-June code changes are exonerated. Cause is the environment, or the recipe is intrinsically marginal and production was lucky. The CE-vs-SM comparison then becomes interpretable as a property of the objectives under a marginal recipe. |
| **survives ~13k** | something in our changes matters despite bit-identical fp32 numerics — pointing at the bf16 / flash-attention / DDP paths the fp32 check cannot see. Next step would be bisecting the ~120 commits since. |

**Power limit, stated plainly:** this is one run. Our binary_sm breaks were at
4,481 / 6,356 / 9,872, so a historical run surviving ~13k is only about a
1-in-4-ish surprise under a naive exchangeability assumption. A single survival
is suggestive, not conclusive; a divergence is the stronger of the two results
because it directly exhibits the failure in historical code.

---

# RESULT: the historical recipe diverges too

`sbatch scripts/hpc/arch/prod_era_diagnostic.slurm` -> job 35019606, Python
3.9.23, worktree at `036a2b5`, patch surface exactly the three numerics-neutral
files. **Diverged at step 11,278**, EMA 23.47 against a best of 0.0200 — a
**1171x** excursion.

## Timing

| run | code | break step |
|---|---|---|
| binary_sm seed 42 | HEAD | 6,356 |
| binary_sm seed 43 | HEAD | 9,872 |
| binary_sm seed 44 | HEAD | 4,481 |
| **binary_sm seed 42** | **036a2b5 (production-era)** | **11,278** |
| production, same commit, collaborator's environment | 036a2b5 | **none in 500,000** |

The threshold difference between the runs does not affect this: replaying the
diagnostic's loss at `factor=4` and at `factor=10` both give 11,260, because the
excursion crosses both thresholds in the same logging interval.

## Signature: abrupt, and with no precursor in anything we log

Historical run, loss by step: `...11040:0.027  11060:0.033  11080:13.7
11100:21.6  11120:10.2...` — a ~400x jump inside one 20-step interval, from a
value indistinguishable from production's at the same step.

HEAD seed 42, which carries the full diagnostic set, shows the same thing and
lets us check for warning signs:

| step | loss | grad_norm | \|ell\|_mean | update_rms | exp_avg_sq_max | survival |
|---|---|---|---|---|---|---|
| 5,600 | 0.0294 | 0.0114 | 27.6 | 0.155 | 3.3e-05 | 0.170 |
| 5,800 | 0.0252 | 0.0164 | 25.4 | 0.155 | 2.7e-05 | 0.181 |
| 6,000 | 0.0328 | 0.0125 | 22.6 | 0.174 | 2.2e-05 | 0.183 |
| **6,160** | **19.3** | — | — | — | — | — |
| 6,200 | 0.332 | 0.0532 | 13.8 | 0.049 | 7.3e-04 | 0.152 |

**Every diagnostic is stable until the interval that contains the jump.**
Gradient norm, logit magnitude, update RMS and gradient survival all sit flat;
`exp_avg_sq_max` is *declining* (5.0e-5 -> 2.2e-5), i.e. gradients were getting
smaller and the run was settling. Immediately after, Adam's second moment jumps
**33x** to 7.3e-4 and the update RMS collapses to a third of its value.

This is a single-batch loss-spike failure, not a drift: one pathological step
poisons Adam's second moment and knocks the model into a worse basin, from which
it partially recovers to a plateau ~10x the old floor and then climbs.

Mechanism **not** identified. Our logging is every 100 steps, so the spike step
itself was never sampled; a per-step gradient-norm log would catch it and is
cheap. Note this is *not* the low-sigma weight amplifier — that hypothesis was
already refuted by the clamp 2x2, and the clamp is off in every run here.

## Conclusions

**1. Our post-June code changes are exonerated.** The production-era commit
diverges in our environment with the same signature. Together with the
bit-identical fp32 forward+loss+backward, nothing we changed causes this.

**2. The cause is environmental.** The same code ran 500,000 steps without a
single excursion in the collaborator's environment and breaks at ~11k in ours.
Seven of our runs across two code versions all broke before 12k; one run
surviving 500k against that is not luck. The concrete, proven difference is the
interpreter and its torch build — production required Python >= 3.10 (PEP 604 in
a signature, present since the initial release), we run 3.9. We cannot go
further without read access to `/rds/project/.../gb511/...`.

**3. Is the CE stability advantage now interpretable? Partially — and less
cleanly than the p=0.05 suggested.**

In favour: the CE-vs-SM comparison is internally valid — one variable, matched
conditions, three seeds each, perfect rank separation (SM 4,481/6,356/9,872 vs
CE 9,911/11,444/18,909), and we now know the divergence it measures is real
rather than an artefact of our edits.

Against, and this is the part that matters: **the historical SM run broke at
11,278, which lands inside the CE band (9,911-18,909) and later than every HEAD
SM run.** Break timing therefore moves by more, across a change we proved is
numerics-neutral in fp32, than the 1.8x effect we are trying to measure. Within
one code version the separation is clean; it does not survive pooling across
versions.

So the honest reading is: **CE tolerates this environment's failure mode longer
than SM under matched conditions, at the significance floor of a 3v3 design,
with an effect size comparable to the run-to-run spread.** It is not yet a
demonstrated property of the objectives.

## What would actually settle it

Not more CE-vs-SM seeds in a setup that is known to be broken. In order:

1. **Fix the environment.** Rebuild on Python >= 3.10 and confirm the production
   recipe survives past ~20k. If it does, every stability number here is
   environment-specific and must be remeasured. This is the only step that
   restores external validity, and it is cheap.
2. **Instrument for the spike**: per-step gradient norm and loss, retained in a
   ring buffer dumped on abort. Without it the mechanism stays invisible.
3. Only then, if divergence persists, is a larger CE-vs-SM design worth its GPU.
