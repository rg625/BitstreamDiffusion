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
