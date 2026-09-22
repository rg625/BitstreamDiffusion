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


---

## Learning rate: what 3e-5 does and does not establish

The from-scratch validation runs `binary_sm` at **lr = 3e-5**, not production's
3e-4, because 3e-4 diverges in **8/8** from-scratch runs measured in this
environment -- including the production-era commit `036a2b5` itself -- while
3e-5 carried all four ordering arms to 15k with zero divergence, where the
3e-4 control had already broken at step 2,323.

**What a surviving 20k run would establish:** that a *stable training regime
exists* in this environment.

**What it would NOT establish:** that 3e-5 is the *right* learning rate for the
objective comparison. It is 10x below the value the production recipe was tuned
around, and no tuning has been done at it.

**Consequence for CE vs SM.** The comparison remains valid, because the two arms
are perfectly matched -- identical init, seed, data, optimiser, EDM weighting,
`p_uncond=0.1`, precision and step budget, with `train.loss_type` the only
difference. But:

- the learning rate is **reported explicitly** on every result;
- absolute accuracies are compared **only against the matched control in the
  same regime**, never against production's 3e-4 trajectory (0.2024 at 200k,
  0.164 at 500k under the karras/DDIM setting). Those numbers come from a
  different optimisation regime and putting them in the same column would
  invite exactly the wrong comparison;
- a lower absolute accuracy at a given step count than production is
  **expected** at a 10x smaller learning rate and is not evidence about either
  objective.

---

## Experiment E2 — from-scratch stability at lr 3e-5. **THE BLOCKER IS SOLVED.**

`binary_sm`, from scratch (`init_from=None`), production-matched
(`p_uncond=0.1`, clamp off, EDM weighting, bf16, batch 512), seed 42,
Python 3.10, **lr 3e-5**: **reached target 20,000 steps with no divergence.**
Final loss 0.0334, val 0.0558.

| recipe | lr | seed | outcome |
|---|---|---|---|
| production-matched, py3.9 | 3e-4 | 42 | diverged @ 6,356 |
| production-matched, py3.10 | 3e-4 | 42 | diverged @ 5,346 |
| production-era commit `036a2b5` | 3e-4 | 42 | diverged @ 11,278 |
| **production-matched, py3.10** | **3e-5** | **42** | **20,000 steps, stable** |

**The learning rate is the identified factor, not the interpreter.** 8/8
from-scratch runs at 3e-4 diverged in this environment, including the
production-era commit itself; every run at 3e-5 has been stable (this, plus all
four ordering arms).

This also closes the environment question in a way the Python 3.10 test could
not. That test falsified the interpreter hypothesis but left the cause open;
the answer is that the recipe is simply unstable at 3e-4 here. Why production
tolerated 3e-4 for 500k steps remains unexplained -- their working tree is
unreadable -- but it is no longer blocking.

### Definitive from-scratch objective comparison — launched

50,000 steps per arm, verified to differ in exactly four keys (`train.loss_type`
plus three output paths):

```
lr=3e-05  seed=42  steps=50000  p_uncond=0.1  clamp=None
weighting=edm  init_from=None  amp=bf16  batch=512
```

Per the recorded caveat: 3e-5 establishes a **stable** regime, not a **tuned**
one. Absolute accuracies are compared only against the matched control in this
regime, never against production's 3e-4 trajectory.

## The epoch-boundary deadlock: it is a collective MISMATCH, and it is deterministic

Tue 15 Sep 2026. Reading the four crash logs side by side (jobs 35542109,
35512411, 35259460, 35423642 — three ordering arms plus a repeat) changes what
this hang is.

**The ranks stop in different places.**

| rank | SeqNum | op | NumelIn |
|------|--------|----|---------|
| 0 | 2516974 | ALLREDUCE | 1,572,864 (a gradient bucket) |
| 1, 2, 3 | 2516971 | ALLREDUCE | 1 (a scalar / barrier) |

Ranks 1–3 report `last completed work: 2516970`, so every rank finished the same
2,516,970 collectives and then diverged: rank 0 went on into the next epoch's
backward pass while ranks 1–3 issued one more scalar collective that rank 0
never issued. That is a **desynchronisation of the collective sequence**, not a
collective that is merely slow — which is why raising the process-group timeout
to 120 min did nothing (the 7,200,013 ms in the log is the full 120 min), and
why disabling the per-epoch checkpoint saves did nothing either. Rank 0 is not
stuck writing a checkpoint; it is stuck in training, waiting for three ranks
that are stuck at the boundary.

**The numbers are identical across jobs and arms.** 2516971 / 2516974 appear in
all four logs - different arms, different nodes, different days. This hang is
deterministic, not flaky.

**It is FIVE epochs, not one** (this corrects the reading above). Every job
completes exactly five epochs and hangs entering the sixth, wherever it resumed:

| job | resumed at epoch | hung entering | completed |
|-----|------------------|---------------|-----------|
| 35512411 | 0 | 5 | 5 |
| 35512408, 35512412 | 5 | 10 | 5 |
| 35542109, 35423642 | 15 | 20 | 5 |

The arithmetic closes: 5 x 22,858 steps = 114,290 steps, and
2,516,971 / 114,290 = 22.0 collectives per optimiser step. Twenty-two buckets
for 133.76M parameters is 6.1M parameters each, 24 MB in fp32 - DDP's default
`bucket_cap_mb=25`. So the collective count, the epoch count and the step count
are one consistent picture, and the "every ~4 hours" in the handoff was really
every ~20 hours: one hang per link, not several.

**Three hypotheses remain confounded** by these logs, because a resume always
restarts at an epoch boundary and every arm runs at the same steps/s: the
trigger could be 5 epochs, or ~2.5M collectives, or ~20 h of process lifetime.
`deadlock_repro.slurm` separates them cheaply and decisively: eight TRUNCATED
epochs is eight epoch boundaries but only ~9k collectives and ten minutes of
runtime. If it hangs, the trigger is the epoch boundary and the repro is now a
ten-minute loop. If it does not, the boundary is innocent and the trigger is
cumulative - a leak or a counter - which is a different and much more specific
bug to hunt.

**What is ruled out by reading the code.** The end-of-epoch collectives are
`_validate_epoch`'s two scalar all-reduces plus the barrier at
`trainers/trainer.py:2396`, all on every rank.
`_maybe_save_resume_ckpt`/`_maybe_save_interval_ckpt` are master-only and
collective-free. The two callbacks actually active in this config
(`SigmaDataEstimator`, `EntropySchedulePlotCallback`) are master-only and
collective-free; the one callback that does call a bare `_barrier()` inside a
conditional (`utils/callbacks/visualization.py:739`) is not in this config's
callback list. `find_unused_parameters=False`, so DDP itself adds no
data-dependent collective. The remaining suspects — `rank0_first()` in
`data/dist_build.py`, which barriers on non-builder ranks only, and the
`dist.broadcast` path in `utils/schedule_controller.py` — need the call site to
be confirmed rather than argued.

**Instrumentation, which is what was missing.** Both mechanisms are free until a
run stalls (`scripts/hpc/arch/deadlock_debug.sh`, sourced by
`ordering_train.slurm` and `token_train.slurm`):

* NCCL flight recorder (`TORCH_NCCL_DUMP_ON_TIMEOUT`) — on timeout each rank
  writes its last collectives with the Python frames that issued them. Decode
  with `experiments/ordering/decode_nccl_trace.py`; line the ranks up by
  `seq_id` and read off the first entry where they stop matching.
* SIGUSR1 stack dump — the launcher watches its own log and, once it stops
  growing for `STALL_MIN` minutes, signals every rank so each writes all its
  Python stacks to `logs/arch/stack_<jobid>_rank<R>.txt` *before* the process
  group times out.

With the recorder in place, `DDP_TIMEOUT_MIN` goes back to 20: a long timeout
now only burns the rest of the reservation.

**And it no longer costs 4 h to reproduce.** `COBIT_STEPS_PER_EPOCH` caps the
epoch at a batch count — identical on every rank, so the ranks reach the
boundary in lockstep exactly as they do naturally — and
`scripts/hpc/arch/deadlock_repro.slurm` drives eight boundaries in a 40-minute
throwaway job that writes no checkpoints. A clean pass would itself be
informative: it would mean the boundary alone is not sufficient and the trigger
needs the epoch's length.

Note for whoever submits: SLURM copies a batch script at submission time, so the
chain links already queued (35588706, 35588707, 35542110, 35512409/10/13/14)
run the **old** launcher and carry none of this. Only newly submitted jobs are
instrumented.

## V-way token run relaunched at effective batch 512 (gradient accumulation)

Wed 16 Sep 2026. The first V-way run answered its question — 0.0255 against a
binary control of 0.0493, a matched comparison — but both arms sat at an
effective batch of 128, a quarter of production's 512, so the answer was
measured at a weak operating point. The batch was not a choice: the token model
materialises a [B, 512, 49153] logits tensor and OOMs at 128 examples/GPU (job
35223498 asked for 12.00 GiB with 3.44 GiB free on an 80 GB A100).

**Gradient accumulation, not more GPUs.** `cfg.train.grad_accum_steps` splits one
optimiser step into N forward passes of `batch_size / (world_size * N)` each.
At 512 with N=4 on four GPUs that is 32 examples/GPU — exactly the per-GPU load
the 128-batch run already proved fits — for four times the effective batch.
`cfg.train.batch_size` keeps its meaning as the **effective** batch (the examples
behind one optimiser step, the number that has to match across arms), and
`global_step` stays an optimiser-step count, so "500k steps" means the same
thing here as in the binary arms.

DDP's all-reduce is deliberately *not* suppressed with `no_sync()` on the
non-final micro-batches. Autograd accumulates into `.grad` before DDP's hook
fires, so each micro-backward all-reduces the running accumulated gradient, and
averaging an already-averaged value across ranks is idempotent — the result is
identical. The cost is (N−1) extra all-reduces per step, a few percent inside
one NVLink node; the benefit is not depending on `no_sync()` interacting
correctly with torch.compile's DDPOptimizer, which rewrites the backward graph
and is exactly the kind of never-exercised path this project has been bitten by
before. `tests/test_grad_accumulation.py` checks the claim numerically
(accumulated gradient == full-batch gradient to 1e-6, and the 1/N scaling is
guarded by a test that shows its absence is wrong by a factor of N).

**The arm it is matched to** is `ord_fs500k_none_s42`: binary bits, from
scratch, 500k steps, batch 512, lr 1e-4, seed 42, p_uncond 0.1 — verified
identical except for the head. Not production's 0.164, which was trained at
lr 3e-4.

**On the hope that a bigger batch fixes the LR instability: the evidence says
no.** `runs/tasks/tinygsm/obj_binary_sm_stab_s42/config.json` records
`lr 3e-4, batch 512`, and that run tripped the divergence guard at step 6,356.
The 8/8 divergences were *already* at batch 512, so they are not a small-batch
artifact and 512 is not a reason to expect 3e-4 to become trainable. Testing a
higher LR at this batch is a legitimate separate arm; it is not something this
run establishes, and changing both batch and LR at once would break the match
with the control.

**Disk.** 47 GB reclaimed before launching: 34 GB of checkpoints from the
falsified-hypothesis probes (`obj_binary_{ce,sm}_stab_*`, `obj_binary_sm_py310_s42`,
`probe_binary_{sm,ce}`) and 13 GB of provably exact duplicates (`step=NNN.pt`
files whose `global_step` equals `last.pt`'s, and `epoch=*.pt` files identical to
`best.pt`). Configs, eval JSONs and TB logs were kept in every case; only `.pt`
weights were removed. Free space 95 GB -> 142 GB.

### Smoke result (job 35629979, 300 steps)

Accumulation works and fits. `[accum] effective batch 512 = 32/GPU x 4 GPUs x 4
accumulation steps`, 212.27M params, `batch_size=512 (Global)`, peak 4.81 GiB
allocated / 30.86 GiB reserved on an 80 GB card, no OOM, clean exit.

The loss tracks the known-good batch-128 run at matched OPTIMISER steps, which
is the check that matters -- accumulation must not change what a step is:

| step | b512 accum 4 | b128 (previous run) |
|------|--------------|---------------------|
| 50   | 44.9 | 39.6 |
| 100  | 44.5 | 55.3 |
| 150  | 41.4 | 49.4 |
| 200  | 43.6 | 40.9 |
| 250  | 20.5 | 22.0 |
| 300  |  9.3 |  6.7 |

Two things the smoke changed in the plan. Throughput is ~2.0-2.5 micro-batches/s
= **~0.5-0.6 optimiser steps/s**, so 500k steps is ~230-275 h: the chain needs
36 h links (token_train.slurm's own header asks for 8 h) and nine of them. And
the 5,000-step resume cadence would put ~2.5 h between last.pt writes, all of
which is lost when a link is killed at its wall limit, so the token run writes
last.pt every 1,000 steps instead.

One cosmetic artefact: an epoch is 91,435 micro-batches, which is not divisible
by 4, so the last 3 micro-batches of each epoch are computed and then discarded
by the next group's `zero_grad`. That is 0.003% of the work and no partial
optimiser step is ever taken.

### The deadlock hit r2l and random again overnight

Both arms died at the epoch-10 boundary (step 228,580 = 10 x 22,858) at 03:28
and 03:36 on 16 Sep, same signature - after five completed epochs each, as every other
occurrence has been. Neither carried the instrumentation: SLURM
copies a batch script at submission time and both jobs, and both of their queued
successors, were submitted before it existed. They also still carry
DDP_TIMEOUT_MIN=120, so each hang burns two hours before the job even dies.

## 18 Sep: the boundary is innocent, and two more hypotheses are dead

**The repro ran clean.** Job 35635430: eight truncated epoch boundaries, 400
steps, `rc=0`, "NO HANG". So the epoch-end code path by itself does not
deadlock. Whatever the trigger is, it is cumulative, and the boundary is only
where it surfaces.

**The token run kills two more candidates, for free.** Job 35635321 has run 23 h
in a single process and reached 51,000 optimiser steps without hanging. At
accumulation 4 and ~34 buckets for 212M parameters that is roughly 7M
collectives -- and even on the ordering model's 22 buckets it is 4.5M. The
ordering arms hang at 2,516,971 collectives and ~20 h. So:

| candidate | status |
|-----------|--------|
| epoch boundary itself | **dead** -- 8 boundaries passed clean |
| ~2.5M collectives | **dead** -- token passed 4.5-7M |
| ~20 h of process lifetime | **dead** -- token ran 23 h |
| ~114k steps / batches per process | survives |
| 5 full-length epochs | survives |

r2l and random did it again on 17-18 Sep: 228,580 -> 342,870 is 114,290 steps,
five epochs, third occurrence at exactly that figure.

Note that both surviving candidates predict the token run will NOT hang: at
0.616 steps/s a 36 h link covers ~80,000 steps, short of 114,290, so its counter
resets at every link. It is no longer a discriminator.

**The instrumentation had a bug and captured nothing.** The stall watchdog
matched "train.py" in the command line to find the ranks. The launcher's own
command line contains "train.py" as an argument, and SIGUSR1 *terminates* a
process with no handler -- so it signalled the torchrun agent, which died
(rc=138) and took the job down at 15 minutes, while the four real ranks (visible
as 3969665-3969668 in that same log's nvidia-smi output) were never signalled.
Every stack file came out 0 bytes and no flight-recorder dump was written,
because the job died before the process group could time out.

Fixed: each rank writes `logs/arch/pid_<job>_rank<R>.txt` as it installs its
handler, and the watchdog signals those pids and nothing else -- a pid file
exists only for a process that can survive the signal. With none published it
signals nothing rather than guessing. `tests/test_deadlock_instrumentation.py`
now starts a real process, checks the helper finds exactly its pid, signals it,
and asserts it *survives* with a non-empty dump.

This lands on the already-queued links automatically: SLURM copies the .slurm
file at submission, but `deadlock_debug.sh` and `train.py` are read at run time.
No resubmission needed.

## 20 Sep: ordering answered at 500k, and the deadlock caught in the act

**Ordering, 500k, full 1319, karras, paired bootstrap vs the matched control.**
Anchor in band (0.1319 / 0.1410 / 0.1228, expected 0.12-0.20), so the decode
path is verified.

| arm | decode | seed | acc | delta vs control | 95% CI | p |
|-----|--------|------|-----|------------------|--------|---|
| control (none) | uniform | 0 | 0.1266 | - | - | - |
| control (none) | uniform | 1 | 0.1342 | - | - | - |
| l2r w=0.25 | matched | 0 | 0.1213 | -0.0053 | [-0.0220, +0.0121] | 0.59 |
| l2r w=0.25 | matched | 1 | 0.1236 | -0.0106 | [-0.0281, +0.0061] | 0.24 |
| l2r w=0.25 | uniform | 0 | 0.0387 | **-0.0879** | [-0.1069, -0.0697] | <1e-4 |
| l2r w=0.25 | uniform | 1 | 0.0425 | **-0.0917** | [-0.1099, -0.0735] | <1e-4 |
| prod anchor | uniform | 0 | 0.1319 | +0.0053 | [-0.0129, +0.0227] | 0.59 |
| prod anchor | uniform | 1 | 0.1410 | +0.0068 | [-0.0114, +0.0250] | 0.49 |

Three results, in order of how much they change the picture:

1. **Training-time L2R ordering does nothing.** 0/2 seeds significant, both
   deltas slightly negative. Consistent with the measured entropy profile: after
   an 11-token fixed prefix, TinyGSM answers are flat at 7.5-8.0 bits with no
   gradient for a directional schedule to exploit.
2. **An ordering-trained model is captive to its decode schedule.** Decode the
   same weights uniformly and accuracy falls by ~70% relative (T1 rises from
   ~933 to ~1180: the programs stop executing). That is a constraint the
   intervention imposes, not a benefit it confers.
3. **lr 1e-4 from scratch matches production.** The control is statistically
   indistinguishable from the anchor. The standing "stable, not tuned" caveat
   can be dropped at 500k: this regime reaches production accuracy.

Seed 2 of the control is missing -- `str(_pred)` raised on a generated program
returning an integer over 4,300 digits, killing the cell after 768 of 1,319
problems. Fixed (`_answer_to_str`, recorded as `<overflow>`); seed 2 needs a
re-run for the third pair.

**The deadlock, finally observed.** The token run hung (it is not immune after
all) and the repaired watchdog found all four ranks. SIGUSR1 still produced
empty dumps -- the ranks are wedged inside CUDA/NCCL where the signal is not
delivered -- but the NCCL flight recorder wrote 1.2 MB per rank, which is what
answered it:

* ranks 1-3: exactly **one** pending collective, `all_reduce_barrier` at
  `trainers/trainer.py:2516`, the end-of-epoch barrier, with Python frames.
* rank 0: **22** pending collectives, all gradient buckets from the *next*
  epoch's training, and its own barrier already `completed`.

Rank 0's barrier completed while ranks 1-3's is still pending, which is only
possible if the ranks' collective streams are **offset**: NCCL matches by issue
order, so rank 0's barrier was matched against a validation all-reduce on the
others. The run was therefore completing mismatched collectives -- reducing the
wrong tensors, silently -- before it wedged.

The offset is ~3 collectives, exactly the epoch boundary's own count (two
`_validate_epoch` all-reduces plus the barrier). The one code path that skips
precisely those three is the `num_train_batches == 0` early `break` at
trainer.py:2474, which returns to the top of the epoch loop without validating
or barriering.

The 8192-entry buffer covers only ~1,170 micro-batches, about 0.3% of an epoch,
so the origin is outside the window and a bigger buffer is not the answer
(~640k collectives per epoch). The cheap instrument is an epoch-boundary
agreement check: all-gather each rank's `num_train_batches` and abort loudly on
disagreement, turning a silent 20-hour deadlock into an immediate, informative
crash.
