# Branch 2 — temporal ordering

**Assessment only. No code changed, no GPU spent.**

---

## 1. How ordering is currently implemented: it isn't

CoBit has **no temporal ordering to vary.** The evidence is in three places and
they agree:

| Location | Fact |
|---|---|
| `ContinuousForwardProcess.sample_sigma(bsz)` | returns shape **`[B]`** — one σ per *example* |
| `trainers/trainer.py` | `t = (1-eps)·rand(B)+eps`; `sigma = get_cumulative_noise(t)` — **`[B]`**, no position axis |
| `models/sdt.py::_SinTimeSigma.forward` | `sigma.log()[:, None] * freq` — **one embedding per example**, broadcast to all positions |

Every one of the 8192 bits (512 tokens × 16) is denoised at **the same noise
level, simultaneously, at every step.** The model is order-agnostic by
construction. There is no left-to-right, no right-to-left, and no random order —
not as a default, but as a structural property.

The position-dependent machinery that *does* exist (RoPE, `abs_pos_mode`,
`n_pos_features`) tells the model *where* a bit is. It does not tell it *when*
that bit is resolved. Those are different things, and only the second is
"temporal ordering".

## 2. Can L→R / R→L / random be tested with the existing architecture?

**No.** There is no ordering variable to set, so this cannot be done with a
sampler flag, a config switch, or a checkpoint swap.

Testing it requires making the noise level **per position**: `σ[B, S]` instead of
`σ[B]`. Ordering is then expressed as a *schedule shape* — positions whose σ
falls first are committed first. Left-to-right is a σ ramp increasing with
position; right-to-left is the reverse; random is a per-example permutation.

That is the AR-Diffusion / Diffusion-Forcing construction, and it touches:

1. **forward process** — `sample_sigma` and the noising must accept `[B, S]`;
2. **σ conditioning** — `_SinTimeSigma` currently maps `[B] → [B, d]`; it needs
   `[B, S] → [B, S, d]`, which changes how the embedding is injected into every
   block;
3. **the loss weighting** — `_sigma_weight` assumes a scalar per example;
4. **the sampler** — the σ schedule becomes a 2-D trajectory, and the entropic
   schedule machinery assumes a single σ per step;
5. **training** — a model trained on scalar σ has never seen a mixed-σ state, so
   every ordering condition needs its own training run.

**This is a training-time architectural change, not a sampler experiment.** The
brief anticipated this possibility; it is the case.

## 3. What this means for the branch

The question "does CoBit depend on temporal ordering?" cannot be asked of the
current model. The honest reformulation is:

> Does *introducing* a temporal ordering — which CoBit currently does not have —
> improve it over simultaneous denoising?

That is a different and more expensive question, because simultaneous denoising
is the control arm and each ordering is a separate training run. It also makes
the comparison inherently 4-armed: simultaneous (control), L→R, R→L, random.

There is a cheaper intermediate worth noting: **per-position σ with a
*uniform* schedule** is mathematically equivalent to the current model, so it
can be used to validate the refactor is behaviour-preserving before any ordering
is imposed — the same no-op discipline the trajectory instrumentation used.

## 4. Smallest valid pilot

Staged, and stage 0 costs nothing:

| Stage | Content | GPU |
|---|---|---|
| 0 | Refactor to per-position σ; assert bit-identical outputs under a uniform schedule | **0** (unit tests) |
| 1 | Train **control** (uniform σ) at reduced budget — establishes the short-run reference | 1 arm |
| 2 | Train **L→R** and **random** at the same budget | 2 arms |
| 3 | R→L only if L→R and random differ from control | 1 arm |

R→L is deliberately deferred: if ordering has no effect, L→R vs random already
shows it, and R→L costs a full run to confirm a null. If ordering *does* matter,
R→L becomes the informative asymmetry test and is worth its cost then.

**Cost: 3 training arms for a first answer, 4 if the effect is real.** Same
per-arm cost as Branch 1, so the same throughput measurement serves both.

## 5. Recommendation on ordering between the two branches

**Branch 1 first.** Three reasons:

1. Branch 1's pilot is *one config line* against a code path that already exists
   and is already tested. Branch 2 needs a σ-conditioning refactor through the
   model, loss, sampler and schedule before a single GPU-hour is useful.
2. Branch 1 has a specific mechanism and a falsifiable prediction (the `D(1−D)`
   gradient suppression, against a measured 80 % saturation). Branch 2 currently
   has a hypothesis with no CoBit-specific evidence behind it.
3. Branch 1 is 2 arms; Branch 2 is 3–4 arms plus a refactor.

If Branch 1 changes the saturation profile materially, that also changes the
premise of Branch 2 — a model that commits bits differently may respond
differently to an imposed commitment order — so running them in this order
avoids having to redo the ordering study.
