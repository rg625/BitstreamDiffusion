# Temporal ordering — design note (per-position σ)

Design only. No code changed. Companion to `docs/temporal_ordering.md`, which
established that **CoBit has no temporal ordering to vary**.

---

## 1–6. Where the single-σ assumption lives

Six places, all of which assume σ is one scalar per example:

| # | Location | Current | Shape |
|---|---|---|---|
| 1 | **σ sampled** | `ContinuousForwardProcess.sample_sigma(bsz)`; trainer draws `t = (1-eps)·rand(B)+eps` | `[B]` |
| 2 | **σ broadcast** | `_sigma_weight`: `sigma.view(-1, 1, 1)` | `[B,1,1]` |
| 3 | **model conditioning** | `_SinTimeSigma.forward`: `sigma.log()[:, None] * freq` → one embedding per example, added once | `[B] → [B,E]` |
| 4 | **forward noising** | `xt = proc.sample_xt(x0, t)`, scalar σ per row | `[B]` |
| 5 | **loss weighting** | `_sigma_weight(cfg, sigma, ndim=3)` → EDM weight `(σ²+σ_d²)/(σ²σ_d²)` | `[B,1,1]` |
| 6 | **sampler** | one σ per step for the whole batch; the entropic schedule is a 1-D sequence | scalar/step |

Note (3) is the substantive one. The σ embedding is produced **once per example**
and injected as a global conditioning vector; positions are distinguished only by
RoPE, which encodes *where*, never *when*.

## 7. Cleanest way to introduce per-position σ

Promote σ from `[B]` to `[B, S_tok]` (per **token**, not per bit — all 16 bits of
a token must resolve together or the codeword is meaningless), then broadcast to
bits at the last moment.

Minimal-diff plan, in dependency order:

1. `sample_sigma` gains an optional `positions` argument; default path returns
   `[B]` **unchanged**.
2. `_SinTimeSigma` accepts `[B]` **or** `[B,S]` and returns `[B,E]` or `[B,S,E]`.
   Injection becomes a per-position add when 2-D. This is the only model change.
3. `_sigma_weight` reshapes on the trailing dims rather than `view(-1,1,1)`.
4. `sample_xt` broadcasts token-σ to bit-σ via `repeat_interleave(bits_per_token)`.
5. Sampler: the schedule becomes `[steps, S_tok]`; the current 1-D schedule is
   the special case where every column is identical.

Every step keeps the `[B]` path as the default branch, so the uniform schedule is
**the same code path with a broadcast**, which is what makes the equivalence test
meaningful rather than a re-implementation compared against itself.

## 8–10. Defining an ordering precisely

An ordering is a **per-token offset of the noise schedule**. Let `u_j ∈ [0,1]` be
the normalised rank of token `j` in the chosen order (0 = resolves first), `t` the
global time, and `w ≥ 0` the *ordering strength*:

```
t_j(t) = clip( t·(1+w) − w·u_j , 0, 1 )        σ_j = σ(t_j)
```

* `w = 0` → `t_j = t` for all j → **exactly the current model**. This is the
  equivalence case and it falls out of the definition, not a special case bolted on.
* `w > 0` → tokens with small `u_j` reach low σ first, i.e. **resolve earlier**.

**Convention, stated explicitly: lower `u_j` ⇒ lower σ earlier ⇒ resolves first.**

* **L→R**: `u_j = j/(S−1)`. Token 0 resolves first. Prompt tokens are clamped
  clean already, so `u` should be ranked over the **suffix only**, or the prompt
  consumes half the ordering range and L→R silently becomes "prompt-first",
  which is vacuous.
* **Random**: `u_j = π(j)/(S−1)` for a permutation π of the suffix positions.
  **Resampled per example per step**, not fixed per example. Fixing it per example
  would make the permutation a learnable property of that example — the model
  could memorise which token comes first — which tests something else entirely.
  Reproducibility comes from seeding the generator on `(global_seed, step, row)`.
* This is an ordering of *resolution times*, not random σ per token. Under a fixed
  permutation the σ field is monotone in rank, which is what distinguishes it from
  noise.

## 11. R→L: defer

`u_j = 1 − j/(S−1)`, trivial to add. Deferred because if simultaneous / L→R /
random show no ordering effect, R→L is a third training run to confirm a null. If
ordering *does* matter, R→L becomes the informative asymmetry test and is worth
its cost then.

## Validation gate — before any ordering run

The refactor must be **numerically identical to the current implementation at
`w = 0`**, since that is mathematically the same model. Tests to add:

| Test | Assertion |
|---|---|
| σ sampling | `w=0` per-position σ equals `[B]` σ broadcast, exactly |
| σ embedding | `_SinTimeSigma([B])` equals `_SinTimeSigma([B,S])[:,0]` when columns are identical |
| forward noising | `sample_xt` bit-identical under a uniform schedule, same RNG draw |
| loss | both arms bit-identical at `w=0` |
| masking | prompt positions excluded identically |
| gradients | `torch.allclose` on parameter grads at `w=0` |
| checkpoint | an existing checkpoint loads and evaluates unchanged |
| RNG | the permutation generator does not perturb the noise stream |

The trajectory instrumentation used exactly this discipline and it caught nothing
— which is the point: it made "no change" checkable rather than asserted.

## Cost

Three arms (control, L→R, random) at the steady-state rate of **0.489 GPU-h/1k
steps**: ~36 GPU-h at 25k, ~73 at 50k, ~147 at 100k. R→L would add a third of that
again. Infrastructure and tests are zero GPU.
