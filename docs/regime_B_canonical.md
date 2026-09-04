# Regime B — the collaborator's canonical CoBit regime

Companion to [`regime_A_original.md`](regime_A_original.md). The two regimes are
**never pooled**; every result JSON now records a `regime` field
(`REGIME_A_ORIGINAL` / `REGIME_B_CANONICAL`), pinned by
`test_both_result_paths_record_the_regime`.

---

## 1. Canonical configuration

Taken from the collaborator's own recorded cell,
`results/gsm8k_passk_shardA/cobit.json`, not from prose.

| Setting | Value |
|---|---|
| checkpoint | `tinigsm_gsm8k/runs/cobit_raw_binary_bits/checkpoints/step=000425000.pt` (**base** run, 425k) |
| sampler | `fkc_em` (`FeynmanKacEulerMaruyamaSampler`) |
| proposal | `edm_churn`, `churn_gamma = 0.41` |
| steps | 1024 |
| β | 1.0 |
| particles | 32, `resampling_policy=never`, `final_resample=0` |
| seed | 42 |
| schedule | entropic |
| `sigma_data` | 0.399844765663147 (from the run's `sigma_data.json`) |
| EMA | enabled |
| precision | fp32 |
| problems | 256 random, shard A (**unavailable on this cluster**) |

**Canonical reference value.** Our full-test-set reproduction:
**0.2813 [0.2570, 0.3055]** on 1319 problems (`replication_audit` cell A).
Their reported 0.2909 on 256 shard-A problems lies inside that interval. This
is the stable anchor; no further compute is spent chasing numerical identity
across different test sets.

**Precision note (resolved).** `_run_fkc_gsm8k` has **no autocast wrapper** — the
FKC path never enters `sample_bits`, which is where bf16 autocast is applied. So
the canonical arm already ran fp32, matching the collaborator, and `--fp32` is a
no-op on that path. One fewer deviation than the audit assumed.

---

## 2. Blocker: the canonical sampler cannot host the guidance study

Established by reading `FeynmanKacEulerMaruyamaSampler` and its
`sample_particles()` signature.

**(a) AutoGuidance and Self-Guidance do not exist under FKC-EM.**
`sample_particles()` accepts no `bad_model`, no `sg_scale`, no `sg_variant`.
Items 3–8 of the canonical guidance screen are **unrunnable without new
implementation**, not merely untested.

**(b) `guidance_scale` is a different mathematical object in each regime.**

| Regime | Meaning of `w` |
|---|---|
| A (DDIM) | linear CFG: `D_u + w(D_c − D_u)` |
| B (FKC) | exponent of a **geometric** average target `q_u^(1−w) q_c^w` (Prop 3.1) |

Under FKC, `w` is an interpolation exponent naturally in [0,1] (w=1 ⇒ pure
conditional). Passing w=12 there is a wild extrapolation, not "strong CFG".
**Comparing "CFG at w" across regimes would compare different operations**, so a
naive port of the Regime A scale sweep is meaningless.

**(c) FKC refuses EDM churn.** `_validate_fkc_settings` raises if
`cfg.evaluation.stochastic.enabled` — "Stochasticity is owned by lambda_zero /
the LambdaProfile". There is therefore **no `--gamma` axis inside FKC**; the γ
sweep must go through `churn_gamma`, and a γ=0 condition inside FKC is not
reachable by the same code path.

---

## 3. Consequence for the plan, and the control that decides it

The guidance study needs a sampler where CFG, AG and SG all exist with the
**same semantics as Regime A** — otherwise cross-regime effect sizes are not
comparable, which is the entire point of the exercise. DDIM + EDM churn is that
sampler.

Substituting it is legitimate **only if the two churn implementations behave
equivalently**. That is what `regime_b_control` measures. This is evidenced
substitution, not the silent substitution the brief forbids.

| Cell | Sampler | Churn | Precision | Purpose |
|---|---|---|---|---|
| *(exists)* | fkc_em | `churn_gamma=0.41` | fp32 | canonical anchor, 0.2813 |
| `rB_ddim_churn041_fp32` | ddim | `gamma=0.41` | fp32 | **the control** |
| `rB_ddim_churn041_bf16` | ddim | `gamma=0.41` | bf16 | precision *under stochasticity* — the audit's null was measured at γ=0 only |

All matched: base 425k, 1024 steps, seed 42, entropic, same `sigma_data`, 1319
problems. ~2.2 GPU-h.

```bash
EXPORTS="ALL,GUID_CFG_W=12,GUID_AG_W=15,GUID_SG_W=2,GUID_BAD_STEP=350000"
sbatch --array=0-1 --export="$EXPORTS,GRID=regime_b_control" \
       scripts/hpc/guidance/array.slurm
```

**Decision rule.**
* If DDIM+churn ≈ FKC-EM (difference within noise) → run the Regime B guidance
  study on DDIM+churn at the canonical checkpoint/steps, anchored to the FKC
  baseline, and document the substitution with this evidence.
* If they differ materially → the difference is itself a finding, and AG/SG must
  be implemented for FKC before the guidance study can proceed. That is a
  substantially larger piece of work and would be scoped separately.

---

## 4. Open, pending the control

Not yet answered, and deliberately not guessed:

* whether the AG and SG sign reversals seen in Regime A reproduce under the
  canonical regime;
* what `w*_CFG(canonical)` is — and, given (b) above, whether the question is
  even well-posed for FKC's geometric-average `w`;
* whether SG's deterministic finite-difference construction is valid on a
  stochastic FKC trajectory at all. Under `edm_churn` the previous state has
  been re-noised, which is the mechanism Regime A's evidence already implicates
  (harm linear in scale, no scale rescues it).
