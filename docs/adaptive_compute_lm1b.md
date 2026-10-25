# Adaptive compute on CoBit-S LM1B: does extra compute have state-dependent value?

*Design and decision rules written 2026-10-19 before any GPU run. Results are appended at the end.*

Implements the "CoBiT Adaptive Compute: Experimental Plan" (Experiments 1–3, offline) on the frozen released LM1B model. Nothing is trained.

Code:
- `experiments/compute_allocation/branch_sampler.py`
- `experiments/compute_allocation/run.py`
- `experiments/compute_allocation/score.py`
- `experiments/compute_allocation/analyze.py`
- launcher: `scripts/hpc/arch/compute_alloc.slurm`
- tests: `tests/test_compute_allocation_branching.py`

## Setup

**Model.** `cobit_s_lm1b_1M_ema.pt` from `gbatzolis/CoBit`, loaded with a strict key check.

**Sampler: the Table 1 γ = 0.200 operating point, as `GenerationDriver` runs it.**
- DDIM on the entropic grid: 255 grid points, i.e. 254 steps plus the final denoise, so NFE = 255.
- EDM churn: `s_churn = 0.2 × 255`, giving γ ≈ 0.2008 per step over the full entropy band; S_noise = 1.003.
- Self-conditioning `carry`, terminal σ = 0.08, bf16 autocast.

`BranchingSampler` reproduces `DDIMSampler.sample` bit for bit when the noise is drawn the production way (test).

**Noise.** A sample's starting noise and the churn noise of step i come from a generator seeded by (seed, chunk, stream, i). Two consequences:
- All branches from one saved state share the churn noise of every original step.
- Extra sub-steps never shift any later noise draw.

**Checkpoints.** The start of each of 8 segments of the step grid, at steps 0, 32, 64, 96, 128, 160, 192, 223. The grid is uniform in entropy-CDF quantile, so each segment holds an equal share of the entropy-rate mass. Checkpoint 0 is the starting noise itself: all states there have the same distribution, so it serves as a **negative control** for state heterogeneity.

**One compute unit = 32 NFE (+12.5%).** The 32 (or 31) steps of the segment after the checkpoint are each integrated in (1+k) sub-steps, log-uniform in σ. Churn is applied once per original step, with that step's γ and noise. So "+k" means k × |segment| extra NFE **on the same noise trajectory**; it is not a different trajectory. After the segment, the reference schedule runs to the end. Branches use k ∈ {0, 1, 2, 4}.

**Two continuation streams per state.** r = 0 reuses the reference's churn noise from the checkpoint on; r = 1 is fresh noise. Within a stream, the k-branches are paired.

**Quality.** Q = −(GPT-2-large NLL per token) of the final sample, scored on the full text exactly as `score_texts` does. Higher is better, and 0.01 nats/token ≈ 0.6 GenPPL at 60. Unigram entropy is recorded per sample, and Δentropy is reported next to every ΔQ.

**State features.** All computed from the first denoiser call of the checkpoint step. That call is identical in every branch, so the features are information an allocator would actually have.
- *State statistics:* bit entropy, fraction of uncertain bits and tokens, confidence mean and variance, per-token entropy (mean, max, top-10% share), and the normalised residual between the state x and the estimate D.
- *Bit-plane entropies:* one per bit position, 15 in total.
- *Dynamics over 1 and 8 steps:* entropy change and its second difference, token flip rate, and bit flip rate.
- *Not included:* hidden-state statistics, because they would need hooks inside the model.

**Funnel.**
- **Pilot:** chunk 0, 256 samples.
- **Phase 2:** chunks 1–3, for 1024 samples in total, only if the pilot is not Case A.

**Cost** (GH200, one GPU): about 13,000 sample-NFE per sample, about 3.3M per chunk.

## Pre-registered reading

### Sanity checks (must pass before anything is interpreted)
- **Anchor:** the reference samples' GenPPL (official sequential scorer) must be within 55–65 per 256-sample chunk, and within 57–63 pooled over 1024, with entropy 4.25–4.37. The paper reports 59.76 ± 0.57 at entropy 4.31.
- **Null replay:** the (k = 0, r = 0) branch replays the reference in a different batch layout. Report the fraction of samples with identical tokens and the mean |ΔQ|. This is the floor for batch-layout non-determinism.

### Experiment 1: heterogeneity (H1)
For each checkpoint s and each k, let M = Q(+k) − Q(+0) within a stream.

- **Signal variance:** Cov over samples of (M from r = 0, M from r = 1). This is the variance of the expected marginal value given the state, because the two continuations are independent given the state.
- **Noise variance:** Var(M from r = 0 − M from r = 1) / 2.
- **Uncertainty:** 2,000-resample bootstrap over samples.

**Decision at k = 1:**

| Case | Condition |
|---|---|
| **B: strong** | At some checkpoint, the bootstrap 95% CI of the signal variance excludes 0, **and** the signal SD ≥ max(\|mean M\|, 0.01 nats/token). States at the same stage then differ by at least as much as the average effect, and by a material amount. |
| **A: little** | The signal-variance CI includes 0 at every checkpoint. State-dependent allocation has no opportunity; at most a timestep schedule. **Stop after the pilot.** |
| **Weak** | Anything in between. Report it, and run Phase 2 to resolve it. |

Checkpoint 0 should come out at a signal variance of about 0. If it doesn't, the noise model is wrong, and the variance estimates elsewhere are not trusted.

### Experiment 2: predictors (H2, H3)
- **Model:** ridge regression, with features standardised inside each fold.
- **Validation:** 5-fold cross-validation grouped by sample, so all of a sample's checkpoints stay in one fold.
- **Target:** fitted on M from one stream and scored by correlation with M from the **other** stream, both directions averaged. The other stream shares none of the fitting noise. The noise ceiling for that correlation is sqrt(signal variance / variance of M), pooled.
- **Feature sets:**
  - A: timestep only (checkpoint indicators)
  - B: state statistics
  - C: B + entropy dynamics
  - D: C + prediction dynamics
  - E: D + bit-plane entropies
  - F: E + timestep
  - B + timestep
- **Also reported:**
  - *within-timestep Spearman correlation*, which measures how well each model ranks states at the same checkpoint (H2);
  - *leave-one-checkpoint-out*, comparing state features only against state features plus log σ (§9 of the plan).

### Experiment 3: offline oracle allocation (granularity, H4)
**Budgets:** B extra units per sample, summed over its 8 checkpoints. B ∈ {1, 2, 4, 8, 16}, i.e. +12.5% to +200% NFE.

**Allocators:**

| Allocator | What it chooses |
|---|---|
| uniform | the same k at every checkpoint; only possible at B = 8 and 16 |
| temporal | one k per checkpoint, the same for every sample (exhaustive search) |
| example | one k per sample, applied at all checkpoints |
| state | one k for each (sample, checkpoint) |
| predicted | as state, but from the cross-fitted ridge predictions (set F) |

The example, state and predicted choices are solved as multiple-choice knapsacks by Lagrangian bisection.

**Selection and scoring.** Allocations are chosen on one stream and scored on the other, both directions averaged, so the oracle carries no winner's-curse bias.

**Assumption.** Gains from different checkpoints of one sample are treated as **additive**. This is unverified: compute at checkpoint 2 changes the state seen at checkpoint 5. A realised run of the winning allocation is needed before any claim about real savings.

**Not measured:** per-token or per-bit allocation. That needs token-local branches; it would be the next step only if state-level allocation pays off.

### Reading the success criteria (§20 of the plan)

| Criterion | Holds if |
|---|---|
| Opportunity | Case B |
| Predictability | The within-timestep Spearman of a feature model is clearly above 0, and a sizeable fraction of the ceiling |
| Exploitability | The state or predicted allocator beats temporal at the same budget, with a gain of at least 0.005 nats/token |

## Results

*(appended after the runs)*
