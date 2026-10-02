# Lab notebook: CoBit adaptive-compute opportunity study

Spec: `PLAN.md` (user's plan, 2026-10-02). Host `csic46`, 1× RTX 4090, conda `sedd`; same
environment, deviations and launch wrapper as `experiments/compute_allocation/NOTEBOOK.md`
(eager denoiser, bf16 autocast + TF32, GPT-2-large eager attention, no Slurm). GPU cap: the
user's 6 GPU-h for the session, shared with the compute-allocation study (1.021 h used before
this study began); `adaptive.py` sums both ledgers before each cell.

## 2026-10-02 20:30 — Design (agreed with the user before starting)

Before running, I showed the user (Run 2 data, batch 0) that a single-rollout ΔQ per sample is
dominated by trajectory chaos: refine[0] mean −0.008 vs per-sample sd 0.25; nochurn[0] (zero extra
compute) sd 0.26; ΔQ correlates −0.2…−0.66 with the reference's own Q (regression to the mean).
The user said "run it". Design changes relative to PLAN.md, all in that message:

1. **Prompts = initial-noise seeds.** LM1B CoBit-S is unconditional. States = samples 0…N−1 of
   the compute-allocation Run 1 reference (seed 42), at its 8 saved checkpoints (steps
   0,32,…,224; σ 76.76 … 0.338). PLAN's "timestep" = checkpoint s.
2. **Compute unit "+c"** = c extra self-conditioning refreshes `x0_hat ← D(x, σ_next, x0_hat)` at
   every step of the 32-step window after the checkpoint (30 steps for s=7): +32c model calls
   (+30c at s=7), 12.5c% of the 255-call baseline. No change to the σ grid, the x update rule, or
   the churn noise; the state is refined at the same noise level. c=1 equals the repo's
   `refined` self-conditioning mode inside the window (unit test). Arms c ∈ {0,1,2,4}; +8 not run.
   This unit was chosen over sub-stepping because sub-stepping changes the trajectory and
   entropy (compute-allocation Run 2).
3. **Q(x_t, c) is estimated from R = 4 rollouts** per (state, arm) with fresh per-step-seeded churn
   noise after the checkpoint. Common random numbers: all arms of a (checkpoint, block) use the
   same churn seed and row layout, so arm differences are paired within each rollout.
   Q = −(GPT-2-large NLL/token) of the final sample (higher = better).
4. **Noise floor without a separate placebo arm:** rollouts {0,1} (half A) and {2,3} (half B)
   give two independent estimates of M(x_t, c) per state. True between-state variance
   τ² = Cov_n(M̂_A, M̂_B) (noise cancels in the covariance). The c = 0 arm itself is the placebo
   (zero extra compute; its rollout spread is the noise).
5. Predictors are fit on half A and scored against half B (so no shared noise / regression to the
   mean between target and evaluation).

Block layout: block k = states 200k…200k+199, rows r·200+j. Phase 1 = block 0 (200 states).
Phase 2 = blocks 1–4 added (1000 states), reusing block 0 unchanged.

## 2026-10-02 20:36–20:39 — Sampler option `sc_refresh` + tests

`segment_overrides={s: {"sc_refresh": c}}` in `DDIMSampler.sample` (non-CFG path; raises under
CFG or without self-conditioning). Tests added to
`experiments/compute_allocation/tests/test_sampler_diagnostic.py`: (i) sc_refresh=1 in every
segment == `sc_refresh_mode="refined"` bit-identically; (ii) sc_refresh=0 is a no-op; NFE =
33 + 8c; (iii) churn noise unchanged and pre-window state identical.

Found while testing: a freshly initialised tiny SDT ignores its self-conditioning input exactly
(zero-initialised layers; logit difference 0.0), which made the "has an effect" assertion fail.
The tiny-model fixture now adds N(0, 0.05²) to every weight so all input paths are live. **All 15
tests pass** (the original 12 now run on the non-degenerate model too).

## 2026-10-02 20:40 — Smoke cell `s7_c1_k0` (job `smoke`, PID 598896)

61 model calls per row (30 steps + final + 30 refreshes) ✔; 800 rows: generation 19.4 s, scoring
9.4 s. ≈ 0.32 s per model call at B = 800. Phase-1 estimate: 6,368 calls ≈ 34 min + scoring
≈ 5 min ≈ 0.67 GPU-h.

## 2026-10-02 20:41 — Phase 1 launched (job `phase1`, PID 601293): blocks 0, all s, c ∈ {0,1,2,4}

Features (`results/features.json`, CPU) from saved states (x, x0_hat = D at step 32s−1) and the
Run 1 step logs: log σ; mean / token-sd / MSB(0–4) / LSB(10–14) bit entropy of x0_hat; fraction of
uncertain bits (0.1<p<0.9) and tokens (mean bit entropy > 0.5); confidence mean/sd; ΔH, Δ²H of the
step-log entropy; fraction of token guesses changed vs 1 and 4 steps earlier; fraction stable for
8 steps; residual rms(x − x0_hat)/σ; rms(x − 0.5)/σ. At s = 0 only log σ and rms(x−0.5)/σ are
defined (no denoiser output yet). Hidden-state statistics not collected (would need model hooks).

## 2026-10-02 20:45 — Pre-registered criteria (written before any rollout result was read)

- **M̂_c(n, s)** = mean over the 4 rollouts of Q(c) − Q(0) (paired per rollout); also M̂_A, M̂_B.
  Main arm for the decision: **c = 4** (largest signal); c = 1, 2 reported. Marginal steps
  Q(1)−Q(0), Q(2)−Q(1), Q(4)−Q(2) and η = ΔQ/ΔNFE also reported.
- **Heterogeneity** per checkpoint: τ_s = sqrt(max(Cov(M̂_A, M̂_B), 0)), bootstrap 95% CI over
  states (2,000 resamples, seed 20261002); split-half correlation r_AB and Spearman–Brown
  reliability of the 4-rollout M̂.
- **Case B (strong within-timestep variation)** if, for at least one checkpoint, the CI of
  Cov(M̂_A, M̂_B) excludes 0 **and** τ_s ≥ 0.01 nats/token (the materiality unit of the previous
  study). **Case A** otherwise. Phase 2 runs only if Case B holds in phase 1, or if the phase-1 CI
  for some checkpoint has τ ≥ 0.01 but includes 0 (underpowered); otherwise stop after phase 1
  (PLAN §16).
- **Entropy confound** (secondary): report mean Δentropy per (s, c) and corr(M̂, Δentropy-hat)
  across states; τ also for an entropy-adjusted M̂ − β·ΔÊ with β from the pooled regression.
- **Predictability**: ridge (λ = 1, standardised features) fit on M̂_A, 5-fold CV by state, scored
  by Spearman ρ against M̂_B; *within-timestep* version uses targets centred per checkpoint.
  Useful if the within-timestep ρ's bootstrap 95% CI excludes 0. Report against the ceiling
  ρ(M̂_A, M̂_B).
- **Exploitability** (oracle, PLAN §11–13): allocations chosen on half A, gains evaluated on half
  B, additivity across checkpoints assumed (not tested). Meaningful if the best adaptive policy
  beats uniform by ≥ 0.01 nats/token per sample at some budget with a bootstrap CI excluding 0.

## 2026-10-02 20:41–21:19 — Phase 1 result (job `phase1`, PID 601293; block 0, 200 states)

32 cells, process time 0.63 GPU-h (ledger `results/gpu_ledger.jsonl`). Analysis
`results/analysis/heterogeneity.json` (CPU).

- **Pre-registered rule → Case B**: cells with Cov(M̂_A, M̂_B) CI excluding 0 and τ ≥ 0.01:
  s2_c4 τ 0.220 [0.026, 0.312]; s3_c2 τ 0.149 [0.043, 0.209]; s5_c1 τ 0.070 [0.007, 0.101].
  12 more cells have τ ≥ 0.01 with CIs including 0. **Caveat noted now:** 3 of 24 cells pass at
  the 95% level with lower bounds just above 0; ~1.2 would pass by chance if τ = 0 everywhere.
  Phase 2 (5× the states) is the planned remedy.
- Placebo (zero-compute) covariances ≈ 0 at every checkpoint, CIs include 0 → estimator sane.
- Mean effects (+4): s0 −0.003, s1 +0.038, **s2 +0.402, s3 +0.265**, s4 +0.066, s5 −0.035,
  s6 −0.048, s7 −0.004 nats/token. Entropy drops with the gain at s2 (−0.158 at +4) and s3
  (−0.032): the same entropy confound as the compute-allocation study. Per-state corr(M̂, ΔÊ) is
  −0.3…−0.7 everywhere; the entropy-adjusted τ for s2_c4 is 0.0, for s3 it stays 0.10–0.12.
- Final quality determined by the state: between-state sd of Q(+0) rises from 0.00 (s0, pure
  noise) to 0.08 (s1), 0.30 (s3), 0.44 (s4), 0.50 (s7) vs single-rollout sd ≈ 0.51.

**Decision (pre-registered rule): Case B → Phase 2.** Blocks 1–4 (states 200–999), same arms.
Estimated 2.5 GPU-h; session total would be ≈ 4.2 of 6 h.

## 2026-10-02 21:21 — Added while phase 2 runs (before any phase-2 result was read)

- **Oracle now reports each policy's Δentropy per sample** (from half-B entropy differences) and an
  entropy flag (|Δentropy| > 0.02, the previous study's rule). Phase-1 preview (debugging the
  figures only) showed that every high-gain policy also lowers entropy (e.g. temporal oracle at a
  25% budget with +4: Δentropy −0.17), so gains must be read with entropy alongside.
- **End-to-end confirmation runs (pre-registered here)**, `adaptive.py confirm`: one policy for all
  1024 Run-1 samples from step 0 with Run 1's chunk/churn seeds, so each sample is paired with its
  Run 1 reference (single rollout, n = 1024). Tests the oracle's additivity assumption:
  P1 = +1 in every segment (= `refined` self-conditioning everywhere, +255 NFE ≈ +100%);
  P2 = +4 in the two checkpoints the temporal oracle picks on phase-2 half A at the 2/8 budget.
  Prediction = Σ over the chosen checkpoints of the rollout mean M̂ (phase 2, 1000 states).
  Additivity is "consistent" if the measured mean ΔQ's 95% CI contains the prediction.
