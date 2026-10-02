# Lab notebook: CoBit-S LM1B compute-allocation diagnostic

Times are local (Europe/London, UTC+01:00). Section refs (§N) are to `PLAN.md`.

## 2026-10-02 17:00 — Session start, environment decision

- Branch `compute-allocation-diagnostic` created from `main` @ `7c93d1d`.
- The task brief assumed Isambard (Slurm, GH200, aarch64). The session actually runs on
  host `csic46`: x86_64, 1x NVIDIA GeForce RTX 4090 (24 GB), no Slurm. The brief's placeholders
  (`<<SLURM_ACCOUNT>>`, `<<GPU_HOUR_CAP>>`, ...) were unfilled.
- **Decisions confirmed by the user (17:05):**
  - Run everything here on the single RTX 4090 (PLAN §12 has a 4090 column). The rule
    "all results from GH200s" becomes "all results from this one RTX 4090"; no pooling with
    GH200 or paper numbers except the README anchor comparison (§5 gate).
  - Compute cap (stop condition): **6 GPU-hours on the 4090**.
  - Storage: repo directory (`runs/`, gitignored) + `HF_HOME=~/hf_cache`. Root disk is 98% full
    (~50 GB free) — keep an eye on it.
  - Python env: conda `sedd` (Python 3.9.18, torch 2.0.1+cu118, transformers 4.38.1,
    flash_attn 2.2.2). Install missing packages into it only; never reinstall torch.
- No Slurm: "jobs" are background processes on this workstation, launched with `nohup`, one GPU.
  Job IDs below are the process IDs plus a log path.

## 2026-10-02 17:10 — Step 0: code reading, discrepancies vs the plan

Files read: PLAN.md, README.md, `configs/lm1b/continuous/eval/rate_eval_seeds.py`,
`diffusion/continuous/samplers.py`, `evaluation/nfe.py`, `evaluation/run_eval.py`,
`evaluation/external_perplexity.py`, `evaluation/text_metrics.py`, plus the call chain
`evaluation/evaluation_drivers/external_ppl.py` → `evaluation/text_generations.py` →
`evaluation/generation_driver.py`, `evaluation/sampling_specs.py`,
`evaluation/compute_entropy_from_caches.py`, `utils/text_decode.py`.

Note: the plan cites `run_eval.py` and `external_perplexity.py` at the repo root; they live
in `evaluation/`.

**Line-number drift** (located by name):

| Symbol | Plan says | Actual |
|---|---|---|
| `_apply_edm_churn` | samplers.py:215 | samplers.py:209 |
| `SigmaSchedule.prepare` | samplers.py:1018 | samplers.py:870 |
| `DDIMSampler.sample` | samplers.py:1715 (loop 1853–1960) | samplers.py:1530 (loop 1619–1792) |
| TF32 enable | run_eval.py:164 | evaluation/run_eval.py:164 (matches) |

**Production config vs §4 — differences (repo wins for "production"):**

1. **Number of steps (important).** The spec targets NFE 256 → `num_steps=255`
   (`steps_for_target_nfe`). `SigmaSchedule.prepare(num_steps=255)` returns **255 σ points**,
   so the loop `for i in range(len(sigmas)-1)` runs **254 Euler steps**, then 1 final denoise:
   **255 model calls**, not 256. `compute_nfe` reports 256 because it counts `num_steps` as
   steps. The plan's "255 Euler steps + final denoise" is off by one.
   *Consequence for §3 segments:* segments 0–6 have 32 steps (steps 0..223); segment 7 has
   **30 steps** (224..253) + the final denoise (plan: 31). Boundary indices 0,32,…,224 unchanged.
2. **σ range.** σ_max is effectively **76.756** (top of the entropy table; `sigma_max=80` is
   clipped to the table), σ_min is the **terminal σ 0.08** (`external_ppl.terminal_sigmas=[0.08]`),
   not 0.002. The entropic grid spans quantiles [F(0.08), F(76.756)=1] of the training CDF.
   (README also says σ_term=0.0794 ≈ 0.08.)
3. **Per-step γ.** `s_churn = 0.2*255 = 51`; per-step γ = `s_churn / num_intervals` =
   51/254 = **0.200787** (not exactly 0.2). Churn window [0.08, 76.756] covers all 254 steps
   (checked numerically). S_noise = 1.003, `window_mode=entropy_cdf`, band [0,1]: match.
4. **Precision (important).** The plan says fp32 + TF32. The generation call in
   `evaluation_drivers/external_ppl.py` hard-codes `use_amp=True, amp_dtype="auto"`, which
   resolves to **bf16 autocast** (`cfg.evaluation.amp_dtype="bf16"`). TF32 is also on.
   The state `x` is fp32; model forward runs under bf16 autocast. Production = bf16 autocast.
5. **`micro_batch_size=512` is the generation batch**, not a GPT-2 scoring batch. GPT-2
   scoring is one text at a time (`score_texts`), so per-sample NLL has no batch effects.
   The per-chunk RNG seed is `(seed*1_000_003 + crc32(tag)*10_007 + chunk_start) mod (2^63-1)`
   — it depends on the chunk layout and the spec tag. I keep **micro-batch 512** (2 chunks of
   512 for N=1024) and the production tag, so initial noise equals production's.
6. **torch.compile**: production uses `torch.compile(mode="default")` on the model.
7. Self-conditioning `carry`: matches. `ati_eta=0`: matches. σ_data=0.5: matches.
8. **Scoring view**: `score_mode="full"`: GPT-2-large scores the full decoded text, which keeps
   BERT special tokens literally (`skip_special_tokens=False`). Token ids = 15-bit int mod 30522.
   Entropy (`compute_entropy_from_caches`) = mean over samples of per-sample unigram entropy of
   the 128 dense BERT ids.
9. Environment differs from README (Python 3.10 / torch 2.4 / transformers ≥4.40); `sedd` has
   older versions. Recorded as a deviation; the anchor gate tests whether it matters.

**Loop state** carried between steps in `DDIMSampler.sample`: `x` (fp32), the
self-conditioning state `x0_hat` (or `x0_hat_c/x0_hat_u` under CFG, unused here), the step
index, and the **global CUDA RNG stream** (churn noise via `torch.randn_like`). σ grid is a
pure function of config. Nothing else (no cached score, ATI uses only the σ grid).

**Data requirements.** With `cfg.cond.enabled=False` the prefix mask is all-False, so
`_build_mask_conditioning` returns `cond_enabled=False` and the LM1B reference batch from the
test loader is never read by the sampler. The LM1B dataset is only needed for the "real data"
reference GenPPL (53.06), which is not part of this diagnostic. Decision: **do not download or
build LM1B**; pass a zero reference tensor. Tokenizer needed: `bert-base-uncased`.
Token-frequency breakdown (§7) will use frequencies in the Run 1 reference samples instead of
LM1B corpus frequencies (deviation, logged).

## 2026-10-02 17:11–17:13 — Step 1: downloads

- Internet is available from this machine (it is the compute node). `HF_HOME=~/hf_cache`,
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (repo is public; no credentials used). Log:
  `experiments/compute_allocation/logs/download.log`.
- Checkpoint `checkpoints/cobit_s_lm1b_1M_ema.pt` from `gbatzolis/CoBit`
  (snapshot `2f2979d7`), symlinked to
  `runs/paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/checkpoints/step=001000000.pt`.
  **Size 533,878,820 bytes. SHA-256 `82094e59d5894c2cc468f0928fe427916c55af0059a44f175a91fe9aa6628b61`**
  (matches the Hub LFS hash).
- `bert-base-uncased` tokenizer files (snapshot `86b5e093`).
- `openai-community/gpt2-large` config/tokenizer/`model.safetensors` (snapshot `32b71b12`).
- Entropy tables `assets/entropy_tables/lm1b/` exist and load: pdf/cdf/sigmas 128 bins, edges 129;
  SHA-256s match the Hub mirror.

## 2026-10-02 17:15–17:21 — Step 2: code changes (§10 items 1–5) and CPU tests

- `diffusion/continuous/samplers.py` (commit `2f5661d`), all opt-in:
  - `churn_noise_seed`: churn noise of original step i drawn from a CUDA/CPU generator seeded
    with `SeedSequence([base_seed, i, 0xC0B17])` (`_churn_step_seed`). In the driver
    `base_seed` = the production chunk seed. Default `None` = global stream (unchanged).
  - `save_states_at` + `state_hook`: snapshot at the *start* of original step i (before churn):
    `x`, `x0_hat` (and `x0_hat_c/_u` for CFG), `i`, the σ grid, CPU and CUDA RNG state.
    `init_state`: resume from such a snapshot (σ grid is checked; in legacy-noise mode the RNG
    state is restored).
  - `segment_overrides={s: {"substeps": K, "gamma": None|g}}`, `segment_len=32`: segment of
    step i is `i // 32`. Sub-steps: K+1 σ points per original step with endpoints exactly
    σ_i, σ_{i+1}, interior points at evenly spaced entropy-CDF quantiles (same quantile grid and
    piecewise-linear inversion as `_inverse_cdf_sample_truncated`;
    `SigmaSchedule.entropic_substep_sigmas`). Churn once at sub-step 0 with the original step's
    γ (computed with the original `num_intervals=254`) and its seeded noise; self-conditioning
    `x0_hat` updated after every sub-step. `gamma` overrides γ for the steps of segment s.
  - `step_callback(i, sub, n_sub, final, sigma, sigma_next, gamma, D, x, eps)`.
- `evaluation/external_perplexity.py`: `score_texts_per_sample` (same arithmetic as
  `score_texts`; summing per-sample values in order reproduces the total).
- Whitespace-insensitive diff of the loop: only 3 original lines changed (import; `range` start;
  `num_intervals` uses a precomputed `n_intervals`, same value). Body only re-indented.
- Tests `experiments/compute_allocation/tests/test_sampler_diagnostic.py`, tiny SDT (85k params,
  8 tokens), real entropy tables, 32 steps, segments of 8. Run on CPU on this machine:
  **12 passed** (17:21). Covers: (1) bit-identity vs the sampler on `main` (loaded from
  `git show main:...`), stochastic/deterministic × carry/refined; (2) null replay from every saved
  state, seeded and legacy-noise modes; (3) refine with 1 sub-step == reference;
  (4) churn noise per step unchanged by refine in any earlier/later segment, drawn once per
  original step at sub-step 0; plus sub-step σ placement (even in quantile), NFE count
  (32+1, +3·8 for refine), nochurn scope, and "branch from saved state == same override from step 0".
- Installed into `sedd`: pytest 8.4.2, scipy 1.13.1, matplotlib 3.9.4 (dry run confirmed torch untouched;
  torch still 2.0.1+cu118).

## 2026-10-02 17:22–17:26 — first GPU launches: two environment problems

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (README) → `RuntimeError: Unrecognized
  CachingAllocator option` on torch 2.0.1. Dropped.
- **torch.compile crashes** in this env: the Inductor/Triton kernel fails with
  `Triton Error [CUDA]: misaligned address` on the 4090 (torch 2.0.1, its bundled Triton).
  **Deviation: run the denoiser eagerly (`--no-compile`) in every job.** Compile only fuses
  kernels; numerics differ at bf16-rounding level. The anchor gate tests the net effect.
- Checkpoint loads with "No EMA found in checkpoint, using raw weights": expected, the released
  file has EMA weights baked in (README Step 1).
- Jobs are launched with `experiments/compute_allocation/run_job.sh <name> <args>` (conda `sedd`,
  `HF_HOME=~/hf_cache`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), logs in
  `experiments/compute_allocation/logs/<name>.log`; GPU time is appended per job to
  `experiments/compute_allocation/results/gpu_ledger.jsonl` (process wall-clock, since the
  GPU is held for the whole process). The driver refuses to start work past the 6 GPU-h cap.

## 2026-10-02 17:27 — Pre-registered operationalisations (written before any Run 1/2 data)

The §7 thresholds are applied exactly as written (material: CI excludes 0 and |ΔNLL/token| ≥ 0.01;
entropy flag |Δentropy| > 0.02; borderline: CI straddles 0.01). Where the plan leaves a quantity
undefined I fix it here, before seeing data (also in `analysis.py` docstring):

- **ΔNLL/token** = Σnll_branch/Σtok_branch − Σnll_ref/Σtok_ref over the 1024 paired samples
  (the change in log GenPPL; 0.01 ≈ +0.6 GenPPL at 60, as §7 says). "Straddles 0.01" = CI
  contains +0.01 or −0.01.
- **Bootstrap**: 10,000 resamples of sample indices, paired (the same indices for ref and
  branch), percentile 95% CI, numpy `default_rng(20261002)`.
- **Δentropy**: mean over samples of the per-sample unigram entropy difference (production
  entropy metric = mean per-sample entropy of the 128 BERT ids).
- **Anchor gate**: README says "GenPPL ≈ 59–60 at entropy ≈ 4.31" (single seed; paper
  59.76 ± 0.57 over 10 seeds). Pass = GenPPL in [58.5, 60.5] (59–60 ± ~1 seed sd of 0.57,
  rounded) **and** entropy within ±0.015 of 4.31.
- **§8 "nochurn costs much more in some segments"** = nochurn[s] is material (ΔNLL > 0, not
  entropy-flagged) **and** exceeds the median nochurn[·] ΔNLL by ≥ 0.01. "nochurn spread evenly"
  = no segment meets that. **"refine[s] is material"** for the step-size row means a material
  *improvement* (ΔNLL < 0), not entropy-flagged. Entropy-flagged branches are reported but not
  counted as material for the verdict (§7: "not interpreted on its own").
- **§8 last row** (mismatch with Run 1): reported if a bottleneck segment is not among the two
  segments holding the most final-settling mass in Run 1.
- **Settling time** of a token = last step t (0..254, 254 = final denoise) at which its guess
  (argmax bits of D, packed 15-bit id) differs from step t−1. **Provisional settling** = first step
  after which the guess is unchanged for 8 steps; "changes after first settling" = guess changes
  again later.
- **Concentration**: per-sample contribution c_i = Δnll_i − r̄_ref·Δtok_i (linearised ΔNLL·Σtok);
  top-10% share = sum of the 103 largest c_i in the direction of the total ÷ total (also |c| version).
- **Token frequency** buckets use rank of the reference token in the Run 1 reference corpus
  (1024×128 tokens): 1–10, 11–100, 101–1000, >1000/unseen. (LM1B corpus not downloaded.)
- **Extra batch b** (borderline segments): seed 42+b with the same chunk layout; a new reference
  trajectory for that batch plus the branch; pooled over batches for the decision.
