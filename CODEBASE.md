# CODEBASE.md — BitstreamDiffusion (CoBit) explained and audited

*Audit date: 2026-09-30, branch `tasks/fkc-temperature` @ `835d921` plus the uncommitted working tree (19 modified files, ~30 untracked).*

This document has two halves: **how the code works** (§1–6) and **what is wrong with it** (§7–8).
Findings are tagged by how they were established:

- **[verified]**: I read the code and, where relevant, reproduced the behaviour.
- **[confirmed]**: a reviewer traced it end to end in code.
- **[likely]**: the mechanism is in the code, but the impact was not measured.
- **[speculative]**: plausible, but not demonstrated.

---

## 0. The one-paragraph version

The repo is the open-source release of the CoBit paper (arXiv:2605.07013). CoBit is a *continuous* diffusion language model:

- **Representation.** Each token id is written as a fixed-width binary codeword. Each bit becomes an analog value in {0, 1}.
- **Training.** A transformer (the "SDT") denoises Gaussian-noised bits under EDM preconditioning.
- **Matched filter.** The network's per-bit logit is added to an analytic *matched-filter* term. That term is the exact Gaussian log-likelihood ratio for one isolated bit, so the network only has to learn the contextual *residual*.
- **Sampling.** Samples come from integrating the probability-flow ODE on an **entropy-rate σ grid**. This is a dataset-specific schedule that puts steps where information is being resolved. Stochastic EDM "churn" is added on top.

On top of that paper code sits roughly as much **research-programme code** of the owner's. It trains CoBit on TinyGSM and evaluates GSM8K pass@k and Sudoku, and it studies the following:

- classifier-free, auto- and self-guidance;
- Feynman–Kac corrector (FKC) SMC and temperature;
- temporal ordering of denoising (left-to-right vs random, per-token σ);
- token-softmax vs bits.

It also includes the SLURM infrastructure to run all of this on CSD3 and Isambard-AI (GH200).

**Size:** ~71k lines of Python, 521 tracked files, 56 config files, 109 scripts, 36 docs, 533 CPU tests.

---

## 1. Repository map

```
train.py                    entry point (torchrun). Loads a config .py, merges saved config on resume, builds Trainer
trainers/trainer.py         2.7k lines: the whole training loop, checkpointing, resume, DDP, validation
models/
  sdt.py                    Sequence Diffusion Transformer: THE model (AdaLN-Zero trunk, per-bit logit head)
  sedd_wrapper.py, backbones/   SEDD baseline (discrete-only in practice)
  autoregressive/gpt.py     AR baseline (KV cache)
diffusion/
  continuous/
    processes.py            VE forward process, σ sampling
    losses.py               binary / token score-interpolation losses, EDM weighting, Triton fused CE
    logit_postprocess.py    matched-filter residual, posterior temperature, codeword sharpening
    samplers.py             2.9k lines: SigmaSchedule + Heun / DDIM / Euler–Maruyama / PC / FKC samplers
    guidance.py             CFG / AutoGuidance / self-guidance algebra + GuidedDenoiser
    smc.py                  ESS, systematic resampling helpers for FKC
    lambda_profiles.py      λ(σ) Langevin-strength profiles (entropy-rate shaped)
    likelihood.py           continuous BPC evaluation (see bug M3)
    ordering.py, ordered_sampler.py   temporal-ordering research (per-token σ, window ordering)
  discrete/                 SEDD-style absorbing/uniform diffusion baselines (Appendix C)
data/
  lm1b.py                   BERT-uncased ids, 128-token blocks → bits
  openwebtext.py            GPT-2 ids → second-stage 16-bit "gpt2id_bpe16" codec → bits
  tinygsm.py, sudoku.py, task_codec.py   task datasets with a prompt-prefix mask
utils/
  schedule_controller.py    online entropy-rate schedule (writes entropy_{pdf,cdf,sigmas,edges}.pt)
  text_decode.py            2k lines: bits → ids → text for every codec
  ema.py, optim.py, ecc_secded.py, callbacks/
evaluation/
  run_eval.py               metric dispatcher (external_ppl, bpc, vlb, mauve, fid, …)
  sampling_specs.py, nfe.py sampler-spec grid + NFE accounting
  external_perplexity.py    GenPPL under gpt2-large
  compute_entropy_from_caches.py   post-hoc unigram entropy
  tasks/                    gsm8k_eval.py, sudoku_eval.py, sandbox_gsm8k.py (executes generated programs)
experiments/{guidance,ordering,tokens}/   analysis scripts (bootstrap CIs, aggregation)
configs/{lm1b,owt,tasks}/   plain-Python get_config() files
scripts/
  lm1b/, owt/               dataset/codec build
  hpc/arch/, hpc/guidance/, tasks/   SLURM launchers per study
  site/                     cluster abstraction (site.sh, submit.sh), transfer.sh, multinode env
docs/                       36 research notes, reports (tex/pdf), handoffs
results/                    committed result JSON/CSV/figures (137 tracked, ~930 untracked)
evaluations/                duplicate of two results/ entries (see H4)
assets/entropy_tables/      shipped entropy-rate schedules (lm1b, owt, owt_medium)
assets/noise_tables/        frozen noise CDF for the window study (NOT tracked; see R1)
```

Symlinked out of the repo (to `/projects/u6ya/...`): `datasets/`, `runs/`, `tinigsm_gsm8k/`.

---

## 2. From text to bits (data)

| Dataset | Tokenizer | Codeword | Sequence | Cache files |
|---|---|---|---|---|
| LM1B | bert-base-uncased (30,522) | raw binary MSB-first (or semantic-rank Gray code / SECDED for ablations) | 128 tokens | `cache_{train,test}_tokens.uint16` + `.meta.json` |
| OWT | GPT-2 → second-stage BPE over GPT-2 ids ("gpt2id_bpe16", V = 65,536) | 16 bits | 1024 | `cache_*_gpt2_flat_eos1.uint16`, `cache_*_gpt2id_bpe16_v65536_base1024_len1024.uint16` |
| TinyGSM | SmolLM ids, padded to 512 | `task_codec.token_ids_to_bits` (MSB-first) | 512, with prompt-length prefix mask | uint16 ids + int32 prompt lengths |
| Sudoku | 12-token vocab | 4 bits | fixed | `.pt` of uint8 |

- **LM1B validation** is carved from the tail of the train cache.
- **OWT `test` split** is the val cache (`data/openwebtext.py:888`).
- **Task datasets** carry a prefix mask. The prompt is kept clean (the "conditioning prefix"), and only the suffix is noised and supervised.

## 3. The model (`models/sdt.py`)

1. **Input.** `_build_content_continuous_bits` (`sdt.py:1015`) computes `c_in · (x_t − 0.5)`, with `c_in = 1/√(σ² + σ_d²)`. When enabled, a self-conditioning channel (the previous x̂₀ estimate) is concatenated.
2. **Patching.** Fourier position features are added. The sequence is then patched into P = bits-per-token chunks, giving one transformer position per token.
3. **Trunk.** `PreNormBlockAda` blocks with AdaLN-Zero conditioning on a sinusoidal embedding of log σ.
   - CoBit-S: 12×768.
   - CoBit-M: 24×1024, 16 heads.
   - Per-bit σ of shape `[B,S]` (used by the ordering experiments) is reduced to a per-token σ for the conditioning.
4. **Head.** `OptimalSkipMLPHead` produces one raw logit per bit.
5. **Matched-filter residual.** This is *outside* the model, in `logit_postprocess.py:115-165`:
   `logit = raw + clip(scale·(x_t − 0.5)/σ², ±30)`.
   Training and sampling both route through `_model_logits_continuous` (`logit_postprocess.py:256`), so they agree. The matched-filter attributes parsed inside `sdt.py:689-718` are dead.

## 4. Training (`train.py` → `trainers/trainer.py`)

**Config and resume**
- `train.py` exec's the config file and calls `get_config()`.
- If `runs/<exp>/checkpoints/last.pt` exists, the **saved `config.json` is merged over the fresh config**. It skips logging/system/evaluation and re-applies an allowlist of layout keys (`RESUME_LAYOUT_KEYS`, uncommitted). This is important: see T2.

**Trainer init** (`trainer.py:709-1214`)
- Loaders are rebuilt with a DistributedSampler at micro-batch `batch_size / (world × accum)`.
- DDP, `torch.compile`, EMA, AdamW with warmup and cosine decay.
- `EntropyScheduleController`, callbacks, auto-resume.

**Step** (`_step_continuous`, `:1478`, ~350 lines)
1. Draw σ. The default mixes a log-normal/log-uniform base with the learned entropy-rate pdf, via a gamma ramp (`schedule_controller.py:242`). Alternatives are per-position ordering σ or window σ.
2. Form `x_t = x₀ + σε`.
3. Build the conditioning prefix, dropped with probability `p_uncond` (this is what enables CFG).
4. Run a self-conditioning pre-pass with probability 0.5.
5. Compute the EDM-weighted binary score loss over the suffix/window mask.
6. Accumulate gradients. Clip, step and update EMA on the last micro-batch.
7. Push per-σ loss into a FIFO. Every N steps, rank 0 rebuilds the entropy-rate histogram and broadcasts it.

**Checkpoints** (`runs/<exp>/checkpoints/`)
- `last.pt`: atomic, rolling.
- `step=%09d.pt`: every 50k steps.
- `best.pt`.
- Entropy tables and a `sigma_data.json` sidecar are written to the run dir.

## 5. Sampling (`diffusion/continuous/samplers.py`)

- **σ grid.** `SigmaSchedule.prepare` (`:1018`) builds either a Karras grid or the **entropic** grid: the inverse CDF of `entropy_cdf.pt`, found in the checkpoint's grandparent directory or in `cfg.evaluation.entropy_run_dir`.
- **Each step:**
  1. Optional EDM churn: `gamma_target`, with S_noise = 1.003 over the entropy band.
  2. Denoise: D = sigmoid(logit).
  3. Score = (D − x)/σ².
  4. Integrate.
- **Decode.** Threshold the bits (or argmax for the token variant), then map through `utils/text_decode.py`.

| Sampler | Lines | Status |
|---|---|---|
| `SigmaSchedule` + helpers | 1–1096 | shared |
| `HeunSampler` | 1102–1641 | legacy: inline CFG copied four times, CFG-only guidance |
| **`DDIMSampler`** | 1648–1990 | **the headline sampler**; all guidance goes through `GuidedDenoiser` |
| `EulerMaruyamaSampler` | 1992–2097 | Langevin noise scaled by λ(σ) |
| `PredictorCorrectorSampler` | 2100–2325 | legacy, with its own copied CFG block |
| `FeynmanKacEulerMaruyamaSampler` | 2328–2858 | FKC/SMC over K particles |

**Guidance** (`guidance.py`)
- `GuidedDenoiser.denoise` batches the conditional, unconditional, AutoGuidance and self-guidance branches into one forward pass per network.
- Self-conditioning state is kept per branch.
- In "hold" mode, self-guidance strips the matched filter before amplifying and re-attaches it afterwards.
- **Scale semantics:** w = 0 means *conditional-only*, and w > 0 means `u + w(c − u)`. So w = 0.01 is almost fully unconditional; sweeps are discontinuous at 0.

**FKC.** Particles are laid out `[B,K,S]`. There are two modes, anneal (β > 1) and CFG-FKC (β = 1 with a geometric-mean target). There are two proposals, `em` and `edm_churn`; the latter is only leading-order exact for β > 1. Resampling is systematic.

**Ordering** (`ordering.py`, `ordered_sampler.py`)
- Each token gets its own σ (clock) *t_j*.
- The old design normalised ranks over the padded suffix. The new "window" design (uncommitted) uses `t_j = clip(τ(1+L/k) − r_j/k)` with absolute ranks and a frozen noise CDF.
- `OrderedSampler` and `WindowSampler` subclass `DDIMSampler` but replace `sample()` wholesale.

## 6. Evaluation, infrastructure, tests

**Paper metrics**
- `evaluation/run_eval.py --metrics external_ppl` generates 1024 samples per sampling spec, then scores GenPPL under gpt2-large (completion-only).
- `compute_entropy_from_caches.py` then computes unigram entropy. OWT is measured on inverse-decoded GPT-2 ids and LM1B on BERT ids.

**GSM8K** (`evaluation/tasks/gsm8k_eval.py`)
- Decode the completion and `exec` it in `sandbox_gsm8k.py`, then call `simple_math_problem()`.
- Compare with the `####` gold answer (tolerance 1e-3). Report accuracy with a bootstrap CI, plus per-problem records for pass@k and maj@k.
- Under FKC, "pass@K" means *any particle correct*. Resampled particles share ancestors, so this is not i.i.d. pass@k.

**Infrastructure**
- `scripts/site/site.sh` detects CSD3 vs Isambard (aarch64 → Isambard; unknown → CSD3).
- `submit.sh` wraps `sbatch` with the site's partition and account, the GPU count read from the launcher's `--gres`, and a wall-time cap check.
- `transfer.sh` is dry-run by default. It checks reachability over SSH, rsyncs from a manifest and checks sha256.

**Safety guards (window study only)**
- `window_train.slurm` writes a `HALT` file on crash, divergence or no progress, and cancels its own pending links.
- `window_watch.sh` is a circuit breaker: 3 crashed runs or a 3000 GPU-h cap writes a global halt. It also cancels jobs silent for 45 minutes.
- Older launchers have only a stall watchdog.

**Tests** (`tests/`, CPU, env `cobit-gh`): **528 passed, 5 failed, 47 s**. All 5 failures come from uncommitted work:
- 4 × `test_objective_probe.py`: the stub at `:166` lacks the new `_window_ordering_enabled` method (`trainer.py:1568`). This is a one-line test fix.
- 1 × `test_gsm8k_robustness.py`: flags `| head -2` at `scripts/hpc/arch/window_train.slurm:83` under pipefail. It is harmless in context (guarded by `|| true`), but the lint is right to flag it.

**Coverage**
- Well covered: trainer mechanics, samplers, FKC, guidance, ordering/window, losses, task codec, GSM8K sandbox, NFE accounting, site config.
- **Not covered:** `run_eval.py`, sampling specs, evaluation drivers, GenPPL, entropy, VLB, MAUVE, `transfer.sh`, `window_watch.sh`, and the whole discrete framework.

---

## 7. Research programme status (as recorded in docs/ and memory)

| Study | Question | Status and headline | Primary source |
|---|---|---|---|
| Paper (LM1B/OWT) | GenPPL–entropy frontier | Released; checkpoints on the HF Hub (`gbatzolis/CoBit`) | README |
| TinyGSM → GSM8K baseline | Can CoBit do math reasoning? | 250k no-CFG checkpoint: 9.2% deterministic, **23.4%** with churn (NFE 512, γ = 0.34) | `docs/RESULTS_TINYGSM_GSM8K.md` |
| Guidance | Do CFG/AG/SG help? | Closed. At γ = 0, CFG w = 12 gives +8 pp. Under churn, CFG must be retuned to w ≈ 2–4 (+1.9 pp); AG goes negative; SG-prev is harmful. Guidance and voting add up. | `docs/findings.md`, `docs/open_questions.md` |
| FKC / temperature (Sudoku) | Principled sharpening | Score temperature gives +2.2 pp. **Every FKC variant is worse than β = 1 voting**, and the weights are uncorrelated with correctness. | `docs/fkc_temperature_findings.md` |
| FKC pass@k (GSM8K) | Same question on GSM8K | The apparent gain from β is a step-size artefact: it reproduces at K = 1 and flips to −5.65 pp at 1024 steps | `docs/HANDOFF_FKC_PASSK.md` |
| Ordering, w = 0.25 | Does L→R ordering help? | **Invalid as an ordering test.** 74% of the suffix is padding, so content got only a 2× σ span. | `docs/experiments_ordering_objective.md` §24 Sep |
| Window ordering (26 runs, 2 seeds) | Order effect at fixed window k | Stopped at 250k (~2,780 GPU-h). l2r is the best order at every k, but the **uniform control still wins** (9.6% vs 8.0%). | same doc (uncommitted section) |
| Tokens vs bits | Does a V-way softmax help? | Stopped 26 Sep, when total Isambard use reached ~3.3k GPU-h. Tokens reach ≈ 0.55× bits. | same doc |

⚠ The last three rows were produced by multi-GPU runs, and bugs **T1–T4 in §8 bear directly on them.**

---

## 8. Audit findings

Ranked by consequence. File references are relative to the repo root.

### 8.1 Findings that may affect results already quoted

**T1. `sigma_data` differs between DDP ranks. [verified]**
- `SigmaDataEstimator` runs **on rank 0 only** (`trainers/trainer.py:1086-1087`). It overwrites `cfg.diffusion.continuous.sigma_data` in place (`utils/callbacks/sigma_data.py:48`).
- The model reads that value on every forward pass (`c_in`, `models/sdt.py:1021`), and so does the EDM loss weight (`diffusion/continuous/losses.py:46`).
- As a result:
  - Rank 0 trains with ≈ 0.40 (bits) or ≈ 0.45 (tokens).
  - Every other rank trains with the configured 0.5.
  - Gradients averaged across ranks mix two preconditionings and two loss weightings.
  - Eval uses rank 0's value from the `sigma_data.json` sidecar.
- Scope: every multi-GPU run since the estimator was added, including the window study and token-vs-bits.
- *Fix:* compute on rank 0 and broadcast, or all-reduce the estimate.

**T2. Probable root cause of the "every 5 epochs" DDP deadlock. [verified mechanism; causal link likely]**
- `EntropySchedulePlotCallback` is registered only on the master (`trainer.py:1088`). Yet it runs forward passes through the **DDP-wrapped** `trainer.model` (`utils/callbacks/entropy_schedule_plot.py:163`).
- DDP defaults to `broadcast_buffers=True`, and the SDT has buffers (`sdt.py:24,193`). So rank 0 issues a collective that the other ranks never match.
- The mitigation (`entropy_plot_every_k_epochs=100000` in `configs/tasks/tinygsm_bits_ordering.py:194`) never reached the hung runs: all three `runs/tasks/tinygsm/ord_fs500k_*/config.json` still say **5**, because the saved config wins on resume (T6).
- The clean repro and `win_*` runs have 100000 and never hung.
- `docs/experiments_ordering_objective.md:355` calls this callback "collective-free", which is wrong.
- *Fix:* use `unwrap_model(trainer.model)` in master-only callbacks, or construct DDP with `broadcast_buffers=False`.

**T3. The random seed is not offset by rank. [verified]**
- `_maybe_set_seed` (`trainer.py:127-139`) seeds every rank identically. Resume also loads rank 0's RNG state into all ranks (`:1353`).
- Consequence: all ranks draw the *same* Gaussian noise, σ, CFG-drop and self-conditioning masks for their different data shards. Rank 0 drifts once a master-only callback consumes RNG.
- Effective noise diversity per step is therefore about 1/world. (The window τ stream does include the rank.)
- This also means T1's rank-0 divergence is not the only asymmetry.

**T4. DistributedSampler ignores `cfg.train.seed`. [verified]**
- It is constructed without `seed=` (`trainer.py:875`), so the seed defaults to 0.
- **Training seeds 42 and 43 therefore see identical data order** in DDP. The two-seed Welch intervals in the window study measure init and noise variance, not data-order variance.

**T5. Random-order evals reuse the same permutations in every batch. [verified]**
- `OrderedSampler.sample()` re-seeds a fresh generator from `order_seed` on every call (`diffusion/continuous/ordered_sampler.py:157-158`, and similarly for `WindowSampler` at `:241`).
- So row *i* of each of the ~21 GSM8K batches gets the same random ordering. The random arm is less random than claimed.

**T6. On resume, the saved config silently overrides edits to the Python config. [confirmed]**
- `train.py:175`. Only the `RESUME_LAYOUT_KEYS` allowlist escapes this.
- This is by design, but it is undocumented at the point of use, and it is what defeated the T2 mitigation.
- *Fix:* at minimum, print a diff of discarded keys.

**T7. A mid-epoch resume skips the rest of that epoch. [confirmed]**
- `_load_checkpoint` returns `epoch + 1` (`trainer.py:1356`), and the data position is not saved.
- Each walltime restart drops up to one epoch of data and its validation. The window docstring's "resume-exact" claim does not hold for data.

**T8. Validation pollutes the online entropy schedule. [confirmed]**
- The FIFO update in `_step_continuous` (`trainer.py:1817`) is not gated on `is_train`.
- The buffer isn't checkpointed.
- With `gamma_max=1.0`, a bin whose pdf reaches 0 can never refill (`utils/schedule_controller.py:501-522`).

**T9. Continuous BPC skips the matched filter. [verified]**
- `diffusion/continuous/likelihood.py:384,407,421,571` call `model(x, σ)` directly and take `sigmoid(raw)`.
- So for every production config (`matched_filter_residual`), the reported bits-per-char is wrong.
- It is still reachable via `run_eval --metrics bpc`.
- It also silently falls back to Karras when tables are missing (`:243,:299`).
- GenPPL, the paper's headline metric, is unaffected.

**T10. The discrete uniform-graph Tweedie step is V× too small. [verified against the SEDD reference]**
- `diffusion/discrete/processes.py:110` uses `score.mean(...)` where SEDD uses `score.sum(...)`.
- It is compounded by `sdt.py:985`, which zeroes the current-token log-score only for the absorbing graph.
- Affects `configs/lm1b/discrete/uniform_*` baselines (Appendix C). The absorbing baselines are fine.

### 8.2 Security and compute-safety

**S1. The GSM8K sandbox is not a sandbox. [verified]**
- `evaluation/tasks/sandbox_gsm8k.py:11-43` calls `exec` in-process with a builtins whitelist. That is escapable via `().__class__.__base__.__subclasses__()`; a reviewer reached `os.listdir` this way.
- It runs as you, on a shared HPC system, with checkpoints and SSH keys reachable.
- Malicious model output is unlikely, but decoded text is arbitrary.

**S2. The sandbox timeout can be defeated, and memory is uncapped. [confirmed]**
- `_Timeout` subclasses `Exception`, so a generated bare `except:` swallows it. The one-shot `setitimer` never re-arms.
- Reproduced: `while True: try: while True: pass / except: pass` hangs.
- `[0]*10**10` can OOM the job.
- Either would burn a GPU allocation until walltime.
- *Fix for S1+S2:* run each program in a subprocess with `resource.setrlimit` (CPU and address space), a scratch working directory and a hard kill.

**S3. `submit.sh:31` mis-parses `D-HH:MM:SS` wall times. [verified]**
- It replaces `-` with `:` and multiplies every field by 60. So `1-00:00:00` becomes 216,000 s, and a legal 24 h request is refused.
- A bare `--time=90` (SLURM: minutes) is read as 90 s.
- `-t` isn't checked at all.

**S4. `submit.sh:44-45` defaults to 4 GPUs when a launcher has no `--gres`. [confirmed]**
- The CPU-only `scripts/tasks/prebuild_tinygsm*.slurm` would reserve, and bill, four GPUs.

**S5. `trust_remote_code=True` on the GenPPL reference model. [likely]**
- `evaluation/external_perplexity.py:109,118`. It is unnecessary for gpt2-large and runs Hub code if the model name changes.

**S6. `torch.load(weights_only=False)` on rsynced checkpoints. [speculative]**
- Acceptable while every file is your own.

### 8.3 Reproducibility

**R1. The frozen noise table for the window study is not in git. [verified]**
- `assets/noise_tables/tinygsm_window_F_v1.pt` is caught by `.gitignore:65 *.pt`, so it doesn't even appear in `git status`.
- The trainer, `WindowSampler` and `configs/tasks/tinygsm_bits_window.py:34` all require it, and it is not in `transfer_manifest.txt`.
- *Fix:* add a `!assets/noise_tables/**/*.pt` exception, as already done for `entropy_tables`.

**R2. Headline results for the ordering, window and token studies exist only in the working tree.**
- `results/ordering/window_summary.json`, `results/tokens/summary_gh200.json`, `results/ordering/train_summary_500k_gh200.json` and ~930 other files are untracked.
- The same goes for the analysis script `experiments/ordering/analyze_window.py` and the whole window kit (`scripts/hpc/arch/window_*.{sh,slurm}`).

**R3. Uncommitted coupling.**
- The `evaluation/tasks/` diff imports `WindowSampler`, which exists only in the uncommitted `ordered_sampler.py`. Committing one without the other breaks eval.
- `order_k=inf` is written to provenance JSON as `Infinity`, which is not strict JSON.

**R4. The recipes live in shell exports, not configs.**
- The correct LRs and step counts for the ordering and token studies are set in `ordering_resubmit.sh` and `logits_gh200.sh`. The config defaults differ (for example, tokens default to 50k steps at 3e-4).
- `tinygsm_bits_window.py` builds its config by mutating `os.environ` and chain-importing four config files.
- `tinygsm_bits_ordering.py` contradicts itself: its header argues for fine-tuning, while the body says from scratch is the real experiment. Its defaults are still fine-tune init at an LR the same file says diverged.

**R5. Statistical caveats on the window study.**
- Welch intervals over 2 training seeds have df ≈ 1–1.8, with no multiple-comparison correction.
- There are 3 k values × 4 contrasts × 5 checkpoints.
- T4 means the seeds don't vary data order.
- `analyze_window.py:117-121` never intersects problem indices, so a partial cell would compare mismatched problems silently. (`experiments/tokens/analyze.py` does intersect.)
- The head-to-head table in the doc is not produced by any committed script.

### 8.4 Latent bugs (not currently on a hot path)

| ID | Where | What |
|---|---|---|
| L1 | `models/sdt.py:147` | Bool attention mask passed to SDPA with inverted polarity. Also `pad_mask_tokens` (`:1165`) is always all-False. Latent because production S % P = 0. |
| L2 | `models/autoregressive/gpt.py:171-177,215` | KV-cache positions freeze once past `max_seq_len`; cached and uncached outputs diverge. |
| L3 | `ordered_sampler.py` | The stochastic settings (γ, `sc_refresh_mode`, `ati_eta`) are swallowed by `**ignored`. `--gamma>0 --sampler_kind window` runs deterministically without warning. Published cells used γ = 0. |
| L4 | `ordered_sampler.py:64` | `_reject_active_guidance` checks only explicit arguments, not the `cfg.evaluation.guidance_scale` fallback that `DDIMSampler` uses. |
| L5 | `guidance.py:762, 667` | SG reads `sigma.reshape(-1)[0]` / `.min()`, which is wrong for per-position σ `[B,S]`. Currently blocked by L4's guard. |
| L6 | `trainer.py:1336-1339, 1393-1396` | `load_state_dict(strict=True)` silently retries `strict=False`. |
| L7 | `utils/text_decode.py:~415` | Invalid codewords are wrapped `% vocab_size`, hiding the invalid-codeword rate. |
| L8 | `data/openwebtext.py:431,512` | The code-cache key omits the tokenizer identity; truncation is uncounted. |
| L9 | `utils/optim.py:150-182` | The first optimiser step runs at the full base LR (warmup is applied after `step`). |
| L10 | `compute_entropy_from_caches.py:303-318` | The non-OWT fallback includes EOS/pad tokens in entropy. |
| L11 | `trainer.py:1940-1947` (uncommitted) | Every validation batch uses the same τ vector. The window seed formula aliases between seeds when accum ≥ 4. |
| L12 | `trainer.py:2069` | The divergence guard is per rank, so ranks can exit asymmetrically. |
| L13 | missing `last.pt` | Silent scratch restart that deletes TB events and overwrites `config.json`, even when `step=*.pt` files exist. This matters after migrations. |

### 8.5 Dead code and duplication

**Dead code**
- `diffusion/discrete/entropy.py` (580 lines).
- `models/backbones/sedd_kv_edit.py` (unconditional `flash_attn` import).
- `autoregression/` (points at a non-existent `configs/autoregressive/`).
- `configs/tasks/tinygsm_bits_cfg_bad.py`.
- `GenerationCallback` / `SigmaGradNormCallback`: imported but never instantiated, so `train.generation.enabled=True` is silently ignored in 5 LM1B configs.
- The `use_flash_attn` flag in `sdt.py`: ignored, with a misleading log line at `:790`.
- The matched-filter attributes in `sdt.py`.

**Duplication**
- Three Karras implementations.
- Three entropy-table loaders with *different strictness*: `samplers.py:778` returns None, `lambda_profiles.py:68` raises, `likelihood.py:171` swallows everything.
- Four paired-bootstrap implementations in `experiments/`.
- Heun's CFG block written out four times.
- ~91% of config lines duplicated verbatim across sibling configs.
- 19 near-duplicate `*_csd3` / `*_isambard` launchers.
- 42 scripts hard-code `rg625` or `GIROLAMI-SL2-GPU`.

**Hot spots**
- `Trainer.__init__` (~500 lines).
- `_step_continuous` (~350 lines).
- `samplers.py` (2.9k lines; a natural split is schedule / conditioning / one file per sampler).
- `text_decode.decode_samples_to_text` (~300 lines).

**Wasted work**
- FKC runs its final decode twice (`samplers.py:2836,2849`); the second pass is invisible to NFE accounting.
- Validation and the self-conditioning pre-pass run with autograd on.
- More than a dozen `.item()` host syncs per sampler step.

### 8.6 Repo hygiene and docs

**Repo hygiene**
- **H1.** A **3.5 GB `core` dump** from a crashed `train.py` (24 Sep) sits in the repo root. It is gitignored, but it counts against the 101 GB home quota. Delete it.
- **H2.** `evaluations/` is a byte-identical tracked copy of `results/fkc_temperature_all_cells.csv` and `results/gsm8k_passk_shardA/`, and nothing references it.
- **H3.** `docs/gpu_budget.md` covers only the ~1000 A100-h guidance programme. It has no entries for the ~3.3k Isambard GPU-h spent on the ordering, window and token studies (of the 10k grant).

**Stale or contradictory docs**
- `temporal_ordering.md` says ordering "isn't implemented".
- `temporal_ordering_design.md` §8–10, and the `ordering.py` module docstring, get the direction of u backwards.
- `experiments_ordering_objective.md`:
  - The "Framing" section is out of date.
  - A1 is still "_pending_".
  - The 20 Sep "L2R does nothing" is not marked retracted in place.
  - The claim that the plot callback is "collective-free" is wrong (T2).
- `GUIDANCE_REPORT.md` §6–9 is superseded by `findings.md` F2/F5–F8.
- `open_questions.md` cites a non-existent `docs/research_status.md`.
- `RESULTS_TINYGSM_GSM8K.md` §5 says guidance is not yet evaluated, and its 23.4% is not reconciled with the handoff's "≈ 29%".
- The FKC class docstring (`samplers.py:2391`) says "no CFG, churn disabled", but both are supported.
- The Sudoku FKC negatives used the approximate `edm_churn` proposal. The code itself warns it is inexact for β > 1, and the docs don't flag this.

---

## 9. Suggested order of work

1. **Before quoting any multi-GPU result again:** fix T1 (broadcast `sigma_data`), T3 (rank offset on the seed) and T4 (`DistributedSampler(seed=cfg.train.seed)`). Then decide whether the window and token-vs-bits conclusions need a caveat or a re-run. Quote GPU-h before re-running anything.
2. **Fix T2** (unwrap the model in master-only callbacks, or set `broadcast_buffers=False`) and make T6 print the discarded keys, so the deadlock stays fixed.
3. **Commit the window study coherently:** the sampler, eval and trainer diffs together, plus the `.gitignore` exception for `assets/noise_tables/` (R1), the analysis scripts and the summary JSONs (R2). Update the 4-line test stub at the same time, so the suite is green.
4. **Move the GSM8K exec into a resource-limited subprocess** (S1, S2).
5. Fix `submit.sh`'s wall-time parser and its GPU default (S3, S4).
6. Housekeeping: delete `core` and `evaluations/`; mark retracted findings in place in the docs; update `gpu_budget.md`.
7. Mark T9 and T10 in code (or fix them). They don't touch the paper's headline GenPPL numbers, but they will bite anyone who runs `--metrics bpc` or the uniform discrete baseline.
