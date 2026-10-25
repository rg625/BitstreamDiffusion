# CPU-only reanalysis (1 Oct 2026)

*Branch `cpu-reanalysis`. No GPU was used. Every script asserts `torch.cuda.is_available() is False`; outputs are in `results/cpu_reanalysis/` and scripts in `experiments/cpu_reanalysis/`. FKC outputs are excluded throughout. Finding IDs (T1, S1, R5, …) refer to `CODEBASE.md` (repo root; the brief said `docs/CODEBASE.md`). Tags: **[verified]** reproduced from data or code; **[likely]** mechanism shown, impact not measured; **[speculative]** plausible, not shown.*

## Summary

1. **The production anchor never moved.** The 9 Sep 0.164 is n=250 (problems 0–249, seed 0, CSD3); the 27 Sep 0.131 is n=1319. Those 250 problems are easier (0.163 vs 0.124), and the 27 Sep seeds average exactly 0.164 on them. [verified]
2. **The 9 Sep "0.2024 at 200k" production point has no source on disk.** No production checkpoint between 150k and 300k was ever evaluated, and none is on disk. The value equals an AG+SG-prev guidance cell (267/1319). Treat the 200k point as unsupported. [verified absent; the match is speculative]
3. **T1 applies to the production 500k run too.** It trained on 4 GPUs at commit `036a2b5` (18 Jun), with the rank-0-only estimator, so ranks 1–3 trained at σ_data = 0.5. [verified mechanism]. TASK 2 finds no measurable effect of the eval-time mismatch on one-step denoising (bits control); the training-time effect is untested
4. **Late training never sees σ < 0.15.** All three learned entropy pdfs have exactly zero mass there, and `entropy_gamma_max = 1.0` (T8). [verified]
5. **Voting only among programs that execute is free and large.** Production goes from 0.131 to 0.201 at k=3 and 0.227 at k=9. Plain majority voting gives nothing, because "no answer" wins the vote. This holds at every arm and checkpoint. [verified]
6. **About 70% of bits programs and 93–97% of token programs do not execute.** In the few that can be diagnosed, bits failures are mostly early stops before `return`, and token failures are mostly corrupted identifiers (NameError). Full program text was never saved. [verified, small biased subsample]
7. **"Token programs that run are more often correct" is partly selection.** Tokens execute on easier problems. On matched problems the gap shrinks from +26 pp to +15 pp [+8, +23], but it doesn't vanish. [verified]
8. **The 9 Sep meeting report has the guidance sign backwards.** Guidance is net-positive *below* a* ≈ 0.35 (CFG w=12: a* = 0.346 [0.310, 0.386]), not above it. [verified]
9. **Fixes are committed:** T1, T3, T4, T5, the safe executor (S1/S2), and the window study (R1–R3). T1, T3, T4 and T5 change behaviour relative to every past multi-GPU run or random-order eval.
10. **Conclusions that need a caveat:** any comparison against "0.164", the 200k point, the window study and token-vs-bits (T1/T3/T4), the window random arm (T5), and the "only positive above 33%" sentence.

---

## TASK 0: Safe executor

`evaluation/tasks/safe_exec.py` runs each program in its own `python -I -S -B` child process:

- RLIMIT_CPU = 5 s (soft; SIGXCPU) and 6 s (hard);
- RLIMIT_AS = 1 GiB;
- RLIMIT_FSIZE = 0, RLIMIT_CORE = 0 and RLIMIT_NPROC = 0;
- an empty environment and a temporary cwd;
- SIGKILL of the process group at a wall-clock deadline.

The limits are set in the child before the program is read, so the program can't raise them again. The child returns a structured `ExecResult`: `status` ∈ {ok, syntax_error, name_error, runtime_error, timeout, memory, no_return}, plus `exception_type`, `value`, `phase` and `no_function`. The builtins whitelist, the math-only import hook, `_extract_code`, `_to_number` and `_extract_text_answer` are unchanged, so scoring is identical.

`gsm8k_eval.py --executor {subprocess,inprocess}` defaults to `subprocess` and is recorded in the result JSON. The FKC path (`_run_fkc_gsm8k`) still uses the in-process sandbox, deliberately (no FKC edits).

**Tests** (`tests/test_safe_exec.py`, 33 tests) cover: the bare-`except` infinite loop (killed in ~1 s at a 1 s budget), `[0]*10**10` (→ memory), MemoryError swallowed by a bare except, writes and forks via the `__subclasses__()` escape (blocked), huge-integer returns, and 14 programs scored identically by both executors.

**Re-scoring check** (`t0_rescore_check.py` → `t0_rescore_check.json`). Full programs were never saved, so an end-to-end re-execution isn't possible. What was checked:

| Check | Result |
|---|---|
| Prod anchor s0 (`results/tokens/eval_gh200/…prod_anchor__ck000500000__s0.json`): stored answers re-scored against gold | 169/1319 = 0.128127, **exactly** as recorded |
| All 24 token-study cells, same re-score | all exact |
| 462 stored programs shorter than 400 chars (complete) re-executed with the new executor | answer and `correct` reproduced 462/462 |
| Old vs new executor on all 2,400 stored texts | 0 disagreements |

[verified] Caveat: the child still runs as you. A whitelist escape can *read* your files; it just can't write, fork, or run unbounded. This is not a jail.

## TASK 7: Answers from existing logs and files

**(a) The production 500k run** (`tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg`):

- **4 GPUs.** `config.json` has `system.world_size = 4`, and every checkpoint holds 4 CUDA RNG states. **1 node** [likely]: 4 devices were visible to rank 0, and CSD3 Ampere nodes have 4 A100s. No training logs are on this cluster.
- **Not before the estimator.** It started at `036a2b5` (2026-06-18, per `docs/prod_era_diagnostic.md`). `SigmaDataEstimator` dates from the initial release (`c998d00`, 10 May), already registered master-only, and the sidecar commit `d6d6f53` (17 Jun) is an ancestor of `036a2b5`.
- **So T1 applies.** Rank 0 trained at 0.3998 and ranks 1–3 at 0.5. [verified mechanism]
- **Other differences from the bits control:**
  - Production reached 500k at epoch 25; the control `ord_gh500k_none_s42` reached it at epoch 21.
  - The control trained at **lr 1e-4** (its `config.json`); production trained at 3e-4. They are not the same recipe.

**(b) Intermediate production checkpoints:**

- **None evaluated.** No evaluation of a production checkpoint between 150k and 300k exists in `results/` or `docs/`. 250k and 350k appear only as AutoGuidance "bad" models (`all_cells.csv`, `bad_checkpoint`).
- **Not on disk.** Only `step=000250000`, `350000`, `425000` and `last.pt` (500k) are on disk; there is no 200k checkpoint.
- **The 0.2024 has no source.** It is hard-coded in `docs/report_since_friday/make_figures.py:54` and quoted in `docs/report_meeting/report.tex:292` and `experiments_ordering_objective.md:248`. It matches, to four decimals, the AG w=15 + SG-prev w=2 cell at 500k (0.20243 = 267/1319, `paired_comparisons.json`).
- **Verdict:** the 200k point can't be shown to come from the same run as the 500k anchor, or from any eval of a 200k production checkpoint. [verified absent; the mis-transcription is speculative]

**(c) Zero-mass `entropy_pdf.pt` bins** (`t7_entropy_bins.py` → `t7_entropy_bins.json`; 128 log-spaced bins over σ ∈ [0.002, 80]):

| Run | Exactly-zero bins | σ range | Also zero |
|---|---|---|---|
| production | 0–51 | [0.0020, 0.148] | bin 125: [62.4, 67.8] |
| bits control `ord_gh500k_none_s42` | 0–51 | [0.0020, 0.148] | bin 127: [73.6, 80.0] |
| token `tok_gh500k_token_ce_s42` | 0–52 | [0.0020, 0.161] | bin 124: [57.5, 62.4] |

There are no near-zero bins below 1e-3 of the peak outside these. All three runs have `entropy_gamma_max = 1.0`, so after the transition σ is drawn only from this pdf: σ < 0.15 is never trained on again, and the bins can't refill (T8). [verified] The low-σ gap may be physically right: at σ = 0.15 a bit is resolved with SNR ≈ 7, and the matched filter dominates there. But it means TASK 2's σ ∈ {0.02, 0.05, 0.1} lie in a band the network hasn't seen since early training.

## TASK 1: Why the anchor moved

`t1_anchor.py` → `t1_anchor.json`.

| | 9 Sep | 27 Sep (token report) |
|---|---|---|
| File | `results/objective/eval/gsm8k_results_stochastic_g0.0_w0.0_s256_sd0.3998_ema1_production_500k.json` | `results/tokens/eval_gh200/…prod_anchor__ck000500000__s{0,1,2}.json` |
| Problems | **250** (idx 0–249) | **1319** |
| Seeds | 0 | 0, 1, 2 |
| Accuracy | 0.164 | 0.1281 / 0.1350 / 0.1312 (mean 0.1314) |
| Cluster / GPU | CSD3, A100 (`gpu-q-4`) | Isambard, GH200 |
| Commit | `7da3d97` (dirty) | `835d921` (dirty) |
| torch / CUDA | 2.8.0+cu128 / 12.8 | 2.8.0+cu129 / 12.9 |
| σ_data (source) | 0.399845 (run sidecar) | 0.399845 (run sidecar) |
| Schedule, sampler, steps, γ, guidance | Karras, DDIM, 256, 0, none | identical |
| EMA, checkpoint, config, prompt path | EMA, `last.pt`, `tinygsm_bits_cfg.py`, `gsm8k_test.json` | identical |

No other recorded field differs. Per-problem comparisons on the shared problems 0–249:

| | Accuracy on idx 0–249 | Accuracy on idx 250–1318 |
|---|---|---|
| 9 Sep | 0.164 | — |
| 27 Sep GH200 DDIM, seeds 0/1/2 | 0.188 / 0.176 / 0.128 (mean **0.164**) | 0.114 / 0.125 / 0.132 |
| CSD3 Karras, 1319 problems (`results/ordering/train_eval/…prod_anchor…uniform__s*`) | **0.164** / 0.188 / 0.136 | 0.124 / 0.130 / 0.120 |
| Pooled over 6 seeds | 0.163 [0.128, 0.201] | 0.124 [0.110, 0.139] |

- **Same cluster, same seed:** CSD3 seed 0 reproduces the 9 Sep result problem-for-problem (0 flips in either direction).
- **Across clusters:** 9 Sep vs 27 Sep seeds flips (wrong→right, right→wrong) by (11, 5), (19, 16), (12, 21); none is significant (McNemar p ≥ 0.16).
- **Seed noise alone:** the three 27 Sep seeds flip among themselves by the same amount: (13, 16), (10, 25), (11, 23).

**Conclusion [verified]:** the 3.3-point gap is entirely the problem subset. The first 250 GSM8K problems are about 4 points easier than the rest. There is no evidence of drift from the cluster, commit, σ_data, schedule or EMA. Any comparison of an n=1319 number against "0.164" is off by about 3 points. The right 1319-problem, 256-step, γ=0 anchor is **0.131–0.132** on either cluster.

## TASK 2: σ_data mismatch, forward passes only

`t2_sigma_data.py` → `t2_sigma_data_{bits_control,production}.json`.

**Design.** 256 held-out TinyGSM examples: rows drawn with seed 0 from the cache's `validation` split, which training never sees. σ ∈ {0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0}.

- **Pairing:** for each example and σ, ε is drawn once and reused in both arms (sidecar σ_data vs 0.5), so x_t is identical. σ_data enters only through c_in.
- **Logits:** via `_model_logits_continuous`, matched filter included; `likelihood.py` is not used (T9).
- **Self-conditioning: disabled.** x̂₀ is zeros on the suffix with the clean prompt injected, exactly the trainer's no-self-conditioning branch, in both arms.
- **Precision and threads:** fp32, 4 threads (the login node caps this user at 1 CPU, hence the ~1.5 h per checkpoint).
- **Metrics:** suffix only, split into content (answer + EOS; mean 139.7 tokens/example) and padding (321.6 tokens/example). Per-bit BCE, bit error rate, and token error rate of thresholded x̂₀. Paired bootstrap over examples (10k).

**Bits control, `ord_gh500k_none_s42`, `last.pt` (step 500k, EMA), sidecar σ_data = 0.399845** (`t2_sigma_data_bits_control.json`; 85.7 min on CPU, which exceeded the 30-min login-node guideline at the owner's request). n = 256 examples per cell; Δ = sidecar − 0.5, paired bootstrap over examples.

| σ | Content BCE: sidecar / 0.5 (Δ [95% CI]) | Content bit error | Content token error | Padding token error: sidecar / 0.5 |
|---|---|---|---|---|
| 0.02 | 1.6e-09 / 1.6e-09 (+5.2e-11 [-1.5e-09, +1.8e-09]) | 0.00000 / 0.00000 (+0.0e+00 [+0.0e+00, +0.0e+00]) | 0.00000 / 0.00000 (+0.0e+00 [+0.0e+00, +0.0e+00]) | 0.00000 / 0.00000 |
| 0.05 | 5.9e-09 / 8.4e-11 (+5.8e-09 [+5.0e-11, +1.7e-08]) | 0.00000 / 0.00000 (+0.0e+00 [+0.0e+00, +0.0e+00]) | 0.00000 / 0.00000 (+0.0e+00 [+0.0e+00, +0.0e+00]) | 0.00000 / 0.00000 |
| 0.1 | 2.8e-15 / 1.0e-15 (+1.8e-15 [-1.8e-16, +5.1e-15]) | 0.00000 / 0.00000 (+0.0e+00 [+0.0e+00, +0.0e+00]) | 0.00000 / 0.00000 (+0.0e+00 [+0.0e+00, +0.0e+00]) | 0.00000 / 0.00000 |
| 0.2 | 4.4e-06 / 4.5e-06 (-9.9e-08 [-4.5e-07, +2.3e-07]) | 1.6e-06 / 1.6e-06 (+0.0e+00 [+0.0e+00, +0.0e+00]) | 0.00002 / 0.00002 (+0.0e+00 [+0.0e+00, +0.0e+00]) | 0.00000 / 0.00000 |
| 0.35 | 0.00051 / 0.00051 (-7.1e-07 [-2.2e-05, +2.1e-05]) | 0.00010 / 0.00010 (-1.5e-06 [-7.7e-06, +3.9e-06]) | 0.00082 / 0.00088 (-5.6e-05 [-1.4e-04, +0.0e+00]) | 0.00000 / 0.00000 |
| 0.5 | 0.00139 / 0.00141 (-1.6e-05 [-3.8e-05, +4.8e-06]) | 0.00046 / 0.00046 (+1.1e-06 [-1.2e-05, +1.4e-05]) | 0.00349 / 0.00359 (-9.5e-05 [-2.6e-04, +4.7e-05]) | 0.00000 / 0.00000 |
| 1.0 | 0.01343 / 0.01346 (-3.5e-05 [-1.0e-04, +2.4e-05]) | 0.00491 / 0.00493 (-1.8e-05 [-9.4e-05, +4.1e-05]) | 0.01984 / 0.01996 (-1.2e-04 [-2.9e-04, +3.9e-05]) | 0.00001 / 0.00001 |

Padding BCE and bit error are below 1e-5 in both arms at every σ, so they aren't tabulated (all values are in the JSON).

**Result [verified for this probe]:**

- **σ ≤ 0.2:** both arms give essentially zero error, on content and padding alike. The matched filter alone resolves the bits.
- **σ = 0.35–1.0:** the sidecar value is very slightly better on every content metric, but no difference is significant.
  - largest content token error difference: σ=1.0, 0.0198 vs 0.0200 (Δ −1.2e-4 [−2.9e-4, +4e-5]);
  - largest relative difference: −6% token error at σ=0.35, CI touching 0.
- **Verdict:** evaluating with the "wrong" σ_data (0.5 instead of 0.40) does not measurably hurt one-step denoising for this model.

**Limits:**

- This is a single denoising step with a clean prompt, and errors are near zero up to σ=1, so the probe has little sensitivity. It doesn't test full sampling from σ=80, where c_in ≈ 1/σ is insensitive to σ_data anyway.
- It says nothing about the *training-time* effect of T1 (ranks mixing two preconditionings and loss weights). Only the 12k-step run (run-first item 1) addresses that.
- The c_in scale differs by only 4% between the arms at σ=1 (0.928 vs 0.894), consistent with the null. [likely]

**Production repeat: skipped.** The run was cancelled by the cluster admins for excessive login-node CPU use, after σ=0.02 (saved in `results/cpu_reanalysis/t2_partial_production.json`). It won't be re-run on the login node. **Missing data: the production rows of this table**; see run-first item 4 for the GPU version.

## TASK 3: Failure taxonomy

`t3_taxonomy.py` → `t3_taxonomy.json`. Scope is all non-FKC cells in `results/tokens/eval_gh200/` (3 seeds each).

**Missing data:** the guidance confirmation cells' per-problem and text files (CSD3 `runs/guidance/`) aren't on this cluster, so they aren't covered.

**Limits of the data.** Every problem has `correct` and the executed `answer` (None = did not run or returned nothing usable), which gives the coarse split below for all 1319 × 3. Program text exists only for the first 100 problems per cell, cut at 400 characters. Of the non-executing stored records, about 88% are exactly 400 chars long, i.e. cut by *logging*; their failure cause is unrecoverable and isn't guessed. The fine classes and truncation flags come from the remaining complete records. Those are biased toward short programs.

| Run | Step | Correct | Wrong answer | Non-exec | Non-exec, cut by logging | Non-exec, complete | Syntax | Name | Other runtime | Timeout/mem | No return | Truncated (tail / no-return / canvas) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bits control | 100k | 0.030 | 0.147 | 0.823 | 219 | 27 | 1 | 13 | 0 | 0 | 13 | 15 (2/14/0) |
| bits control | 200k | 0.072 | 0.158 | 0.770 | 207 | 29 | 3 | 11 | 0 | 0 | 15 | 15 (2/15/0) |
| bits control | 300k | 0.096 | 0.174 | 0.730 | 199 | 17 | 1 | 7 | 0 | 0 | 9 | 9 (0/9/0) |
| bits control | 400k | 0.117 | 0.173 | 0.709 | 189 | 28 | 1 | 6 | 0 | 0 | 21 | 21 (0/21/0) |
| production | 500k | 0.131 | 0.180 | 0.689 | 192 | 18 | 0 | 5 | 0 | 0 | 13 | 13 (1/13/0) |
| token `tok_gh500k` | 100k | 0.016 | 0.017 | 0.967 | 226 | 65 | 0 | 49 | 0 | 0 | 16 | 18 (2/18/0) |
| token `tok_gh500k` | 200k | 0.041 | 0.031 | 0.929 | 230 | 48 | 0 | 42 | 0 | 0 | 6 | 7 (0/7/0) |
| token `tok_fs500k` (b128) | 500k | 0.023 | 0.019 | 0.958 | 234 | 54 | 1 | 40 | 1 | 0 | 12 | 16 (5/15/0) |

The coarse fractions are over 1319 × 3 programs. The fine counts are over the first 100 problems × 3 seeds.

**Truncation flags.** A program is *tail* if it stops mid-statement, *no-return* if `simple_math_problem` parses but has no `return` with a value, and *canvas* if it fills the 512-token canvas.

**Findings:**

- **What fails, by arm.** Among complete non-executing programs, bits failures are mostly early stops before `return`, all flagged truncated (bits control 58 of 101, production 13 of 18). Token failures are mostly NameErrors that are **not** truncated (125 of 167).
- **What the NameErrors are.** The identifiers are corrupted between definition and use (`meters_per_sprint` → `meters_per_printprint`, `blue_fiber` → `blue_added`). That is a long-range coherence failure, not running out of canvas.
- **Truncated fraction** of complete non-executing programs: 0.59 (bits control, 60/101), 0.72 (production, 13/18), 0.25 (tokens, 41/167). [verified on the subsample; [likely] for the full population]
- **Canvas end:** no program reached it.
- **Timeouts and memory:** none. The S2 hazards haven't occurred in practice.
- **No mismatches:** no program recorded as non-executing executes now.

**Length.** Characters, censored at 400. Medians are over stored records.

| Arm | Correct: median, share ≥400 | Wrong answer: median, share ≥400 | Non-exec: median, share ≥400 |
|---|---|---|---|
| bits 400k | 399.5, 0.50 | ≥400, 0.69 | ≥400, 0.87 |
| production | ≥400, 0.51 | ≥400, 0.67 | ≥400, 0.91 |
| token 200k | 350, 0.38 | 379, 0.11 | ≥400, 0.83 |

Failing programs are longer than correct ones in every arm: 83–92% of non-executing records are ≥400 chars, vs 28–51% of correct ones. [verified, censored] Token-level lengths aren't recoverable.

## TASK 4: Voting that skips non-executing programs

`t4_voting.py` → `t4_voting.json`. Same cells as TASK 3, with the 3 seeds as samples (problem indices intersected; n = 1319 for every row). Ties get expected credit under a uniform random tie-break. Deltas are vs pass@1, with a paired bootstrap over problems.

| Run | Step | Exec rate | pass@1 | maj@3 (Δ) | maj@3, exec-only (Δ) | pass@3 (Δ) |
|---|---|---|---|---|---|---|
| bits control | 100k | 0.177 | 0.0298 | 0.0248 (−0.005 [−0.009, −0.001]) | 0.0570 (**+0.027** [+0.021, +0.034]) | 0.0713 (+0.041) |
| bits control | 200k | 0.230 | 0.0720 | 0.0655 (−0.007 [−0.012, −0.001]) | 0.1274 (**+0.055** [+0.047, +0.064]) | 0.1509 (+0.079) |
| bits control | 300k | 0.270 | 0.0960 | 0.0910 (−0.005 [−0.011, +0.001]) | 0.1600 (**+0.064** [+0.055, +0.074]) | 0.1918 (+0.096) |
| bits control | 400k | 0.291 | 0.1173 | 0.1082 (−0.009 [−0.016, −0.002]) | 0.1966 (**+0.079** [+0.069, +0.090]) | 0.2350 (+0.118) |
| production | 500k | 0.311 | 0.1314 | 0.1312 (−0.000 [−0.007, +0.006]) | 0.2010 (**+0.070** [+0.060, +0.080]) | 0.2365 (+0.105) |
| token `tok_gh500k` | 100k | 0.033 | 0.0162 | 0.0104 (−0.006 [−0.009, −0.003]) | 0.0353 (**+0.019** [+0.013, +0.025]) | 0.0371 (+0.021) |
| token `tok_gh500k` | 200k | 0.071 | 0.0407 | 0.0270 (−0.014 [−0.018, −0.009]) | 0.0853 (**+0.045** [+0.036, +0.053]) | 0.0895 (+0.049) |
| token `tok_fs500k` | 500k | 0.042 | 0.0230 | 0.0154 (−0.008 [−0.011, −0.004]) | 0.0493 (**+0.026** [+0.020, +0.033]) | 0.0508 (+0.028) |

**Production k-curve.** 9 samples per problem: three 3-seed sets at Karras, 256 steps, γ=0 (GH200 DDIM; GH200 ordered-uniform; CSD3 ordered-uniform). The duplicate `window_eval_gh200` copy is dropped. pass@k uses the unbiased estimator; maj@k is averaged over all k-subsets. [likely]: the three sets use near-equivalent, not identical, samplers.

| k | pass@k | maj@k (plain) | maj@k (exec-only) | exec-only − pass@1 |
|---|---|---|---|---|
| 1 | 0.132 | 0.132 | 0.132 | — |
| 2 | 0.186 | 0.132 | 0.171 | +0.039 [+0.035, +0.043] |
| 3 | 0.220 | 0.130 | 0.192 | +0.061 [+0.054, +0.068] |
| 5 | 0.258 | 0.131 | 0.214 | +0.082 [+0.073, +0.092] |
| 9 | 0.288 | 0.132 | 0.227 | +0.095 [+0.083, +0.107] |

**Findings:**

- **Plain voting is useless here.** "No answer" is the modal outcome, so plain majority voting is flat or harmful in every row.
- **Filtering to executing programs recovers most of the oracle gap.** At k=3, 66–70% of the pass@3 − pass@1 headroom for bits and production, 91–95% for tokens. [verified]
- **Compute-matched baselines need the filter.** Any comparison against "maj@k" (e.g. guidance + voting in `findings.md`) should state which vote it used. The executing-only vote costs nothing beyond the samples.
- **No larger-k data:** no non-FKC run on disk has more than 3 samples per problem from a single sampler.

## TASK 5: Token vs bits on matched problems

`t5_selection.py` → `t5_selection.json`. 200k cells: `tok_gh500k_token_ce_s42` vs the bits control `ord_gh500k_none_s42`, 3 seeds each, with problem indices intersected across all six cells and the production cells (1319 common).

| Subset | Arm | Executing programs | Correct | Acc. given exec, pooled | Acc. given exec, problem-weighted |
|---|---|---|---|---|---|
| All problems | token | 282 | 161 | **0.571** [0.505, 0.636] | — |
| All problems | bits | 912 | 285 | **0.312** [0.276, 0.349] | — |
| Both arms ≥1 exec (160 problems) | token | 216 | 133 | 0.616 [0.541, 0.684] | **0.559** [0.486, 0.632] |
| Both arms ≥1 exec (160 problems) | bits | 296 | 127 | 0.429 [0.359, 0.498] | **0.405** [0.339, 0.474] |

On matched problems, token − bits is +0.154 [+0.078, +0.230] problem-weighted (p < 1e-4) and +0.187 [+0.113, +0.260] pooled.

**Difficulty of the problems each arm runs.** "Production accuracy" is the mean over the 3 GH200 production seeds.

| Problems | n | Prompt tokens (mean) | Production accuracy | Gold `<<calc>>` steps |
|---|---|---|---|---|
| all | 1319 | 62.0 | 0.131 | 3.25 |
| token executes (any seed) | 214 | 48.5 | **0.354** | 2.67 |
| bits executes (any seed) | 606 | 51.8 | **0.238** | 2.88 |
| both execute | 160 | 46.2 | 0.423 | 2.61 |
| neither executes | 659 | 71.9 | 0.032 | 3.62 |

**Conclusion.** There is a real selection effect: the token arm only gets programs to run on short, easy problems [verified]. Controlling for it on the 160 matched problems shrinks the gap from 26 to 15 points, but it stays clearly positive. Within those problems, tokens still execute on fewer seeds (216 vs 296 programs), so within-problem selection — tokens running only on the easier seeds — isn't excluded [likely]. The token arm's deficit (0.55× bits overall) is an *execution* deficit, not a reasoning one: when a token program runs, it is right more often.

## TASK 6: Guidance fix/break arithmetic

`t6_fix_break.py` → `t6_fix_break.json`.

**Missing data:** the per-problem confirmation records (CSD3 `runs/guidance/`), so the n=1069 holdout can't be rebuilt.

**What is on disk.** `results/guidance/paired_comparisons.json` holds, per (config, seed), the paired Δ and the number of discordant problems on the **full n=1319** (512 steps, γ=0). That recovers fix and break counts exactly: fix − break = nΔ and fix + break = discordant. Each entry was matched to its seed through `all_cells.csv`, with duplicates dropped.

**Break-even.** The net gain is fix·(1−a) − break·a, which is positive iff a < a* = fix/(fix+break), with fix and break as rates. Guidance therefore helps **below** a*.

**CIs.** The binomial CIs treat each seed's fix and break counts as independent binomials. They ignore that the seeds share problems, so they are too narrow; the seed range is shown alongside.

| Config | Seeds | Fix rate (of baseline-wrong) | Break rate (of baseline-correct) | a* [binomial 95%] | a* seed range |
|---|---|---|---|---|---|
| CFG w=12 | 3 | 0.135 [0.124, 0.147] | 0.255 [0.219, 0.292] | **0.346** [0.310, 0.386] | 0.304–0.396 |
| CFG w=7 | 3 | 0.126 [0.115, 0.137] | 0.234 [0.199, 0.270] | **0.351** [0.313, 0.392] | 0.325–0.387 |
| CFG w=4 | 3 | 0.107 [0.097, 0.118] | 0.214 [0.181, 0.248] | **0.335** [0.296, 0.379] | 0.299–0.362 |
| CFG w=20 | 3 | 0.136 [0.124, 0.147] | 0.281 [0.245, 0.319] | 0.325 [0.293, 0.362] | 0.308–0.339 |
| SG-prev w=2 | 3 | 0.076 [0.067, 0.085] | 0.122 [0.095, 0.151] | **0.382** [0.327, 0.447] | 0.340–0.440 |
| SG-prev w=1 | 1 | 0.052 | 0.097 | 0.352 [0.252, 0.500] | — |
| AG w=15, bad=350k | 3 | 0.122 [0.111, 0.133] | 0.281 [0.245, 0.319] | **0.303** [0.270, 0.338] | 0.277–0.319 |
| AG w=6, bad=250k | 3 | 0.092 [0.083, 0.102] | 0.281 [0.245, 0.319] | 0.247 [0.218, 0.281] | 0.220–0.265 |

The baseline accuracy is 0.1385 (3 seeds). The identical 0.281 break rates for CFG w=20 and both AG configs are a coincidence (154/548 in each, from different per-seed counts).

**Churn regime.** `paired_comparisons.json` has no paired counts for the churn cells (`stoch_confirm`, `churn_anatomy` in `all_cells.csv`), so a* there can only be recomputed from the percentages in `docs/tiny_gsm8k_error_analysis.md` §3 (doc-derived, no CIs):

| Regime | Baseline | CFG | a* | Predicted |
|---|---|---|---|---|
| A, γ=0.3 | 0.252 | w=2 | 0.310 | positive (baseline < a*) |
| A, γ=0.3 | 0.252 | w=4 | 0.314 | positive |
| B, γ=0.41 | 0.316 | w=2 | 0.332 | marginally positive (observed +1 problem) |
| B, γ=0.41 | 0.316 | w=8 | 0.222 | negative (observed −10) |

**Sign check [verified].** `docs/report_meeting/report.tex:87` (9 Sep meeting report) says guidance is "only positive above a break-even baseline accuracy of ≈33%". The arithmetic gives the opposite: positive only *below* a* (0.33–0.35 for CFG, 0.30–0.38 across CFG, SG-prev and AG w=15), and net-harmful above it. `docs/tiny_gsm8k_error_analysis.md:86` already states it correctly ("turns net-harmful once baseline accuracy exceeds ≈33%"). The a* values above are consistent with that 33%.

**Proposed correction (not applied, awaiting your approval):**

> ~~…so its net value is arithmetic and only positive above a break-even baseline accuracy of ${\approx}33\%$.~~
> …so its net value is arithmetic: it is positive only while baseline accuracy is \emph{below} a break-even $a^* = \text{fix}/(\text{fix}+\text{break}) \approx 33$--$35\%$ (CFG $w{=}12$: $a^*{=}0.35$, 95\% CI $[0.31, 0.39]$, $n{=}1319$, 3 seeds), and it turns net-harmful above it.

## TASK 8: Code fixes

Every commit below was followed by a full CPU suite run, passing each time (566 → 568 → 574 → 575 → 579 passed, 0 failed).

| Commit | What | Changes behaviour vs past runs? |
|---|---|---|
| `3fa86b3` | S1/S2 safe executor (TASK 0) | No (scoring identical) |
| `86b7406` | **Window study, committed coherently:** sampler, trainer and eval diffs; `assets/noise_tables/tinygsm_window_F_v1.pt` plus a `.gitignore` exception; `analyze_window.py`, `window_sigma_field.py`; the `window_*` scripts; `window_summary.json`, `window_sigma_field.json`; `test_window_ordering.py`. Also fixes the `_window_ordering_enabled` stub (4 failures) and the `\| head -2` pipefail lint (1 failure). | No |
| `8d02c96` | `results/tokens/summary_gh200.json`, `train_summary_500k_gh200.json`, plus their producer `experiments/tokens/analyze.py` (R2) | No |
| `acd0562` | `train.py` `RESUME_LAYOUT_KEYS` (pre-existing uncommitted change, now committed separately) | No |
| `90824b8` | **T1:** broadcast rank 0's σ_data in `train()` after `on_train_begin` | **Yes:** ranks 1..N-1 now train at the estimate, not 0.5 |
| `204b5b5` | **T3:** per-rank seed `seed + 1_000_003·rank` after model/EMA/optimizer construction; DDP resume re-seeds from (seed, rank, step) instead of cloning rank 0's RNG | **Yes:** multi-GPU noise streams change; single-GPU unchanged |
| `e5cd9a2` | **T4:** `DistributedSampler(seed=cfg.train.seed)` | **Yes:** data order changes for any seed ≠ 0 |
| `c7fa61a` | **T5:** Ordered/WindowSampler generator seeded with `order_seed + 1_000_003·batch_index` | **Yes:** random-arm batches after the first get new permutations |

Tests are in `tests/test_rank_seeding.py` (13 tests, including a 3-process gloo broadcast on CPU) and `tests/test_safe_exec.py`.

**Not committed (out of scope, left as found):**
- the site/migration diffs (`scripts/site/*`, `scripts/hpc/guidance/env.sh`, `test_site_config.py`, `MIGRATION_ISAMBARD.md`); the window launchers source `site.sh`/`env.sh`, so they depend on these;
- `docs/experiments_ordering_objective.md`;
- `ordering_eval.slurm`, `token_train.slurm`, `deadlock_repro.slurm`;
- the ~900 per-cell eval JSONs.

**Still open:** T2 (deadlock) and R5 (`analyze_window.py` doesn't intersect problem indices).

## Run first when compute returns

GPU-h estimates use the measured ~2.0 it/s on 4 A100s (`docs/prod_era_diagnostic.md`) and ~240 s per 1319-problem, 256-step eval on one GH200 (`efficiency.wall_clock_s`). GH200 training throughput wasn't measured here.

1. **12k-step, lr 3e-4, from-scratch test of T1/T3/T4**, 4 GPUs, production recipe: about **7 GPU-h per seed**, ~20 GPU-h for 3 seeds. Every lr 3e-4 run on our side diverged between 4,481 and 11,278 steps (8/8; `experiments_ordering_objective.md`, `env_validation_plan.md`), while production — which also carried T1/T3/T4 — ran 500k cleanly. Surviving 12k with the fixes would point at T1/T3 as a contributor; diverging again would exonerate them. Also confirm the logged `[sigma_data] all ranks use …` matches the sidecar. This decides whether the window and token-vs-bits conclusions need re-runs, before any is quoted again.
2. **Save full program text in `gsm8k_eval`, then re-evaluate the production anchor** (3 seeds × 1319): about **0.2 GPU-h**. A small code change is needed first (all texts, uncut). It makes TASK 3's taxonomy complete instead of a 12% subsample.
3. **Production exec-only maj@k up to k=16** (16 seeds × 1319, 256 steps): about **1.1 GPU-h**. This turns TASK 4's free +7–9 pp into a single-sampler curve.
4. **TASK 2 for production and the token run, on GPU only** (not on a login node: the CPU attempt was cancelled by the admins). Under **0.1 GPU-h** each (3,584 forwards). Low priority: the bits-control result is a clean null.
5. **Only after (1):** decide on window/token re-runs with the fixed trainer. Quote GPU-h per arm before launching (the window study used ~2,780 GPU-h for 26 runs to 250k).
