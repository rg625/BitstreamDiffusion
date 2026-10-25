# Compute-allocation diagnostic: CoBit-S on LM1B at fixed NFE

*Plan only; nothing has been run. Drafted 2026-10-02.*

## 1. Question

The budget is fixed at the production setting: NFE 256, the entropic σ grid and churn γ = 0.2. With that budget, **which part of the denoising trajectory limits sample quality, and why?**

"Why" means one of three causes:

- **Step size:** the steps in that region are too coarse.
- **Uncorrected errors:** mistakes made there are never revised.
- **The model:** neither more resolution nor more correction helps, so the limit is the network itself.

Out of scope: NFE scaling, training or fine-tuning, schedule search, adaptive controllers. See §11.

## 2. What the sampler does now

The production LM1B sampler is configured in `configs/lm1b/continuous/eval/rate_eval_seeds.py`, spec γ = 0.200.

- **Steps.** 255 Euler steps on the probability-flow ODE plus one final denoise, so NFE = 256 (`evaluation/nfe.py`).
- **Equal cost per step.** Every NFE is one forward pass over 128 tokens × 15 bits. So the only thing that decides where compute goes is the σ grid.
- **The σ grid.** `SigmaSchedule.prepare(schedule="entropic")` (`diffusion/continuous/samplers.py:1018`) samples the inverse of the training entropy-rate CDF (`assets/entropy_tables/lm1b/`) at evenly spaced quantiles. Each step therefore covers an equal share of entropy-rate mass.
- **Churn.** EDM churn (`_apply_edm_churn`, `samplers.py:215`) runs on every step, because the entropy band is set to its full range [0, 1]. Each step uses γ = 0.2 and S_noise = 1.003, and churn costs no NFE.
- **Self-conditioning.** It runs in `carry` mode: the previous step's x̂₀ is passed in at no NFE cost.

Two results are already known:

| Change at NFE 256 | GenPPL effect on LM1B | Source |
|---|---|---|
| Karras grid → entropic grid | improves by 30–40 points | README, Figure 4 in the paper |
| Deterministic → stochastic (churn) | 82.90 → 59.76 | README Table 1 |

So *where* steps go and *whether errors get corrected* both matter a lot. What isn't known is **which region the current grid under-serves, and by which of the three causes.**

## 3. Design idea

**Segments.** Cut the 255 steps into 8 consecutive segments of 32 steps each. The last segment has 31 steps plus the final denoise. Because the grid is uniform in entropy-CDF quantile, each segment resolves an equal share (1/8) of the entropy-rate mass and gets an equal share of compute. **The entropic grid therefore assumes every segment deserves equal compute. This plan tests that assumption directly.**

**Paired branches.** Every intervention starts from a saved state of one reference trajectory and runs to the end. The branch and the reference share every random number except the one being intervened on. Each sample is then its own control, so a single batch of 1024 samples is enough, and branches that start partway through cost only the remaining steps.

## 4. Fixed settings, the same in every run

| Setting | Value |
|---|---|
| Checkpoint | `cobit_s_lm1b_1M_ema.pt` from `gbatzolis/CoBit`, saved to `runs/paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/checkpoints/step=001000000.pt` |
| Entropy tables | `assets/entropy_tables/lm1b/` |
| σ range, σ_data | σ_min = 0.002, σ_max = 80, σ_data = 0.5 |
| Sampler | `ddim_entropic`, 255 steps + final denoise (NFE 256), `sc_refresh_mode="carry"`, `ati_eta=0` |
| Churn | γ = 0.200, S_noise = 1.003, `window_mode="entropy_cdf"`, band [0.0, 1.0] |
| Samples | 1024, seed 42, fixed batch size and order in every branch |
| Scoring | `openai-community/gpt2-large`, `external_ppl.micro_batch_size` set to 64–128 to fit in 24 GB, plus unigram token entropy |
| Precision | as released: fp32 with TF32 matmuls (`run_eval.py:164`) |

## 5. Run 1: anchor and logged reference

**Purpose**
- Check that the setup reproduces the paper.
- Measure the real cost of one run.
- Produce the reference trajectory and saved states that every later branch starts from.

**What it does.** One stochastic sampling run of 1024 samples with the §4 settings. During the run it records:

- **Boundary states:** at the start of each segment (steps 0, 32, …, 224), the full sampler state. That is x, **the self-conditioning carry state**, and the step index. Without the self-conditioning state, a branch can't replay the reference.
- **Per-step token guesses:** each token's most likely id from the denoiser's estimate D at every step (int16, about 67 MB).
- **Per-step summaries:** mean per-bit entropy of D for each token, and the size of the churn noise.
- **Final samples:** decoded text, per-sample GPT-2-large NLL and token count, and per-sample unigram entropy.

**Pass criterion.** GenPPL within the single-seed range the README states (≈ 59–60) and entropy ≈ 4.31. If it fails, stop and debug before anything else.

**CPU analysis, no GPU needed**
- **Settling time:** for each token, the last step at which its guess changes. Plot a histogram by segment.
- **Changes after first settling:** per segment, the fraction of tokens whose guess changes again after first settling. Errors that do get revised show up here.
- **Bit position:** settling time broken down by bit position, from most to least significant of the 15 bits.

**What it shows.** Where decisions are actually made, compared with the uniform 32 steps per segment. It describes; it doesn't establish cause.

**Cost.** About one cell, roughly 0.03 GH200 GPU-h (estimated; this run measures it).

## 6. Run 2: per-segment probes

**Purpose.** Find the limiting segment and its cause.

**Branches.** Each branch starts from a saved boundary state of Run 1, changes only segment s, and then follows the reference settings to the end.

| Branch | Change, applied in segment s only | Extra NFE | What it measures |
|---|---|---|---|
| `refine[s]`, s = 0…7 | Each of the 32 steps is split into 4 sub-steps placed evenly in entropy-CDF quantile. Churn is applied once per original step, with the **same γ and the same noise** as the reference. | +96 | Error caused by step size in s |
| `nochurn[s]`, s = 0…7 | γ = 0 for the steps in s | 0 | How much error correction in s contributes |
| `null` | Replay the reference unchanged from the segment-0 state | 0 | Noise floor: how far an unchanged replay drifts because of GPU non-determinism |

That makes 17 branches in total, each producing 1024 final samples.

**What is held identical between a branch and the reference.** The starting noise, the churn noise of every original step, the batch layout and the checkpoint. The only difference is the change made in segment s.

**Cost.** Summed over all branches, about 3,300 NFE per sample. That equals about 13 full cells, roughly 0.4 GH200 GPU-h. GPT-2 scoring of 17 × 1024 samples adds less than 0.05.

## 7. Metrics and statistics (Runs 1 and 2)

Each branch is compared sample by sample against the reference. Per branch, record:

- **ΔNLL/token:** the change in mean GPT-2-large NLL per token (main metric). An increase means worse. As a guide, 0.01 nats is about 0.6 GenPPL at 60, roughly one sampling-seed standard deviation.
- **Δentropy:** the change in mean unigram entropy.
- **Tokens changed:** the fraction of tokens that differ from the reference, after subtracting the `null` floor.

**Uncertainty.** 95% confidence intervals from a 10,000-resample bootstrap over the 1024 paired samples.

**When an effect counts.** Fixed in advance:

- **Material:** the confidence interval excludes 0 **and** |ΔNLL/token| ≥ 0.01.
- **Entropy flag:** a branch with |Δentropy| > 0.02 is flagged. Its ΔNLL is reported but not interpreted on its own, because lower entropy lowers NLL without meaning better samples.
- **Borderline:** if a segment's interval straddles the 0.01 line, run a second 1024-sample batch for **that segment only**.

**Spread across samples and tokens, from the same data, CPU only**
- **Concentration:** is a segment's effect spread over all samples or carried by a few? Report the share of the total ΔNLL coming from the top 10% of samples.
- **Which tokens change:** position in the sequence, token frequency, and bit position.

## 8. How to read Run 2

| Observed pattern | Conclusion |
|---|---|
| `refine[s]` is material for one or a few segments | **Step size** is the bottleneck in those segments. The equal-compute assumption fails there, and those segments need more steps. |
| `nochurn[s]` costs much more in some segments, while `refine[s]` there is not material | **Uncorrected errors** are the bottleneck there. What that region needs is revision of earlier decisions, not finer steps. |
| Both are material in the same segment | Both causes are present. Compare the sizes of the two effects. |
| No `refine[s]` is material, and `nochurn[s]` is spread evenly across segments | **The model** is the limit. Moving compute around at NFE 256 won't help. Stop here and skip Run 3. |
| The pattern doesn't match where Run 1 shows decisions being made | Entropy rate isn't measuring impact: the regions that matter most aren't the ones resolving the most information. Report this as a separate finding. |

## 9. Run 3, optional: test the bottleneck at fixed compute

Run 2 changes compute in one segment as a probe. Run 3 is the only test that the located bottleneck limits quality **while total compute stays at 256**. Which version to run depends on the Run 2 result:

| Run 2 result | The single comparison, NFE fixed at 256 |
|---|---|
| Step size in segment s* | Move 32 steps from the segment with the smallest `refine` effect into s*, and re-sample the inverse CDF at the new per-segment density. Compare with the unmodified entropic grid. |
| Uncorrected errors in s* | Raise γ inside s*, and lower it elsewhere to keep entropy fixed. This costs no NFE. Compare with uniform γ = 0.2. |
| Model | Skip Run 3. |

**Design.** 2 conditions (modified vs baseline) × 3 values of γ (bracketing entropy 4.31, so both can be compared at matched entropy by interpolation) × 3 sampling seeds × 1024 samples. Report GenPPL at entropy 4.31, with a confidence interval from paired-seed differences.

**Cost.** 18 cells, about 0.55 GH200 GPU-h.

## 10. Code changes needed (no GPU needed to develop them)

Everything goes in `DDIMSampler.sample` (`samplers.py:1715`; loop at `:1853–1960`) plus a new driver script. The default behaviour must stay bit-identical.

1. **Per-step noise from its own seed.** Draw the churn noise for original step i from a generator seeded with `(base_seed, i)`, instead of one running global stream. Otherwise extra sub-steps shift every later draw and the pairing breaks. The starting noise is covered by the saved state.
2. **Save and restore state.** Add an optional hook that saves `(x, sc_state, i)` at given step indices, and an entry point that starts the loop from a saved state at step i.
3. **Per-segment changes.** An optional `segment_overrides = {s: {"substeps": 4, "gamma": None|0.0}}` setting. For sub-steps, apply churn once at the first sub-step of each original step, using that step's seeded noise; the self-conditioning state updates at every sub-step.
4. **Per-step logging.** An optional callback that receives `(i, sigma, D)` and stores token guesses and per-bit entropy summaries.
5. **Per-sample scoring.** `external_perplexity.py` already computes NLL and token count for each sequence before averaging. Write those per-sample values out, together with per-sample entropy (`evaluation.text_metrics.token_unigram_entropy_from_token_ids`).
6. **Driver.** `experiments/compute_allocation/diagnose.py` with three subcommands:
   - `reference` (Run 1);
   - `probe --segment s --kind refine|nochurn|null` (Run 2), written so a rerun skips finished branches, one output file per branch;
   - `analyze` (CPU only: bootstrap, settling-time and concentration analyses).

**Tests (CPU, small model).** Keep each test short; on Isambard, don't run them on the login node for long.

- With no overrides, the output is identical to the current sampler.
- `null` replay from a saved state matches the uninterrupted run exactly on CPU.
- `refine` with 1 sub-step is identical to the reference.
- The churn noise drawn for step i doesn't depend on how many sub-steps came before it.

## 11. Deliberately left out

- **NFE scaling.** Not the question.
- **Deterministic-sampler arm.** Production is stochastic, and the 82.9 vs 59.76 gap is already known.
- **Other ways of spending extra compute** (Heun, restart, Langevin corrector, `refined` self-conditioning). `refine` and `nochurn` already separate the two causes.
- **Multiple seeds and γ grids in Runs 1–2.** The paired comparison makes them unnecessary; seeds are added only for borderline segments.
- **Schedule search, adaptive controllers, per-token policies, training.**

## 12. Cost and execution

The per-cell cost is estimated from TinyGSM throughput scaled to LM1B's shorter sequences; Run 1 replaces it with a measured figure.

| | GH200 GPU-h | One RTX 4090 (24 GB) |
|---|---|---|
| Run 1 | ~0.03 | ~5–10 min |
| Run 2 | ~0.4 | ~1–2 h |
| Run 3 (optional) | ~0.55 | ~1–2 h |
| **Total** | **~0.5 (Runs 1–2), ~1 with Run 3** | **~2–4 h** |

**Order:** Run 1 → check the anchor → Run 2 → analyze → decide on Run 3.

**Regime.** All runs are on one GPU type. Results from a 4090 are not pooled with GH200 numbers or with the paper's 59.76; Run 1 is the anchor for comparison.

## 13. Deliverables

1. **Settling-time figure (Run 1):** where tokens settle, by segment and bit position, against the uniform step allocation.
2. **Per-segment result table (Run 2):** ΔNLL/token for `refine[s]` and `nochurn[s]`, with confidence intervals and entropy flags.
3. **One-paragraph verdict:** the bottleneck segment or segments and the cause (step size, uncorrected errors or the model), using §8.
4. **Concentration result:** whether the bottleneck sits in a few samples, tokens or bit positions, or affects everything evenly.
5. **Optional (Run 3):** the fixed-NFE comparison at matched entropy confirming or refuting the bottleneck.
