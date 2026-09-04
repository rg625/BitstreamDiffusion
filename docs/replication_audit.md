# Replication audit — our 0.1385 vs the collaborator's ~0.29

**Question.** Our DDIM baseline scores 0.1385 (512 steps) and 0.1390 (1024 steps)
at γ=0. The collaborator reports ~0.29 single-sample and a **+7.03 point** gain
from 512→1024 steps. Both cannot be descriptions of the same experiment.

**Method.** Every row below was established by reading the repository, the
recorded result JSONs and the checkpoint files — not by inference from the PDFs.
The collaborator's canonical cell is `results/gsm8k_passk_shardA/cobit.json`,
which records its own full configuration.

---

## 1. The two configurations, side by side

| Axis | Collaborator | Ours | How verified | Status |
|---|---|---|---|---|
| checkpoint | `cobit_raw_binary_bits` **base run**, step 425000 | `cobit_raw_binary_bits_cfg` **CFG run**, step 500000 | md5 of both files differ; `global_step` read from each | **DIFFERENT** |
| training config | `tinygsm_bits.py` | `tinygsm_bits_cfg.py` | full flattened config diff | **equivalent for sampling** — differs only in `cond.p_uncond` (0.0 → 0.1, training-only) and paths |
| `sigma_data` | 0.399844765663147 | 0.399844765663147 | both `sigma_data.json` sidecars read | **MATCHED** |
| EMA | `--ema 1` | default `apply_ema=True` | both checkpoints contain an `ema` block | **MATCHED** |
| seed | 42 | 42 | — | **MATCHED** |
| schedule | entropic | entropic | — | **MATCHED** |
| **churn knob** | `--churn_gamma 0.41` | `--gamma` | argparse + call sites | **DIFFERENT CODE PATHS** (see §2) |
| DDIM/EDM churn | **off** | on at our γ | `configure_stochastic` branch | **DIFFERENT** |
| sampler | `fkc_em` (Euler–Maruyama) | `ddim` | — | **DIFFERENT** |
| steps | 1024 | 512 | — | **DIFFERENT**, and a churn-dose axis (§2) |
| particles | 32, `resampling_policy=never` | 1 | `total_resample_events=0`, `min_ess=32/32` | **inert by construction** (§3) |
| problem set | 256 random, `data_gsm8k_shardA_256.json` | 1319 full test set | file absent from this cluster | **NOT REPRODUCIBLE** |
| precision | fp32 | **bf16 autocast** | `cfg.evaluation.use_amp=True`, `amp_dtype=bf16` | **DIFFERENT** |
| batch size | 2 | 64 | — | different, expected inert |

---

## 2. The finding that explains most of the gap

**`--gamma` and `--churn_gamma` are different knobs on different code paths.**

* `--gamma` → `configure_stochastic()` → `cfg.evaluation.stochastic` → EDM churn
  *inside the DDIM sampler*. This is the axis **we** swept.
* `--churn_gamma` → `fkc_churn_gamma` → the per-step churn of the FKC
  **`edm_churn` proposal**. This is the axis **they** used.

Their command passes `--churn_gamma 0.41` and never passes `--gamma`, so `--gamma`
takes its default of 0.0 and `configure_stochastic` disables the DDIM churn
outright (`if mode == "deterministic" or gamma <= 0.0: st.enabled = False`).

So the two studies both describe their sampler as "churn at 0.41", but reach it
through different implementations. Naively setting `--gamma 0.41` with
`sampler_kind=fkc_em` would configure the **opposite** of their run.

**Consequence for their +7.03 "step count" result.** `configure_stochastic` sets
`s_churn = γ·(N−1)`, and EDM takes `γ_i = min(S_churn/N, √2−1)`, so per-step
churn is roughly constant and **total injected noise scales with the step count**.
With γ pinned at 0.41 in every one of their cells, doubling 512→1024 doubles the
number of churn injections. Their step-count effect is therefore confounded with
a churn-dose effect. Our own measurement isolates it: at γ=0, 512→1024 gives
**+0.0005 [−0.0013,+0.0023]** — more steps alone buy nothing.

---

## 3. Why particle count is not an axis

They ran `resampling_policy=never` at β=1.0 with `final_resample=0`, and the
recorded cell confirms `total_resample_events=0` and `min_ess=32.0` out of 32.
The 32 particles are therefore independent draws with uniform weights, and the
number they quote (0.2909) is `particle_mean_accuracy` — the mean over
particles, whose expectation equals the K=1 single-sample accuracy. Running
K=32 would cost 32× and measure the same quantity, so the audit uses K=1.

---

## 4. What our existing data already says about the dominant axis

From `stoch_confirm` and `stoch_screen` (our checkpoint, DDIM, our churn path):

| γ | exact match | problems |
|---|---|---|
| 0.0 | 0.1367 | 1319 |
| 0.2 | 0.2421 | 1319 |
| 0.3 | 0.2568 | 1319 |
| 0.41 | 0.2720 | 250 |

**Stochasticity alone moves the baseline from 0.137 to ~0.27**, which closes all
but roughly two points of the gap to their 0.2909 — before any of the other four
axes are considered.

---

## 5. Ablation tree (grid `replication_audit`, 7 cells, ~5 GPU-h)

Starts at their configuration and removes one axis per cell, so each change in
accuracy is attributable to a single factor. All cells: 1319 problems, seed 42.

| Cell | Checkpoint | Sampler | Churn | Steps | Isolates |
|---|---|---|---|---|---|
| A | base 425k | fkc_em | `churn_gamma=0.41` | 1024 | their configuration |
| B | CFG 500k | fkc_em | `churn_gamma=0.41` | 1024 | checkpoint / training dropout |
| C | CFG 500k | ddim | `gamma=0.41` | 1024 | churn implementation + solver |
| D | CFG 500k | ddim | `gamma=0.41` | 512 | step count = churn dose |
| E | CFG 500k | ddim | `gamma=0.41` | 256 | step count = churn dose |
| F | CFG 500k | ddim | `gamma=0` | 1024 | **already measured: 0.1390** |
| G | base 425k | ddim | `gamma=0` | 512 | checkpoint at our setting |
| H | CFG 500k | ddim | `gamma=0` | 512 | precision (fp32 vs bf16) |

Submit:

```bash
EXPORTS="ALL,GUID_CFG_W=12,GUID_AG_W=15,GUID_SG_W=2,GUID_BAD_STEP=350000"
sbatch --array=0-6 --export="$EXPORTS,GRID=replication_audit" \
       scripts/hpc/guidance/array.slurm
```

---

## 6. Conclusion (interim)

**C — the results are not directly comparable.** They differ on five axes
simultaneously (checkpoint, churn implementation, solver, step count, problem
set), plus precision. Neither number is wrong; they describe different
experiments.

The dominant axis is **stochasticity**, and this is already established rather
than conjectured: our own γ sweep reproduces ~0.27 from a 0.137 baseline with no
other change. The residual ~2 points is what the ablation tree is for.

One substantive correction falls out for **their** study rather than ours: because
`s_churn` scales with the step count, their headline "+7.03 points from 512→1024
steps" cannot be read as a pure integration-resolution result. At γ=0 the same
doubling is worth +0.0005 in our hands.

**This conclusion is interim** — §5 has not run. It will be upgraded to A, B, C
or D once those seven cells complete. What is already final: `sigma_data`, EMA,
seed and schedule all match, and the config difference is provably
sampling-irrelevant, so none of those can explain the gap.

**Not reproducible at all:** their 256-problem shard A lives at
`/home/gb511/s-flm/data_gsm8k_shardA_256.json` and is not on this cluster. Every
audit cell uses the full 1319 problems instead, so a residual problem-set term
remains unquantified. Obtaining that file is the one external dependency.
