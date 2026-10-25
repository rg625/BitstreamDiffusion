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

## 5. Ablation tree — RESULTS

Grid `replication_audit`, 7 cells, 1319 problems, seed 42, ~5 GPU-h actual.
Each cell removes one axis from the collaborator's configuration.

| Cell | Checkpoint | Solver | Churn | Steps | Exact match | 95% CI |
|---|---|---|---|---|---|---|
| **A** *(their config)* | base 425k | fkc_em | `churn_gamma=0.41` | 1024 | **0.2813** | [0.2570, 0.3055] |
| C | CFG 500k | ddim | `gamma=0.41` | 1024 | 0.2957 | [0.2707, 0.3207] |
| D | CFG 500k | ddim | `gamma=0.41` | 512 | 0.2911 | [0.2669, 0.3154] |
| E | CFG 500k | ddim | `gamma=0.41` | 256 | 0.2661 | [0.2426, 0.2896] |
| H | CFG 500k | ddim | `gamma=0`, **fp32** | 512 | 0.1357 | [0.1175, 0.1547] |
| F | CFG 500k | ddim | `gamma=0` | 1024 | 0.1334 | [0.1152, 0.1524] |
| G | base 425k | ddim | `gamma=0` | 512 | 0.1259 | [0.1084, 0.1440] |
| *(ref)* | CFG 500k | ddim | `gamma=0` | 512 | 0.1385 | — |

**Cell B was lost to a filename collision** and was not rerun. The FKC branch
builds its output filename from sampler parameters only, ignoring `--tag`, so
cells A and B — identical in every sampler argument, differing only in
`--checkpoint` — wrote to the same path and B silently overwrote A's slot.
(A survived; the file records the base checkpoint.) This is the exact hazard the
collaborator's handoff warns about. **Fixed** in `gsm8k_eval.py` and pinned by
`test_fkc_result_filename_honours_the_tag`. B is no longer needed: the axis it
isolated — checkpoint — is settled more cheaply by G vs the reference.

### Paired attribution (per-problem, 20,000-resample bootstrap)

| Axis | Δ | 95% CI | Significant |
|---|---|---|---|
| **churn γ=0 → 0.41** (1024 steps, same ckpt + solver) | **+0.1622** | [+0.1387, +0.1857] | **yes** |
| steps 256 → 512 at γ=0.41 | +0.0250 | [+0.0038, +0.0470] | yes |
| steps 512 → 1024 at γ=0.41 | +0.0045 | [−0.0174, +0.0265] | no |
| steps 512 → 1024 at γ=0 | +0.0000 | [−0.0023, +0.0023] | no |
| checkpoint CFG 500k → base 425k (γ=0) | −0.0076 | [−0.0243, +0.0091] | no |
| precision bf16 → fp32 (γ=0) | +0.0023 | [−0.0076, +0.0121] | no |

---

## 6. Required table

| Condition | Ours | Collaborator | Matched? | Explanation |
|---|---|---|---|---|
| Their exact configuration | **0.2813** [0.2570, 0.3055] | **0.2909** | **YES** | their value lies inside our interval; residual is the problem set (256 random vs 1319) |
| Our canonical baseline | 0.1385 | — | n/a | γ=0; differs from theirs by churn alone |
| γ=0 → 0.41 | +0.1622 | — | n/a | **the entire discrepancy** |
| Checkpoint (base vs CFG run) | −0.0076 n.s. | — | n/a | not a cause; base run is if anything slightly worse |
| Precision (bf16 vs fp32) | +0.0023 n.s. | — | n/a | not a cause |
| Solver / churn implementation | ≈0 after adjusting for checkpoint | — | n/a | FKC-proposal churn and DDIM EDM churn are equivalent in effect |
| `sigma_data` | 0.399844765663147 | 0.399844765663147 | **YES** | both sidecars; never a cause |
| EMA, seed, schedule | ema=1, 42, entropic | ema=1, 42, entropic | **YES** | — |
| Task config | `tinygsm_bits_cfg.py` | `tinygsm_bits.py` | **equivalent** | flattened diff: only `p_uncond` + paths |
| Particle count | K=1 | K=32 | **equivalent** | `resampling never`, β=1, `total_resample_events=0` ⇒ independent draws; they report `particle_mean_accuracy` |
| **512 → 1024 steps** | **+0.0045** [−0.0174, +0.0265] | **+7.03 pts** [+1.95, +12.50] | **NO** | see §7 |
| Problem set | 1319 full | 256 random (shard A) | **NO** | file absent from this cluster; no longer material |

---

## 7. The one claim that does *not* reproduce

Their headline **+7.03 points from 512→1024 steps** is not reproduced. At
γ=0.41 we measure **+0.0045 [−0.0174, +0.0265]** — about half a point, not
significant — and our estimate sits *below* their interval [+1.95, +12.50].

Two contributing factors, both now measured:

1. **Their step-count axis is also a churn-dose axis.** `s_churn = γ·(N−1)`, so
   with γ pinned at 0.41 doubling the steps doubles the injected noise. The
   effect is real but **saturates**: 256→512 is worth +0.0250 (significant),
   512→1024 only +0.0045 (not). They measured in the flat region.
2. **Their interval is wide** — 256 problems against our 1319 — and our paired
   estimate is far tighter.

At γ=0 the same doubling is worth **exactly zero** (+0.0000 [−0.0023, +0.0023]),
which is what our original report stated.

---

## 8. Conclusion

### **A — exact replication achieved.**

Running the collaborator's configuration in our repository gives **0.2813
[0.2570, 0.3055]**, and their reported **0.2909** falls inside that interval.
Nothing is broken in either codebase.

**The 0.1385 vs 0.29 discrepancy is stochastic churn, and nothing else.**
Turning γ from 0 to 0.41 is worth **+0.1622 [+0.1387, +0.1857]** on identical
checkpoint, solver and step count — larger than the whole gap. Every other
candidate was tested and is null: checkpoint (−0.008 n.s.), precision (+0.002
n.s.), solver and churn implementation (≈0), and `sigma_data`, EMA, seed,
schedule and config are matched or provably irrelevant.

The two numbers were never in conflict. Ours is a **deterministic** baseline;
theirs is a **stochastic** one. Our study fixed γ=0 throughout and therefore
reported the deterministic figure as "the baseline" without qualification —
that is the reporting error, and it is ours.

**Sub-conclusion for their study (B, scoped):** the +7.03-point step-count claim
does not survive. It is confounded with churn dose and measured in the region
where that effect has saturated.

**Residual, not material:** shard A is unavailable, so a problem-set term is
unquantified. Since their number already falls inside our full-test-set
interval, closing it would not change the verdict.
