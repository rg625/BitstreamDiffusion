# Guidance experiments — CSD3 runbook

Exact commands, in the order they should be run. Everything assumes:

```bash
cd /rds/user/rg625/hpc-work/BitstreamDiffusion
export COBIT_PYTHON=/home/rg625/miniforge3/envs/sedd/bin/python
export PYTHONPATH=$PWD
```

Account `GIROLAMI-SL2-GPU`, partition `ampere` (A100-80GB). Check the balance
before a large array: `mybalance`.

---

## 0. Preconditions

```bash
# All four CFG-run checkpoints must load (they are ~2.1 GB each).
ls -la tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/

# The SmolLM tokenizer must be cached locally: compute nodes are OFFLINE.
ls hf_cache/hub/models--HuggingFaceTB--SmolLM-135M
```

If the tokenizer cache is missing, populate it **from the login node** (which
does have outbound network) before submitting anything:

```bash
HF_HOME=$PWD/hf_cache $COBIT_PYTHON -c \
  "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M')"
```

---

## 1. Tests (free, run first)

```bash
PYTHONPATH=. $COBIT_PYTHON -m pytest tests/ -q
```

Expect 225 passed + 1 pre-existing unrelated failure
(`test_codeword_lowT_full_codebook_is_perbit_map`).

---

## 2. Smoke test — always before a sweep

```bash
sbatch scripts/hpc/guidance/smoke.slurm
```

Runs baseline / CFG / AG / SG-prev / SG-exact / CFG+AG+SG-prev on 16 problems
at 32 steps, then validates the outputs and **fails the job** if a cell is
missing, non-finite, lacks provenance, or is indistinguishable from baseline.
Its accuracies are meaningless — do not report them.

Read the per-cell wall clock from the log; it sets the `--time` and array width
for everything below.

---

## 3. Review a grid before submitting it

```bash
$COBIT_PYTHON -m experiments.guidance.grids list
$COBIT_PYTHON -m experiments.guidance.grids show ag
```

---

## 4. Sweeps

Array bounds come from the grid itself, so they cannot drift out of sync:

```bash
submit () {   # submit <grid> [extra sbatch args...]
  local g="$1"; shift
  local n=$($COBIT_PYTHON -m experiments.guidance.grids size "$g")
  echo "grid $g: $n cells -> --array=0-$((n-1))"
  sbatch --array=0-$((n-1)) --export=ALL,GRID="$g" "$@" \
         scripts/hpc/guidance/array.slurm
}
```

Staged, cheapest and most informative first:

```bash
submit cfg_coarse        # 12 cells  — Phase 10, CFG scale
submit ag                # 31 cells  — Phase 11, badness x scale
submit sg                # 36 cells  — Phase 12, SG-prev vs SG-exact x NFE
submit sg_delta          # 10 cells  — SG log-sigma offset sensitivity
submit sg_mf             #  6 cells  — matched-filter hold-vs-vary ablation
submit cfg_stochastic    #  6 cells  — Phase 15, CFG under EDM churn
```

Throttle concurrency on a busy queue with `%`:

```bash
submit ag --array=0-30%8
```

Cells are **idempotent** — a cell whose result JSON exists is skipped — so a
partially completed array can be resubmitted wholesale.

---

## 5. Aggregate and analyse

```bash
$COBIT_PYTHON -m experiments.guidance.aggregate runs/guidance --out results/guidance
$COBIT_PYTHON -m experiments.guidance.analyse  results/guidance
```

Writes `all_cells.csv`, `per_step.csv`, `paired_comparisons.json`,
`factorial_effects.json` and the figures. Regenerate rather than editing.

---

## 6. Factorial and NFE grids (after the single-axis optima are known)

These two grids need one operating point per axis, which only the sweeps above
can supply — so they are read from the environment rather than hard-coded.
Set them to whatever the single-axis sweeps found, and **export them for the
job too**, or the array will build a different grid than you reviewed:

The single-axis confirmations have now supplied those operating points
(§7 results): **CFG w=12, AG w=15 with bad=step 350000, SG-prev w=2**.

```bash
export GUID_CFG_W=12 GUID_AG_W=15 GUID_SG_W=2 GUID_BAD_STEP=350000

$COBIT_PYTHON -m experiments.guidance.grids show factorial_confirm   # review first
$COBIT_PYTHON -m experiments.guidance.grids show nfe
```

**The two grids still outstanding.** Submit exactly these (the `GUID_*` values
must be in `--export`, not merely in your shell, or the array silently builds a
*different* grid than the one you reviewed — the defaults are w=3/1.5/0.5):

```bash
EXPORTS="ALL,GUID_CFG_W=12,GUID_AG_W=15,GUID_SG_W=2,GUID_BAD_STEP=350000"

# Phase 14 — quality vs compute. 28 cells, n=250, NFE 8..512. Cheap (~3 GPU-h).
sbatch --array=0-27 --export="$EXPORTS,GRID=nfe" \
       scripts/hpc/guidance/array.slurm

# Phase 13 — the 2x2x3 factorial at confirmation grade.
# 36 cells = 12 conditions x 3 seeds, 1319 problems, 512 steps (~18 GPU-h).
sbatch --array=0-35 --export="$EXPORTS,GRID=factorial_confirm" \
       scripts/hpc/guidance/array.slurm
```

### The two control grids — submit these FIRST

They are cheap and they decide whether the headline findings survive; the
factorial and NFE grids are only worth their GPU-hours if these come back the
right way.

```bash
# 1. Is SG-prev guidance, or just a 2nd-order solver? 15 cells, ~7 GPU-h.
#    Heun-256 (512 NFE) vs SG-prev-512 (512 NFE) is the discriminating pair.
sbatch --array=0-14 --export="$EXPORTS,GRID=solver_control" \
       scripts/hpc/guidance/array.slurm

# 2. maj@2 / maj@3: the deployable compute-matched baseline. 4 cells, ~2 GPU-h.
#    Seeds 45-48, disjoint from the confirmation grids on purpose.
sbatch --array=0-3 --export="$EXPORTS,GRID=compute_control" \
       scripts/hpc/guidance/array.slurm

# 3. Cheap ablations. 3 + 4 cells, well under 1 GPU-h each.
sbatch --array=0-2 --export="$EXPORTS,GRID=null_ablation" \
       scripts/hpc/guidance/array.slurm
sbatch --array=0-3 --export="$EXPORTS,GRID=ag_ema" \
       scripts/hpc/guidance/array.slurm
```

Then re-run §5 plus:

```bash
$COBIT_PYTHON -m experiments.guidance.compute_matched results/guidance
```

which picks up maj@k automatically once `per_problem.answer` is present (runs
from before that change report pass@k only).

Cells are idempotent (a cell whose result JSON exists is skipped), so a
partially failed array can be resubmitted wholesale. Afterwards re-run §5;
`analyse.py` estimates the main effects and the two- and three-way interactions
automatically once the factorial grid is complete — until then it prints
`factorial grid incomplete; interaction effects not estimated`.

`--export=ALL` in `submit` carries the `GUID_*` variables into the job.
`GUID_BAD_STEP` is validated against the files on disk and fails loudly if the
checkpoint is missing.

---

## 7. Confirmation (Phase 16)

`grid_confirm(cells, seeds=...)` promotes a shortlist to the full 1319-problem
set at >=512 steps across seeds, keeping problems and initial noise matched so
comparisons stay paired.

---

## 8. Optional: train a weaker AutoGuidance model

```bash
sbatch scripts/hpc/guidance/train_bad_model.slurm
```

**Read the header first.** It needs the tokenized TinyGSM cache
(`datasets/tinygsm/*.meta.json`), which is not in this checkout. And the good
run was configured with `every_steps=25_000`; its eval directories show
checkpoints existed at 100k/150k/200k/225k/300k/350k with accuracies spanning
13.0% -> 25.6%. Recovering those gives a wider, already-measured badness axis
at zero GPU cost. Train only if they are genuinely gone.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `env.sh: No such file or directory` | SLURM runs the script from `/var/spool/...`; `$(dirname "$0")` is not the repo. Scripts resolve `PROJECT_DIR` instead — don't reintroduce `dirname`. |
| Job hangs at tokenizer load | `hf_cache` missing the SmolLM tokenizer; compute nodes are offline. Populate it from the login node (§0). |
| `PytorchStreamReader ... failed finding central directory` | Checkpoint is truncated or still being copied. Check `lsof <file>` for a writer before blaming the code. |
| Every guidance direction is 0.0 | The checkpoint is untrained. SDT uses AdaLN-zero, so at init the output is input-independent. The smoke validator reports this explicitly. |
| `--ag_scale > 0 requires --bad_checkpoint` | AutoGuidance needs a second model; pass one. |
