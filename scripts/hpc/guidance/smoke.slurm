#!/bin/bash
#SBATCH --job-name=cobit_guid_smoke
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/guidance/smoke_%j.log
#
# One cheap job that exercises every code path a sweep will use, before any
# sweep is submitted:
#
#   1. the environment resolves and CUDA initialises
#   2. the good checkpoint loads and EMA applies
#   3. the AutoGuidance bad checkpoint loads as a second network
#   4. baseline / CFG / AG / SG-prev / SG-exact / the 3-way combination all run
#   5. results are written with full provenance and are machine-readable
#   6. resource usage and per-cell wall clock are reported, so the sweep's
#      --time and array width can be set from measurement rather than guesswork
#
# Deliberately tiny: 16 problems, 32 steps. It is a plumbing test, not an
# experiment -- the accuracies it prints are not meaningful and must not be
# reported as results.
#
#     sbatch scripts/hpc/guidance/smoke.slurm
set -euo pipefail

source "$(dirname "$0")/env.sh"

OUT_DIR="${OUT_DIR:-runs/guidance/smoke}"
LIMIT="${LIMIT:-16}"
STEPS="${STEPS:-32}"
BS="${BS:-16}"
mkdir -p "$OUT_DIR" logs/guidance

CONFIG="configs/tasks/tinygsm_bits_cfg.py"
RUN="tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg"
GOOD="$RUN/checkpoints/step=000425000.pt"
BAD="${BAD:-$RUN/checkpoints/step=000250000.pt}"

for f in "$GOOD" "$BAD"; do
  [ -f "$f" ] || { echo "[smoke] missing checkpoint: $f" >&2; exit 2; }
done

common=(--config "$CONFIG" --checkpoint "$GOOD"
        --sampler deterministic --sampler_kind ddim --schedule entropic --gamma 0.0
        --steps "$STEPS" --limit "$LIMIT" --batch_size "$BS" --seed 42
        --out_dir "$OUT_DIR" --collect_diagnostics)

run_cell () {
  local tag="$1"; shift
  echo ""
  echo "=================== smoke cell: $tag ==================="
  local t0=$(date +%s)
  srun "$COBIT_PYTHON" -m evaluation.tasks.gsm8k_eval "${common[@]}" --tag "$tag" "$@"
  echo "[smoke] cell '$tag' took $(( $(date +%s) - t0 ))s"
}

run_cell smoke_baseline   --guidance_scale 0
run_cell smoke_cfg        --guidance_scale 3
run_cell smoke_ag         --guidance_scale 0 --ag_scale 2 --bad_checkpoint "$BAD"
run_cell smoke_sgprev     --guidance_scale 0 --sg_scale 1 --sg_variant prev
run_cell smoke_sgexact    --guidance_scale 0 --sg_scale 1 --sg_variant exact
run_cell smoke_cfg_ag_sg  --guidance_scale 3 --ag_scale 2 --bad_checkpoint "$BAD" \
                          --sg_scale 1 --sg_variant prev

echo ""
echo "=================== smoke summary ==================="
"$COBIT_PYTHON" -m experiments.guidance.aggregate "$OUT_DIR" --check-smoke
