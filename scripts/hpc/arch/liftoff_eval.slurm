#!/bin/bash
#SBATCH --job-name=cobit_liftoff
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# WHERE DOES GSM8K ACCURACY LIFT OFF ZERO?
#
# We know 5,000 steps -> 0.0000 and 200,000 -> 0.2024, with nothing in between.
# That gap is what makes "how long must we train?" unanswerable, and it decides
# whether a CE-vs-SM performance comparison costs ~25 or ~600 GPU-h.
#
# Uses PRE-BREAK checkpoints only, so every model here is healthy:
#   obj_binary_sm      @10k  (original pilot, broke at 19,540)
#   obj_binary_ce_stab @10k  (production-matched, broke at 11,444)
# Identical eval settings to the 5k cells and the production anchor:
# karras, DDIM, gamma=0, 256 steps, EMA, seed 0, 250 problems.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch results/objective/eval
LIMIT="${LIFTOFF_LIMIT:-250}"
run () {
  local ck="$1" tag="$2"
  [ -f "$ck" ] || { echo "[liftoff] MISSING $ck"; return; }
  echo "[liftoff] ---- $tag"
  "$COBIT_PYTHON" evaluation/tasks/gsm8k_eval.py \
    --config configs/tasks/tinygsm_bits_cfg.py --checkpoint "$ck" --ema 1 \
    --schedule karras --sampler_kind ddim --gamma 0.0 \
    --steps 256 --limit "$LIMIT" --seed 0 \
    --out_dir results/objective/eval --regime liftoff --tag "$tag" \
    || echo "[liftoff] $tag FAILED"
}
run runs/tasks/tinygsm/obj_binary_sm/checkpoints/step=000010000.pt        sm_10k_prebreak
run runs/tasks/tinygsm/obj_binary_ce_stab_s42/checkpoints/step=000010000.pt ce_10k_prebreak
echo "[liftoff] finished $(date)"
