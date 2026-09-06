#!/bin/bash
#SBATCH --job-name=cobit_objsmoke
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# 5k-step SMOKE TEST for the CE/SM pilot. ~2.5 GPU-h per arm (~37 min wall on
# 4 GPUs), ~5 GPU-h for both. Run this BEFORE committing the 98 GPU-h pilot.
#
#   sbatch --export=ALL,OBJ_LOSS=binary_sm scripts/hpc/arch/objective_smoke.slurm
#   sbatch --export=ALL,OBJ_LOSS=binary_ce scripts/hpc/arch/objective_smoke.slurm
#
# WHAT IT DISCRIMINATES
#   The first pilot's CE arm diverged at step 6,580 and the SM arm at 19,540.
#   5k steps therefore does NOT prove stability on its own -- it is a floor, not
#   a certificate. What it does establish:
#     * both arms reach 5k with the loss still falling (the old CE arm was
#       already at 3.9e3 by 10k, so a broken CE arm shows up well before 5k);
#     * the divergence guard is armed and has not misfired on healthy noise;
#     * the objective probe is actually writing objective/* scalars -- the bug
#       that voided the entire first pilot;
#     * the weight clamp is active and bounded.
#   PASS/FAIL is judged on those four, not on accuracy.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
mkdir -p logs/arch

export OBJ_LOSS="${OBJ_LOSS:?set OBJ_LOSS=binary_sm or binary_ce}"
export OBJ_STEPS="${OBJ_STEPS:-5000}"
export OBJ_PROBE_EVERY="${OBJ_PROBE_EVERY:-100}"   # denser than the pilot's 500
export OBJ_CKPT_EVERY="${OBJ_CKPT_EVERY:-2500}"
export OBJ_TAG="${OBJ_TAG:-smoke}"
CFG="configs/tasks/tinygsm_bits_objective.py"

echo "[smoke] arm=$OBJ_LOSS steps=$OBJ_STEPS probe_every=$OBJ_PROBE_EVERY"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader

if ! compgen -G "datasets/tinygsm/*capall*.meta.json" > /dev/null; then
  echo "[smoke] FATAL: full-corpus cache (capall) not found." >&2
  exit 1
fi

srun "$COBIT_PYTHON" -m torch.distributed.run \
     --standalone --nproc_per_node=4 train.py --config "$CFG"
echo "[smoke] arm=$OBJ_LOSS finished at $(date)"
