#!/bin/bash
#SBATCH --job-name=cobit_obj
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# binary_sm vs binary_ce, 100k steps, one arm per job.
#
#   sbatch --export=ALL,OBJ_LOSS=binary_sm scripts/hpc/arch/objective_pilot.slurm
#   sbatch --export=ALL,OBJ_LOSS=binary_ce scripts/hpc/arch/objective_pilot.slurm
#
# ~49 GPU-h per arm at the measured steady-state 0.489 GPU-h/1k steps, i.e.
# ~12.3 h wall on 4 GPUs -- just inside the 12 h limit, so a resume may be
# needed for the tail. last.pt is written every 5000 steps, so resubmitting the
# same line continues from there.
#
# REQUIRES the FULL corpus cache (data.max_train_examples = None). The capped
# 20k probe cache is NOT sufficient: 100k steps at batch 512 is 51M samples and
# would cycle a 20k-row cache ~2500 times.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
mkdir -p logs/arch

export OBJ_LOSS="${OBJ_LOSS:?set OBJ_LOSS=binary_sm or binary_ce}"
export OBJ_STEPS="${OBJ_STEPS:-100000}"
CFG="configs/tasks/tinygsm_bits_objective.py"

echo "[obj] arm=$OBJ_LOSS steps=$OBJ_STEPS"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader

# Refuse the capped cache outright: training 100k steps on 20k rows would run to
# completion and produce a meaningless comparison rather than an error.
if ! compgen -G "datasets/tinygsm/*capall*.meta.json" > /dev/null; then
  echo "[obj] FATAL: full-corpus cache (capall) not found in datasets/tinygsm/." >&2
  echo "[obj] Build it first; the capped probe cache must not be used here." >&2
  exit 1
fi

srun "$COBIT_PYTHON" -m torch.distributed.run \
     --standalone --nproc_per_node=4 train.py --config "$CFG"
echo "[obj] arm=$OBJ_LOSS finished at $(date)"
