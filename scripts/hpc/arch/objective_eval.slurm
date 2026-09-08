#!/bin/bash
#SBATCH --job-name=cobit_objeval
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# OBJECTIVE (binary_sm vs binary_ce) -- TASK PERFORMANCE AT A MATCHED BUDGET
#
# Both arms come from the production-matched stability runs (p_uncond=0.1,
# clamp OFF, guard armed, seed 42), at step 5,000 -- the largest checkpoint
# both arms share, and BEFORE either broke (sm broke at 6,356; ce at 11,444).
# So this is a like-for-like comparison of the two objectives at an identical
# training budget, on identical evaluation settings.
#
# Schedule is KARRAS, not entropic, ON PURPOSE: the entropic schedule is fitted
# per run from that run's own entropy tables, so it would differ between the
# arms and confound the objective with its sigma schedule. Karras is analytic
# and identical for both.
#
#   sbatch --export=ALL,OBJ_EVAL_LIMIT=250 scripts/hpc/arch/objective_eval.slurm
#
# EXPECTATION, stated in advance: production needed 500k steps for 13.85%.
# At 5k neither arm is remotely converged and both may score ~0. That is a
# legitimate result -- it would mean the task-performance question is not
# answerable at the budget this environment permits -- and it is recorded as
# such rather than being dressed up.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
mkdir -p logs/arch results/objective/eval

LIMIT="${OBJ_EVAL_LIMIT:-250}"
STEPS="${OBJ_EVAL_STEPS:-256}"
SEED="${OBJ_EVAL_SEED:-0}"
echo "[objeval] commit $(git rev-parse --short HEAD) limit=$LIMIT steps=$STEPS seed=$SEED"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader

for ARM in binary_sm binary_ce; do
  CK="runs/tasks/tinygsm/obj_${ARM}_stab_s42/checkpoints/step=000005000.pt"
  if [ ! -f "$CK" ]; then echo "[objeval] MISSING $CK" >&2; continue; fi
  echo "[objeval] ---- $ARM @5k"
  "$COBIT_PYTHON" evaluation/tasks/gsm8k_eval.py \
      --config configs/tasks/tinygsm_bits_cfg.py \
      --checkpoint "$CK" --ema 1 \
      --schedule karras --sampler_kind ddim --gamma 0.0 \
      --steps "$STEPS" --limit "$LIMIT" --seed "$SEED" \
      --out_dir results/objective/eval \
      --regime objective_5k --tag "${ARM}_5k" || echo "[objeval] $ARM FAILED"
done
echo "[objeval] finished at $(date)"
