#!/bin/bash
#SBATCH --job-name=cobit_ordtrain
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# TRAINING-TIME temporal ordering. Four arms, run in parallel:
#   ORD_MODE=none   ORD_W=0     (control, uniform sigma)
#   ORD_MODE=l2r    ORD_W=0.25
#   ORD_MODE=r2l    ORD_W=0.25
#   ORD_MODE=random ORD_W=0.25
# All start from the SAME production 500k checkpoint with fresh optimiser
# state, same seed, same budget, same everything but cfg.train.ordering.
#
#   for m in none l2r r2l random; do
#     W=0.25; [ "$m" = none ] && W=0
#     sbatch --export=ALL,ORD_MODE=$m,ORD_W=$W,ORD_TAG=ft scripts/hpc/arch/ordering_train.slurm
#   done
#
# Python 3.10 (sedd310) by default -- set COBIT_PYTHON to override. Do NOT mix
# 3.9 and 3.10 results in one comparison.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch

export ORD_MODE="${ORD_MODE:?set ORD_MODE=none|l2r|r2l|random}"
export ORD_W="${ORD_W:?set ORD_W}"
export ORD_STEPS="${ORD_STEPS:-30000}"
export ORD_SEED="${ORD_SEED:-42}"
export ORD_TAG="${ORD_TAG:-ft}"
CFG="configs/tasks/tinygsm_bits_ordering.py"

echo "[ord] commit $(git rev-parse --short HEAD)"
echo "[ord] python $($COBIT_PYTHON -V 2>&1)"
echo "[ord] mode=$ORD_MODE w=$ORD_W steps=$ORD_STEPS seed=$ORD_SEED"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader
compgen -G "datasets/tinygsm/*capall*.meta.json" > /dev/null || { echo "[ord] FATAL: no corpus" >&2; exit 1; }

set +e
srun "$COBIT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
     train.py --config "$CFG"
rc=$?
set -e
[ $rc -ne 0 ] && echo "[ord] rc=$rc -- if the log shows [divergence-guard], this arm DIVERGED (a result, not a crash)"
echo "[ord] finished $(date)"
exit 0
