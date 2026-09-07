#!/bin/bash
#SBATCH --job-name=cobit_objstab
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# STABILITY STUDY: binary_sm vs binary_ce, clamp OFF, production-matched,
# 20k steps, >=3 seeds per arm.
#
# PRIMARY ENDPOINT: time-to-divergence. NOT gradient survival -- the 5k smoke
# showed all four arms sitting at grad_survival ~0.18 before any event, and the
# unclamped SM arm broke at 4,720 with survival still 0.1765 afterwards. The
# endpoint the original pilot was built around does not separate the arms.
#
#   for s in 42 43 44; do
#     sbatch --export=ALL,OBJ_LOSS=binary_sm,OBJ_SEED=$s,OBJ_TAG=stab \
#            scripts/hpc/arch/objective_stability.slurm
#     sbatch --export=ALL,OBJ_LOSS=binary_ce,OBJ_SEED=$s,OBJ_TAG=stab \
#            scripts/hpc/arch/objective_stability.slurm
#   done
#
# BUDGET: 20k steps = 9.8 GPU-h at the measured 0.489 GPU-h/1k, ~2.7 h wall on
# 4 GPUs. Six runs is 58.7 GPU-h worst case. Expected is lower: with the guard
# at factor=4 a diverging SM arm aborts ~400 steps after the break (both smoke
# breaks were before 5k), so the SM arms should cost ~2.5 GPU-h each and the
# realistic total is ~37 GPU-h.
#
# The clamp stays OFF. It is exonerated as the cause (both SM arms broke with
# and without it) and it has no demonstrated benefit; OFF also reproduces the
# healthy 500k production recipe exactly.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
mkdir -p logs/arch

export OBJ_LOSS="${OBJ_LOSS:?set OBJ_LOSS=binary_sm or binary_ce}"
export OBJ_SEED="${OBJ_SEED:?set OBJ_SEED, the replication axis of this study}"
export OBJ_STEPS="${OBJ_STEPS:-20000}"
export OBJ_TAG="${OBJ_TAG:-stab}"
export OBJ_PROBE_EVERY="${OBJ_PROBE_EVERY:-100}"
export OBJ_CKPT_EVERY="${OBJ_CKPT_EVERY:-5000}"
CFG="configs/tasks/tinygsm_bits_objective.py"

# The clamp must stay off; refuse rather than silently reintroduce a variable
# that the 2x2 already exonerated.
if [ -n "${OBJ_WMAX:-}" ] && [ "${OBJ_WMAX}" != "none" ] && [ "${OBJ_WMAX}" != "off" ]; then
  echo "[stab] FATAL: OBJ_WMAX=${OBJ_WMAX} set. This study runs clamp OFF." >&2
  exit 1
fi

echo "[stab] arm=$OBJ_LOSS seed=$OBJ_SEED steps=$OBJ_STEPS clamp=off"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader

if ! compgen -G "datasets/tinygsm/*capall*.meta.json" > /dev/null; then
  echo "[stab] FATAL: full-corpus cache (capall) not found." >&2
  exit 1
fi

set +e
srun "$COBIT_PYTHON" -m torch.distributed.run \
     --standalone --nproc_per_node=4 train.py --config "$CFG"
rc=$?
set -e
# The divergence guard exits non-zero ON PURPOSE. That is a RESULT (the arm
# diverged), not an infrastructure failure, and it must not look like one.
if [ $rc -ne 0 ]; then
  echo "[stab] arm=$OBJ_LOSS seed=$OBJ_SEED exited rc=$rc"
  echo "[stab] if the log shows [divergence-guard], this arm DIVERGED -- that is the measurement."
fi
echo "[stab] arm=$OBJ_LOSS seed=$OBJ_SEED finished at $(date)"
exit 0
