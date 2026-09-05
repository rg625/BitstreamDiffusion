#!/bin/bash
#SBATCH --job-name=cobit_bad_train
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/guidance/bad_train_%j.log
#
# Train a deliberately under-trained "bad" CoBit model for AutoGuidance.
#
# Same architecture, same data, same p_uncond=0.1 as the good run -- only the
# step budget differs, so the AG direction isolates training quality. Keeps
# every 25k checkpoint, giving the badness ladder {25k, 50k, 75k, 100k} that a
# single bad model could not provide.
#
#   sbatch scripts/hpc/guidance/train_bad_model.slurm
#   sbatch --dependency=afterany:<jobid> scripts/hpc/guidance/train_bad_model.slurm   # next link
#
# BEFORE SUBMITTING, read this:
#
#   This job needs the tokenized TinyGSM cache (datasets/tinygsm/*.meta.json),
#   which is NOT present in this checkout. Building it requires downloading
#   TinyGSM and running scripts/tasks/prebuild_task.py first -- see
#   scripts/tasks/prebuild_tinygsm_csd3.slurm. Budget for that separately.
#
#   The cheaper alternative is almost certainly better: the good run was
#   configured with checkpointing every 25k steps, and its evaluation
#   directories show checkpoints existed at 100k/150k/200k/225k/300k/350k with
#   measured accuracies spanning 13.0% -> 25.6%. Recovering those from wherever
#   the current checkpoint upload originates gives a WIDER, already-validated
#   badness axis at zero GPU cost. Only train if they are truly unavailable.
set -euo pipefail

# SLURM copies the batch script to /var/spool/slurm/slurmd/job<id>/slurm_script,
# so $(dirname "$0") points at the spool directory, NOT the repository. Resolve
# the environment from PROJECT_DIR instead.
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"

CONFIG="configs/tasks/tinygsm_bits_cfg_bad.py"
RUN_DIR="runs/tasks/tinygsm/cobit_raw_binary_bits_cfg_bad"
export BAD_TOTAL_STEPS="${BAD_TOTAL_STEPS:-100000}"
FINAL_CKPT="$RUN_DIR/checkpoints/step=$(printf '%09d' "$BAD_TOTAL_STEPS").pt"

mkdir -p logs/guidance

# --- safeguard: the tokenized cache must exist -------------------------------
if ! ls datasets/tinygsm/*.meta.json >/dev/null 2>&1; then
  echo "ERROR: no TinyGSM cache (datasets/tinygsm/*.meta.json). Refusing to train." >&2
  echo "       Build it first (see scripts/tasks/prebuild_tinygsm_csd3.slurm)." >&2
  exit 1
fi

# --- stop condition: already done --------------------------------------------
if [ -f "$FINAL_CKPT" ]; then
  echo "[bad-train] $FINAL_CKPT already exists; nothing to do."
  exit 0
fi

echo "[bad-train] target steps=$BAD_TOTAL_STEPS  run_dir=$RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Not bare `torchrun`: the conda env is not on PATH inside the srun step, so it
# dies with "execve(): torchrun: No such file or directory" (cost the first
# throughput probe two jobs). Go through the interpreter env.sh already resolved.
srun "$COBIT_PYTHON" -m torch.distributed.run \
     --standalone --nnodes=1 --nproc_per_node=4 --max-restarts=3 \
     train.py --config "$CONFIG"

echo "[bad-train] link finished at $(date). Checkpoints:"
ls -la "$RUN_DIR/checkpoints" 2>/dev/null || true
if [ ! -f "$FINAL_CKPT" ]; then
  echo "[bad-train] not yet at $BAD_TOTAL_STEPS steps -- submit another link with:"
  echo "  sbatch --dependency=afterany:${SLURM_JOB_ID:-JOBID} $0"
fi
