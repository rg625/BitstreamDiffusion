#!/bin/bash
#SBATCH --job-name=cobit_verify
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/site/%x_%j.log
#
# GATE 1: the device half of the environment check, which a login node cannot
# run (no GPU driver there). Partition and account come from submit.sh:
#
#   COBIT_SITE=isambard COBIT_ACCOUNT=<project> COBIT_PYTHON=<env>/bin/python \
#     bash scripts/site/submit.sh scripts/site/verify_gpu.slurm
#
# Pass condition: the log ends in "[verify] GATE 1 PASS". Roughly 5 minutes on
# 4 GPUs, i.e. well under half a GPU-hour.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?submit through scripts/site/submit.sh, which exports PROJECT_DIR}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/site

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
rc=0
"$COBIT_PYTHON" scripts/site/verify_env.py --require-gpu || rc=1
srun "$COBIT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$COBIT_GPUS_PER_NODE" \
     scripts/site/verify_env.py --nccl || rc=1
if [ $rc -eq 0 ]; then echo "[verify] GATE 1 PASS"; else echo "[verify] GATE 1 FAIL"; fi
exit $rc
