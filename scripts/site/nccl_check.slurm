#!/bin/bash
#SBATCH --job-name=cobit_nccl
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/site/%x_%j.log
#
# Multi-node NCCL check: one node's bandwidth first (the NVLink reference),
# then all nodes of the allocation. Submit through scripts/site/submit.sh:
#
#   bash scripts/site/submit.sh scripts/site/nccl_check.slurm [--nodes=N]
#
# PASS: the multi-node line says correct=True, the log shows NCCL using
# "AWS Libfabric" (NET/OFI, provider cxi) and not NET/Socket, and busbw is
# tens of GB/s rather than ~1.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?submit through scripts/site/submit.sh}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
source scripts/site/multinode_env.sh
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET

echo "[nccl] single node:"
srun --nodes=1 --ntasks=1 "$COBIT_PYTHON" -m torch.distributed.run --standalone \
     --nproc_per_node=4 scripts/site/nccl_check.py 2>&1 | grep -E "nccl_check|NET/|Using network" | sort -u | head -20 || true

echo "[nccl] all $SLURM_JOB_NUM_NODES nodes:"
srun "$COBIT_PYTHON" -m torch.distributed.run --nnodes="$SLURM_JOB_NUM_NODES" \
     --nproc_per_node=4 --rdzv_backend=c10d --rdzv_id="$SLURM_JOB_ID" \
     --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" scripts/site/nccl_check.py \
     2>&1 | grep -E "nccl_check|NET/|Using network|rror" | sort -u | head -30 || true
echo "[nccl] finished $(date)"
