#!/bin/bash
#SBATCH --job-name=cobit_m_tinygsm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# CoBit-M (462M) on TinyGSM: configs/tasks/tinygsm_bits_cobit_m.py.
# Submit through the site wrapper (partition/account/CPUs):
#
#   bash scripts/site/submit.sh scripts/hpc/arch/cobit_m_tinygsm.slurm --nodes=2
#
# Global batch is 512 whatever the node count (64/GPU on 2 nodes, 128/GPU on
# 1); COBIT_M_ACCUM splits it further if memory needs it. Chained 24 h jobs
# resume from last.pt (every 2k steps). The run stops at COBIT_M_STOP_AT_STEP
# (250k); continue to 500k with --export=ALL,COBIT_M_STOP_AT_STEP=500000.
#
# Smoke/throughput probe (own run dir, stops early, one timed numbered save):
#   --export=ALL,COBIT_M_TAG=smoke_n2,COBIT_M_STOP_AT_STEP=600,COBIT_M_CKPT_STEPS=500,COBIT_M_MEMLOG=1
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch
# train.py applies TINYGSM_TOTAL_STEPS to optim.total_steps after the resume
# merge; for this run that would re-lay the 500k cosine.
unset TINYGSM_TOTAL_STEPS
export COBIT_M_TAG="${COBIT_M_TAG:-s42}" COBIT_M_STOP_AT_STEP="${COBIT_M_STOP_AT_STEP:-250000}"
export COBIT_M_CKPT_STEPS="${COBIT_M_CKPT_STEPS:-100000,200000,250000}" COBIT_M_ACCUM="${COBIT_M_ACCUM:-1}"
CONFIG=configs/tasks/tinygsm_bits_cobit_m.py

source "$PROJECT_DIR/scripts/hpc/arch/deadlock_debug.sh"
export DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-20}"
enable_nccl_flight_recorder "$PROJECT_DIR/logs/arch"
start_stall_watchdog "$PROJECT_DIR/logs/arch/${SLURM_JOB_NAME:-cobit_m_tinygsm}_${SLURM_JOB_ID:-local}.log" "${STALL_MIN:-15}"
trap stop_stall_watchdog EXIT
NNODES="${SLURM_JOB_NUM_NODES:-1}"
echo "[cobit_m] commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- trainers train.py configs || echo ' (dirty)') | python $($COBIT_PYTHON -V 2>&1)"
echo "[cobit_m] tag=$COBIT_M_TAG stop=$COBIT_M_STOP_AT_STEP ckpts=$COBIT_M_CKPT_STEPS nodes=$NNODES accum=$COBIT_M_ACCUM -> $((512 / (4 * NNODES * COBIT_M_ACCUM)))/GPU"
echo "[cobit_m] start $(date +%s) $(date)"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader
compgen -G "datasets/tinygsm/*capall*.meta.json" >/dev/null || { echo "[cobit_m] FATAL no corpus" >&2; exit 1; }
MEMLOG=""
if [ "${COBIT_M_MEMLOG:-0}" = "1" ]; then
  MEMLOG="$PROJECT_DIR/logs/arch/${SLURM_JOB_NAME:-cobit_m_tinygsm}_${SLURM_JOB_ID:-local}_mem.csv"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 2 > "$MEMLOG" &
  MEMLOG_PID=$!
fi
set +e
if [ "$NNODES" -gt 1 ]; then
  source "$PROJECT_DIR/scripts/site/multinode_env.sh"
  srun "$COBIT_PYTHON" -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node=4 \
       --rdzv_backend=c10d --rdzv_id="$SLURM_JOB_ID" --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
       train.py --config "$CONFIG"
else
  srun "$COBIT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
       train.py --config "$CONFIG"
fi
rc=$?; set -e
[ $rc -ne 0 ] && echo "[cobit_m] rc=$rc -- check for OOM or [divergence-guard]"
if [ -n "$MEMLOG" ]; then
  kill "$MEMLOG_PID" 2>/dev/null || true
  awk -F', ' '{if ($2 > m[$1]) m[$1] = $2; t = $3} END {for (g in m) printf "[cobit_m] peak memory.used gpu%s: %d / %d MiB\n", g, m[g], t}' "$MEMLOG" | sort
fi
echo "[cobit_m] end $(date +%s) $(date)"; exit 0
