#!/bin/bash
#SBATCH --job-name=cobit_calloc
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# Adaptive-compute study on CoBit-S LM1B (frozen, released 1M EMA checkpoint):
# reference trajectory + counterfactual compute branches + GPT-2-large scoring,
# one CHUNK of CA_N samples at a time. Submit through scripts/site/submit.sh.
#
#   CA_CHUNKS="0"        space-separated chunk ids, run in order
#   CA_N=256             samples per chunk
#
# Idempotent: every reference / branch / score file is written once and skipped
# on a rerun, so a killed job resumes where it stopped.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
CA_CHUNKS="${CA_CHUNKS:?set CA_CHUNKS}"
CA_N="${CA_N:-256}"
mkdir -p logs/arch
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for c in $CA_CHUNKS; do
  echo "[calloc] chunk $c n=$CA_N  $(date -Is)"
  "$COBIT_PYTHON" -m experiments.compute_allocation.run chunk --chunk "$c" --n "$CA_N"
done
echo "[calloc] done $(date -Is)"
