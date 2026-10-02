#!/usr/bin/env bash
# Launch one adaptive.py job on the local RTX 4090 (this host has no Slurm).
# Usage: experiments/adaptive_compute/run_job.sh <job-name> <adaptive.py args...>
# Writes experiments/adaptive_compute/logs/<job-name>.log and prints the PID.
set -eo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="$1"; shift
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate sedd
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
# (README's PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is not supported by torch 2.0.1)
LOG="$REPO/experiments/adaptive_compute/logs/$NAME.log"
mkdir -p "$(dirname "$LOG")"
cd "$REPO"
{
  echo "=== $NAME start $(date -Is) host=$(hostname) args: $*"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "$LOG"
nohup bash -c "python experiments/adaptive_compute/adaptive.py $* ; echo \"=== $NAME exit \$? \$(date -Is)\"" >> "$LOG" 2>&1 &
echo "$!"
