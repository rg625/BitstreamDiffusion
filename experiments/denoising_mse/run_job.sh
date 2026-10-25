#!/usr/bin/env bash
# Launch mse_curves.py runs on the local RTX 4090 (no Slurm on this host), sequentially.
# Usage: experiments/denoising_mse/run_job.sh <job-name> "<args run 1>" ["<args run 2>" ...]
# Writes experiments/denoising_mse/logs/<job-name>.log and prints the PID.
set -eo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="$1"; shift
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate sedd
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
LOG="$REPO/experiments/denoising_mse/logs/$NAME.log"
mkdir -p "$(dirname "$LOG")"
cd "$REPO"
{
  echo "=== $NAME start $(date -Is) host=$(hostname)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "$LOG"
CMDS=""
for a in "$@"; do
  CMDS+="echo \"=== run: $a \$(date -Is)\"; python experiments/denoising_mse/mse_curves.py $a || exit \$?; "
done
nohup bash -c "$CMDS echo \"=== $NAME exit \$? \$(date -Is)\"" >> "$LOG" 2>&1 &
echo "$!"
