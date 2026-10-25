#!/usr/bin/env bash
# Compute-demand curves (A: +2 SC refreshes per 32-step segment, CoBit-S) and model switching (B: CoBit-M on part
# of the trajectory) on GSM8K, seeds 0 and 1; per-problem correctness for difficulty stratification.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs"
for seed in 0 1; do
  echo "=== cd seed $seed start $(date -Is)" >> "$L/tg_cd.log"
  python -W ignore compute_demand.py --configs all --seed $seed >> "$L/tg_cd.log" 2>&1
  echo "=== cd seed $seed exit $? $(date -Is)" >> "$L/tg_cd.log"
done
echo "=== cd done $(date -Is)" >> "$L/tg_cd.log"
