#!/usr/bin/env bash
# Global score temperature (DDIMSampler score_temp_tau, b71eb71; diag variant stemp<tau>[u]) sweep on GSM8K,
# CoBit-S, seeds 0 and 1 (paired with the base runs): sharpened tau grid + unsharpened-final-step variants.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs/tg_stemp.log"
run() { echo "=== $* start $(date -Is)" >> "$L"; python -W ignore diagnostics.py "$@" >> "$L" 2>&1; echo "=== exit $? $(date -Is)" >> "$L"; }
run --model S --variants stemp0.95,stemp0.9,stemp0.85,stemp0.8,stemp0.7,stemp0.5,stemp0.9u,stemp0.8u --seeds 0,1
echo "=== stemp S done $(date -Is)" >> "$L"
