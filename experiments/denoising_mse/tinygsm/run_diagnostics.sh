#!/usr/bin/env bash
# GSM8K diagnostics (diagnostics.py): Maj@N (base, seeds 0-31), SC reset / decay, confidence clamping, staggered noise
# (2 seeds each), CoBit-S first, then CoBit-M.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs/tg_diag.log"
V="screset4.0,screset2.5,screset1.3,scdecay0.5,scdecay0.8,scdecay0.9,clampS0.99,clampP0.99,clampS0.99s1.0,clampP0.99s1.0,stagger0.0,stagger0.05,stagger0.1,stagger0.2"
run() { echo "=== $* start $(date -Is)" >> "$L"; python -W ignore diagnostics.py "$@" >> "$L" 2>&1; echo "=== exit $? $(date -Is)" >> "$L"; }
run --model S --variants base --seeds 0-31
run --model S --variants "$V" --seeds 0,1
run --model M --variants base --seeds 0,1
run --model M --variants "$V" --seeds 0,1
run --model M --variants base --seeds 2-31
echo "=== diag done $(date -Is)" >> "$L"
