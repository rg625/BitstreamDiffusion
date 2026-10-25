#!/usr/bin/env bash
# Stage 1 of the temperature x nucleus sweep (token-space decode, diag_sampler.tok*): CoBit-S, seeds 0 and 1, paired
# with the base runs. Temperature only, top-p only at T=1, a joint grid, and sigma-gated (sigma <= 2.5) versions.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs/tg_temp.log"
run() { echo "=== $* start $(date -Is)" >> "$L"; python -W ignore diagnostics.py "$@" >> "$L" 2>&1; echo "=== exit $? $(date -Is)" >> "$L"; }
run --model S --variants tok1.0p1.0,tok0.7p1.0,tok0.5p1.0,tok0.3p1.0,tok0.2p1.0,tok0.1p1.0 --seeds 0,1
run --model S --variants tok1.0p0.95,tok1.0p0.9,tok1.0p0.8 --seeds 0,1
run --model S --variants tok0.5p0.9,tok0.3p0.9,tok0.5p0.8,tok0.3p0.8 --seeds 0,1
run --model S --variants tok0.3p1.0s2.5,tok0.5p0.9s2.5 --seeds 0,1
echo "=== temp stage 1 done $(date -Is)" >> "$L"
