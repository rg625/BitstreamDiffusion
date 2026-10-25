#!/usr/bin/env bash
# Stage 2 of the temperature x nucleus sweep: CoBit-M on the token-space control and the best stage-1 cells (seeds 0,1),
# then Maj@8 at the best temperature (seeds 2-7) for both models.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs/tg_temp.log"
run() { echo "=== $* start $(date -Is)" >> "$L"; python -W ignore diagnostics.py "$@" >> "$L" 2>&1; echo "=== exit $? $(date -Is)" >> "$L"; }
run --model M --variants tok1.0p1.0,tok0.2p1.0,tok0.5p0.8 --seeds 0,1
run --model S --variants tok0.2p1.0 --seeds 2-7
run --model M --variants tok0.2p1.0 --seeds 2-7
echo "=== temp stage 2 done $(date -Is)" >> "$L"
