#!/usr/bin/env bash
# TinyGSM CoBit-S runs, queued after the GSM8K RoPE comparison.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs"
until grep -q "=== done" "$L/tg_gsm8k_bf16.log" 2>/dev/null; do sleep 30; done
run() { local n="$1"; shift; echo "=== $n start $(date -Is)" >> "$L/tg_chain.log"; python -W ignore "$@" > "$L/$n.log" 2>&1;
        echo "=== $n exit $? $(date -Is)" >> "$L/tg_chain.log"; }
run tg_static_S mse_tinygsm.py --model S --n-seq 4096 --grid 129 --tag static
run tg_true_S traj_tinygsm.py --mode truechain --models S --n 512 --tag true_S
echo "=== tg chain done $(date -Is)" >> "$L/tg_chain.log"
