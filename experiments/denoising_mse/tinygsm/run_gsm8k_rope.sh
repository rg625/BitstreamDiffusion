#!/usr/bin/env bash
# GSM8K accuracy of TinyGSM CoBit-S 250k at the branch's 256-NFE operating point, eager bf16 RoPE vs fp32 RoPE.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs"
python -W ignore gsm8k_eval.py --model S --gamma 0.31 --steps 256 --no-rope-fix > "$L/tg_gsm8k_bf16.log" 2>&1
python -W ignore gsm8k_eval.py --model S --gamma 0.31 --steps 256 > "$L/tg_gsm8k_fp32.log" 2>&1
echo "=== done $(date -Is)" >> "$L/tg_gsm8k_bf16.log"
