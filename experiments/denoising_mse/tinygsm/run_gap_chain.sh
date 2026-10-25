#!/usr/bin/env bash
# Remaining runs to match the OWT study exactly: token-position runs (2048 x 97, as the OWT tok run) and
# GSM8K for CoBit-S on CoBit-M's schedule (the 2 x 2 schedule table).
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs"
run() { local n="$1"; shift; echo "=== $n start $(date -Is)" >> "$L/tg_gap.log"; python -W ignore "$@" > "$L/$n.log" 2>&1;
        echo "=== $n exit $? $(date -Is)" >> "$L/tg_gap.log"; }
run tg_gsm8k_S_schedM gsm8k_eval.py --model S --sched M --gamma 0.31 --steps 256
run tg_tok_S mse_tinygsm.py --model S --n-seq 2048 --grid 97 --tag tok
run tg_tok_M mse_tinygsm.py --model M --n-seq 2048 --grid 97 --tag tok
echo "=== tg gap done $(date -Is)" >> "$L/tg_gap.log"
