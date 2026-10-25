#!/usr/bin/env bash
# TinyGSM CoBit-S vs CoBit-M experiments (all float32 RoPE; shared CoBit-S schedule for paired runs).
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs"
run() { local n="$1"; shift; echo "=== $n start $(date -Is)" >> "$L/tgM_chain.log"; python -W ignore "$@" > "$L/$n.log" 2>&1;
        local rc=$?; echo "=== $n exit $rc $(date -Is)" >> "$L/tgM_chain.log"; return $rc; }
run tgM_gsm8k   gsm8k_eval.py --model M --gamma 0.31 --steps 256 || exit 1
run tgM_paired  traj_tinygsm.py --mode paired --a S --b M --n 1319 --tag paired_SM || exit 1
run tgM_localS  traj_tinygsm.py --mode local --a S --b M --n 512 --tag local_S || exit 1
run tgM_localM  traj_tinygsm.py --mode local --a M --b S --n 512 --tag local_M || exit 1
run tgM_true    traj_tinygsm.py --mode truechain --models S,M --n 512 --tag true_SM || exit 1
run tgM_static  mse_tinygsm.py --model M --n-seq 4096 --grid 129 --tag static || exit 1
echo "=== tgM chain done $(date -Is)" >> "$L/tgM_chain.log"
