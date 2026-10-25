#!/usr/bin/env bash
# Sequential reruns with the float32-RoPE fix (rope_fix.py). Logs in experiments/denoising_mse/logs/.
set -o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO"
L=experiments/denoising_mse/logs
run() { local name="$1"; shift; echo "=== $name start $(date -Is): $*" >> "$L/chain.log";
        python "$@" > "$L/$name.log" 2>&1; local rc=$?; echo "=== $name exit $rc $(date -Is)" >> "$L/chain.log"; return $rc; }
run traj_main  experiments/denoising_mse/trajectories.py --n 1024 --batch 32 --chunk 64 --tag main || exit 1
run traj_local experiments/denoising_mse/trajectories.py --n 512 --batch 32 --chunk 64 --tag local --local || exit 1
run mse_main_S experiments/denoising_mse/mse_curves.py --model S --n-seq 4096 --grid 129 --tag main || exit 1
run mse_main_M experiments/denoising_mse/mse_curves.py --model M --n-seq 4096 --grid 129 --tag main || exit 1
run mse_tok_S  experiments/denoising_mse/mse_curves.py --model S --n-seq 2048 --grid 97 --tag tok || exit 1
run mse_tok_M  experiments/denoising_mse/mse_curves.py --model M --n-seq 2048 --grid 97 --tag tok || exit 1
echo "=== chain done $(date -Is)" >> "$L/chain.log"
