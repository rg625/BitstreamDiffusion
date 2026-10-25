#!/usr/bin/env bash
# Follow-up runs: same-input decomposition along CoBit-M's trajectories, and ground-truth chains.
set -o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO"
L=experiments/denoising_mse/logs
run() { local name="$1"; shift; echo "=== $name start $(date -Is): $*" >> "$L/chain2.log";
        python -W ignore "$@" > "$L/$name.log" 2>&1; local rc=$?; echo "=== $name exit $rc $(date -Is)" >> "$L/chain2.log"; return $rc; }
run traj_localM experiments/denoising_mse/trajectories.py --n 512 --batch 32 --chunk 64 --tag localM --local --ref M || exit 1
run traj_true   experiments/denoising_mse/trajectories.py --n 512 --batch 32 --chunk 64 --tag truechain --truechain || exit 1
echo "=== chain2 done $(date -Is)" >> "$L/chain2.log"
