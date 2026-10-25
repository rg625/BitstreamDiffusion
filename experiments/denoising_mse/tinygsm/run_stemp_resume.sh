#!/usr/bin/env bash
# Resume the tau = 0.6 S/M switching study after the disk-full crash (24 Oct 08:27): wait until >= 30 GB is free on /,
# then run the missing configurations (finished ones are skipped by compute_demand.py).
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
L="$REPO/experiments/denoising_mse/logs/tg_stemp.log"
until [ "$(df --output=avail -B1G / | tail -1 | tr -d ' ')" -ge 30 ]; do sleep 300; done
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
run() { echo "=== $* start $(date -Is)" >> "$L"; python -W ignore "$@" >> "$L" 2>&1; echo "=== exit $? $(date -Is)" >> "$L"; }
echo "=== resume (disk has space) $(date -Is)" >> "$L"
CFG="base,M,Mseg0,Mseg1,Mseg2,Mseg3,Mseg4,Mseg5,Mseg6,Mseg7,MthenS64,MthenS128,MthenS192,SthenM64,SthenM128,SthenM192"
for seed in 0 1; do
  run compute_demand.py --configs "$CFG" --seed $seed --score-temp-tau 0.6
done
echo "=== stemp study done $(date -Is)" >> "$L"
