#!/usr/bin/env bash
# Score-temperature study, continued: finish the CoBit-S tau grid near the optimum, sweep CoBit-M, pick the best common
# tau (pick_tau.py) and rerun the S/M model-switching study (compute_demand.py, configs B + base) at that tau.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
cd "$REPO/experiments/denoising_mse/tinygsm"
L="$REPO/experiments/denoising_mse/logs/tg_stemp.log"
run() { echo "=== $* start $(date -Is)" >> "$L"; python -W ignore "$@" >> "$L" 2>&1; echo "=== exit $? $(date -Is)" >> "$L"; }
run diagnostics.py --model S --variants stemp0.75,stemp0.65,stemp0.6 --seeds 0,1
run diagnostics.py --model M --variants stemp0.9,stemp0.8,stemp0.7,stemp0.6 --seeds 0,1
python pick_tau.py >> "$L" 2>&1
TAU=$(python pick_tau.py | tail -1)
echo "=== switching study at tau=$TAU $(date -Is)" >> "$L"
CFG="base,M,Mseg0,Mseg1,Mseg2,Mseg3,Mseg4,Mseg5,Mseg6,Mseg7,MthenS64,MthenS128,MthenS192,SthenM64,SthenM128,SthenM192"
for seed in 0 1; do
  run compute_demand.py --configs "$CFG" --seed $seed --score-temp-tau "$TAU"
done
echo "=== stemp study done $(date -Is)" >> "$L"
