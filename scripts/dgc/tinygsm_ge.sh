#!/bin/bash
# Gamma-dependent Euler study on TinyGSM -> GSM8K: build the grids from the forward profiles and submit one
# scoretemp_eval cell per (model, grid, steps). Production sampler only (DDIM + EDM churn 0.41, carry, T = 1),
# seed 0, all 1319 problems, paired per problem across grids.
#
#   bash scripts/dgc/tinygsm_ge.sh grids            # CPU
#   bash scripts/dgc/tinygsm_ge.sh submit [--go]    # dry run unless --go
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${COBIT_PYTHON:-/projects/u6ya/raricu.u6ya/miniforge3/envs/cobit-gh/bin/python}
R=results/dgc/gamma_euler; G=$R/grids; OUT=results/dgc/gsm8k
GAMMA=0.41; IV=63,127,255,1023; STEPS="64 128 256 1024"
declare -A RUN=([s500]=runs/tasks/tinygsm/cobit_raw_binary_bits [m250]=runs/tasks/tinygsm/cobit_m_bits_s42)
declare -A CK=([s500]=${RUN[s500]}/checkpoints/step=000500000.pt [m250]=${RUN[m250]}/checkpoints/step=000250000.pt)
declare -A CFG=([s500]=configs/tasks/tinygsm_bits.py [m250]=configs/tasks/tinygsm_bits_cobit_m.py)
declare -A SD=([s500]=",ST_SIGMA_DATA=0.399844765663147" [m250]="")
declare -A WALL=([s500]=00:45:00 [m250]=01:45:00)
EXC=nid010003,nid010029,nid010033,nid010038,nid010040

case "${1:?grids|submit}" in
  grids)
    mkdir -p "$G"
    for m in ${MODELS:-s500 m250}; do
      P=$R/profiles/tinygsm_$m.npz
      b() { "$PY" scripts/dgc/build_gamma_grid.py "$@" --intervals $IV --sigma_lo 0.002 --sigma_hi 80 | sed "s|^|[$m ${1}] |"; }
      b table --table "${RUN[$m]}" --out "$G/tinygsm_${m}_table.npz"
      b dgc_profile --profile "$P" --out "$G/tinygsm_${m}_dgc_profile.npz"
      b gamma --profile "$P" --gamma $GAMMA --out "$G/tinygsm_${m}_gamma_g$GAMMA.npz"
    done ;;
  submit)
    GO=0; [ "${2:-}" = --go ] && GO=1; N=0
    for m in ${MODELS:-s500 m250}; do for arm in table dgc_profile gamma_g$GAMMA; do for st in $STEPS; do
      label="ge_${m}_${arm}_t1_g${GAMMA}_n${st}_l1319"
      if ls "$OUT"/gsm8k_results_*"__$label".json >/dev/null 2>&1; then echo "skip $label (done)"; continue; fi
      [ -f "$G/tinygsm_${m}_$arm.npz" ] || { echo "missing grid $G/tinygsm_${m}_$arm.npz" >&2; exit 1; }
      exp="ALL,ST_ARM=$m,ST_CFG=${CFG[$m]},ST_CK=${CK[$m]},ST_W=0,ST_TAU=1,ST_GAMMA=$GAMMA,ST_STEPS=$st,ST_LIMIT=1319"
      exp="$exp,ST_SCHEDULE=grid_file,ST_GRID_FILE=$G/tinygsm_${m}_$arm.npz,ST_ENTROPY_DIR=${RUN[$m]},ST_OUT_DIR=$OUT"
      exp="$exp,ST_LABEL=__$label${SD[$m]}"
      if [ $GO = 1 ]; then
        echo "$(bash scripts/site/submit.sh scripts/hpc/arch/scoretemp_eval.slurm --parsable --job-name=$label \
               --time=${WALL[$m]} --exclude=$EXC --export="$exp" 2>/dev/null) $label"
      else echo "DRY $label"; fi
      N=$((N + 1))
    done; done; done
    echo "$N cells $([ $GO = 1 ] && echo submitted || echo '(dry run)')" ;;
esac
