#!/bin/bash
# Gamma-dependent Euler study: build the profile-based grids and write the run manifests for one dataset (CPU).
#
#   bash scripts/dgc/gamma_euler_plan.sh lm1b|owt
#
# Theory sweep: literal Euler (euler_lambda, null self-conditioning) at fixed S in {1, 4}, five grids.
# Production sweep: the DDIM/EDM sampler (ddim_grid, carry) at the paper's churn gammas, five grids.
# Lines are seed-major, so every finished prefix holds complete paired seeds.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=${1:?lm1b|owt}
PY=${COBIT_PYTHON:-/projects/u6ya/raricu.u6ya/miniforge3/envs/cobit-gh/bin/python}
R=results/dgc/gamma_euler; G=$R/grids; M=$R/manifests; P=$R/profiles/${DATA}_cobit_s.npz
mkdir -p "$G" "$M"
case $DATA in lm1b) GAMMAS="0.185 0.2" ;; owt) GAMMAS="0.13 0.175 0.18" ;; esac
SEEDS="42 43 44 45 46"; NFES=64/128/256; IV=63,127,255
b() { "$PY" scripts/dgc/build_gamma_grid.py "$@" --intervals $IV | sed "s|^|[$DATA ${1}] |"; }

[ -f "$G/${DATA}_table.npz" ] || b table --table assets/entropy_tables/$DATA --out "$G/${DATA}_table.npz"
for a in log_uniform karras; do [ -f "$G/${DATA}_$a.npz" ] || b $a --out "$G/${DATA}_$a.npz"; done
b dgc_profile --profile "$P" --out "$G/${DATA}_dgc_profile.npz"
for S in 1 4; do b gamma --profile "$P" --S $S --out "$G/${DATA}_gamma_S$S.npz"; done
for g in $GAMMAS; do b gamma --profile "$P" --gamma $g --out "$G/${DATA}_gamma_g$g.npz"; done

: > "$M/theory_$DATA.txt"
for s in $SEEDS; do for S in 1 4; do for a in table gamma_S$S dgc_profile log_uniform karras; do
  echo "GE_DATA=$DATA GE_ARM=$a GE_GRID=$G/${DATA}_$a.npz GE_SAMPLER=euler_lambda GE_S=$S GE_NFES=$NFES EVAL_SEED=$s"
done; done; done >> "$M/theory_$DATA.txt"
: > "$M/prod_$DATA.txt"
for s in $SEEDS; do for g in $GAMMAS; do for a in table gamma_g$g dgc_profile log_uniform karras; do
  echo "GE_DATA=$DATA GE_ARM=$a GE_GRID=$G/${DATA}_$a.npz GE_SAMPLER=ddim_grid GE_GAMMA=$g GE_NFES=$NFES EVAL_SEED=$s"
done; done; done >> "$M/prod_$DATA.txt"
wc -l "$M/theory_$DATA.txt" "$M/prod_$DATA.txt"
