#!/bin/bash
# Blocks A-E of the sampler study (docs/pdf_reports/sampler_math.pdf, entropy_profile.pdf), 13 Oct 2026.
# CoBit-S unguided 500k EMA, entropic grid, 1024 steps, all 1319 problems, seed 0, one GPU per cell.
#
#   bash scripts/hpc/arch/scoretemp_blocks.sh            # dry run
#   bash scripts/hpc/arch/scoretemp_blocks.sh --go       # submit (skips cells with a result or a queued job)
#   bash scripts/hpc/arch/scoretemp_blocks.sh --go dfull # block D at full size (after its smoke)
#
#   A  unsharpened last step; churn 0.41 and gamma 0; tau {0.5,0.3,0.2}; per-step diagnostics
#   B  ancestral sampler (no churn), unsharpened last step; tau {1,0.7,0.5,0.3,0.2,0.1}
#   C  churn gamma {0.1,0.2,0.3} x tau {1,0.5,0.3,0.2,0.1}, unsharpened last step (tau=1 = paired baseline)
#   D  DeepMind logit temperature (token space, full logit), churn 0.41, T {0.5,0.3,0.1}; smoke first (64 problems)
#   E  end at sigma_min 0.1, churn 0.41: tau {1,0.7,0.5,0.3,0.2} sharpened + tau {0.3,0.2} unsharpened
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$PROJECT_DIR"
GO=0; [ "${1:-}" = "--go" ] && GO=1
WHAT="${2:-main}"
OUT=results/scoretemp/blocks_gh200
LAUNCHER=scripts/hpc/arch/scoretemp_eval.slurm
CK=runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000500000.pt
BASE="ALL,ST_CFG=configs/tasks/tinygsm_bits.py,ST_CK=$CK,ST_W=0,ST_STEPS=1024,ST_SCHEDULE=entropic,ST_SIGMA_DATA=0.399844765663147,ST_OUT_DIR=$OUT"
mkdir -p "$OUT"
QUEUED="$(timeout 60 squeue -h -u "$USER" -o "%j")" || { echo "squeue failed -- STOPPING" >&2; exit 1; }
N=0

cell () {  # $1=arm $2=tau(or T) $3=gamma $4=wall $5=extra exports
  local label="$1_w0_t$2_g$3_n1024"
  if ls "$OUT"/gsm8k_results_*"__$label"*.json >/dev/null 2>&1 || grep -qx "st_$label" <<<"$QUEUED"; then
    echo "skip $label"; return
  fi
  local tau="$2"; [ -n "${5:-}" ] && [[ "$5" == *ST_POST_TEMP* ]] && tau=1
  local ex="$BASE,ST_ARM=$1,ST_TAU=$tau,ST_GAMMA=$3,ST_LABEL=__$label${5:+,$5}"
  if [ $GO -eq 1 ]; then
    id=$(bash scripts/site/submit.sh "$LAUNCHER" --parsable --job-name="st_$label" --time="$4" --export="$ex" 2>/dev/null)
    echo "$id st_$label"
  else
    echo "DRY st_$label ($4) $5"
  fi
  N=$((N + 1))
}

if [ "$WHAT" = main ]; then
  for g in 0.41 0; do for t in 0.5 0.3 0.2; do
    cell s500a "$t" "$g" 02:00:00 "ST_ST_FINAL=unsharpened,ST_DIAG=1"; done; done
  for t in 1 0.7 0.5 0.3 0.2 0.1; do
    cell s500b "$t" 0 01:30:00 "ST_SAMPLER_KIND=ancestral,ST_ST_FINAL=unsharpened"; done
  for g in 0.1 0.2 0.3; do for t in 1 0.5 0.3 0.2 0.1; do
    cell s500c "$t" "$g" 01:30:00 "ST_ST_FINAL=unsharpened"; done; done
  for t in 1 0.7 0.5 0.3 0.2; do
    cell s500e "$t" 0.41 01:30:00 "ST_SIGMA_MIN=0.1"; done
  for t in 0.3 0.2; do
    cell s500eu "$t" 0.41 01:30:00 "ST_SIGMA_MIN=0.1,ST_ST_FINAL=unsharpened"; done
  # block D smoke: one temperature on the first 64 problems, to measure the codeword-softmax cost
  cell s500dsmoke 0.3 0.41 01:00:00 "ST_POST_TEMP=0.3,ST_LIMIT=64"
elif [ "$WHAT" = dfull ]; then
  for T in 0.5 0.3 0.1; do cell s500d "$T" 0.41 01:30:00 "ST_POST_TEMP=$T"; done
fi
echo "$N cells $([ $GO -eq 1 ] && echo submitted || echo '(dry run)')"
