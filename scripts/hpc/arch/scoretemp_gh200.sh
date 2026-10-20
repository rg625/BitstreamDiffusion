#!/bin/bash
# Sigma-style ("global") score temperature on GSM8K, ON ISAMBARD.
# Spec: Georgios's note "Testing Sigma-style score temperature on CoBit-S"
# (11 Oct 2026), plus two additions agreed 12 Oct: a w=12 column and a 512-step
# control.
#
#   bash scripts/hpc/arch/scoretemp_gh200.sh plan            # list cells, submit nothing
#   bash scripts/hpc/arch/scoretemp_gh200.sh submit [--go]   # gates + afterok sweep
#   bash scripts/hpc/arch/scoretemp_gh200.sh report          # tables + paired deltas
#
# Idempotent: a cell with a result JSON or a queued/running job is skipped.
#
# ARMS (seed 0, all 1319 problems, Karras, EMA, --score_temp_mode global):
#   s425    CoBit-S unguided @425k (no sigma_data sidecar here -> 0.399844765663147)
#     A  gamma=0.41, 1024 steps, tau in TAUS            (Georgios Exp. 1)
#     B  gamma=0,    1024 steps, tau in TAUS            (his gamma=0 check, all tau
#                                                        so it needs no second round)
#     C  gamma=0.41,  512 steps, tau in TAUS            (addition: step-size control)
#   cfg425  CoBit-S p_uncond=0.1 @425k, gamma=0.41, 1024 steps, per-step diagnostics
#     w in {1,2,3,5,9,12} x tau in {1,.5,.3,.2,.1}, plus (1, 0.05)  (Exp. 2 + w=12)
#   m250    CoBit-M (462M) unguided @250k, gamma=0.41, 1024 steps, tau in TAUS
#
# GATES: per arm, tau=1 on 128 problems at 128 steps must score >= 0.03 before
# any of that arm's cells start (a sampler bug once scored 0.000 vs a true 0.164).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$PROJECT_DIR"
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../site/site.sh" >/dev/null
PY="$COBIT_PYTHON"

PHASE="${1:?usage: scoretemp_gh200.sh plan|submit|report [--go]}"; shift
GO=0; [ "${1:-}" = "--go" ] && GO=1

OUT="results/scoretemp/eval_gh200"
GATE_OUT="$OUT/gate"
LAUNCHER="scripts/hpc/arch/scoretemp_eval.slurm"
TAUS="1 0.7 0.5 0.3 0.2 0.1 0.05"
CFG_WS="1 2 3 5 9 12"
CFG_TAUS="1 0.5 0.3 0.2 0.1"
SD_S="0.399844765663147"

S_CFG=configs/tasks/tinygsm_bits.py
S_CK=runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000425000.pt
C_CFG=configs/tasks/tinygsm_bits_cfg.py
C_CK=tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/step=000425000.pt
M_CFG=configs/tasks/tinygsm_bits_cobit_m.py
M_CK=runs/tasks/tinygsm/cobit_m_bits_s42/checkpoints/step=000250000.pt

mkdir -p "$OUT" "$GATE_OUT"

if [ "$PHASE" = report ]; then
  exec "$PY" experiments/scoretemp/report.py "$OUT"
fi

queued_names () {  # a failed squeue is UNKNOWN, never "nothing queued"
  timeout 60 squeue -h -u "$USER" -o "%j" || { echo "[st] squeue failed -- STOPPING" >&2; exit 1; }
}
QUEUED="$(queued_names)"

submit () {  # $1=job name $2=wall $3=exports [$4=dependency]
  local out rc dep=()
  [ -n "${4:-}" ] && dep=(--dependency="afterok:$4" --kill-on-invalid-dep=yes)
  if [ $GO -eq 0 ]; then echo "[st] DRY  $1  ($2) ${4:+after $4}" >&2; echo "DRY"; return; fi
  set +e
  out=$(bash scripts/site/submit.sh "$LAUNCHER" --parsable --job-name="$1" --time="$2" \
        ${dep[@]+"${dep[@]}"} --export="$3" 2>/dev/null); rc=$?
  set -e
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    echo "[st] submit rc=$rc out='$out' for $1 -- STOPPING; check squeue before re-running" >&2
    exit 1
  fi
  echo "[st] job $out  $1 ${4:+(after $4)}" >&2
  echo "$out"
}

have () {  # $1=label -> result exists or job queued
  ls "$OUT"/gsm8k_results_*"__$1"*.json >/dev/null 2>&1 && return 0
  grep -qx "st_$1" <<<"$QUEUED"
}

N=0
cell () {  # $1=arm $2=cfg $3=ck $4=w $5=tau $6=gamma $7=steps $8=wall $9=gate_job [$10=sd] [$11=diag]
  local label="$1_w$4_t$5_g$6_n$7"
  if have "$label"; then echo "[st] skip $label (done or queued)" >&2; return; fi
  local ex="ALL,ST_ARM=$1,ST_CFG=$2,ST_CK=$3,ST_W=$4,ST_TAU=$5,ST_GAMMA=$6,ST_STEPS=$7,ST_LABEL=__$label,ST_OUT_DIR=$OUT"
  [ -n "${10:-}" ] && ex+=",ST_SIGMA_DATA=${10}"
  [ -n "${11:-}" ] && ex+=",ST_DIAG=${11}"
  if [ "$PHASE" = plan ]; then echo "  $label  wall=$8"; else submit "st_$label" "$8" "$ex" "$9" >/dev/null; fi
  N=$((N + 1))
}

gate () {  # $1=arm $2=cfg $3=ck $4=w [$5=sd] -> prints job id
  local label="gate_$1"
  if ls "$GATE_OUT"/gsm8k_results_*"__$label"*.json >/dev/null 2>&1; then
    echo "[st] gate $1 already has a result; re-check it by hand or delete it" >&2; echo ""; return
  fi
  local ex="ALL,ST_ARM=$1,ST_CFG=$2,ST_CK=$3,ST_W=$4,ST_TAU=1,ST_GAMMA=0.41,ST_STEPS=128,ST_LIMIT=128,ST_LABEL=__$label,ST_OUT_DIR=$GATE_OUT,ST_GATE_MIN_ACC=0.03"
  [ -n "${5:-}" ] && ex+=",ST_SIGMA_DATA=$5"
  [ "$PHASE" = plan ] && { echo "  gate $1" >&2; echo ""; return; }
  submit "st_gate_$1" "00:30:00" "$ex"
}

for c in "$S_CK" "$C_CK" "$M_CK" "$S_CFG" "$C_CFG" "$M_CFG"; do
  [ -f "$c" ] || { echo "[st] FATAL: missing $c" >&2; exit 1; }
done

G_S=$(gate s425 "$S_CFG" "$S_CK" 0 "$SD_S" | tail -1)
G_C=$(gate cfg425 "$C_CFG" "$C_CK" 3 "$SD_S" | tail -1)
G_M=$(gate m250 "$M_CFG" "$M_CK" 0 | tail -1)

for t in $TAUS; do
  cell s425 "$S_CFG" "$S_CK" 0 "$t" 0.41 1024 01:30:00 "$G_S" "$SD_S"
  cell s425 "$S_CFG" "$S_CK" 0 "$t" 0    1024 01:30:00 "$G_S" "$SD_S"
  cell s425 "$S_CFG" "$S_CK" 0 "$t" 0.41  512 01:00:00 "$G_S" "$SD_S"
done
for w in $CFG_WS; do
  for t in $CFG_TAUS; do
    cell cfg425 "$C_CFG" "$C_CK" "$w" "$t" 0.41 1024 02:30:00 "$G_C" "$SD_S" 1
  done
done
cell cfg425 "$C_CFG" "$C_CK" 1 0.05 0.41 1024 02:30:00 "$G_C" "$SD_S" 1
for t in $TAUS; do
  cell m250 "$M_CFG" "$M_CK" 0 "$t" 0.41 1024 04:00:00 "$G_M"
done
echo "[st] $PHASE: $N cells$([ $GO -eq 0 ] && [ "$PHASE" = submit ] && echo ' (dry run; add --go)')"
