#!/bin/bash
# Evaluate the production anchor and all four 500k ordering arms ON ISAMBARD.
#
#   bash scripts/hpc/arch/ordering_eval_gh200.sh gate3          # dry run
#   bash scripts/hpc/arch/ordering_eval_gh200.sh gate3 --go     # anchor only (gate 3)
#   bash scripts/hpc/arch/ordering_eval_gh200.sh arms  --go     # the four arms
#
# All four arms were TRAINED on CSD3; that is common to them. Their evaluation
# must be common too, so every cell here -- including control and l2r, which
# already have CSD3 evals -- is re-run on GH200 into a FRESH directory and
# compared only within it. Never mix these JSONs with results/ordering/train_eval/
# (CSD3): a different BLAS, attention kernel and reduction order is a regime
# change, and regimes are never pooled.
#
# Phase `arms` refuses to submit until gate 3 has passed IN THIS DIRECTORY:
# three anchor seeds, each in 0.12-0.20. A sampler bug once scored 0.000
# against a true 0.164 and was caught only by an anchor; on new hardware that
# class of failure is more likely, so no arm is spent before the anchor lands.
#
# Karras throughout: the entropic schedule is refitted per run and would
# confound training ordering with decoding schedule.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$PROJECT_DIR"
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../site/site.sh" >/dev/null   # COBIT_PYTHON / COBIT_ACCOUNT, or refuse
PY="$COBIT_PYTHON"

PHASE="${1:?usage: ordering_eval_gh200.sh gate3|arms [--go]}"
GO=0; [ "${2:-}" = "--go" ] && GO=1
OUT="results/ordering/train_eval_gh200"
REGIME="ordering_train__eval_gh200"
ANCHOR="tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/last.pt"
ARMS=(ord_fs500k_none_s42:none:0 ord_fs500k_l2r_w0.25_s42:l2r:0.25
      ord_fs500k_r2l_w0.25_s42:r2l:0.25 ord_fs500k_random_w0.25_s42:random:0.25)

global_step () {  # from the checkpoint, never the file name or a progress bar
  "$PY" - "$1" <<'PYEOF'
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["global_step"]))
except Exception:
    print("unknown")
PYEOF
}

submit () {  # $1=exports $2=--time
  local out rc
  set +e
  out=$(bash scripts/site/submit.sh scripts/hpc/arch/ordering_eval.slurm \
        --parsable --time="$2" --export="$1"); rc=$?
  set -e
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    echo "[eval] submit rc=$rc out='$out' -- STOPPING. rc!=0 is UNKNOWN, not 'not queued':" >&2
    echo "[eval] check squeue -u \$USER before re-running, or two jobs share one output." >&2
    exit 1
  fi
  echo "[eval] job $out  ($1)"
}

[ -f datasets/gsm8k/gsm8k_test.json ] || { echo "MISSING datasets/gsm8k/gsm8k_test.json" >&2; exit 1; }
COMMON="ALL,EV_OUT_DIR=$OUT,EV_REGIME=$REGIME"

case "$PHASE" in
  gate3)
    [ -f "$ANCHOR" ] || { echo "MISSING anchor $ANCHOR" >&2; exit 1; }
    echo "  anchor: step=$(global_step "$ANCHOR")  -> $OUT"
    [ $GO -eq 1 ] || { echo "dry run -- would submit 1 job (anchor, 3 seeds). Add --go."; exit 0; }
    submit "$COMMON,EV_RUN=prod_anchor,EV_MODE=none,EV_W=0,EV_CK_PATH=$ANCHOR" 04:00:00
    echo "[eval] gate 3 passes when all three seeds land in 0.12-0.20:"
    echo "  $PY experiments/ordering/analyze_train.py --dir $OUT"
    ;;
  arms)
    "$PY" - "$OUT" <<'PYEOF' || { echo "[eval] gate 3 has not passed in $OUT -- refusing to spend arm hours" >&2; exit 1; }
import glob, json, sys
lo, hi = 0.12, 0.20
accs = {}
for f in glob.glob(f"{sys.argv[1]}/*.json"):
    d = json.load(open(f))
    if "cobit_raw_binary_bits_cfg/checkpoints" in str(d.get("checkpoint")) and d.get("num_examples") == 1319:
        accs[int(d.get("seed") or 0)] = float(d["accuracy"])
print(f"  anchor on GH200: {dict(sorted(accs.items()))}  band {lo}-{hi}")
sys.exit(0 if set(accs) >= {0, 1, 2} and all(lo <= a <= hi for a in accs.values()) else 1)
PYEOF
    for spec in "${ARMS[@]}"; do
      IFS=: read -r run mode w <<<"$spec"
      ck="runs/tasks/tinygsm/$run/checkpoints/last.pt"
      [ -f "$ck" ] || { echo "MISSING $ck" >&2; exit 1; }
      st=$(global_step "$ck"); echo "  $run: step=$st"
      [ "$st" = "500000" ] || { echo "  -> NOT at 500000 (got '$st'), refusing" >&2; exit 1; }
    done
    [ $GO -eq 1 ] || { echo "dry run -- would submit 4 jobs (21 cells). Add --go."; exit 0; }
    for spec in "${ARMS[@]}"; do
      IFS=: read -r run mode w <<<"$spec"
      # 6 cells (matched + uniform x 3 seeds) for the ordered arms, 3 for control.
      submit "$COMMON,EV_RUN=$run,EV_MODE=$mode,EV_W=$w" "$([ "$mode" = none ] && echo 04:00:00 || echo 08:00:00)"
    done
    echo "[eval] when all four finish:"
    echo "  $PY experiments/ordering/analyze_train.py --dir $OUT --control ord_fs500k_none_s42 \\"
    echo "     --out results/ordering/train_summary_500k_gh200.json"
    ;;
  *) echo "unknown phase $PHASE (gate3|arms)" >&2; exit 1 ;;
esac
