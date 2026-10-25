#!/bin/bash
# Evaluate the two FINISHED 500k ordering arms on the full 1319-problem set.
#
#   bash scripts/hpc/arch/ordering_eval_500k.sh          # dry run
#   bash scripts/hpc/arch/ordering_eval_500k.sh --go
#
# This answers the ordering task's primary question -- does ordering help at
# all? -- because control and l2r are the matched pair: both from scratch, both
# 500,000 steps, batch 512, lr 1e-4, seed 42, p_uncond 0.1, differing only in
# cfg.train.ordering.
#
# THREE JOBS, twelve cells, ~1 h each:
#   control    3 cells (uniform decode x 3 seeds; for w=0 "as trained" IS uniform)
#   l2r        6 cells (matched + uniform decode x 3 seeds)
#   anchor     3 cells (production 500k checkpoint, 0.164 expected)
#
# The anchor is not optional. A sampler bug once returned probabilities from
# sigma_{N-1} instead of a final denoise and scored 0.000 against a true 0.164;
# it was caught only because an anchor was in the batch. analyze_train.py now
# refuses to present a clean table without one.
#
# Karras schedule throughout: the entropic schedule is refitted per run from
# that run's own tables, so it would differ between arms and confound training
# ordering with decoding schedule.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
cd "$PROJECT_DIR"

GO=0; [ "${1:-}" = "--go" ] && GO=1
ANCHOR="tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/last.pt"

# Refuse to evaluate an arm that has not actually finished: a checkpoint short
# of 500,000 would be compared as though it were the endpoint.
check_step () {  # $1=checkpoint  $2=expected step
  local got
  got=$("${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}" - "$1" <<'PYEOF'
import sys, torch
try:
    d = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
    print(int(d.get("global_step", -1)))
except Exception as e:
    print(-1)
PYEOF
)
  echo "$got"
}

for arm in ord_fs500k_none_s42 ord_fs500k_l2r_w0.25_s42; do
  ck="runs/tasks/tinygsm/$arm/checkpoints/last.pt"
  [ -f "$ck" ] || { echo "MISSING $ck" >&2; exit 1; }
  st=$(check_step "$ck")
  echo "  $arm: step=$st"
  [ "$st" = "500000" ] || { echo "  -> NOT at 500000, refusing" >&2; exit 1; }
done
[ -f "$ANCHOR" ] || { echo "MISSING anchor $ANCHOR" >&2; exit 1; }
echo "  anchor: step=$(check_step "$ANCHOR")"

if [ $GO -eq 0 ]; then
  echo
  echo "dry run -- would submit 3 jobs (control, l2r, anchor). Re-run with --go."
  exit 0
fi

submit () {  # $1=exports
  local out rc
  out=$(sbatch --parsable --export="$1" scripts/hpc/arch/ordering_eval.slurm); rc=$?
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    echo "[eval] sbatch rc=$rc out='$out' -- STOPPING (rc!=0 is UNKNOWN, check squeue)" >&2
    exit 1
  fi
  echo "[eval] job $out"
}

submit "ALL,EV_RUN=ord_fs500k_none_s42,EV_MODE=none,EV_W=0"
submit "ALL,EV_RUN=ord_fs500k_l2r_w0.25_s42,EV_MODE=l2r,EV_W=0.25"
submit "ALL,EV_RUN=prod_anchor,EV_MODE=none,EV_W=0,EV_CK_PATH=$ANCHOR"

echo
echo "[eval] when all three finish:"
echo "  python experiments/ordering/analyze_train.py \\"
echo "     --dir results/ordering/train_eval --control ord_fs500k_none_s42 \\"
echo "     --out results/ordering/train_summary_500k.json"
