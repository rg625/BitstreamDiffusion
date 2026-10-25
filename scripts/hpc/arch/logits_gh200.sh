#!/bin/bash
# THE GRANT QUESTION: does a V-way token softmax beat raw bits at a real
# operating point? Both arms from scratch ON ISAMBARD, matched in everything but
# the head: 500k optimiser steps, effective batch 512, lr 1e-4, seed 42,
# p_uncond 0.1.
#
#   bash scripts/hpc/arch/logits_gh200.sh smoke [--accum N] [--go]  # 300 steps at accum 1 and 2 (or just N)
#   bash scripts/hpc/arch/logits_gh200.sh chain --accum N [--go]    # both arms, chained
#   bash scripts/hpc/arch/logits_gh200.sh status                    # global_step of both arms
#
# --nodes N (token arm only) spreads the SAME effective batch over N nodes:
# 512 / (4 x N x accum) per GPU. Per-node throughput is flat in the per-GPU
# batch (measured 24 Sep: ~710 samples/s at 64, 80, 96 and 112/GPU; the GPU is
# saturated), so nodes buy wall-clock and the optimiser step is unchanged.
# Multi-node smokes are tagged smoke_gh_n<N>_a<accum>, and `chain --nodes N`
# requires that smoke, not the single-node one.
#
# NOT the CSD3 runs. tok_b512_token_ce_s42 (167k, CSD3) and ord_fs500k_none_s42
# (500k, CSD3) are both on disk here, and train.py resumes from whatever last.pt
# the run directory holds -- so reusing either tag would continue a CSD3 run on
# GH200 and mix machines INSIDE one arm. Both arms here are tagged gh500k:
#   runs/tasks/tinygsm/tok_gh500k_token_ce_s42     V-way token softmax
#   runs/tasks/tinygsm/ord_gh500k_none_s42         binary control
#
# ACCUMULATION. It existed only because [B,512,49153] logits OOM at 128/GPU on
# an 80 GB A100. The effective batch is 512 at any TOK_ACCUM, and global_step is
# an optimiser-step count, so the choice changes memory and speed, not what a
# step is. Smoke both, pick the smallest that fits with headroom, and pass it to
# `chain`, which refuses an accum whose smoke did not finish cleanly.
#
# THE HANG. On CSD3 every process hung entering its sixth epoch (114,290 steps).
# It costs wall-clock only: last.pt is written before the corrupted window, the
# 20-min process-group timeout kills the job, and the next afterany link resumes.
# Links are therefore sized on ~114k steps each, not on the 24 h wall cap; a
# surplus link that starts after 500k exits at once (trainer.py's resume guard).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$PROJECT_DIR"
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../site/site.sh" >/dev/null   # COBIT_PYTHON / COBIT_ACCOUNT, or refuse
PY="$COBIT_PYTHON"

PHASE="${1:?usage: logits_gh200.sh smoke|chain|status [--accum N] [--go]}"; shift
GO=0; ACCUM=""; NODES=1
while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift ;;
    --accum) ACCUM="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done

TOK_RUN="runs/tasks/tinygsm/tok_gh500k_token_ce_s42"
CTL_RUN="runs/tasks/tinygsm/ord_gh500k_none_s42"
TOK_LINKS="${TOK_LINKS:-7}"   # ~114k steps per link if the hang reproduces; 5 needed, 2 spare
CTL_LINKS="${CTL_LINKS:-6}"
WALL="${WALL:-24:00:00}"      # workq_qos cap

global_step () {  # from last.pt, never the progress bar (it resets every link)
  [ -f "$1/checkpoints/last.pt" ] || { echo "none"; return; }
  "$PY" - "$1/checkpoints/last.pt" <<'PYEOF'
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["global_step"]))
except Exception:
    print("unknown")
PYEOF
}

# A failed squeue is "unknown", not "no jobs": a timeout once returned empty and
# was read as zero jobs.
jobs_named () {
  local out
  out=$(timeout 60 squeue -u "$USER" -h -n "$1" -o "%i %T") || { echo "UNKNOWN"; return; }
  echo "$out"
}

submit () {  # $1=launcher, rest = sbatch args. Prints the job id or stops.
  local launcher="$1" out rc; shift
  set +e
  out=$(bash scripts/site/submit.sh "$launcher" --parsable "$@"); rc=$?
  set -e
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    echo "[logits] submit rc=$rc out='$out' -- STOPPING. rc!=0 is UNKNOWN, not 'not queued':" >&2
    echo "[logits] check squeue -u \$USER before re-running, or two jobs share one run dir." >&2
    exit 1
  fi
  echo "$out"
}

smoke_tag () {  # $1=accum -> the smoke's TOK_TAG at the current --nodes
  if [ "$NODES" = 1 ]; then echo "smoke_gh_a$1"; else echo "smoke_gh_n${NODES}_a$1"; fi
}

smoke_ok () {  # $1=accum. The newest log of that smoke must show a clean finish.
  local log tag
  tag=$(smoke_tag "$1")
  log=$(grep -l "tag=$tag\b" logs/arch/cobit_toktrain_*.log 2>/dev/null | xargs -r ls -t | head -1 || true)
  [ -n "$log" ] || { echo "no smoke log for accum $1"; return 1; }
  if grep -qE "OutOfMemoryError|CUDA out of memory|Traceback|\[tok\] rc=" "$log"; then
    echo "smoke accum $1 FAILED: $log"; return 1
  fi
  grep -q "\[tok\] finished" "$log" || { echo "smoke accum $1 has not finished: $log"; return 1; }
  echo "smoke accum $1 clean: $log"
}

case "$PHASE" in
  smoke)
    # --accum N re-runs just one of the two (e.g. after a node-level failure).
    SMOKES="${ACCUM:-1 2}"
    for a in $SMOKES; do
      d="runs/tasks/tinygsm/tok_$(smoke_tag "$a")_token_ce_s42"
      # A leftover smoke dir would RESUME at step 300 and exit at once, which
      # looks exactly like a clean pass.
      [ -e "$d" ] && { echo "REFUSING: $d exists -- remove it first (rm -r $d)" >&2; exit 1; }
    done
    echo "would submit smoke(s) on $NODES node(s) at TOK_ACCUM in {$SMOKES} (-> 512/(4x$NODES x accum)/GPU), 300 steps, 1 h wall"
    [ $GO -eq 1 ] || { echo "dry run. Add --go."; exit 0; }
    for a in $SMOKES; do
      j=$(submit scripts/hpc/arch/token_train.slurm --time=01:00:00 --nodes="$NODES" \
          --export=ALL,TOK_LOSS=token_ce,TOK_BATCH=512,TOK_ACCUM=$a,TOK_LR=1e-4,TOK_SEED=42,TOK_STEPS=300,TOK_TAG=$(smoke_tag "$a"),TOK_CKPT_EVERY=1000000,TOK_RESUME_EVERY=1000000)
      echo "[logits] smoke accum $a: job $j"
    done
    echo "[logits] then: grep -E 'accum\\]|peak|it/s' logs/arch/cobit_toktrain_<job>.log | tail"
    ;;
  chain)
    case "$ACCUM" in 1|2|4) ;; *) echo "--accum 1|2|4 required (pick from the smoke)" >&2; exit 1 ;; esac
    smoke_ok "$ACCUM" || { echo "REFUSING: accum $ACCUM has no clean smoke" >&2; exit 1; }
    # ONLY_TOK=1: only the token arm is being (re)chained, so only its jobs and
    # its last.pt are checked; the control chain keeps running untouched.
    if [ "${ONLY_TOK:-0}" = 1 ]; then NAMES=(cobit_gh_tok); RUNS=("$TOK_RUN"); else
      NAMES=(cobit_gh_tok cobit_gh_ctl); RUNS=("$TOK_RUN" "$CTL_RUN"); fi
    for n in "${NAMES[@]}"; do
      q=$(jobs_named "$n")
      [ "$q" = "UNKNOWN" ] && { echo "squeue failed -- state UNKNOWN, STOPPING" >&2; exit 1; }
      [ -n "$q" ] && { echo "REFUSING: $n already queued/running: $q" >&2; exit 1; }
    done
    for r in "${RUNS[@]}"; do
      st=$(global_step "$r"); echo "  $r: step=$st"
      # Starting a chain onto an existing last.pt is only right when it is THIS
      # chain's own checkpoint; say so explicitly rather than resuming silently.
      if [ "$st" != "none" ] && [ "${RESUME_OK:-0}" != "1" ]; then
        echo "REFUSING: $r already has last.pt (step $st). If it is this Isambard chain's own" >&2
        echo "          and you mean to extend it, re-run with RESUME_OK=1." >&2
        exit 1
      fi
    done
    echo "  token: $NODES node(s), accum $ACCUM ($((512 / (4 * NODES * ACCUM)))/GPU), $TOK_LINKS links x $WALL"
    [ "${ONLY_TOK:-0}" = 1 ] && echo "  control: untouched (ONLY_TOK=1)" || echo "  control: $CTL_LINKS links x $WALL"
    [ $GO -eq 1 ] || { echo "dry run. Add --go."; exit 0; }
    # Resume cadence in optimiser steps. 1,000 was ~12 min on one node; on N
    # nodes it is 12/N min, and every write stalls all ranks for a 3.4 GB save,
    # so it scales with N to keep ~12 min between writes.
    TOK_EX="ALL,TOK_LOSS=token_ce,TOK_BATCH=512,TOK_ACCUM=$ACCUM,TOK_LR=1e-4,TOK_SEED=42,TOK_STEPS=500000,TOK_TAG=gh500k,TOK_CKPT_EVERY=100000,TOK_RESUME_EVERY=$((1000 * NODES))"
    CTL_EX="ALL,ORD_MODE=none,ORD_W=0,ORD_INIT=none,ORD_LR=1e-4,ORD_SEED=42,ORD_STEPS=500000,ORD_TAG=gh500k,ORD_CKPT_EVERY=100000"
    chain () {  # $1=launcher $2=job name $3=exports $4=links $5=nodes
      local prev="" i j dep
      for i in $(seq 1 "$4"); do
        dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
        j=$(submit "$1" --job-name="$2" --time="$WALL" --nodes="$5" ${dep[@]+"${dep[@]}"} --export="$3")
        echo "[logits] $2 link $i: job $j${prev:+ (afterany:$prev)}"
        prev="$j"
      done
    }
    chain scripts/hpc/arch/token_train.slurm cobit_gh_tok "$TOK_EX" "$TOK_LINKS" "$NODES"
    # ONLY_TOK=1 re-chains the token arm alone (e.g. to move it to more nodes)
    # and leaves a running control chain untouched.
    [ "${ONLY_TOK:-0}" = 1 ] || chain scripts/hpc/arch/ordering_train.slurm cobit_gh_ctl "$CTL_EX" "$CTL_LINKS" 1
    echo "[logits] progress: bash scripts/hpc/arch/logits_gh200.sh status"
    ;;
  status)
    for r in "$TOK_RUN" "$CTL_RUN"; do echo "  $r: global_step=$(global_step "$r")"; done
    for n in cobit_gh_tok cobit_gh_ctl; do echo "  $n: $(jobs_named "$n" | tr '\n' ' ')"; done
    ;;
  *) echo "unknown phase $PHASE (smoke|chain|status)" >&2; exit 1 ;;
esac
