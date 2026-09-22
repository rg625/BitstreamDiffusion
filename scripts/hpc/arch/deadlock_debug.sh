#!/bin/bash
# Epoch-boundary deadlock instrumentation. Sourced by the multi-GPU training
# launchers; costs nothing until a run actually stalls.
#
# Every ordering arm (and the token run) hangs at an epoch boundary every few
# hours. The NCCL watchdog signature is asymmetric:
#
#   rank 0     SeqNum=2516974 OpType=ALLREDUCE NumelIn=1572864   (gradient bucket)
#   ranks 1-3  SeqNum=2516971 OpType=ALLREDUCE NumelIn=1         (scalar / barrier)
#
# i.e. rank 0 is three collectives AHEAD: the ranks disagree about how many
# collectives the epoch boundary contains. The watchdog names the op but not the
# Python line that issued it, and raising the process-group timeout to 120 min
# did not help (it is a genuine mismatch, not a slow collective), so what is
# missing is *which call site* each rank is in. Two mechanisms provide that:
#
#   1. NCCL flight recorder -- TORCH_NCCL_DUMP_ON_TIMEOUT writes a per-rank ring
#      buffer of the last N collectives (op, sizes, state, Python frames) when
#      the process group times out. Decode with
#      experiments/ordering/decode_nccl_trace.py.
#   2. SIGUSR1 stack dump -- start_stall_watchdog watches the job log and, once
#      it has stopped growing, signals every rank so each one writes all its
#      Python thread stacks to logs/arch/stack_<jobid>_rank<R>.txt BEFORE the
#      process group times out and tears the job down.
#
# Keep DDP_TIMEOUT_MIN modest (20-30): with the flight recorder in place a fast
# timeout is preferable, since each hang costs the rest of the wall clock.

enable_nccl_flight_recorder() {
  local out_dir="${1:-logs/arch}"
  mkdir -p "$out_dir"
  export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-8192}"
  # torch 2.8 renamed these; it still honours the old names but warns on every
  # rank. Set both so the run works whichever the installed torch expects.
  export TORCH_FR_BUFFER_SIZE="$TORCH_NCCL_TRACE_BUFFER_SIZE"
  export TORCH_NCCL_DUMP_ON_TIMEOUT=1
  export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${out_dir}/nccl_trace_${SLURM_JOB_ID:-local}_rank"
  # Python frames for each recorded collective; this is what identifies the
  # mismatched call site. C++ frames are deliberately off (symbolisation is slow).
  export TORCH_NCCL_TRACE_CPP_STACK="${TORCH_NCCL_TRACE_CPP_STACK:-0}"
  export TORCH_FR_CPP_STACK="$TORCH_NCCL_TRACE_CPP_STACK"
  export COBIT_STACK_DIR="$out_dir"
  echo "[debug] NCCL flight recorder on: buffer=$TORCH_NCCL_TRACE_BUFFER_SIZE dump=${TORCH_NCCL_DEBUG_INFO_TEMP_FILE}*"
}

# Pids of the training ranks, from the files each rank writes once its SIGUSR1
# handler is installed (train.py:_install_stack_dumper).
#
# This used to walk the process tree and match "train.py" in the command line.
# That was wrong twice over: the launcher's own command line contains "train.py"
# as an argument, and SIGUSR1 TERMINATES a process that has no handler. So the
# watchdog killed the torchrun agent (rc=138), took the job down at 15 minutes
# and captured nothing, while the four real ranks were never signalled at all.
# A pid file is written only by a process that has already installed the
# handler, so signalling one is always safe.
_dd_rank_pids() {
  local dir="${COBIT_STACK_DIR:-logs/arch}" job="${SLURM_JOB_ID:-local}" f pid
  for f in "$dir"/pid_"$job"_rank*.txt; do
    [ -f "$f" ] || continue
    pid=$(cat "$f" 2>/dev/null) || continue
    [ -n "$pid" ] && [ -d "/proc/$pid" ] && echo "$pid"
  done
}

# start_stall_watchdog <logfile> [stall_minutes]
# Backgrounds a poller that SIGUSR1s every rank when <logfile> stops growing.
start_stall_watchdog() {
  local logfile="$1"
  local stall_min="${2:-${STALL_MIN:-15}}"
  local parent=$$
  (
    local last_size=-1 last_change=$SECONDS sz now
    while true; do
      sleep 60
      [ -f "$logfile" ] || continue
      sz=$(stat -c %s "$logfile" 2>/dev/null || echo -1)
      if [ "$sz" != "$last_size" ]; then
        last_size="$sz"; last_change=$SECONDS
        continue
      fi
      now=$SECONDS
      if [ $(( now - last_change )) -ge $(( stall_min * 60 )) ]; then
        echo "[stall-watchdog] $(date): log has not grown for ${stall_min}m -- dumping stacks"
        local pids n
        pids=$(_dd_rank_pids | tr '\n' ' ')
        n=$(echo $pids | wc -w)
        echo "[stall-watchdog] rank pids: ${pids:-<none>} (n=$n)"
        if [ "$n" -eq 0 ]; then
          # Never fall back to guessing: a wrong SIGUSR1 kills the job.
          echo "[stall-watchdog] no rank pid files -- NOT signalling anything." >&2
          echo "[stall-watchdog] expected ${COBIT_STACK_DIR:-logs/arch}/pid_${SLURM_JOB_ID:-local}_rank*.txt" >&2
        else
          for p in $pids; do
            kill -USR1 "$p" 2>/dev/null && echo "[stall-watchdog] SIGUSR1 -> $p"
          done
        fi
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null || true
        echo "[stall-watchdog] stacks in ${COBIT_STACK_DIR:-logs/arch}/stack_${SLURM_JOB_ID:-local}_rank*.txt"
        last_change=$SECONDS   # re-arm; dump again if the stall persists
      fi
    done
  ) &
  STALL_WATCHDOG_PID=$!
  echo "[debug] stall watchdog pid=$STALL_WATCHDOG_PID log=$logfile threshold=${stall_min}m"
}

stop_stall_watchdog() {
  [ -n "${STALL_WATCHDOG_PID:-}" ] && kill "$STALL_WATCHDOG_PID" 2>/dev/null || true
}
