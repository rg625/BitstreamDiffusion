#!/bin/bash
# One watch tick for the window-ordering study. Safe to run as often as you like.
#
#   bash scripts/hpc/arch/window_watch.sh          # report + safe actions
#   WATCH_DRY=1 bash scripts/hpc/arch/window_watch.sh   # report only
#
# Prints one line per run, then ALERT / ACTION lines. The only actions it takes
# are ones that STOP spending:
#   * circuit breaker: >= BREAKER distinct runs HALTed by a crash (not a
#     divergence) -> write WINDOW_GLOBAL_HALT and cancel every PENDING win_/wev_
#     job. A systemic bug must not burn 26 nodes.
#   * budget cap: GPU-h used by win_/wev_ jobs since launch >= CAP_GPUH ->
#     same. Warns at 80%.
#   * a RUNNING win_ job whose log has not grown for SILENT_MIN minutes is
#     cancelled (SIGTERM: its next link resumes, and the launcher's
#     no-progress rule halts a run that keeps doing it).
# and one that spends, gated: `window_study.sh eval --go` (anchor-gated, one
# attempt per checkpoint, skips HALTed runs) unless a global halt is present.
# It never clears a HALT and never resubmits training: that is a human decision.
set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$PROJECT_DIR"
source scripts/site/site.sh >/dev/null 2>&1
PY="$COBIT_PYTHON"
SINCE="${SINCE:-2026-09-24T16:00:00}"
CAP_GPUH="${CAP_GPUH:-3000}"
BREAKER="${BREAKER:-3}"
SILENT_MIN="${SILENT_MIN:-45}"
DRY="${WATCH_DRY:-0}"
GH=runs/tasks/tinygsm/WINDOW_GLOBAL_HALT
act () { echo "ACTION $*"; }

global_halt () {   # $1 reason
  [ "$DRY" = 1 ] && { echo "ALERT (dry) would global-halt: $1"; return; }
  echo "$(date -Is) $1" >> "$GH"
  ids=$(squeue -u "$USER" -h -t PENDING -o "%i %j" | awk '$2 ~ /^(win_|wev_)/ {print $1}')
  [ -n "$ids" ] && scancel $ids
  act "GLOBAL HALT ($1): cancelled $(echo $ids | wc -w) pending jobs; running jobs left to finish their own epilogue"
}

# ---- queue snapshot ------------------------------------------------------
Q=$(squeue -u "$USER" -h -o "%i|%j|%T|%M" 2>/dev/null) || { echo "ALERT squeue failed -- no actions this tick"; exit 0; }

# ---- per-run table -------------------------------------------------------
crashed=0; diverged=0; done_n=0; total=0
printf "%-26s %8s %6s %-14s %s\n" run step it/s jobs note
for d in runs/tasks/tinygsm/win_*; do
  [ -d "$d" ] || continue
  r=$(basename "$d"); total=$((total+1))
  st="-"; [ -f "$d/checkpoints/last.pt" ] && st=$("$PY" - "$d/checkpoints/last.pt" <<'PYEOF' 2>/dev/null
import sys, torch
print(int(torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["global_step"]))
PYEOF
)
  jobs=$(awk -F'|' -v n="win_$r" '$2==n {print $3}' <<<"$Q" | sort | uniq -c | awk '{printf "%s%s ", $1, substr($2,1,1)}')
  note=""; speed="-"
  runj=$(awk -F'|' -v n="win_$r" '$2==n && $3=="RUNNING" {print $1}' <<<"$Q" | head -1)
  if [ -n "$runj" ]; then
    log="logs/arch/win_${r}_${runj}.log"
    if [ -f "$log" ]; then
      speed=$(tr '\r' '\n' < "$log" | grep -oE '[0-9.]+it/s' | tail -1 | tr -d 'it/s')
      age=$(( ( $(date +%s) - $(stat -c %Y "$log") ) / 60 ))
      if [ "$age" -ge "$SILENT_MIN" ]; then
        note="SILENT ${age}m"
        if [ "$DRY" != 1 ]; then scancel "$runj"; act "cancelled silent job $runj ($r, log idle ${age} min)"; fi
      fi
      e=$(grep -m1 -E "Traceback|OutOfMemory|non-finite" "$log" 2>/dev/null)
      [ -n "$e" ] && note="$note log-error: ${e:0:80}"
    fi
  fi
  if [ -f "$d/HALT" ]; then
    reason=$(tail -1 "$d/HALT" | cut -d' ' -f3-)
    note="HALT: ${reason:0:90}"
    case "$reason" in DIVERGED*) diverged=$((diverged+1)) ;; *) crashed=$((crashed+1)) ;; esac
  fi
  [ "$st" != "-" ] && [ "$st" -ge "${WIN_STEPS:-300000}" ] 2>/dev/null && { done_n=$((done_n+1)); note="DONE $note"; }
  printf "%-26s %8s %6s %-14s %s\n" "$r" "$st" "${speed:--}" "${jobs:-none}" "$note"
done
echo "runs: $total  done: $done_n  halted-crash: $crashed  diverged: $diverged"

# ---- budget ledger -------------------------------------------------------
gpuh=$(sacct -u "$USER" -S "$SINCE" -X -n -P --format=JobName,AllocTRES,ElapsedRaw 2>/dev/null |
  awk -F'|' '$1 ~ /^(win_|wev_)/ { if (match($2,/gres\/gpu=([0-9]+)/,m)) s+=m[1]*$3 } END {printf "%.0f", s/3600}')
echo "GPU-h since $SINCE (win_+wev_): ${gpuh:-?} / cap $CAP_GPUH"
if [ -n "$gpuh" ]; then
  [ "$gpuh" -ge "$CAP_GPUH" ] && [ ! -f "$GH" ] && global_halt "budget cap reached: ${gpuh} GPU-h >= $CAP_GPUH"
  [ "$gpuh" -ge $((CAP_GPUH * 8 / 10)) ] && [ "$gpuh" -lt "$CAP_GPUH" ] && echo "ALERT budget at >=80% of cap"
fi

# ---- circuit breaker -----------------------------------------------------
if [ "$crashed" -ge "$BREAKER" ] && [ ! -f "$GH" ]; then
  global_halt "circuit breaker: $crashed runs halted by crashes"
fi
[ -f "$GH" ] && echo "ALERT GLOBAL HALT in force: $(tail -1 "$GH")"
[ "$crashed" -gt 0 ] && echo "ALERT $crashed run(s) halted by a crash -- inspect HALT + log before any restart"

# ---- evaluations (gated) -------------------------------------------------
if [ ! -f "$GH" ]; then
  if [ "$DRY" = 1 ]; then bash scripts/hpc/arch/window_study.sh eval 2>&1 | sed 's/^/[eval] /'
  else bash scripts/hpc/arch/window_study.sh eval --go 2>&1 | sed 's/^/[eval] /'; fi
fi
nev=$(awk -F'|' '$2 ~ /^wev_/' <<<"$Q" | wc -l)
echo "eval jobs queued/running: $nev"
