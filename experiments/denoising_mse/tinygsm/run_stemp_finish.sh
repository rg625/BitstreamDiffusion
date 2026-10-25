#!/bin/bash
# wait for the tau=0.6 switching study, then analyse it
cd "$(dirname "$0")"
while kill -0 217489 2>/dev/null; do sleep 120; done
L=../logs/tg_stemp.log
{ echo "=== analysis $(date)"; python analyze_compute_demand.py --tag st0.6 && python compare_switching.py --tag st0.6; echo "=== analysis done $(date)"; } >> $L 2>&1
