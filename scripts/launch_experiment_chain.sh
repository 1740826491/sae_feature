#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/mnt/workspace/Hrj/gemma_hierarchy}
BASE=$ROOT/results/dclm_1m_9b_hyper_doc_1125
ROUTING=$BASE/offline_routing/routing_overlap_summary.csv
CAP=$BASE/overlap_capacity
LOG=$BASE/experiment_chain.log
echo "[$(date)] waiting for stage 3" >> "$LOG"
while [ ! -s "$ROUTING" ]; do sleep 60; done
echo "[$(date)] stage 3 detected; starting stage 5" >> "$LOG"
python "$ROOT/run_overlap_capacity.py" --overlap-csv "$ROUTING" --out "$CAP" >> "$LOG" 2>&1
touch "$CAP/STAGE_5_COMPLETE"
echo "[$(date)] stage 5 complete" >> "$LOG"
