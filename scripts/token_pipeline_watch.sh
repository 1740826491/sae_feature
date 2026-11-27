#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/mnt/workspace/Hrj/gemma_hierarchy}
ACT=$ROOT/results/dclm_1m_9b_token_activations_1126
HIER=$ROOT/results/dclm_1m_9b_token_hyper_1126
OUT=$ROOT/results/token_properties_1126
LOG=$ROOT/results/token_properties_1126.log
mkdir -p "$OUT"
echo "[$(date)] waiting for token hierarchy" >> "$LOG"
while [ ! -s "$HIER/complete.json" ]; do sleep 60; done
echo "[$(date)] hierarchy ready; running E1-E4/E6" >> "$LOG"
python "$ROOT/run_token_properties.py" --activation "$ACT" --out "$OUT" >> "$LOG" 2>&1
echo "[$(date)] token properties complete" > "$OUT/PIPELINE_COMPLETE"
