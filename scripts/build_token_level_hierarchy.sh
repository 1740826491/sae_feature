#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/mnt/workspace/Hrj/gemma_hierarchy}
ACT=$ROOT/results/dclm_1m_9b_token_activations_1126
STATS=$ROOT/results/dclm_1m_9b_token_hyper_stats_1126
HIER=$ROOT/results/dclm_1m_9b_token_hyper_1126
LOG=$ROOT/results/dclm_1m_9b_token_hyper_1126.log
while [ "$(find "$ACT" -maxdepth 1 -name 'progress_rank*.json' 2>/dev/null | wc -l)" -lt 8 ]; do sleep 60; done
python "$ROOT/build_token_level_stats.py" --input "$ACT" --output "$STATS" >> "$LOG" 2>&1
python "$ROOT/run_hifer_doc_hierarchy.py" --input "$STATS" --output "$HIER" >> "$LOG" 2>&1
touch "$HIER/TOKEN_LEVEL_HIERARCHY_COMPLETE"
