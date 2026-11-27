#!/usr/bin/env bash
set -euo pipefail

# Single resumable driver. It never silently skips a stage.
ROOT=/mnt/workspace/Hrj/gemma_hierarchy
HIER=$ROOT/results/dclm_1m_9b_hyper_doc_1125
PURITY=$ROOT/results/sft_domain_8way_gemma9b_1125/document_domain_purity.csv
OUT=$ROOT/results/overnight_hierarchy_pipeline_1126
mkdir -p "$OUT/logs"

wait_for_file() { local f=$1; while [[ ! -s "$f" ]]; do echo "waiting_for=$f"; sleep 60; done; }
write_status() { printf '%s\n' "$2" > "$OUT/$1.status"; }

write_status stage1 RUNNING
wait_for_file "$HIER/complete.json"
wait_for_file "$HIER/feature_depth.csv"
write_status stage1 COMPLETE

write_status stage2 RUNNING
python3 /mnt/workspace/Hrj/gemma_hierarchy/merge_hierarchy_domain_purity.py \
  --hierarchy "$HIER/feature_depth.csv" --purity "$PURITY" --out "$OUT/domain_purity" \
  > "$OUT/logs/domain_purity.log" 2>&1
write_status stage2 COMPLETE

write_status stage3 RUNNING
python3 "$ROOT/run_offline_routing_overlap.py" --activation "$ROOT/results/dclm_1m_9b_residpost_top150_car_A" --depth "$HIER/feature_depth.csv" --out "$OUT/overlap" > "$OUT/logs/overlap.log" 2>&1
write_status stage3 COMPLETE
write_status stage4 RUNNING
python3 "$ROOT/run_fixed_assignment_capacity.py" --overlap "$OUT/overlap" --out "$OUT/capacity" > "$OUT/logs/capacity.log" 2>&1
write_status stage4 COMPLETE
write_status stage5 RUNNING
python3 "$ROOT/run_overlap_matched_synthetic.py" --overlap "$OUT/overlap" --out "$OUT/synthetic_capacity" > "$OUT/logs/synthetic_capacity.log" 2>&1
write_status stage5 COMPLETE
write_status pipeline COMPLETE
