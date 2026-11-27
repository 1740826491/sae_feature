#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/workspace/Hrj/gemma_hierarchy
OUT=$ROOT/results/overlap_efficiency_real_matched_1126
SAE=$ROOT/models/gemma-scope-9b-pt-res/layer_27/width_131k/average_l0_56/params.npz
TABLE=$ROOT/results/dclm_1m_9b_hyper_doc_1125/document_routing_overlap_v2_main_1126/feature_vs_domain.csv
mkdir -p "$OUT/logs"
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Full-scale representative cell: same 3000 steps/batch/config as formal cells.
python "$ROOT/run_overlap_efficiency_sweep.py" --sae "$SAE" --overlap-table "$TABLE" --out "$OUT" \
  --only-overlap real_feature_depth1 --only-width 32 --only-seed 11 \
  > "$OUT/logs/representative.log" 2>&1
python - "$OUT" <<'PY'
import json,math,sys
from pathlib import Path
p=Path(sys.argv[1]); fs=list((p/'real_feature_depth1').glob('w32_*_s11/result.json'))
assert len(fs)==4, len(fs)
for f in fs:
 r=json.loads(f.read_text()); assert math.isfinite(r['test_nmse']) and r['steps']==3000
(p/'REPRESENTATIVE_COMPLETE').write_text('4 formal-scale cells passed\n')
PY

# Eight devices, disjoint resumable cell partitions.
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$worker nohup python "$ROOT/run_overlap_efficiency_sweep.py" \
    --sae "$SAE" --overlap-table "$TABLE" --out "$OUT" --worker "$worker" --workers 8 \
    > "$OUT/logs/worker_${worker}.log" 2>&1 &
  echo $! > "$OUT/logs/worker_${worker}.pid"
done
wait
python "$ROOT/summarize_overlap_efficiency.py" --input "$OUT" --out "$OUT/summary"
touch "$OUT/PIPELINE_COMPLETE"
