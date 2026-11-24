#!/usr/bin/env bash
set -euo pipefail
cd /mnt/workspace/Hrj/gemma_hierarchy/k_sweep_1124a
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK
python - <<'PY'
import json, math
from pathlib import Path
rows=[json.loads(p.read_text()) for p in Path('smoke_v2').glob('*/result.json')]
assert len(rows)==10 and all(math.isfinite(r['test']['capacity']) for r in rows)
PY
mkdir -p formal
# Exactly one full-budget pilot; workers skip its completed result.
CUDA_VISIBLE_DEVICES=0 python -u run.py --worker 0 --workers 30 --out formal > pilot.log 2>&1
python - <<'PY'
import json, math
from pathlib import Path
r=json.loads(Path('formal/k2_specified_11/result.json').read_text())
assert r['steps']==6000 and math.isfinite(r['test']['nmse']) and r['train_seconds']>0
print('Pilot complete',r,flush=True)
PY
pids=()
for worker in {0..7}; do
    CUDA_VISIBLE_DEVICES=$worker python -u run.py --worker "$worker" --workers 8 --out formal > "worker_$worker.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
