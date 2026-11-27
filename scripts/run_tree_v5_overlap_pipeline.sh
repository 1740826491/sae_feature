#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/workspace/Hrj/tree_moe_overlap_exact
REP_PID_FILE="$ROOT/representative.pid"
REP_OUT="$ROOT/results_v5_overlap_representative"
FORMAL_OUT="$ROOT/results_v5_overlap_formal"
while kill -0 "$(cat "$REP_PID_FILE")" 2>/dev/null; do sleep 20; done
python - "$REP_OUT" <<'PY'
import json,math,sys
from pathlib import Path
p=Path(sys.argv[1]); rows=[json.loads(f.read_text()) for f in p.glob('*.json')]
assert len(rows)==35,(len(rows),'representative cells missing')
assert not [r for r in rows if 'error' in r], [r for r in rows if 'error' in r]
methods={r['cfg']['model']+':'+r['cfg'].get('routing','dense')+':'+r['cfg'].get('gate','') for r in rows}
required={'dense:dense:','tree:perfect:','tree:independent:','tree:learned:hier','tree:learned:flat3'}
assert required<=methods,(required-methods,methods)
for r in rows:
 assert r['curve'] and all(math.isfinite(float(x['weighted'])) for x in r['curve'])
Path(p/'REPRESENTATIVE_COMPLETE').write_text('35 full-scale cells passed\n')
PY
python "$ROOT/gen_tree_v5_overlap_configs.py" --spec-root "$ROOT/spec_v5_overlap" --out "$ROOT/configs_formal.jsonl"
mkdir -p "$FORMAL_OUT" "$ROOT/formal_logs"
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$worker nohup python "$ROOT/tmoe.py" --spec "$ROOT/spec_v5_overlap/real" \
    --configs "$ROOT/configs_formal.jsonl" --out "$FORMAL_OUT" --worker "$worker" --workers 8 \
    > "$ROOT/formal_logs/worker_${worker}.log" 2>&1 &
  echo $! > "$ROOT/formal_logs/worker_${worker}.pid"
done
touch "$ROOT/FORMAL_LAUNCHED"
