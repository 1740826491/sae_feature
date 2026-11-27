#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/mnt/workspace/Hrj/gemma_hierarchy}
RUN=${RUN:-sft_8way_1m_token_1126}
ACT="$ROOT/results/${RUN}_activations"
STATS="$ROOT/results/${RUN}_stats"
HIER="$ROOT/results/${RUN}_hierarchy"
PURITY="$ROOT/results/${RUN}_domain_purity"
PROPS="$ROOT/results/${RUN}_properties"
SMOKE="$ROOT/results/${RUN}_smoke"
LOG="$ROOT/results/${RUN}_pipeline.log"
MODEL="$ROOT/models/gemma-2-9b"
SAE="$ROOT/models/gemma-scope-9b-pt-res/layer_27/width_131k/average_l0_56/params.npz"
DATA="$ROOT/sft_domain_8way_1125"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
export PPU_HOME=${PPU_HOME:-/usr/local/PPU_SDK}
mkdir -p "$ACT" "$STATS" "$HIER" "$PURITY" "$PROPS"
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
require(){ test -e "$1" || { log "missing required path: $1"; exit 2; }; }
require "$MODEL/config.json";require "$SAE";require "$DATA/00_bbh.txt"
if [ ! -s "$SMOKE/SMOKE_COMPLETE" ]; then
 log "stage 0: starting 8-rank smoke"
 rm -rf "$SMOKE";mkdir -p "$SMOKE"
 torchrun --standalone --nproc_per_node=8 "$ROOT/gemma_sft_token_runner.py" --model "$MODEL" --sae "$SAE" --data "$DATA" --out "$SMOKE" --layer 27 --max-tokens 16000 --batch-size 8 --topk 150 >>"$LOG" 2>&1
 python - "$SMOKE" <<'PY'
import json,sys,numpy as np
from pathlib import Path
p=Path(sys.argv[1]);progress=list(p.glob('progress_rank*.json'));assert len(progress)==8,len(progress)
rows=bad=0
for q in (p/'token_activations').glob('*.npz'):
 v=np.load(q)['activation_values'];rows+=len(v);bad+=np.isnan(v).all(1).sum()
valid=rows-bad
assert rows>0 and valid/rows>=.25,(rows,bad,valid)
(p/'SMOKE_COMPLETE').write_text(json.dumps({'rows':rows,'all_nan_rows':int(bad),'valid_rows':int(valid),'valid_fraction':valid/rows}))
PY
 log "stage 0 complete"
fi
if [ ! -e "$ACT/FORWARD_COMPLETE" ]; then
 log "stage 1: starting SFT token forward"
 torchrun --standalone --nproc_per_node=8 "$ROOT/gemma_sft_token_runner.py" --model "$MODEL" --sae "$SAE" --data "$DATA" --out "$ACT" --layer 27 --max-tokens 1000000 --batch-size 8 --topk 150 >>"$LOG" 2>&1
 test "$(find "$ACT" -maxdepth 1 -name 'progress_rank*.json' | wc -l)" -eq 8
 touch "$ACT/FORWARD_COMPLETE";log "stage 1 complete"
fi
if [ ! -e "$STATS/STATS_COMPLETE" ]; then
 log "stage 2: building token frequency/coactivation"
 python "$ROOT/build_token_level_stats.py" --input "$ACT" --output "$STATS" >>"$LOG" 2>&1
 touch "$STATS/STATS_COMPLETE";log "stage 2 complete"
fi
if [ ! -e "$HIER/TOKEN_LEVEL_HIERARCHY_COMPLETE" ]; then
 log "stage 3: building token hierarchy"
 python "$ROOT/run_hifer_doc_hierarchy.py" --input "$STATS" --output "$HIER" >>"$LOG" 2>&1
 touch "$HIER/TOKEN_LEVEL_HIERARCHY_COMPLETE";log "stage 3 complete"
fi
if [ ! -s "$PURITY/complete.json" ]; then
 log "stage 4: computing token domain purity"
 python "$ROOT/compute_sft_token_domain_purity.py" --activation "$ACT" --hierarchy "$HIER" --out "$PURITY" >>"$LOG" 2>&1
 log "stage 4 complete"
fi
if [ ! -e "$PROPS/PIPELINE_COMPLETE" ]; then
 log "stage 5: running preliminary token properties E1-E4/E6"
 python "$ROOT/run_token_properties.py" --activation "$ACT" --out "$PROPS" >>"$LOG" 2>&1
 touch "$PROPS/PIPELINE_COMPLETE";log "stage 5 complete"
fi
touch "$ROOT/results/${RUN}_PIPELINE_COMPLETE";log "all stages complete"
