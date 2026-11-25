#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${LIBRARY_PATH:-}"

ROOT=/mnt/workspace/Hrj/gemma_hierarchy
MODEL="$ROOT/models/gemma-2-9b"
SAE="$ROOT/models/gemma-scope-9b-pt-res/layer_27/width_131k/average_l0_56/params.npz"
DATA=/mnt/workspace/dclm
OUT="$ROOT/results/dclm_25m_9b_residpost_full_1125"
LOG="$ROOT/logs/dclm_25m_9b_residpost_full_1125.log"

mkdir -p "$OUT" "$ROOT/logs"
cat > "$OUT/manifest.txt" <<EOF
model=$MODEL
sae=$SAE
layer=27
hook=full model.model.layers[27] output
data=$DATA
max_tokens=25000000
seq_len=256
batch_size=8
topk=150
save_doc_features=true
EOF

source /usr/local/PPU_SDK/envsetup.sh ppu
exec torchrun --standalone --nproc_per_node=8 \
  "$ROOT/gemma_hierarchy_runner_residpost.py" \
  --model "$MODEL" \
  --sae "$SAE" \
  --data "$DATA" \
  --out "$OUT" \
  --layer 27 \
  --max-tokens 25000000 \
  --seq-len 256 \
  --batch-size 8 \
  --topk 150 \
  --save-doc-features
