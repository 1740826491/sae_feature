#!/usr/bin/env bash
set -euo pipefail
cd /mnt/workspace/Hrj/gemma_hierarchy/k_sweep_1124a
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK
OUT=formal_freq_lora_alt_1124
mkdir -p "$OUT"
pids=()
for worker in {0..7}; do
  CUDA_VISIBLE_DEVICES="$worker" python -u run_moe_classification_lora_freq.py \
    --worker "$worker" --workers 8 --out "$OUT" \
    > "formal_freq_lora_worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf 'FINAL_STATUS=%s\n' "$status"
exit "$status"
