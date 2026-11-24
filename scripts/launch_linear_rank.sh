#!/usr/bin/env bash
set -eu
cd /mnt/workspace/Hrj/gemma_hierarchy/k_sweep_1124a
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK
mkdir -p linear_rank
CUDA_VISIBLE_DEVICES=0 python run_linear_rank.py --smoke --out linear_rank/smoke > linear_rank/smoke.log 2>&1
CUDA_VISIBLE_DEVICES=0 python run_linear_rank.py --workers 100 --worker 0 --out linear_rank/formal > linear_rank/pilot.log 2>&1
python -c 'import json,pathlib; r=json.loads(next(pathlib.Path("linear_rank/formal").glob("l0.0_d597_r0_dense_11/result.json")).read_text());print(r,flush=True);assert r["mix"]<0.1'
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python run_linear_rank.py --workers 8 --worker $i --out linear_rank/formal > linear_rank/worker_$i.log 2>&1 &
done
wait
