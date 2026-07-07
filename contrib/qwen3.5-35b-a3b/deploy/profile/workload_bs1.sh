#!/bin/bash
source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null
export QWEN35_MODEL_PATH=/models/Qwen3.5-35B-A3B
export DN_NKI=1 MOE_SPARSE=1 GQATAIL=1 DNBATCHED_V2=1 PROFILE_STEPS=5
exec torchrun --nproc-per-node=4 /work/Qwen3.6-35B-A3B/static_decode_35b.py \
  --model-path /models/Qwen3.5-35B-A3B --max-seq-len 2048 --num-tokens 4 \
  --num-layers 40 --graph-splits 1 --batch-size 1
