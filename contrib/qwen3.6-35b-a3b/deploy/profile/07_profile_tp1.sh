#!/bin/bash
# Single-core device profile (neuron-nki-profiling skill approach): run the model
# at TP=1 + NOREDUCE=1 so the decode NEFF has NO collectives → capturable on one
# core with the skill's clean flags. Gives the per-engine/per-op COMPUTE breakdown
# (DeltaNet/MoE/GQATAIL/projections) — not the cross-core collective cost, but
# that's the actionable bottleneck view. Uses fewer layers to keep compile short
# (the per-op attribution is per-layer-type, repeats every 4 layers).
set -u
IMG=${ECR_REGISTRY}/concourse-release-0461d3b:latest
OUT=/mnt/nvme/prof_tp1; NL=${NL:-8}
docker rm -f q35_prof q35_cap 2>/dev/null; docker ps -q | xargs -r docker kill 2>/dev/null
docker rm -f q35_ne_ui 2>/dev/null; sleep 2
rm -rf "$OUT"; mkdir -p "$OUT"
echo "=== (1) emit TP=1 NOREDUCE decode NEFF (NL=$NL) on core 0 ==="
docker run --rm --name q35_prof --privileged --device=/dev/neuron0 \
  -e QWEN35_MODEL_PATH=/models/Qwen3.5-35B-A3B \
  -e DN_NKI=1 -e MOE_SPARSE=1 -e GQATAIL=1 -e DNBATCHED_V2=1 -e NOREDUCE=1 -e PROFILE_STEPS=5 \
  -e NEURON_RT_INSPECT_ENABLE=1 -e NEURON_RT_INSPECT_DEVICE_PROFILE=1 -e NEURON_RT_INSPECT_OUTPUT_DIR=/out \
  -e NEURON_RT_VISIBLE_CORES=0 \
  -v /home/ubuntu/qwen35:/work -v /mnt/nvme/Qwen3.5-35B-A3B:/models/Qwen3.5-35B-A3B -v "$OUT":/out \
  -w /work/Qwen3.6-35B-A3B "$IMG" bash -lc "
  source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null
  torchrun --nproc-per-node=1 static_decode_35b.py --model-path /models/Qwen3.5-35B-A3B \
    --max-seq-len 2048 --num-tokens 4 --num-layers $NL --graph-splits 1 --batch-size 1 2>&1 | tail -3
" 2>&1 | grep -ivE "nccl|CCOM|OFI|Warning" | tail -5
NEFF=$(find "$OUT" -name "*.neff" -printf '%s %p\n' | sort -n | tail -1 | awk '{print $2}')
echo "NEFF=$NEFF ($(stat -c%s "$NEFF" 2>/dev/null)B)"
echo "=== (2) single-core capture (skill flags) ==="
docker run --rm --name q35_cap --privileged --device=/dev/neuron0 -e NEURON_RT_VISIBLE_CORES=0 \
  -v /mnt/nvme:/mnt/nvme "$IMG" bash -lc \
  "/opt/aws/neuron/bin/neuron-explorer capture -n '$NEFF' -s $OUT/tp1.ntff --profile-nth-exec=2 --enable-dge-notifs 2>&1 | tail -12"
ls -la "$OUT"/tp1.ntff 2>/dev/null && echo "NTFF OK" || echo "no NTFF"
echo "=== (3) summary-json metrics ==="
docker run --rm -v /mnt/nvme:/mnt/nvme "$IMG" bash -lc \
  "/opt/aws/neuron/bin/neuron-explorer view --output-format summary-json -n '$NEFF' -s $OUT/tp1.ntff 2>/dev/null > $OUT/metrics.json; \
   python3 -c \"import json; m=json.load(open('$OUT/metrics.json')); [print(f'{k}: {m[k]}') for k in sorted(m) if any(s in k for s in ['latency','engine','percent','hbm','bytes','intensity','count'])]\" 2>&1 | head -30"
