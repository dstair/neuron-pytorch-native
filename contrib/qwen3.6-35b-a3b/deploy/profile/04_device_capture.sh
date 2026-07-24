#!/bin/bash
# Device profile (engine/instruction breakdown): (1) emit the big decode NEFF via
# an inline NEURON_RT_INSPECT_DEVICE_PROFILE run (PROFILE_STEPS=5), then (2)
# capture-replay it with 4-worker collectives → device NTFF. NEFF is per-rank;
# replay rank 0's. See [[reference-neuron-explorer-ui]] / FP8 profile memory.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
require_qwen35_model

IMG="$QWEN35_NATIVE_IMAGE"
OUT="$QWEN35_PROFILE_ROOT/prof_dev"
docker rm -f q35_prof q35_cap 2>/dev/null; docker ps -q | xargs -r docker kill 2>/dev/null
docker rm -f q35_ne_ui 2>/dev/null; sleep 2   # free the device (UI holds none, but be safe)
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== (1) emit device NEFF+NTFF (inline NEURON_RT_INSPECT_DEVICE_PROFILE) ==="
docker run --rm --name q35_prof --privileged --device=/dev/neuron0 \
  -e QWEN35_MODEL_PATH=/models/Qwen3.5-35B-A3B \
  -e DN_NKI=1 -e MOE_SPARSE=1 -e GQATAIL=1 -e DNBATCHED_V2=1 -e PROFILE_STEPS=5 \
  -e NEURON_RT_INSPECT_ENABLE=1 -e NEURON_RT_INSPECT_DEVICE_PROFILE=1 \
  -e NEURON_RT_INSPECT_OUTPUT_DIR=/out \
  -v "$QWEN35_SOURCE_DIR":/work:ro \
  -v "$QWEN35_MODEL_DIR":/models/Qwen3.5-35B-A3B:ro \
  -v "$OUT":/out \
  -w /work "$IMG" bash -lc '
  source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null
  torchrun --nproc-per-node=4 static_decode_35b.py --model-path /models/Qwen3.5-35B-A3B \
    --max-seq-len 2048 --num-tokens 4 --num-layers 40 --graph-splits 1 --batch-size 1 2>&1 | tail -3
' 2>&1 | grep -ivE "nccl|CCOM|OFI|Warning" | tail -6
echo "=== emitted NEFFs ==="; find "$OUT" -name "*.neff" | sort | head
NEFF=$(find "$OUT" -name "*.neff" | sort | head -1)
if [[ -z "$NEFF" ]]; then
  echo "error: no NEFF found under $OUT" >&2
  exit 1
fi
echo "replay NEFF=$NEFF"

echo "=== (2) capture-replay (4-worker collectives) → device NTFF ==="
docker run --rm --name q35_cap --privileged --device=/dev/neuron0 \
  -v "$OUT":"$OUT" "$IMG" bash -lc "
  /opt/aws/neuron/bin/neuron-explorer capture -n '$NEFF' \
    -s '$OUT/device.ntff' --collectives-worker-count 4 -r 4 --ignore-exec-errors 2>&1 | tail -8
"
echo "=== device NTFF ==="; ls -la "$OUT"/device*.ntff 2>/dev/null
