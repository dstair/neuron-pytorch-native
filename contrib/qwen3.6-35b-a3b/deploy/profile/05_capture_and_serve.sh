#!/bin/bash
# Capture-replay the LARGEST emitted NEFF (the real decode graph) → device NTFF,
# then (re)start the UI and upload BOTH device + system profiles under myself.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

IMG="$QWEN35_NATIVE_IMAGE"
DEV="$QWEN35_PROFILE_ROOT/prof_dev"
SYS="$QWEN35_PROFILE_ROOT/prof_bs1"
DATA_DIR="$QWEN35_PROFILE_ROOT/ne_data"
UI_LOG="$QWEN35_PROFILE_ROOT/ne_ui.log"
NS=myself
docker rm -f q35_cap q35_ne_ui 2>/dev/null; docker ps -q | xargs -r docker kill 2>/dev/null; sleep 2
# largest NEFF = the decode graph (rank 0's dir)
NEFF=$(find "$DEV" -name "*.neff" -printf '%s %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
if [[ -z "$NEFF" ]]; then
  echo "error: no NEFF found under $DEV" >&2
  exit 1
fi
echo "replay NEFF=$NEFF ($(stat -c%s "$NEFF" 2>/dev/null) bytes)"
echo "=== capture-replay (4-worker collectives) ==="
docker run --rm --name q35_cap --privileged --device=/dev/neuron0 \
  -v "$QWEN35_PROFILE_ROOT":"$QWEN35_PROFILE_ROOT" "$IMG" \
  bash -lc "/opt/aws/neuron/bin/neuron-explorer capture -n '$NEFF' -s $DEV/device.ntff \
    --collectives-worker-count 4 -r 4 --ignore-exec-errors 2>&1 | tail -10"
ls -la "$DEV"/device.ntff 2>/dev/null || echo "NO device.ntff produced"

echo "=== (re)start UI ==="
mkdir -p "$DATA_DIR"
docker run -d --name q35_ne_ui --network host \
  -v "$QWEN35_PROFILE_ROOT":"$QWEN35_PROFILE_ROOT" "$IMG" \
  bash -lc "/opt/aws/neuron/bin/neuron-explorer view --data-path '$DATA_DIR' --port 3001 > '$UI_LOG' 2>&1"
sleep 30
echo "=== upload device profile ==="
[ -f "$DEV"/device.ntff ] && docker exec q35_ne_ui /opt/aws/neuron/bin/neuron-explorer upload \
  --endpoint http://localhost:3002 --neff "$NEFF" --ntff "$DEV"/device.ntff \
  --name bs1_banked --namespace "$NS" --uploader "$NS" --overwrite --wait 2>&1 | tail -4
echo "=== upload system profile ==="
TRACE=$(find "$SYS" -name ntrace.pb 2>/dev/null | head -1)
if [[ -n "$TRACE" ]]; then
  SYSDIR=$(dirname "$TRACE")
  docker exec q35_ne_ui /opt/aws/neuron/bin/neuron-explorer upload \
    --endpoint http://localhost:3002 --profile-directory "$SYSDIR" \
    --name bs1_banked_system --namespace "$NS" --uploader "$NS" --overwrite --wait \
    2>&1 | tail -4
else
  echo "no system profile found under $SYS"
fi
echo "=== registered profiles ==="
docker exec q35_ne_ui bash -lc "curl -s 'http://localhost:3002/api/v1/profiles/search?userName=$NS&uploader=$NS' -H 'x-user-id: $NS'" 2>&1 | python3 -c "import json,sys; d=json.load(sys.stdin); [print(p['profile_name'], p['type'], p['status']) for p in (d.get('data') or [])]" 2>/dev/null || echo "(search parse failed)"
