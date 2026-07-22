#!/bin/bash
# Step: start the neuron-explorer web UI (detached, --network host so an SSH
# -L 3001/3002 forward reaches it), then UPLOAD the captured profiles into the
# running server under namespace/uploader=myself (the viewer's x-user-id), which
# is the actual registration step — `view`/--data-path alone leaves the list EMPTY.
# See [[reference-neuron-explorer-ui]].
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

IMG="$QWEN35_NATIVE_IMAGE"
OUT="$QWEN35_PROFILE_ROOT/prof_bs1"
DATA_DIR="$QWEN35_PROFILE_ROOT/ne_data"
UI_LOG="$QWEN35_PROFILE_ROOT/ne_ui.log"
NS=myself
# locate device NEFF/NTFF + the system profile dir from the inspect output
NEFF=$(find "$OUT" -name "*.neff" 2>/dev/null | head -1)
NTFF=$(find "$OUT" -name "*.ntff" 2>/dev/null | head -1)
TRACE=$(find "$OUT" -name "ntrace.pb" 2>/dev/null | head -1)
SYSDIR=""
if [[ -n "$TRACE" ]]; then
  SYSDIR=$(dirname "$TRACE")
fi
echo "NEFF=$NEFF"; echo "NTFF=$NTFF"; echo "SYSDIR=$SYSDIR"

docker rm -f q35_ne_ui 2>/dev/null
mkdir -p "$DATA_DIR"
# Serve UI detached (binds 127.0.0.1:3001 UI + :3002 API). Re-ingests on launch.
docker run -d --name q35_ne_ui --network host \
  -v "$QWEN35_PROFILE_ROOT":"$QWEN35_PROFILE_ROOT" "$IMG" bash -lc \
  "/opt/aws/neuron/bin/neuron-explorer view --data-path '$DATA_DIR' --port 3001 2>&1 | tee '$UI_LOG'"
echo "waiting for API on :3002..."; sleep 25
docker exec q35_ne_ui bash -lc "curl -s -o /dev/null -w 'API %{http_code}\n' http://localhost:3002/ 2>&1" || true

# Register the device profile under the viewer's identity.
if [ -n "$NEFF" ] && [ -n "$NTFF" ]; then
  docker exec q35_ne_ui /opt/aws/neuron/bin/neuron-explorer upload \
    --endpoint http://localhost:3002 --neff "$NEFF" --ntff "$NTFF" \
    --name bs1_banked --namespace "$NS" --uploader "$NS" --overwrite --wait \
    2>&1 | tail -4
fi
# Register the system profile.
if [ -n "$SYSDIR" ]; then
  docker exec q35_ne_ui /opt/aws/neuron/bin/neuron-explorer upload \
    --endpoint http://localhost:3002 --profile-directory "$SYSDIR" \
    --name bs1_banked_system --namespace "$NS" --uploader "$NS" --overwrite --wait \
    2>&1 | tail -4
fi
echo "=== UI up on 3001/3002. Forward: ssh -L 3001:localhost:3001 -L 3002:localhost:3002 ... ==="
