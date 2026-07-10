#!/bin/bash
# Capture-replay the LARGEST emitted NEFF (the real decode graph) → device NTFF,
# then (re)start the UI and upload BOTH device + system profiles under myself.
set -u
IMG=${ECR_REGISTRY}/concourse-release-0461d3b:latest
DEV=/mnt/nvme/prof_dev; SYS=/mnt/nvme/prof_bs1; NS=myself
docker rm -f q35_cap q35_ne_ui 2>/dev/null; docker ps -q | xargs -r docker kill 2>/dev/null; sleep 2
# largest NEFF = the decode graph (rank 0's dir)
NEFF=$(find "$DEV" -name "*.neff" -printf '%s %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
echo "replay NEFF=$NEFF ($(stat -c%s "$NEFF" 2>/dev/null) bytes)"
echo "=== capture-replay (4-worker collectives) ==="
docker run --rm --name q35_cap --privileged --device=/dev/neuron0 -v /mnt/nvme:/mnt/nvme "$IMG" \
  bash -lc "/opt/aws/neuron/bin/neuron-explorer capture -n '$NEFF' -s $DEV/device.ntff \
    --collectives-worker-count 4 -r 4 --ignore-exec-errors 2>&1 | tail -10"
ls -la "$DEV"/device.ntff 2>/dev/null || echo "NO device.ntff produced"

echo "=== (re)start UI ==="
mkdir -p /mnt/nvme/ne_data
docker run -d --name q35_ne_ui --network host -v /mnt/nvme:/mnt/nvme "$IMG" \
  bash -lc "/opt/aws/neuron/bin/neuron-explorer view --data-path /mnt/nvme/ne_data --port 3001 > /mnt/nvme/ne_ui.log 2>&1"
sleep 30
echo "=== upload device profile ==="
[ -f "$DEV"/device.ntff ] && docker exec q35_ne_ui /opt/aws/neuron/bin/neuron-explorer upload \
  --endpoint http://localhost:3002 --neff "$NEFF" --ntff "$DEV"/device.ntff \
  --name bs1_banked --namespace "$NS" --uploader "$NS" --overwrite --wait 2>&1 | tail -4
echo "=== upload system profile ==="
SYSDIR=$(dirname "$(find "$SYS" -name ntrace.pb | head -1)")
docker exec q35_ne_ui /opt/aws/neuron/bin/neuron-explorer upload \
  --endpoint http://localhost:3002 --profile-directory "$SYSDIR" \
  --name bs1_banked_system --namespace "$NS" --uploader "$NS" --overwrite --wait 2>&1 | tail -4
echo "=== registered profiles ==="
docker exec q35_ne_ui bash -lc "curl -s 'http://localhost:3002/api/v1/profiles/search?userName=$NS&uploader=$NS' -H 'x-user-id: $NS'" 2>&1 | python3 -c "import json,sys; d=json.load(sys.stdin); [print(p['profile_name'], p['type'], p['status']) for p in (d.get('data') or [])]" 2>/dev/null || echo "(search parse failed)"
