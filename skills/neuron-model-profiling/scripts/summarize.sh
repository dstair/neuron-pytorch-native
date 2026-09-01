#!/usr/bin/env bash
# Engine-level summary for one captured region.
#   usage: summarize.sh <tag>   ->  $W/summ_<tag>.json
#
# summary-json prints to STDOUT; --output-file is REJECTED for this format (version-bound).
# A `view` run on a ~1 GB NTFF takes minutes. Empty output != failure; wait.
set -uo pipefail
TAG="${1:?usage: summarize.sh <tag>}"
W=${W:-/mnt/nvme/lnc1-work/profile}
IMAGE=${IMAGE:-421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest}

docker run --rm --privileged -v /opt/aws/neuron/lib:/host_neuron_lib:ro \
  -v "$W":/work -w /work "$IMAGE" bash -lc "
    export LD_LIBRARY_PATH=/host_neuron_lib:\$LD_LIBRARY_PATH
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    neuron-explorer view --output-format summary-json \
      -n /work/${TAG}.neff -s /work/${TAG}_rank_0_exec_2.ntff > /work/summ_${TAG}.json
  " > "$W/summ_${TAG}.log" 2>&1
echo "$TAG summarize exit=$? -> $W/summ_${TAG}.json ($(wc -c < "$W/summ_${TAG}.json" 2>/dev/null || echo 0) bytes)"
