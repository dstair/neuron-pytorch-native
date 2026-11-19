#!/usr/bin/env bash
# Runs ON THE BOX. Launches the native-PyTorch DLC (STACK=public|beta6) with the device,
# the work volume, and the clean env, then runs a command inside it.
#
#   ./container.sh exec python3 -m sqlft.train --layers 4 ...
#   ./container.sh exec torchrun --nproc_per_node 4 labs/lab_collectives.py
#   ./container.sh shell
#   ./container.sh clean-cache      # MUST be sudo -- the cache is root-owned (see below)
#   ./container.sh versions
#
# Env knobs: STACK (public|beta6, default public), LNC (1|2, default 2), WORLD (ranks,
# default 1), NAME, IMAGE, CORES (pin a single-process job to these logical cores, e.g.
# CORES=2 -- lets independent evals share the chip one core each; never set it for
# torchrun jobs, the backend assigns cores per rank itself), SQLFT_MODEL (checkpoint dir
# inside the container, default /work/models/Qwen3.5-4B).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/load_env.sh"
WORK_ROOT="${WORK_ROOT:-$(cd "$HERE/../.." && pwd)}"   # e.g. /mnt/nvme/qwen35-training
SRC_REL="src"

# STACK=public (default): the public-beta TorchNeuron DLC, Neuron SDK 2.33.0.
# STACK=beta6: the pre-GA build behind every number measured before 2026-10-21. It is not
# public, so its reference comes from QWEN35_TRAIN_BETA6_IMAGE (.env), never from git.
# Each stack gets its own NEFF cache so an A/B never serves one compiler's NEFF to the other.
STACK="${STACK:-public}"
case "$STACK" in
  public) BASE_IMAGE="public.ecr.aws/neuron/torch-neuronx@sha256:b4648437b8c7dbfa2acde18301a4bb4ae9658da5a412da75c59693ed4fd308ec" ;;
  beta6)  require_env QWEN35_TRAIN_BETA6_IMAGE "pre-GA DLC reference, pinned by digest"
          BASE_IMAGE="$QWEN35_TRAIN_BETA6_IMAGE" ;;
  *) echo "unknown STACK=$STACK (public|beta6)" >&2; exit 2 ;;
esac
CACHE_DIR="$WORK_ROOT/cache-$STACK"

# The derived image from env/Dockerfile (DLC + transformers/datasets/duckdb/kernels).
# Falls back to the bare DLC so lab_smoke.py works before setup_box.sh has ever run.
if [[ -z "${IMAGE:-}" ]]; then
  if sudo docker image inspect "qwen35-train:$STACK" >/dev/null 2>&1; then
    IMAGE="qwen35-train:$STACK"
  else
    IMAGE="$BASE_IMAGE"
  fi
fi
NAME="${NAME:-qwen35_train_$$}"
LNC="${LNC:-2}"
WORLD="${WORLD:-1}"

mkdir -p "$CACHE_DIR" "$WORK_ROOT/runs"

docker_run() {
  # --privileged is what gets /dev/neuron0 in; without it torch_neuronx fails at
  # _get_platform_target with "No neuron device available".
  #
  # We deliberately do NOT mount the host /opt/aws/neuron, even though the older scripts
  # in this repo do. beta6 bundles its own NRT and its _C.so needs a symbol the host's
  # runtime-lib 2.34.10 does not export:
  #     ImportError: _C.so: undefined symbol: nrta_tensor_cast, version NRT_3.0.0
  # Set MOUNT_HOST_NEURON=1 to restore the old behaviour if a future image needs it.
  local mounts=()
  [[ "${MOUNT_HOST_NEURON:-0}" == 1 ]] && mounts+=(-v /opt/aws/neuron:/opt/aws/neuron:ro)

  # $CACHE_DIR lands on /tmp so it also absorbs neuronx-cc's hardcoded /tmp/neuron_backend.
  # /var/tmp is shared from the host too: NKI kernel binaries are cached there
  # (nki-intermediate-cache) and the compiled modules in the SHARED /tmp reference them by
  # path. With a per-container /var/tmp, concurrent containers (parallel evals) fail with
  # NCC_EVRF059 "Kernel file ... does not exist on the host" -- measured 2026-11-09.
  mkdir -p "$CACHE_DIR/var_tmp"
  sudo docker run --rm --name "$NAME" --privileged --network host --ipc host \
    -v "$CACHE_DIR/var_tmp":/var/tmp \
    -e LNC="$LNC" -e WORLD="$WORLD" -e HOST_CC="${HOST_CC:-0}" \
    -e CORES="${CORES:-}" \
    -e SQLFT_MODEL="${SQLFT_MODEL:-/work/models/Qwen3.5-4B}" \
    "${mounts[@]}" \
    -v "$WORK_ROOT":/work \
    -v "$CACHE_DIR":/tmp \
    -w "/work/$SRC_REL" \
    "$@"
}

IN_CONTAINER_PREAMBLE='
  source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
  source /work/'"$SRC_REL"'/env/run_env.sh
  export PYTHONPATH="/work/'"$SRC_REL"'${PYTHONPATH:+:$PYTHONPATH}"
'

case "${1:-}" in
  exec)
    shift
    [[ $# -gt 0 ]] || { echo "exec needs a command" >&2; exit 2; }
    printf -v cmd '%q ' "$@"
    docker_run -i "$IMAGE" bash -lc "$IN_CONTAINER_PREAMBLE
      exec $cmd"
    ;;

  shell)
    docker_run -it "$IMAGE" bash -lc "$IN_CONTAINER_PREAMBLE
      exec bash"
    ;;

  bash)   # run a shell snippet (needed by setup_box.sh; quoting differs from exec)
    shift
    docker_run -i "$IMAGE" bash -lc "$IN_CONTAINER_PREAMBLE
      $*"
    ;;

  clean-cache)
    # A --privileged container creates the cache as root. A plain `rm -rf` fails
    # SILENTLY and the previous NEFF is served on every later run -- this invalidated
    # about six consecutive experiments once. Always sudo, before any run you will quote.
    # beta6 names: neuron_cache / neuron_backend / nbackend. The public stack's are
    # neff_cache / hlo_cache / local_cache -- the old list silently cleaned nothing there.
    sudo rm -rf "$CACHE_DIR"/neuron_cache "$CACHE_DIR"/neuron_backend "$CACHE_DIR"/nbackend \
      "$CACHE_DIR"/neff_cache "$CACHE_DIR"/hlo_cache "$CACHE_DIR"/local_cache
    # Each device-OOM writes three dump files here (the driver hardcodes /tmp). An OOM that
    # retries instead of failing fast leaves hundreds of them.
    sudo find "$CACHE_DIR" -maxdepth 1 \
      \( -name 'neuron_mem_table_*' -o -name 'nrt_mem_log_*' -o -name 'neuron_mem_log_*' \) \
      -delete 2>/dev/null || true
    echo "cleared NEFF cache and OOM dumps under $CACHE_DIR"
    ;;

  versions)
    docker_run -i -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$IMAGE" bash -lc "$IN_CONTAINER_PREAMBLE
      python3 -c '
import importlib.metadata as md, torch
print(\"torch\", torch.__version__)
for p in [\"torch-neuronx\",\"neuronx-cc\",\"nki\",\"transformers\",\"datasets\",\"safetensors\",\"duckdb\",\"kernels\",\"peft\"]:
    try: print(p, md.version(p))
    except Exception: print(p, \"MISSING\")
'"
    ;;

  *) echo "usage: $0 {exec CMD...|bash SNIPPET|shell|clean-cache|versions}" >&2; exit 2 ;;
esac
