# Source before any Neuron run on a trn2 box with LNC=2 (the default on trn2.3xlarge).
#   NEURON_VENV   path to a venv with torch-neuronx / neuronx-cc (its bin/ must be on PATH, or compiles fail
#                 with "posix_spawn failed")
#   ASR_WORK      where ref/, model/ and artifacts/ outputs go (default: this directory)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${NEURON_VENV:-}" ]; then source "$NEURON_VENV/bin/activate"; fi
export ASR_WORK="${ASR_WORK:-$HERE}"
# Without this, torch-neuronx >= 2.14 compiles some built-in NKI kernels for LNC=1 -> NCC_EVRF066.
export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}"
export NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---hbm-scratchpad-page-size=1024}"
export NEURON_SCRATCHPAD_PAGE_SIZE=1024
export USE_NEURON_SCHEDULER_V2=1
export TORCH_NEURONX_NEFF_CACHE_DIR="${TORCH_NEURONX_NEFF_CACHE_DIR:-$ASR_WORK/neff-cache}"
# Local snapshot (or a HF hub id) for Qwen/Qwen3-ASR-0.6B-hf; cpu_ref.py and run.py both read this.
export QASR_MODEL="${QASR_MODEL:-$ASR_WORK/model}"
