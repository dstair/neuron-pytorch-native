#!/usr/bin/env bash
# The CLEAN environment for every number we intend to quote. Sourced by container.sh.
#
# Per skills/trainium-run-discipline rule 1: debug flags belong in a SEPARATE script,
# never behind a flag in this one. NEURON_LAUNCH_BLOCKING + DGE notifications measured
# ~15x slower on a 40-layer graph and produced two published wrong numbers. If you need
# diagnosis, use debug_env.sh and do not quote its timings.
#
# LNC: this box is 1 Trn2 chip / 96 GB.
#   LNC=2 (default) -> 4 logical cores @ 24 GB   <- proven regime, use for correctness/LoRA
#   LNC=1           -> 8 logical cores @ 12 GB   <- use only when rank COUNT is the point

LNC="${LNC:-2}"
WORLD="${WORLD:-1}"

# Prepend rather than replace: the public DLC also puts EFA and OpenMPI libs on this path.
export LD_LIBRARY_PATH=/opt/aws/neuron/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# beta6 reads NEURON_LOGICAL_NC_CONFIG. NEURON_RT_VIRTUAL_CORE_SIZE -- which the beta3-era
# FSDP recipe used -- appears nowhere in this build's sources; setting it does nothing.
export NEURON_LOGICAL_NC_CONFIG="$LNC"
if [[ -n "${CORES:-}" ]]; then
  # One process pinned to explicit cores (parallel single-core evals). NUM_CORES and
  # VISIBLE_CORES are mutually exclusive in the runtime.
  export NEURON_RT_VISIBLE_CORES="$CORES"
else
  export NEURON_RT_NUM_CORES="$WORLD"
fi
export NEURON_CC_FLAGS="--target trn2 --lnc $LNC --model-type=transformer --auto-cast=none"

# Persisted NEFF/HLO cache. $CACHE_DIR on the host is mounted at /tmp, which also
# absorbs neuronx-cc's hardcoded /tmp/neuron_backend (it can otherwise fill the root disk).
export NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache

# Keep host fallback to ops the backend genuinely cannot do, so a silent CPU fallback
# does not quietly become the thing we are timing.
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1

# Host-side collective communication. beta6 hard-errors on HOST_CC without STREAMS, so they
# are only ever set as a PAIR, and only opt-in (HOST_CC=1) because streams change execution
# semantics and must not be on for a single-core baseline. Under test as a candidate fix for
# FSDP2's in-graph allgather failing with
#   NMGR:dlr_build_and_load_cc_resources  Failed to build and load collectives resources
# even though bare eager all_gather succeeds.
if [[ "${HOST_CC:-0}" == 1 ]]; then
  export TORCH_NEURONX_ENABLE_STREAMS=1
  export TORCH_NEURONX_ENABLE_HOST_CC=1
fi

# Deliberately NOT set here (measured against this build, 2026-09-23):
#   TORCH_NEURONX_ENABLE_ASYNC_NRT -- a beta3 flag; absent from beta6 sources, dead.

export HF_HOME=/work/hf
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
