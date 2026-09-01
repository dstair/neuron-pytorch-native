#!/usr/bin/env bash
# Cross-compile the FP8 CTE prefill graph on a trn1.32xlarge (495 GB host RAM) for
# target trn2/LNC=1, so high-batch configs that wedge the trn2.3xlarge's 128 GB host
# can be compiled at all. The NEFF cache is then restored on the trn2 for the timed
# run (compile host != benchmark host -- never quote a number from here).
#
# This is the FP8 counterpart of deploy/compile_prefill_trn2.sh, whose hard-coded
# BF16 BS=2/20k invocation predates the FP8 CTE path. Every graph-affecting env var
# below is copied VERBATIM from deploy/run_prefill_fp8_ob.sh, because the persistent
# cache key is sha256(HLO) x sha256(target|...|compiler_version|flags): any drift in
# source, flags, or container image silently produces a cache MISS on the trn2 and
# you re-pay the whole compile natively.
#
# The only additions are cross-target, and none of them touch the traced graph:
#   LD_PRELOAD=<shim>                 fakes the instance family INSIDE the cache-key
#                                     constructor only, so the key says trn2
#   NEURON_PLATFORM_TARGET_OVERRIDE   compile for trn2
#   QWEN35_CACHE_PLATFORM_TARGET      what the shim writes into the key
#   CROSS_TARGET_COMPILE_ONLY=1       routes the eager seed/layout tensors through CPU
#                                     so no wrong-target NEFF executes before the
#                                     prefill graphs are traced
#
# TP=8/LNC=1 ONLY. A TP=4/LNC=2 invocation dies in ~40 s at init_process_group --
# LNC=2 is a Trn2-only logical-core config trn1 hardware cannot bring up.
#
# Expected ending: the trn2-target NEFFs compile and persist, then FAIL to load on
# trn1 cores ("Invalid NEFF" / DMA_ABORT / NRT status 1203). That is SUCCESS for this
# script -- the cache write finalizes at compile time, before load. Gate on the NEFF
# count in the cache, never on the exit code.
#
# usage: xc_prefill_fp8_trn1.sh [bs] [n] [fp8] [layers] [splits] [bucket] [blk] [maxtok] [ob] [hoist] [dbginfo]
# default = the BS=16 sibling of the 4,289.2 tok/s record (bs4 n10000 l40 s4 bc1024 mt0 ob1 ho1)
#
# dbginfo=1 adds XLA_IR_DEBUG / XLA_HLO_DEBUG / NEURON_FRAMEWORK_DEBUG so the NEFF
# carries nki_source_location and a profile can attribute instructions to source lines.
# Without it ~36% of instruction time lands in <unattributed> (measured 2026-08-22 at
# BS=6), which is enough to make a component ranking meaningless -- bwmm read as 1.54 ms
# purely because its source locations were missing.
#
# These flags CHANGE THE HLO, hence the cache key, so a dbginfo NEFF is a DIFFERENT
# cache entry from the shipping one. Hence the -dbg tag suffix: a debug NEFF must never
# be served to a timed run, and a profiling compile must never invalidate a timing
# cache. Profile with it; never quote a throughput number from it.
set -uo pipefail
BS="${1:-16}"; N="${2:-10000}"; FP8="${3:-1}"; LAYERS="${4:-40}"; SPLITS="${5:-4}"
BUCKET="${6:-1024}"; BLK="${7:-512}"; MAXTOK="${8:-0}"
OB="${9:-1}"; HOIST="${10:-1}"; DBGINFO="${11:-0}"
MAXSEQ=$(( ( (N + BUCKET - 1) / BUCKET ) * BUCKET ))
if [[ "$DBGINFO" == "1" ]]; then
  DBG_ENV=(-e XLA_IR_DEBUG=1 -e XLA_HLO_DEBUG=1 -e NEURON_FRAMEWORK_DEBUG=1)
  DBG_TAG="-dbg"
else
  DBG_ENV=()
  DBG_TAG=""
fi

WORK=/mnt/nvme/lnc1-work
IMAGE=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
MODEL=/mnt/nvme/Qwen3.5-35B-A3B
NKILIB=$WORK/nki-library
SRC=$WORK/src/contrib/qwen3.6-35b-a3b
SHIM=$WORK/shim/libnrt_platform_target_override.so
TAG=bs${BS}-n${N}-l${LAYERS}-s${SPLITS}-bc${BUCKET}-blk${BLK}-mt${MAXTOK}-fp8${FP8}-ob${OB}-ho${HOIST:-d}${DBG_TAG}
LOG=$WORK/logs/xc_prefill_${TAG}.log
CACHE=$WORK/cache-xc-${TAG}
NAME=q35-xc-prefill-${TAG}

mkdir -p "$WORK/logs" "$(dirname "$SHIM")"

# The nkilib moe_cte kernel hard-asserts NUM_SHARDS == 2 and blocks TP=8/LNC=1 with
# [NCC_INKI016]. This checkout is copied from the trn2 with the patch already applied;
# re-verify rather than assume, because an unpatched compile fails ~25 min in.
if grep -q "only work on TRN2" "$NKILIB/src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_I.py"; then
  echo "FATAL: $NKILIB is NOT patched for LNC=1" | tee "$LOG"; exit 1
fi
# The output-block FP8 hoist lives in nkilib too (patches/nkilib-output-block-quant.patch).
# Without it OB=1/HOIST=1 silently compiles the OLD op count and the cache is useless.
if [[ "$OB" == "1" ]] && ! grep -q "is_output_block_quant" \
     "$NKILIB/src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_I.py"; then
  echo "FATAL: $NKILIB lacks the output-block-quant patch but OB=1" | tee "$LOG"; exit 1
fi

# Build the cross-target shim OUTSIDE the source tree: $SRC is byte-compared against
# the trn2 to prove cache-key equality, so nothing may be written into it.
if [[ ! -f "$SHIM" ]]; then
  QWEN35_NATIVE_IMAGE="$IMAGE" "$SRC/deploy/cross_compile/build_shim.sh" "$SHIM" \
    >>"$LOG" 2>&1 || { echo "FATAL: shim build failed (see $LOG)"; exit 1; }
fi

# Preflight: prove the shim actually rewrites the family INSIDE the cache-key
# constructor. This is the single point of failure that would otherwise waste the
# whole compile by silently writing trn1-keyed entries that miss on the trn2.
#
# Do NOT gate on `_C._get_platform_target()` -- the shim is deliberately selective
# (it checks its own backtrace for CompilationCacheKey + CompileOnlyKernelExecution),
# so that call correctly returns the PHYSICAL target trn1 even when the shim is
# working perfectly. Instead run the shipped validate_cache_override.py, which forces
# a real cache-key construction via _C.compile_graph(), and with SHIM_DEBUG=1 assert
# the shim's own line reports the override. NRT family 5 == trn2 (2 == trn1).
PREFLIGHT=$(docker run --rm --privileged \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e LD_PRELOAD=/opt/qwen35/libnrt_platform_target_override.so \
  -e QWEN35_CACHE_PLATFORM_TARGET=trn2 -e NEURON_LOGICAL_NC_CONFIG=1 \
  -e QWEN35_PLATFORM_TARGET_SHIM_DEBUG=1 \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$SHIM":/opt/qwen35/libnrt_platform_target_override.so:ro \
  -v "$SRC/deploy/cross_compile/validate_cache_override.py":/opt/qwen35/v.py:ro \
  "$IMAGE" bash -lc 'source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    python /opt/qwen35/v.py' 2>&1)
printf '%s\n' "$PREFLIGHT" | grep -F 'qwen35-platform-override' >>"$LOG" 2>&1
if ! printf '%s\n' "$PREFLIGHT" |
     grep -qE 'qwen35-platform-override .*target=trn2 overridden_family=5'; then
  { echo "FATAL: shim did not override the cache key to trn2 (family 5)."
    printf '%s\n' "$PREFLIGHT" | tail -20; } | tee -a "$LOG"
  exit 1
fi
PLATFORM=trn2

docker rm -f "$NAME" >/dev/null 2>&1 || true
# A --privileged container creates the cache root-owned; a non-sudo rm fails SILENTLY
# and a stale NEFF is served forever after. Always sudo-clear.
sudo rm -rf "$CACHE" "$WORK/nbackend"
mkdir -p "$CACHE"

{
  echo "START $(date -u +%FT%TZ) tag=$TAG"
  echo "  host=$(hostname) image=$(docker image inspect --format '{{.Id}}' "$IMAGE")"
  echo "  maxseq=$MAXSEQ splits=$SPLITS blk=$BLK maxtok=$MAXTOK ob=$OB hoist=$HOIST"
  echo "  platform_preflight=$PLATFORM cache=$CACHE dbginfo=$DBGINFO"
  echo "  src_py_md5=$(cd "$SRC" && find . -name '*.py' -not -path '*/__pycache__/*' \
        | sort | xargs md5sum | md5sum | awk '{print $1}')"
} | tee -a "$LOG"

docker run --rm --name "$NAME" --privileged --network host --ipc host \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e LD_PRELOAD=/opt/qwen35/libnrt_platform_target_override.so \
  -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e QWEN35_CACHE_PLATFORM_TARGET=trn2 \
  -e CROSS_TARGET_COMPILE_ONLY=1 \
  -e CROSS_TARGET_MARKER_DIR=/tmp/cross-target-compile \
  -e NEURON_LOGICAL_NC_CONFIG=1 -e QWEN35_LNC=1 \
  -e NEURON_SCRATCHPAD_PAGE_SIZE=256 \
  -e NEURON_CC_FLAGS="--target trn2 --lnc 1 --optlevel 1 --hbm-scratchpad-page-size 256" \
  -e NKI_ENABLE_TRACE_CACHE=0 \
  -e MOE_CTE=1 -e MOE_CTE_NKI_PACK=1 -e MOE_CTE_BLOCK=${BLK} -e MOE_CTE_MAX_TOKENS=${MAXTOK} -e MOE_CTE_FP8=${FP8} \
  -e MOE_CTE_FP8_OUTPUT_BLOCK="${OB}" -e MOE_CTE_FP8_OB_HOIST="${HOIST}" \
  -e GQA_CTE_PREFILL=1 -e GQA_DYNAMIC_ROPE_KV=1 \
  -e DN_CHUNK_NKI=1 -e CHUNK_SIZE=32 -e DN_STABLE_C32=0 -e DN_PACK_C32=1 -e DN_PACK_N=4 \
  -e DN_NKI=1 -e GQATAIL=1 -e PREFILL_FINGERPRINT=1 \
  -e DN_K_HEADS=2 -e DN_V_HEADS=4 -e GQA_Q_HEADS=2 \
  -e PREFILL_SHARDED_LM_HEAD=1 -e PREFILL_SHARDED_EMBED=1 \
  ${DBG_ENV[@]+"${DBG_ENV[@]}"} \
  -e PYTHONPATH=/nki-library/src/nkilib_src \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$SHIM":/opt/qwen35/libnrt_platform_target_override.so:ro \
  -v "$SRC":/work:ro -v "$NKILIB":/nki-library:ro \
  -v "$MODEL":/models/Qwen3.5-35B-A3B:ro \
  -v "$CACHE":/tmp \
  -w /work "$IMAGE" bash -lc "
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    torchrun --nproc-per-node=8 --max-restarts=0 static_decode_35b.py \
      --model-path /models/Qwen3.5-35B-A3B \
      --batch-size ${BS} --num-layers ${LAYERS} --max-seq-len ${MAXSEQ} \
      --prefill-bench ${N} --bucket-chunk ${BUCKET} \
      --bucket-compile 1 --prefill-splits ${SPLITS} --skip-compile
  " >> "$LOG" 2>&1
RC=$?
NEFFS=$(sudo find "$CACHE/neff_cache" -name '*.neff' 2>/dev/null | wc -l)
BIG=$(sudo find "$CACHE/neff_cache" -name '*.neff' -size +1M 2>/dev/null | wc -l)
# Distinct size classes = distinct graphs; each should have one NEFF per rank.
CLASSES=$(sudo find "$CACHE/neff_cache" -name '*.neff' -size +1M -printf '%s\n' 2>/dev/null \
  | awk '{printf "%d\n", $1/1048576}' | sort -u | wc -l)
EXPECTED=$(( SPLITS * 8 ))
echo "DOCKER_EXIT=$RC neffs=$NEFFS big_neffs=$BIG size_classes=$CLASSES expected_big=$EXPECTED $(date -u +%FT%TZ)" \
  | tee -a "$LOG"

# Exit code is NOT the gate: a trn2 NEFF cannot load on trn1, so a successful
# cross-compile ALWAYS ends non-zero (rank N SIGSEGVs on load with "additional fields
# for dynamic dma on v2 arch missing from neff", which SIGTERMs its siblings). The cache
# write finalizes at compile time, before load, so that ending is expected.
#
# But "$BIG -ge 1" was FAR too lenient: on 2026-09-01 a run whose ranks were SIGTERM'd
# partway emitted 25 of 32 large NEFFs and still exited 0. That matters differently by
# purpose, so make the purpose explicit:
#   - restoring a TIMING cache needs every NEFF; a missing one is a mid-run cache miss
#     (or a silent recompile on the trn2, whose 128 GB host is why we cross-compile).
#   - PROFILING needs only one representative per size class, since capture-replay
#     replays a single NEFF across 8 simulated workers.
# XC_ALLOW_PARTIAL=1 selects the profiling gate. Default demands the full set.
if [[ "$BIG" -ge "$EXPECTED" ]]; then
  echo "GATE PASS: complete ($BIG/$EXPECTED large NEFFs, $CLASSES size classes)" | tee -a "$LOG"
  exit 0
fi
if [[ "${XC_ALLOW_PARTIAL:-0}" == "1" && "$CLASSES" -ge 1 && "$BIG" -ge 1 ]]; then
  echo "GATE PASS (PARTIAL, profiling only): $BIG/$EXPECTED large NEFFs in $CLASSES size" \
       "classes. Enough to capture-replay; NOT enough to restore as a timing cache." | tee -a "$LOG"
  exit 0
fi
echo "GATE FAIL: only $BIG/$EXPECTED large NEFFs ($CLASSES size classes)." \
     "Set XC_ALLOW_PARTIAL=1 if this is for profiling. Otherwise retry, and consider a" \
     "higher --prefill-splits: more splits = smaller per-segment graph = less walrus RSS." \
  | tee -a "$LOG"
exit 1
