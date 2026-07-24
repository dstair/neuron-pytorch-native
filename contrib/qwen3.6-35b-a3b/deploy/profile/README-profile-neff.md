# Profiling the 40-layer TP=8 BS=128 FP8 decode graph

End-to-end recipe to capture a source-attributed full-graph profile of the
Qwen3.6-35B-A3B FP8 decode graph. Verified working 2026-07-23 (tiled BS=128 FP8:
Vector 170 ms / DMA 162 ms / total 353 ms, 66 MB debug NEFF).

The 40-layer TP=8 graph is too large for direct on-device profiling
(`NEURON_RT_INSPECT_DEVICE_PROFILE=1` exhausts HBM; `--enable-dge-notifs`
overflows). The working method is: **cross-compile a debug NEFF on trn1, then
collective-replay-capture it on trn2.**

## 1. Cross-compile the debug NEFF (trn1.32xlarge, us-east-2)

Use `deploy/compile_decode_fp8_trn2.sh`. It ALREADY uses the correct harness
(`test_decode_fullgraph_device.py`) — do **not** switch it to `static_decode_35b.py`:
static_decode hangs at the cross-host execute step and is SIGTERM'd before the
fullgraph NEFF's persistent-cache write finalizes, so only ~52 KB NKI-kernel NEFFs
persist. `test_decode_fullgraph_device.py` catches the expected "Invalid NEFF"
error after compile, writes rank markers, and exits clean → the ~66 MB fullgraph
NEFF survives.

```bash
cd <repo>/contrib/qwen3.6-35b-a3b
export QWEN35_NATIVE_IMAGE=<concourse DLC image>
export QWEN35_MODEL_DIR=/mnt/nvme/models/Qwen3.5-35B-A3B
export QWEN35_FP8_MODEL_DIR=/mnt/nvme/models/Qwen3.5-35B-A3B-FP8   # FP8 mode REQUIRES this
export QWEN35_NKILIB_DIR=/mnt/nvme/nki-library
export QWEN35_XC_NEURON_CC_FLAGS="--target trn2 --lnc 1"          # O2 (match shipped); do NOT add --optlevel 1
export XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 NEURON_FRAMEWORK_DEBUG=1     # source attribution
DN_TILED_CONV=1 deploy/compile_decode_fp8_trn2.sh \
  --mode fp8 --fp8-impl block_pow2_coalesced \
  --layers 40 --batch-size 128 --direct-state-output on \
  --compile-concurrency 8 --cache-dir /mnt/nvme/xc-cache/tiled-prof-dbg
```

Success = eight ~66 MB `*.neff` under `<cache-dir>/neff_cache/` (one per rank) +
8 `rank*.done` markers. A result of only ~52 KB NEFFs means the harness regressed
to static_decode. Transfer the largest NEFF to trn2 (S3; from the dev box use
`aws s3api put-object`, not `aws s3 cp`).

## 2. Capture-replay on trn2 (trn2.3xlarge, ap-southeast-4)

Runs in the DLC. Needs the host neuron lib mount (driver mismatch) AND
`NEURON_LOGICAL_NC_CONFIG=1` (8-core LNC=1; without it `nrt_init` fails — the
instance is native LNC=2 = 4 cores).

```bash
docker run --rm --privileged --device=/dev/neuron0 \
  -v /opt/aws/neuron/lib:/host_neuron_lib:ro -e NEURON_LOGICAL_NC_CONFIG=1 \
  -v $PWD:/work -w /work <DLC> bash -c '
  export LD_LIBRARY_PATH=/host_neuron_lib:$LD_LIBRARY_PATH
  neuron-explorer capture -n /work/fullgraph.neff -s /work/fg.ntff \
    --collectives-worker-count 8 -r 8 -i 0 --single-io \
    --profile-nth-exec=2 --ignore-exec-errors'   # DGE OFF (no --enable-dge-notifs)
```

Gotcha: capture writes `<base>_rank_0_exec_2.ntff`, NOT the `-s` name you passed.

## 3. Ingest + query

`neuron-explorer view` takes >2 min on a ~1 GB NTFF — wait for it. Use the
CONTAINER-relative NTFF path (`/work/...`), not the host path.

```bash
# Aggregate engine times + HBM (fast, reliable):
neuron-explorer view --output-format summary-json -n /work/fullgraph.neff \
  -s /work/<base>_rank_0_exec_2.ntff > summ.json
# Per-source attribution (parquet), then query with profile/prof_query.py:
neuron-explorer view --ingest-only --data-path /nedata -n /work/fullgraph.neff \
  -s /work/<base>_rank_0_exec_2.ntff
```

`perfect_pipeline = max(per-engine active time)`,
`serialization_gap = total_time - perfect_pipeline`. DGE-off replay leaves
`DmaPacket` incomplete → use Summary `hbm_read_bytes`/`hbm_write_bytes` for
traffic, not the packet byte total.
