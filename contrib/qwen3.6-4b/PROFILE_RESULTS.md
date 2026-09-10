# Qwen3.5-4B LNC=1 profile

Profiled September 10, 2026 on the `trn2.3xlarge` used for the throughput
measurements. At capture time, the target was the fastest configuration:
eight TP=1/LNC=1 replicas, BS=32 each, with FP8 MLP and vocabulary projections
and BF16 attention projections. That configuration measured
**748.1 aggregate tok/s**.

## Method and scope

`neuron-explorer` captured the qualified full-graph NEFF on one LNC=1 core,
then ingested the NTFF into parquet for the queries in `analyze_profile.py`.
The standalone replay took 296.61 ms, or 107.9 tok/s at BS=32, close to the
109.5 tok/s isolated qualification result.

The capture intentionally runs one representative graph alone. It describes
the kernels used by every replica, but it does not reproduce the shared-HBM
contention of the eight-copy run. The measured DP8 mean is 93.5 tok/s/core,
13.3% below the standalone replay.

Data-quality caveats:

- 92.4% of instruction rows have source locations.
- `DmaPacketAggregated` covers 24.6% of packet bytes.
- Source-attributed DMA covers 17.8% of packet bytes.
- Ingestion warned that DGE packet counts exceeded DMA trace entries on all
  three DMA engines. Whole-profile packet totals and engine times are used
  below; source-level DMA and spill rankings are directional.

The retained host artifacts are:

- NTFF: `/mnt/nvme/qwen4b-work/profiles/dp8-lnc1-prebuilt-capture/fullgraph_exec_2.ntff`
- Summary and query output: the adjacent `metrics.json` and `analysis.json`
- Parquet: `/mnt/nvme/qwen4b-work/profiles/ne-data/profiles/global/qwen4b-dp8-lnc1-prebuilt@latest`

## Performance bounds

| Measurement | Time | Share of total |
|---|---:|---:|
| Total graph | 296.61 ms | 100.0% |
| GpSimd active | **168.84 ms** | **56.9%** |
| Tensor Engine active | 145.88 ms | 49.2% |
| DMA active | 123.84 ms | 41.8% |
| Scalar active | 80.83 ms | 27.3% |
| Vector active | 72.82 ms | 24.5% |
| Perfectly overlapped lower bound | 168.84 ms | 56.9% |
| Serialization/overlap gap | **127.77 ms** | **43.1%** |

GpSimd is the longest-active engine. Its trace is dominated by
`DMA_DIRECT2D`; the largest source-attributed interval is
`fp8_matmul.py:127`, at 116.79 ms of merged activity. GpSimd causal analysis
is still limited, so this identifies the measured hotspot rather than proving
that one source line accounts for the full wall-clock gap.

## DMA

- 13.31 GB moved in 11.49 million packets.
- Achieved bandwidth while DMA was active was 107.4 GB/s, **24.7%** of the
  435 GB/s per-core peak.
- 57.3% of bytes used transfers no larger than 4 KiB; 91.0% of packets were at
  most 1 KiB.
- Software-dynamic DMA carried 89.6% of packet bytes.
- Profile summary spill traffic was **6.07 GB**: 2.63 GB save and 3.44 GB
  reload.
- Raw DMA bytes were 1.48x the input/output tensor-byte lower bound.

The strongest measured opportunity is therefore to make the selective FP8
path issue fewer, wider transfers and keep intermediates on-chip for longer.

## Tensor Engine

- Adjusted hardware work was 613.47 GFLOP.
- Transpose work was 366.15 GFLOP, **59.7%** of adjusted hardware work.
- Achieved throughput while active was 4.21 TFLOP/s, **5.35%** of peak.
- 96.6% of transpose FLOPs lack a source location and appear to be
  compiler-generated layout work.

Transpose FLOPs are large, but their ideal peak cost is only 4.66 ms. Layout
cleanup should accompany wider fused kernels; it is not a substitute for
addressing the GpSimd/DMA scheduling and overlap gap.

## Recommended throughput work

1. Fuse gate/up/SwiGLU/down into a batched FP8 MLP kernel. Load the activation
   once, use wide contiguous weight tiles, and retain the intermediate in SBUF.
2. Fuse FP8 vocabulary projection, scaling, and local argmax so full logits
   are not written to and reread from HBM.
3. Coalesce or hoist the software-dynamic transfers around
   `fp8_matmul.py:127`; measure packet count, median transfer size, and GpSimd
   active time after each change.
4. Reduce live ranges and materializations that drive the 6.07 GB spill
   round-trip. Partial attribution points first to DeltaNet and GQA state/cache
   handling, but those rankings need a higher-coverage trace before a targeted
   rewrite.
5. Re-run DP8 after single-core improvements. A standalone gain that increases
   HBM demand may disappear when two cores share each HBM bank.

The first two items directly extend the fused-kernel approach from the 35B
implementation and are the best candidates to improve the 748.1 tok/s result.

## Implemented follow-up

The two primary recommendations were implemented after this capture:

- Fused gate/up/SwiGLU/down: **898.8 DP8 tok/s**
- Added fused vocabulary projection/scaling/argmax: **935.1 DP8 tok/s**

The final result is 25.0% above the profiled 748.1 tok/s baseline, with the
same 6.57 GiB module footprint.

## Reproduction

On the Trainium2 host:

```bash
export QWEN4B_RUNTIME_TEMPLATE=<qualified-lnc1-runtime-cache>
export QWEN4B_PROFILE_ROOT=<profile-output-directory>
deploy/profile_dp8_lnc1_trn2.sh
```

The launcher captures an existing full-graph NEFF instead of enabling
framework source-debug flags, which would invalidate the cache and trigger a
full eight-replica recompile.
