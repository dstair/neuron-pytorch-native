# Qwen3.5-4B Trainium2 decode results

Measured September 9–10, 2026 on one `trn2.3xlarge` in `ap-southeast-4`.
All runs use sequence bucket 256.

Toolchain:

- PyTorch 2.12.1
- torch-neuronx 2.12.3.0.1636 (beta-5)
- Neuron runtime 2.34.10
- `--target trn2 --lnc 1|2 --optlevel 1`, matching the topology

Each row uses the same full 32-layer graph, an untimed cold compile/first
execution followed by three explicit warmups, then 20 synchronized timed
decode steps. Throughput is aggregate batch tokens per second.

| Weight path | TPOT | Throughput | Module GiB/rank | First-step agreement vs BF16 |
|---|---:|---:|---:|---:|
| BF16 | **188.35 ms** | **679.6 tok/s** | 3.39 | reference |
| FP8 vocabulary head, BF16 backbone | 194.39 ms | 658.5 tok/s | 3.25 | 119/128 (93.0%) |
| FP8 all projections | 223.26 ms | 573.3 tok/s | **2.42** | 106/128 (82.8%) |

All three runs populated every row of all 24 DeltaNet recurrent states
(`131072` nonzero rows per layer), so the result does not come from skipped
state updates.

## Conclusions

### LNC=1 data-parallel results

Eight independent TP=1 graphs were bound one per physical core. Each GQA layer
uses four distinct 4-Q/1-KV kernel calls and cache buffers, and all 32 GQA cache
groups plus all 24 DeltaNet states passed occupancy checks.

| Topology and weights | Batch | Timed steps | Aggregate throughput | Module GiB/core |
|---|---:|---:|---:|---:|
| DP=8, TP=1, fused FP8 MLP + fused head argmax | 32/core | 50 | **935.1 tok/s** | 6.57 |
| DP=8, TP=1, fused FP8 MLP + generic head | 32/core | 50 | **898.8 tok/s** | 6.57 |
| DP=8, TP=1, generic FP8 MLP + head | 32/core | 50 | 748.1 tok/s | 6.57 |
| generic FP8 MLP + head, first run | 32/core | 20 | 737.7 tok/s | 6.57 |
| DP=8, TP=1, all projections FP8 | 32/core | 20 | 667.0 tok/s | **5.36** |
| DP=8, TP=1, all projections FP8 | 64/core | 20 | 641.0 tok/s | 5.61 |

The fused MLP raises DP8 throughput by 20.1% over the generic selective-FP8
path. Fusing vocabulary projection, scaling, and argmax adds another 4.0%,
for a total **25.0% improvement** over 748.1 tok/s. The final result is 37.6%
above the TP=4 BF16 control and 63.1% above TP=4 all-FP8.

On one isolated LNC=1 core, the same progression measured 109.5 tok/s
(generic), 150.7 tok/s (fused MLP), and 164.8 tok/s (fused MLP and head).
The standalone fused MLP device test was bit-exact against the three generic
FP8 calls. A cached full-model control preserved all 32 first-step IDs; after
nine stateful executions, 31/32 final IDs matched. The fused head returned
identical IDs in its standalone test and preserved both first and final IDs
relative to the fused-MLP full graph. Every run populated all DeltaNet state
rows and GQA cache groups.

The two LNC=1 cores sharing each 24-GiB HBM bank divide throughput unevenly,
but their pair totals are stable. Increasing batch from 32 to 64 raises
single-core throughput from 96.5 to 108.9 tok/s but reduces the eight-replica
aggregate because pair contention dominates.

### LNC=2 data-parallel result

Under LNC=2, logical core IDs 0–3 each combine a pair of physical
NeuronCores. Four independent full TP=1 models were bound one per logical core.
BS=64 per replica preserves the DP8 winner's aggregate batch of 256.

| Topology and weights | Batch | Timed steps | Mean TPOT | Aggregate throughput | Module GiB/replica |
|---|---:|---:|---:|---:|---:|
| DP=4, TP=1, LNC=2, FP8 MLP + head, BF16 attention | 64/replica | 50 | 499.04 ms | **513.0 tok/s** | 6.82 |

Replica throughput ranged only from 128.238 to 128.259 tok/s. All 24
DeltaNet states had all `262144` rows populated and all 32 GQA cache groups
had all 64 batch rows populated. Replica 0 agreed with the TP=4 BF16 control
on 55/64 first-step tokens and exactly matched the first 32 tokens from the
same selective-FP8 DP8 path.

The generic-FP8 LNC=2 layout is 45.1% slower than the fused DP8/LNC1 winner
and 24.5% slower than the TP=4 BF16 control. Per-replica throughput rises
17.1% over the original LNC=1 selective-FP8 BS=32 qualification, but each
replica consumes twice as many physical cores; the extra per-model throughput
does not offset halving the replica count.

The fastest tested configuration is therefore DP=8/TP=1/LNC=1 at BS=32 per
replica, with the fused native FP8 MLP and vocabulary-argmax kernels enabled.

Quantizing every projection cuts module storage by 28.8%, but reduces
throughput by 15.6%. The generic per-linear FP8 custom calls do not recover
their boundary, scaling, and output-materialization overhead on this dense 4B
graph. This matches the failure mode already documented by the 27B work:
FP8 is a capacity win until dequantization and adjacent operations are fused
into wider kernels.

Use the fused DP8 `mlp-head` recipe for maximum aggregate throughput, TP=4
BF16 for a model-parallel replica, and full FP8 when minimum module footprint
matters more than speed.

## Profile (pre-fusion DP8 winner, 748.1 tok/s)

Captured September 10, 2026 with `profiling/profile_dp8_lnc1_trn2.sh`: one
representative full-graph NEFF replayed alone on one LNC=1 core, then queried
with `profiling/analyze_profile.py`. The replay took 296.61 ms (107.9 tok/s at
BS=32). It does **not** reproduce the shared-HBM contention of the eight-copy
run, whose mean is 13.3% lower per core.

| Engine | Active | Share of 296.61 ms |
|---|---:|---:|
| GpSimd | **168.84 ms** | **56.9%** |
| Tensor | 145.88 ms | 49.2% |
| DMA | 123.84 ms | 41.8% |
| Scalar | 80.83 ms | 27.3% |
| Vector | 72.82 ms | 24.5% |
| Serialization/overlap gap | **127.77 ms** | **43.1%** |

- GpSimd was dominated by `DMA_DIRECT2D` from the generic `fp8_matmul` kernel
  (116.79 ms merged at its line 127).
- DMA reached only 24.7% of the 435 GB/s per-core peak. 57.3% of bytes moved in
  transfers of at most 4 KiB, and spill traffic was 6.07 GB (2.63 GB save, 3.44 GB
  reload).
- Transposes were 59.7% of Tensor Engine work, 96.6% of it compiler-generated,
  but worth only 4.66 ms at peak.
- Caveat: source-attributed DMA covered only 17.8% of packet bytes, so per-line DMA
  and spill rankings are directional.

That profile motivated the two fused kernels, which replaced every generic FP8
call on the winner path and raised DP8 from 748.1 to 935.1 tok/s. The raw NTFF,
`metrics.json`/`analysis.json` and the parquet ingest are on the trn2 host under
`/mnt/nvme/qwen4b-work/profiles/` (instance store, so gone after a stop/start).

## Open work

1. Profile the fused 935.1 tok/s winner. The profile above predates the fused
   kernels, so the next bottleneck is unmeasured.
2. Re-run the batch sweep around BS=32 with the fused kernels.
3. Reduce the 6.07 GB spill round-trip. Partial attribution points at DeltaNet
   and GQA state/cache handling; it needs a higher-coverage trace first.
4. Fuse the DeltaNet QKV/Z/A/B projections into the batched DeltaNet kernel,
   loading and transposing the activation once per layer. A *standalone* FP8
   in_proj kernel measured 4–12× slower than compiled BF16 on the 27B, so this
   only pays off if fused.
5. Add prefill (the 35B chunked DeltaNet and CTE/flash GQA paths) and integrate
   the decode graph with a serving stack.

## Reproducing

All rows use `deploy/bench_trn2.sh` (see its header for the environment
variables). DP topologies compile one replica into a runtime template, then
`run` clones it per core.

| Row | Commands |
|---|---|
| DP8 fused MLP + head, 935.1 | `compile dp8` then `run dp8` |
| DP8 fused MLP, generic head, 898.8 | `QWEN4B_FUSED_FP8_HEAD=0` on both |
| DP8 generic FP8, 748.1 | `QWEN4B_FUSED_FP8_MLP=0 QWEN4B_FUSED_FP8_HEAD=0` on both |
| DP8 all-FP8, BS=32 / 64 | `QWEN4B_FP8_SCOPE=all` (+ `QWEN4B_BATCH_SIZE=64`), fused off, `QWEN4B_BENCH_ITERS=20` |
| DP4 LNC=2, 513.0 | `compile dp4-lnc2` then `run dp4-lnc2` |
| TP=4 BF16 / FP8 head / all-FP8 | `run tp4 --dtype bf16\|fp8-lm-head\|fp8 --batch-size 128 --max-seq-len 256 --warmup 3 --bench-iters 20` |

Use a separate `QWEN4B_RUNTIME_TEMPLATE` for each topology/scope/fused/batch
combination. Every replica copies the whole template, and the profiler captures
the largest NEFF it finds there.
