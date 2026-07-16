# AGENT.md — Qwen3.6-35B-A3B (MoE) on Trainium2, PyTorch Native

Guidance for an AI coding agent (Claude or similar) bringing this model up on AWS
Trainium2 in a fresh environment. It captures the architecture, the design
conventions, the validated run recipes, and — most importantly — the hard-won
gotchas that are expensive to rediscover. Pair it with `README.md` (results) and
the code in this directory.

> Naming: published as `Qwen/Qwen3.6-35B-A3B` on Hugging Face; the architecture
> class is `Qwen3_5MoeForConditionalGeneration` (`model_type: qwen3_5_moe`). "3.5"
> = arch family, "3.6" = release. Same architecture; this code runs on the HF
> checkpoint (see the weights note below).

## What this is

A PyTorch-Native (`torch.compile(fullgraph=True, backend="neuron")`) inference
harness for a 35B sparse-MoE hybrid model: 40 layers = [DeltaNet ×3, GQA ×1] ×10,
256-expert top-8 MoE on every layer, hidden 2048, TP=4 on one `trn2.3xlarge`
(LNC=2, 96 GB device / ~24 GB per core). Verified dims live in `model_dims.py`
(single source of truth, config-driven). Routing math and full shapes are in
`README.md`.

## Design conventions (read before editing)

- **One static forward → one NEFF.** `static_decode_35b.py` expresses the whole
  40-layer decode (and prefill) as a fixed-shape function that compiles to a single
  NEFF. This is what avoids the per-op eager-dispatch overhead that dominates
  latency. Do not introduce data-dependent shapes into the compiled path.
- **Heavy ops must be opaque NKI kernels, not pure torch.** neuronx-cc's tiler
  cannot handle the DeltaNet recurrence or GQA attention when they're exposed as
  pure-torch einsums at scale — see the compile gotchas below. Custom kernels live
  in `kernels/`, each registered as a `torch.ops.*` custom op via a `*_ops.py`
  companion so it drops into the graph without a graph break.
- **Manual TP.** Weights are sharded per-core by hand (colwise for q/k/v/gate/up
  and experts, rowwise for output/down) with functional all-reduces at known
  boundaries. KV heads (2) don't divide TP=4 → each KV head is replicated across
  `world_size // 2` cores.
- **CPU oracle before device.** Validate kernel and routing math on a CPU reference
  first (`kernels/tests/test_moe_oracle_cpu.py` proves the masked-dense MoE ≡
  canonical `Qwen3MoeSparseMoeBlock` to ~3e-9). Cheap; catches most bugs before any
  device compile.

## Environment

- **You need the PyTorch-Native DLC image**, not a host XLA venv. Host
  `torch-neuronx` venvs on the DLAMI import `torch_xla` (the XLA lowering path);
  Native adds a `neuron` device via PrivateUse1 and is only in the Native DLC. The
  harness reads weights with a dependency-free safetensors parser (`st_reader.py`),
  so it does not need `transformers`/`safetensors` in the image.
- **⚠️ Runtime/driver version match.** The Neuron runtime bundled in the DLC image
  must be compatible with the host's kernel driver (`aws-neuronx-dkms`). A mismatch
  fails `nrt_init()` even single-core with `ucode_ll_create ... error 6` /
  `Copy from buffer to memory failed`. If the host driver is *older* than the
  image's runtime, bind-mount the host runtime over the image's:
  `-v /opt/aws/neuron:/opt/aws/neuron:ro -e LD_LIBRARY_PATH=/opt/aws/neuron/lib`.
  Diagnose by checking whether the host's own stack initializes the device (a tiny
  XLA `xm.xla_device()` tensor op) — if that works, the device is fine and it's
  purely a runtime-in-image mismatch.
- **Weights packaging.** This code's loader is validated against a 14-shard /
  1811-tensor packing of the checkpoint. The current HF repo may be re-sharded
  (e.g. 26 shards / 1169 tensors, with fused expert `gate_up`); the parameters are
  the same model but `st_reader.py` / `model_dims.py` expect the packing they were
  written for. If loading a differently-packed copy, update the reader accordingly
  (dims/routing/RoPE are unchanged — no perf difference).
- **CTE GQA needs a newer nkilib than the current DLC bundle.** The bundled
  `attention_cte` rejects head-dim 256. The validated source is nki-library commit
  `1ee625782cb1bf91b40bccab741a82c726445080`, exposed with
  `PYTHONPATH=<nki-library>/src/nkilib_src`. Any replacement must support
  head-dim 256, prefix K/V, and runtime `prior_used_len`.
- **Preserve complete compiler cache roots.** Native prefill compile products land
  under the host directory mounted as container `/tmp`, notably `hlo_cache`,
  `neff_cache`, and NKI compiler subtrees. Use `deploy/cache/push.sh` and
  `pull.sh` to stage/restore the full tree through S3; a bare `.neff` is not a
  reusable cache entry. Restore before starting the container, keep the `/tmp`
  mount and image/compiler flags/graph/shapes/TP/LNC topology identical, then use
  `inspect.sh <cache-dir> <run-log>` plus process monitoring to diagnose misses.
  In this Native stack the completed NEFFs are under `neff_cache`, not just the
  configured `NEURON_COMPILE_CACHE_URL`; a log can lack explicit hit markers, so
  no active `neuronx-cc`/`walrus_driver` work during the matching run is required
  corroboration.

## Run recipes (validated)

Decode benchmark, full 40 layers, TP=4, recommended flags:
```
DN_NKI=1 GQATAIL=1 \
  torchrun --nproc-per-node=4 static_decode_35b.py \
    --model-path <weights> --max-seq-len <S> --num-layers 40 \
    --graph-splits 1 --batch-size <BS> --bench --bench-iters 30
```
Coherence check (real prompt, greedy): add `--num-tokens 16 --prompt-ids
"760,6511,314,9338,369"` ("The capital of France is") and drop `--skip-prefill`;
expect it to echo the prompt back (the documented greedy loop).

CPU correctness (no device): `python3 kernels/tests/test_moe_oracle_cpu.py --tokens 8`.

Compiled long-context prefill, N=20000, BS=1:
```
PYTHONPATH=<nki-library>/src/nkilib_src \
MOE_CTE=1 MOE_CTE_NKI_PACK=1 GQA_CTE_PREFILL=1 GQA_DYNAMIC_ROPE_KV=1 \
DN_CHUNK_NKI=1 CHUNK_SIZE=16 DN_NKI=1 GQATAIL=1 \
  torchrun --nproc-per-node=4 static_decode_35b.py \
    --model-path <weights> --max-seq-len 20480 --num-layers 40 \
    --prefill-bench 20000 --bucket-chunk 1024 --bucket-compile 1 \
    --prefill-splits 4 --skip-compile
```

### Flags that matter

| Flag | Effect |
|---|---|
| `DN_NKI=1` | DeltaNet NKI kernel — **required past ~20 layers** (see gotcha) |
| `GQATAIL=1` | Fused GQA attention-tail kernel — also collapses long-context compile |
| `MOE_SPARSE=1` | True-sparse MoE dispatch — ~2× at BS=1; do NOT use at BS≥16 (see gotcha) |
| `DNBATCHED_V2=1` | DMA-coalesced batched DeltaNet decode |
| `MOE_FP8=1` | FP8 experts — memory lever only (see FP8 gotcha) |
| `GQA_FLASH_PREFILL=1`, `DN_CHUNK_NKI=1` | Prefill kernels (see prefill recipe) |
| `GQA_CTE_PREFILL=1`, `GQA_DYNAMIC_ROPE_KV=1` | Prefix-aware compiled GQA prefill |
| `MOE_CTE_NKI_PACK=1` | Fuse stable NKI route packing into the CTE MoE call |

## Hard-won gotchas (each cost real time)

- **Pure-torch DeltaNet decode does not compile past ~20 layers.** neuronx-cc trips
  a `PGTiling` assertion on the recurrent-state einsum. `DN_NKI=1` (opaque kernel)
  cures it — the full 40 layers then compile. Not TP, not MoE (both ruled out).
- **Long-context cold-compile: use `GQATAIL=1`.** A naïve 20k decode graph took
  ~2.7 hours to compile because pure-torch GQA attention is exposed to the tiler.
  `GQATAIL=1` (opaque attention-tail kernel) collapses it to ~10 min. Memory fits
  at ~19.1 GB/core at 20k regardless (KV is small; weights dominate).
- **Long-context batch is HBM-bound, not throughput-bound.** Fixed ~19.1 GB/core of
  weights leaves little of the 24 GB/core for KV cache. At seq=10k the batch ceiling
  is BS=8 (BS=16 OOMs at NEFF load: `NRT_RESOURCE: Failed to allocate resource`); at
  seq=20k only BS=1 fits. Short context (seq=256) scales to BS=32. So the lever to
  raise the long-context batch ceiling is *reducing weight bytes* (FP8), not tuning.
- **FP8 is a memory/capacity lever, NOT a decode-latency win.** It halves expert
  weights (~19→11 GB/core) and is CPU-coherent, but every latency attempt regressed
  (BS=1 grouped-matvec ~2.2× slower; a fused-MoE FP8 path hit a dtype/compile wall).
  Cause: at BS=1 FP8 turns wide fused GEMMs into many tiny per-expert matvecs and
  dispatch overhead dwarfs the bandwidth saved. Trn2's `nc_matmul` wants *legacy*
  `float8_e4m3`; torch only has `e4m3fn` — a real plumbing gap. Treat FP8 as the way
  to unlock higher batch at long context (capacity), not to speed up a single stream.
- **Sparse MoE does not scale to BS≥16.** `MOE_SPARSE=1` gathers `T·K` experts;
  the gathered graph explodes the *host* compiler memory at BS≥16 (F137 / OOM /
  host wedge). Use masked-dense MoE for BS≥16; sparse only wins BS≤4.
- **Historical prefill baseline: eager sequence bucketing.**
  `--bucket-chunk 2048 --bucket-compile 0` with `GQA_FLASH_PREFILL=1 DN_CHUNK_NKI=1`
  plus pad-token masking gives coherent 20k prefill with no OOM and ~9 min compile
  (~259 prompt tok/s). Two kernel bugs had to be fixed for correctness: (1) pad
  tokens (id 0) corrupt the carried DeltaNet state — zero their `beta`/`g`; (2) the
  chunked-DeltaNet L2-norm used `x·rsqrt(‖x‖²+eps)` but must match
  `x/max(‖x‖,eps)` for near-zero rows. Random-input kernel tests missed both — test
  on real post-conv/proj distributions.
- **Use `MOE_CTE=1`, not `MOE_NKILIB=1`, for prefill.** `moe_tkg` maps tokens
  to the NKI partition dimension and requires 128-token calls. The
  `nkilib.core.moe.moe_cte` adapter in this directory routes long-token inputs
  into expert blocks and keeps the MoE opaque to Dynamo. At S=2048, TP=4,
  B=512 it measured 71-73 ms for one eager layer and 266.5 ms for four eager
  layers, versus 193.5 ms and 793.3 ms for pure Torch. Four compiled CTE layers
  measured 198.2 ms / 10,334 tok/s and loaded without the old descriptor failure.
- **DeltaNet prefill currently needs `CHUNK_SIZE=16` at 20k context.** C=64 passes random
  tests but its conditionally stable doubling inverse returns an inaccurate
  recurrent state on real layer-0 inputs (state max error 1.69) and produces
  NaNs by layer 5 at S=2048. C=32 made all 40 layers finite at S=2048, but a
  full S=20000 run with 512-token buckets deterministically became non-finite
  at bucket 21, layer 18. The layer-17 hidden and all incoming recurrent states
  were finite; layer 18 emitted non-finite hidden values while its returned
  state stayed finite. This reproduced in eager mode, so it is not compiler
  drift. C=16 remained finite through all 40 layers and all 40 buckets in two
  S=20000 eager passes, measuring 71391.7 ms / 280.1 tok/s with a stable
  fingerprint. The matched compiled run measured 20886.0 ms / 957.6 tok/s and
  retained the eager top-5 set. Do not increase chunk size without a real-weight,
  full-context finite/coherence test.
- **Old C32 can be catastrophically wrong while still finite.** Immutable
  layer-18/bucket-21 replay agrees with the CPU reference near 1e-6 on TP ranks
  0/1/3, but rank 2 has output error 2.96e34 and state error 3.06e36.
  RMSNorm masked the raw output scale, so final finiteness alone is not a
  sufficient gate. A block-factorized C32 inverse reduced a CPU stability test
  from roughly 1.0 error to 1.2e-4, but its first Trn2 compile hit an SBUF
  allocation/scheduling conflict involving a `32x128` tile. Resolve allocation,
  then require all-rank capture replay before a full 20k benchmark.
- **Split compiled prefill into coarse 10-layer NEFFs.** A monolithic 20-layer
  CTE graph generated 5,440,131 instructions and failed the compiler's 5,000,000
  limit. `--prefill-splits 2` compiled and loaded two 10-layer segments; use
  `--prefill-splits 4` for 40 layers. The segment linkers used about 10-12 GiB
  each, far below the old pure-Torch-MoE linker blow-up.
  Full 40-layer S=2048 measured 2017.5 ms / 1015 tok/s, versus 2702.9 ms /
  757.7 tok/s eager CTE. The compiled and eager top-5 sets were identical.
- **Compiled/eager fingerprint drift comes from fused bf16 round points.** The
  DeltaNet NKI call itself is bit-exact eager versus compiled. Separate qkv
  projection and conv graphs are also exact, but the combined compiled graph
  carries f32 accumulator values through `.float()` consumers instead of
  materializing eager bf16 intermediates. This changes q/k by about 1.5e-2, v
  by 3.1e-2, and g by 5.0e-2. Treat full-model token coherence as the acceptance
  test; do not attribute this drift to the CTE router or DeltaNet custom call.
- **Real-prompt C=32 coherence is validated in eager mode.** With
  `"The capital of France is"` (IDs `760,6511,314,9338,369`), full 40-layer
  eager CTE prefill generated IDs `11751,369,264,3177,314`, corresponding to
  a coherent continuation about Paris. A compiled short-prompt check is not a
  cache hit: `_dn_valid_len=5` specializes a different pad-mask graph and would
  require another full compile.
- **Use `GQA_DYNAMIC_ROPE_KV=1` to reuse compiled prefill across offsets.** Plain
  tensor indexing for dynamic RoPE/KV positions failed in NRT, while a Python
  `q_base` specialized one graph per bucket. The fused `gqa_rope_kv_dynamic` NKI
  op instead consumes a runtime int32 offset, applies RoPE, and performs aliased
  KV writes. A four-layer S=4096 run compiled once and processed both 2048-token
  offsets, then measured 390.5 ms / 10,488.9 tok/s with identical warm/timed
  fingerprints. At full 40-layer depth with four 10-layer segments, the same
  two-bucket test compiled in 39.3 minutes and measured 3646.2 ms /
  1123.4 tok/s; no second-offset compilation occurred. Its S=2048 eager output
  was exactly equal to the old static indexing path. Keep the static path as a
  control, but use the dynamic path for compiled long-context prefill.
- **Pass DeltaNet valid length as runtime metadata too.** A Python
  `_dn_valid_len` would specialize a second graph set for the padded final
  bucket at S=20000. With the dynamic GQA path enabled, the harness now passes
  runtime int32 `q_base` and `valid_len` tensors. At S=2304, the dynamic eager
  path was exactly equal to the static eager pad mask, and one four-layer cold
  compile served both a 2048-token full bucket and a 256-token partial bucket.
  The timed compiled pass measured 391.5 ms / 5884.6 tok/s and retained the
  eager top-5 set.
- **Prefill compilation has two independent resource ceilings.** Four concurrent
  TP linkers exhausted 128 GB host RAM for 20/40-layer pure-torch MoE graphs.
  NVMe-backed `/tmp` plus 128 GB swap allowed a four-layer nkilib graph to compile,
  peaking around 31-34 GB RSS per linker and 26 GB swap. Separately, pure-torch
  four-layer NEFF load hit the hard DMA vring ceiling at 16,777,200 descriptors.
  Disabling compilation around MoE reduced linker memory but produced 81 resident
  NEFF fragments and hit the same cumulative descriptor ceiling.
- **S=20480 with 2048-token compiled buckets still exceeds the descriptor
  ceiling.** Dynamic offset/valid-length metadata removes graph specialization,
  but the long-KV flash-attention expansion makes each 10-layer segment much
  larger than at max_seq=4096. The first segment loaded; loading the second hit
  the cumulative `16,777,200` vring descriptor limit. Linkers reached 25-32 GiB
  RSS/rank and used about 50 GiB swap. More host RAM cannot fix this runtime
  limit; reduce query bucket size or reduce descriptors inside the GQA kernel.
- **A 512-token bucket clears the long-context descriptor ceiling.** The full
  40-layer C=32 graph compiled and loaded at max_seq=20480, then measured
  16014.0 ms / 1248.9 tok/s at N=20000. That run was numerically invalid due
  to the independent C=32 DeltaNet failure above. Linker RSS was only about
  3 GiB/rank. The corrected C=16 run measured 20886.0 ms / 957.6 tok/s with a
  finite, repeatable fingerprint and the same top-5 as eager C=16. This is the
  validated local-flash control: 3.70x the original 259 tok/s baseline.
- **Use nkilib CTE attention instead of fixed-KMAX flash at 20k.** The local
  flash kernel always scans all 20,480 KV rows and measured 11.66-11.69 ms per
  production-shape GQA call, independent of the used prefix. Prefix-aware
  `attention_cte` measured 0.77-0.81 ms at prior lengths 0, 10,240, and 19,968
  (14.5-15.1x faster), with cosine >=0.999975 against local flash. A matched
  four-layer eager control improved 694.2 to 651.6 ms and retained the same
  top-5. The full 40-layer compiled C16 run measured 17854.9 ms / 1120.1 tok/s,
  1.17x over the 957.6 tok/s flash control and 4.32x over the original baseline.
  Warm/timed fingerprints were identical and finite. The real prompt
  `"The capital of France is"` generated
  `[11751,369,264,3177,314,1880]`; its first five IDs exactly match the prior
  validated continuation. Use `GQA_CTE_PREFILL=1` instead of
  `GQA_FLASH_PREFILL=1`; the flags are mutually exclusive.
- **CTE makes bucket 2048 loadable, but bucket 1024 is faster.** Replacing the
  expanded local flash kernel removed the descriptor ceiling: all four
  10-layer bucket-2048 segments compiled and loaded at max_seq=20480. The
  result was finite/repeatable but measured 20631.9 ms / 969.4 tok/s, slower
  than bucket 512 at 17854.9 ms / 1120.1 tok/s. Cold warmup took 3471 seconds
  and linkers reached roughly 15-18 GiB/rank. Isolated CTE attention remained
  efficient at active sizes 512/1024/2048 (0.79/1.28/2.11 ms), so the
  regression comes from the larger surrounding compiled graph, not attention.
  The matched bucket-1024 run measured 17374.1 ms / 1151.1 tok/s, versus
  17854.9 ms / 1120.1 tok/s at bucket 512. Use bucket 1024; do not assume still
  fewer buckets are faster.
- **The Torch-routed batched prefill baseline does not improve throughput.**
  Homogeneous BS=2
  with independent DeltaNet, convolution, and KV state passed a four-layer
  partial-bucket isolation test against two independent BS=1 executions
  (cosine >=0.999936 for logits and every carried state). Full S=20000 loaded
  and returned finite state, but measured 41069.0 ms / 974.0 aggregate tok/s,
  versus 17374.1 ms / 1151.1 tok/s at BS=1. Batch latency grew 2.36x for 2x
  tokens, so BS=4 was gated off. The C16 DeltaNet kernel is explicitly
  sequential over `B*V_HEADS`; an isolated 1024-token call scaled
  11.57-11.93 to 22.58-22.92 ms. Isolated CTE expert compute stayed near
  6.7-7.0 ms from 1024 to 2048 flattened tokens.
- **The BS=2 regression is the MoE route prefix scan, then DeltaNet.** A matched
  full-segment trace measured 188.9 ms at BS=1 and 471.0 ms at BS=2. The ten
  `pack_local_routes()` scans at `moe_cte_adapter.py:50` grew from 61.3 to
  258.3 ms and from 24.9 to 105.7 GB of attributed HBM traffic. The compiler
  lowers `group_hot.cumsum(dim=0)` to HLO `reduce-window` using TensorE
  MATMUL/LDWEIGHTS. DeltaNet grew from 82.4 to 165.6 ms; together these explain
  99% of the matched segment increase. The other segment shape measured
  485.7 ms, with 263.7 ms in the same scan and 145.6 ms in its seven DeltaNet
  calls. DMA is the slowest engine (261-281 ms active per segment);
  collectives are only about 3 ms. Replace/fuse the route scan before any BS=4
  attempt, and do not pursue a larger batch until BS=2 exceeds the BS=1
  aggregate rate.
- **The fused NKI route packer is the validated BS=1 optimum.**
  `MOE_CTE_NKI_PACK=1` keeps route metadata private to the existing CTE custom
  call and uses stable four-lane `nonzero_with_count` compaction plus direct
  DMA. It passed 96 exact metadata cases, distributed fused/precomputed CTE
  equivalence, and four-layer BS=2 isolation. Standalone 8,192/16,384-route
  packing measured 2.420/3.777 ms (1.56x). Three hot BS=1 S=20000 runs measured
  13.4834/13.4967/13.4878 seconds: **13.4878 s / 1482.8 tok/s median**. A
  matched segment fell from 188.89 to 122.26 ms, total HBM from 31.96 to
  3.30 GB, and route HBM from 24.94 GB to 45.9 MB, with no `reduce-window`.
  Keep the Torch fallback; default-on and BS=4 decisions wait for full BS=2
  throughput and profile results.
- **DeltaNet C16 is scheduling-bound, not HBM-bound.** At S=512 it measured
  5.284 ms with model MFU 0.145%, instruction MFU 0.478%, and MBU 0.497%.
  ScalarE/VectorE/TensorE occupancy was 74.3/73.2/65.7%, DMA occupancy 15.8%,
  and transposes represented 9.34% of hardware FLOPs. Optimize overlap,
  transpose placement, and live ranges before pursuing HBM bandwidth changes.
- **CTE block-size/static-loop experiments did not help.** B=256 with a reserved
  static expert block measured 74.7 ms for one eager layer; B=512 static measured
  73.4 ms, both slightly slower than the original dynamic B=512 path (~72 ms).
  Keep `MOE_CTE_BLOCK=512` and dynamic routing unless a profile shows a new reason
  to revisit it.
- **Fresh multi-segment compiles can stall on local-cache locks.** The first
  40-layer/split-4 run spent repeated 1200-second intervals waiting on stale
  `/tmp/local_cache/locks/*.lock` files before breaking them. Total warmup was
  3137 seconds, dominated by lock waits rather than compilation. Preserve the
  completed NVMe `/tmp` cache and clean stale locks before a deliberate cold run.
- **Local detailed record:** when present, see
  `experiments/compiled_prefill_2026-07-14.md`. The directory is intentionally
  gitignored because it contains operational instance paths and log locations.
- **Measure synced TPOT, not enqueue time.** The Neuron eager backend dispatches
  async; a timing loop without `torch.neuron` synchronize measures enqueue, not
  execution (this produced bogus ~1 ms "TPOT" figures historically that were really
  ~35 ms). Always synchronize before timing.
- **Device-smoke at the real `max_seq`.** Some tiling limits only appear at large
  seq; a 512-token smoke can pass while 2048/20k fails.

## Device / ops hygiene

- **Free the device before a new run.** A live container holds all 4 cores →
  "Logical Neuron Core(s) not available". Kill stray containers first.
- **Cap per-config compile time** and watch host RAM on high-BS or long-seq
  compiles — a runaway neuronx-cc compile can exhaust host memory and wedge the box
  (recover by reboot). Instance-store NVMe is typically not in fstab → remount after
  any reboot (data survives reboot, lost only on stop/terminate).

## Correctness reference (oracle)

The validated NxDI implementation (`aws-neuron/neuronx-distributed-inference`
PR #60, torch-xla) is the correctness oracle: 100% token-match vs CPU, ~18.4 ms/tok
BS=1 on trn2.3xlarge TP=4. Its MoE uses a non-portable NxDI library module, which is
why this harness carries its own MoE kernels + CPU oracle. Use it to validate token
output, not as an architecture to copy.

## Reuse map

| Piece | Source |
|---|---|
| DeltaNet / GQA / RoPE / RMSNorm / TP / compile harness | the sibling `qwen3.6-27b` (dense) — retune head counts |
| MoE (masked-dense grouped-bmm, expert-parallel) | `kernels/` here + `test_moe_oracle_cpu.py` |
| Correctness oracle | this package's CPU oracle + NxDI PR #60 token match |
