# Qwen3.6-35B-A3B (MoE) on Trainium2 (PyTorch Native)

A PyTorch-Native inference implementation of the sparse-MoE **Qwen3.6-35B-A3B**
model (~3B active parameters of 35B) on a single Trainium2 device
(`trn2.3xlarge`, TP=4, LNC=2). It shares the DeltaNet + GQA backbone structure of
the [dense 27B](../qwen3.6-27b) but replaces the dense MLP with a 256-expert
top-8 mixture of experts, and targets a fixed long-context (20,000-token) regime.

> **Naming.** The model is published as
> [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) on Hugging
> Face, but its architecture class is `Qwen3_5MoeForConditionalGeneration`
> (`model_type: qwen3_5_moe`) — "3.5" names the architecture family, "3.6" names
> the release. The two are the same architecture, so this code runs on the HF
> checkpoint unchanged.

The whole 40-layer decode model compiles to a single NEFF via
`torch.compile(fullgraph=True, backend="neuron")`. The validated long-context
prefill path uses four coarse 10-layer regions. With compiler 2.25 and DGE
enabled, a single 40-layer prefill region also cross-compiles. The fastest
prefill runs at **TP=8/LNC=1** once both replicated `[V, H]` vocab tensors are
load-time sharded to fit the ~12 GB/rank budget (3,456.8 tok/s, +24.5%; see the
prefill throughput table and the TP=8/LNC=1 resolution below). On the **beta-4
container** that became **3,645.0 tok/s** (+5.4% over 3,456.8, same source, same page
size 256 — a clean compiler-only win, verified numerically bit-identical). On beta-4 the
in-place rope-KV write, now **correct** (one cache buffer per GQA layer), reached
**4,002.1 tok/s** (+9.8% over 3,645.0). The current best is **4,097.9 tok/s** on the
**beta-5 container** (built 2026-08-05; +2.4% over 4,002.1, identical
source, bit-identical fingerprint, reproduced at 4,095.2) — another clean compiler-only
win. Decode's current best is **681.3 tok/s** at BS=128 (beta-5, O1, `DN_PERLAYER_STATE`;
+7.8% over beta-4's 632.0; 582.3 tok/s at seq=1024); see the beta-5 section below.

The previously headlined **3,889.4 tok/s** is **withdrawn** — that build of the
in-place rope-KV rewrite **dropped ~60% of its KV-cache writes**, so part of its gain
was work not done. The mechanism (only the first in-place mutation of a buffer per
traced graph survives) and the fix are below; the fixed path is both faster than the
defective one and populates the cache 100%, at the exact reference logits fingerprint.

## Architecture

```
40 layers = [DeltaNet × 3, GQA × 1] × 10   (full-attention every 4th layer)
hidden 2048, vocab 248320, RMSNorm eps 1e-6, RoPE partial-64 @ theta 1e7

DeltaNet (30 layers): 16 K-heads, 32 V-heads, k/v dim 128
  in_proj_qkv [8192,2048] = q(2048)||k(2048)||v(4096); in_proj_z [4096,2048]
  in_proj_a/b [32,2048]; A_log/dt_bias [32]; depthwise conv1d [8192,1,4]

GQA (10 layers): 16 Q-heads, 2 KV-heads, head_dim 256, sigmoid output gate
  q_proj [8192,2048] (query||gate); k/v_proj [512,2048]; o_proj [2048,4096]
  per-head q_norm / k_norm [256]

MoE (all 40 layers): 256 experts, top-8, moe_inter 512, + shared expert
  experts.gate_up_proj [256,1024,2048] (gate||up fused); experts.down_proj [256,2048,512]
  router gate.weight [256,2048]
  shared_expert.{gate,up} [512,2048], down [2048,512]; shared_expert_gate [1,2048]
```

Routing (canonical Qwen3-MoE, validated):

```
logits = x @ router.T;  w = softmax(logits, float);  w, sel = topk(w, 8)
w /= w.sum(-1)                                    # norm_topk_prob = True
routed = sum_j w[:,j] * expert_sel[:,j](x)
out = routed + sigmoid(x @ shared_gate.T) * SwiGLU_shared(x)
```

Verified architecture constants and the `tp_dims(world_size)` sharding plan live in
`model_dims.py`. KV heads (2) don't divide TP=4, so each KV head is replicated
across `world_size // 2` cores.

## Layout

```
static_decode_35b.py    the static decode/prefill forward (all 40 layers) + compile
                        harness + manual TP sharding + benchmark entry point
model_dims.py           verified architecture constants and TP sharding dims
deltanet_decode.py      DeltaNet recurrent decode step
chunked_prefill.py      chunked DeltaNet prefill reference
st_reader.py            safetensors weight reader / sharder
kernels/                NKI kernels + torch.ops registrations (*_ops.py)
  deltanet_full_batched_35b*   batched DeltaNet decode (8 v-heads/core)
  deltanet_chunked_prefill_35b*  chunked-prefill DeltaNet
  gqa_tail_35b*, gqa_flash_prefill_35b*  fused GQA decode tail + local flash prefill
  gqa_cte_35b*, gqa_rope_kv_35b*  nkilib CTE attention + dynamic RoPE/KV update
  fp8_group_matvec*    FP8 grouped matvec for MoE experts
  tests/               repeatable CPU/device checks with pass/fail assertions
debug/                  reusable isolation, capture, and numerical diagnostics
deploy/profile/         device-profiling capture + neuron-explorer UI scripts
deploy/compile_prefill_trn2.sh
                        reproducible Trn1-to-Trn2 prefill compile driver
deploy/cross_compile/   scoped Trn2 cache-key override and validation helpers
experiments/            ignored local journals, resume notes, and benchmark tools
```

Keep active runtime implementations in `kernels/`. A script belongs in
`kernels/tests/` when it is a repeatable regression with an explicit pass/fail
result; diagnostics that print intermediate evidence for manual interpretation
belong in `debug/`. The ignored `experiments/` directory is for machine-specific
records and temporary investigation tooling. Superseded code should normally
remain in Git history rather than accumulating in a `legacy/` directory.

## De-risking: the MoE CPU oracle

The #1 risk — sparse MoE routing under `torch.compile` static shapes — is retired
before any device time by `kernels/tests/test_moe_oracle_cpu.py`. On the real
layer-0 weights, the **masked-dense grouped-bmm** formulation we run on Neuron is
numerically identical (to ~3e-9) to a HF-sparse reference, to the canonical
`transformers` `Qwen3MoeSparseMoeBlock`, and to the expert-parallel sharded form.

```bash
python3 kernels/tests/test_moe_oracle_cpu.py --tokens 8
python3 kernels/tests/test_moe_sparse_eq.py
python3 kernels/tests/test_moe_decode_tp_cpu.py
```

## Running

```bash
# Full 40-layer decode benchmark, BS=1, recommended decode flags
DN_NKI=1 MOE_SPARSE=1 MOE_DECODE_TP=1 GQATAIL=1 DNBATCHED_V2=1 \
  torchrun --nproc-per-node=4 static_decode_35b.py \
    --num-layers 40 --max-seq-len 2048 --batch-size 1 --bench

# Validated compiled prefill, N=20000, BS=1
PYTHONPATH=<nki-library>/src/nkilib_src \
MOE_CTE=1 GQA_CTE_PREFILL=1 GQA_DYNAMIC_ROPE_KV=1 \
DN_CHUNK_NKI=1 CHUNK_SIZE=16 DN_NKI=1 GQATAIL=1 \
  torchrun --nproc-per-node=4 static_decode_35b.py \
    --num-layers 40 --max-seq-len 20480 --prefill-bench 20000 \
    --bucket-chunk 1024 --bucket-compile 1 --prefill-splits 4 --skip-compile

# CPU correctness (no device)
python3 static_decode_35b.py --cpu --num-layers 40
```

Set `QWEN35_MODEL_PATH` (or `--model-path`) to the weights directory.

The scripts under `deploy/profile/` load the repository's ignored `.env`
automatically. Configure `QWEN35_NATIVE_IMAGE`, `QWEN35_MODEL_DIR`, and
`QWEN35_PROFILE_ROOT`; source mounts are derived from the scripts' location.

### Fused block-W8 high-batch decode

`MOE_FUSED_W8=fp8|int8` enables the experimental all-expert NKI path for
BS=32/64/128/256 full-graph decode. It reads only routed-expert gate/up/down
weights and BF16 `weight_scale_inv` tensors from the official FP8 checkpoint;
router, shared expert, attention, DeltaNet, embeddings, and the LM head still
come from the BF16 checkpoint. Set the second checkpoint with
`QWEN35_FP8_MODEL_PATH` or `--expert-model-path`.

The loader reads safetensors E4M3FN bytes without requiring a PyTorch FP8
dtype. `MOE_FUSED_W8_FP8_IMPL=row` uses the nkilib row-scaled scheduler,
`dual` preserves the official weights exactly as two legacy-E4M3 planes, and
`block_pow2` maps each 128x128 source block to one native legacy-E4M3 plane by
an exact scale exponent shift. `block_pow2_coalesced` retains those exact
128x128 scales but uses 128x512 native-E4M3 slabs, rotating buffers, and
BS-dependent TensorE column packing; it supports BS=32/64/128. Direct E4M3FN
matmul is not supported by this Trn2 toolchain and bitcasting codes
`0x78..0x7e` is invalid. The `int8` fallback uses symmetric signed INT8 per
source block. The path is mutually exclusive with `MOE_SPARSE`,
`MOE_DECODE_TP`, `MOE_CTE`, `MOE_CTE_NKI_PACK`, `MOE_NKILIB`, and the older
`MOE_FP8` path. It also requires `DECODE_FULLGRAPH=1`, `--skip-prefill`, and
one static compile per batch shape.

Run the host-side conversion/routing suite first:

```bash
pytest -q kernels/tests/test_moe_w8_cpu.py
```

Then establish a two-layer BS=32 device reference and candidate. These commands
assume TP=8/LNC=1:

```bash
export QWEN35_MODEL_PATH=<bf16-weights>
export QWEN35_FP8_MODEL_PATH=<official-fp8-weights>
export DN_NKI=1 DNBATCHED_V2=1 DN_DIRECT_STATE_OUT=1
export GQATAIL=1 GQA_STATEFUL_KV=1
export DN_K_HEADS=2 DN_V_HEADS=4 GQA_Q_HEADS=2
export DECODE_FULLGRAPH=1 DECODE_SHARDED_LM_HEAD=1
export NEURON_LOGICAL_NC_CONFIG=1
export NEURON_CC_FLAGS="--target trn2 --lnc 1"

MOE_OFFICIAL_FP8_REFERENCE=1 torchrun --nproc-per-node=8 \
  kernels/tests/test_decode_fullgraph_device.py \
  --mode sharded --output-dir /tmp/q35-w8-reference \
  --model-path "$QWEN35_MODEL_PATH" \
  --expert-model-path "$QWEN35_FP8_MODEL_PATH" \
  --num-layers 2 --batch-size 32

MOE_FUSED_W8=fp8 \
MOE_FUSED_W8_FP8_IMPL=block_pow2_coalesced \
torchrun --nproc-per-node=8 \
  kernels/tests/test_decode_fullgraph_device.py \
  --mode sharded --output-dir /tmp/q35-w8-fp8 \
  --model-path "$QWEN35_MODEL_PATH" \
  --expert-model-path "$QWEN35_FP8_MODEL_PATH" \
  --num-layers 2 --batch-size 32

python3 kernels/tests/test_decode_fullgraph_device.py --world-size 8 \
  --compare /tmp/q35-w8-reference /tmp/q35-w8-fp8 --quantized-compare
```

The isolated custom-op check compares official real weights against the
quantized CPU reference. Run a one-expert smoke before the production 32 local
experts:

```bash
python3 kernels/tests/test_moe_fused_w8_device.py \
  --mode fp8 --batch-sizes 32 \
  --expert-model-path "$QWEN35_FP8_MODEL_PATH" --expert-count 32
```

On `trn2.3xlarge`, all 16 CPU tests passed and the real-weight isolated
E4M3FN kernel passed with both one and 32 local experts. Two-layer exact-FP8
full-graph correctness passed on all ranks: logits cosine was
0.999775-0.999834, relative L2 was 1.37-1.79%, all 32 greedy IDs matched, and
DeltaNet and convolution state relative errors stayed at or below 0.251%.
Symmetric INT8 failed these numerical gates.

The initial 40-layer result was invalid because `DN_DIRECT_STATE_OUT=1`
discarded the state tensors returned by the compiled graph and relied on parent
slice mutation. Assigning the returned DeltaNet and convolution states
explicitly made four-layer direct and non-direct runs bit-identical. The fixed
40-layer exact-FP8 comparison had step-0 logits cosine 0.99921-0.99952 and
relative L2 2.80-3.76%, with state relative errors of 0.67-1.14%. One of 32
greedy IDs differed on a 0.0625 reference margin that became an exact tie.

Teacher-forcing step 1 with the candidate state and reference token separated
state recurrence from token-choice divergence. Logits improved to cosine
0.99960-0.99972 and relative L2 1.96-2.52%; state relative errors were
0.54-1.73%. One row again differed only at a 0.125 reference margin that became
a tie. Continuous numerical correctness therefore passes, while exact greedy
IDs remain sensitive to BF16 near-ties.

The exact path still failed the throughput gate:

| Layers | BS | Path | TPOT (ms) | aggregate tok/s | module GB/rank |
|--:|--:|--|--:|--:|--:|
| 2 | 32 | BF16 hybrid control | **7.93** | **4,033** | 2.47 |
| 2 | 32 | Exact E4M3FN, tile-local BF16 decode | 32.26 | 992 | 2.27 |
| 40 | 32 | BF16 hybrid control | **99.80** | **320.6** | 10.80 |
| 40 | 32 | Exact E4M3FN, tile-local BF16 decode | 618.72 | 51.7 | 6.84 |

The fixed 40-layer path is 6.20x slower than BF16 at BS=32 despite saving
3.96 GiB/rank. BS=64 and BS=128 were not attempted because the requested
progression required BS=32 to pass both correctness and throughput. Earlier
pre-fix executions at those shapes are not correctness-qualified results.

A native TensorE experiment requantized the official blocks to legacy E4M3.
Its isolated one-expert kernel matched its quantized CPU reference (cosine
0.9999948, normalized RMSE 0.326%), but the requantized result versus exact
official FP8 had cosine 0.9996023 and normalized RMSE 7.16%. At four layers,
step-0 logits relative L2 was 4.14-4.27% with three of 32 greedy IDs different;
it was rejected. Symmetric INT8 remains rejected by the same numerical gates.

The follow-up `block_pow2` experiment avoids arbitrary requantization. Every
real 128x128 block contained an extended E4M3FN code, so its payload was
divided by two and its BF16 scale doubled. Across all 256 experts in layers 0,
1, and 39, cosine was 1.000000000, normalized RMSE was 0.0000363%, 99.99% of
values were exact, and no value clipped. The production 32-local-expert NKI
check measured cosine 0.9999943 and NRMSE 0.33595% against its CPU reference,
and cosine 0.9999955 and NRMSE 0.29175% against exact official-FP8 output.

Full-graph correctness passed at two and four layers with all 32 greedy IDs
matching:

| Layers | Worst logits cosine | Worst logits rel. L2 | Worst state rel. L2 | TPOT | Aggregate tok/s |
|--:|--:|--:|--:|--:|--:|
| 2 | 0.999770 | 1.81% | 0.28% | 22.70 ms | 1,409.9 |
| 4 | 0.999721 | 2.08% | 1.29% | 42.29 ms | 756.7 |

The two-layer BF16 control remains 7.93 ms and 4,033 tok/s, so the native
block kernel is 2.86x slower and fails the throughput gate. A matched rank-0
profile explains the regression:

| Metric | BF16 | `block_pow2` |
|--|--:|--:|
| Device execution | 6.366 ms | 23.761 ms |
| HBM reads | 589.2 MB | 395.6 MB |
| DMA bytes | 622.2 MB | 428.9 MB |
| DMA transfers | 3,548 | 27,962 |
| DMA active time | 3.741 ms | 12.111 ms |
| GPSIMD active time | 2.223 ms | 17.676 ms |
| TensorE occupancy | 45.8% | 18.0% |

Explorer reported a DGE packet-count mismatch, so the HBM and DMA byte counts
are directional. The traffic reduction is real, but the manually scheduled
all-expert kernel fragments DMA and spends too much time in GPSIMD. The
40-layer compile was therefore not attempted. A viable next step would consume
the same block-power-of-two weights in nkilib's faster all-expert scheduler
while retaining 128x128 scales.

The separately selectable `block_pow2_coalesced` follow-up adapts the
nki-library `7a5b6f9` ring-buffer and column-packing schedule without modifying
nki-library. Gate/up weights are packed as `[E,H,2,I]`, down remains
`[E,I,H]`, and repeated BF16 scales are stored as `[E,128,N]` in projection,
contraction-block, output-block order. It loads each expert's scales once,
uses two rotating 128x512 weight/PSUM slots, and applies each 128-column scale
before contraction accumulation. Affinity is applied once when the expert
output scratch is added to the routed FP32 result.

A required CPU-only shared-scale experiment searched BF16 scales at factors
0.70 through 1.00 of `absmax/240` for every 128x512 slab. It was rejected:

| Experts | Weight cosine | Weight NRMSE | CPU MoE cosine | CPU MoE NRMSE |
|--:|--:|--:|--:|--:|
| 1 | 0.999729233 | 2.32694% | 0.999223446 | 3.94366% |
| 32 | 0.999732771 | 2.31183% | 0.999326253 | 3.67027% |

The CPU MoE missed both the 0.9995 cosine and 3.5% NRMSE gates, so no
`block_group512` production mode was added.

All 40 CPU tests pass. Official-weight isolated checks pass for 32 local
experts at BS=32/64/128 with kernel-reference NRMSE of 0.00252%, 0.00124%,
and 0.00277%. The final source snapshot is `18e91693f453`; its two-layer
BS=32 cache archive has SHA256
`5f11acc791a864ab57e68a38c2cb5e61dbb2250575ec0cb0cac6f2139d35ed6f`.
Against the official-FP8-dequantized hybrid reference, the two-layer run has
worst logits cosine 0.99991870, logits relative L2 0.47813%, state relative L2
0.24132%, and all 32 greedy IDs match. Three warmups plus 30 synchronized
iterations measured 10.67 ms TPOT and 2,998.0 aggregate tok/s.

The matched rank-0 Explorer result is:

| Metric | BF16 hybrid | `block_pow2` | `block_pow2_coalesced` |
|--|--:|--:|--:|
| Device execution | 6.366 ms | 23.761 ms | 9.563 ms |
| HBM reads | 589.2 MB | 395.6 MB | 395.6 MB |
| HBM writes | 29.5 MB | 29.8 MB | 29.8 MB |
| DMA bytes | 622.2 MB | 428.9 MB | 428.9 MB |
| DMA transfers | 3,548 | 27,962 | 5,531 |
| DMA active time | 3.741 ms | 12.111 ms | 3.892 ms |
| GPSIMD active time | 2.223 ms | 17.676 ms | 3.427 ms |
| TensorE occupancy | 45.8% | 18.0% | 25.48% |

Versus `block_pow2`, DMA transfers fell 80.22% and GPSIMD active time fell
80.61%. TPOT, accuracy, DMA, and GPSIMD gates pass, but TensorE occupancy
misses the required 30%. Explorer also reports a DGE packet-count mismatch, so
HBM and DMA byte figures remain directional. The gated 40-layer compile was
not run, and BS=64/128 full-graph progression is not qualified.

### Reusable compiler cache

The Native compiler stores the reusable prefill artifacts beneath the complete
host directory mounted as `/tmp` in the container. That directory contains
`hlo_cache`, `neff_cache`, and NKI compiler subtrees. Do not archive or restore
only a `.neff` file, and do not point the runtime at an arbitrary one.

Set these ignored `.env` values on the compile and reuse hosts:

```bash
export QWEN35_COMPILER_CACHE_DIR=/mnt/nvme/qwen35-prefill-cache
export QWEN35_COMPILER_CACHE_S3_URI=s3://YOUR-BUCKET/neuron-compile-cache/qwen35
```

After a successful cold compile, stage an immutable cache key:

```bash
deploy/cache/push.sh bs4-c16-s20000-direct512
```

On a future host, restore it before starting the Native container:

```bash
deploy/cache/pull.sh bs4-c16-s20000-direct512
deploy/cache/inspect.sh
```

Mount `QWEN35_COMPILER_CACHE_DIR` at `/tmp` inside the container exactly as in
the producing run. A persistent-cache hit requires the same captured graph,
input shapes, compiler/image version, compiler flags, Neuron generation, and
TP/LNC topology. The scripts refuse to mix cache contents by default; use
`--replace` only to discard an existing target or deliberately refresh a key.
The optional second argument to `inspect.sh` scans a run log for cache-hit,
cache-miss, and backend-compiler markers. Native logs are sometimes
inconclusive; corroborate a cache hit by confirming no active `neuronx-cc` or
`walrus_driver` process appears during the run.

### Trn1-to-Trn2 full-depth prefill compile

`deploy/compile_prefill_trn2.sh` uses a high-memory Trn1 host to compile the
prefill graph for Trn2. It defaults to TP=8/LNC=1, BS=2, N=20,000,
bucket=1024, 40 layers, optlevel 1, and a Trn2 target. The cache-key shim changes
only the Torch NeuronX persistent-cache identity; direct runtime queries still
report the physical Trn1 host. This makes the complete cache reusable on Trn2.

Configure `QWEN35_NATIVE_IMAGE`, `QWEN35_MODEL_DIR`, and `QWEN35_NKILIB_DIR` in
the ignored `.env`, then use a separate cache root for each graph shape:

```bash
# One compiled region containing all 40 layers.
deploy/compile_prefill_trn2.sh \
  --layers 40 --splits 1 --tp 8 --lnc 1 \
  --cache-platform-target trn2 --scratchpad-page-size-mb 64 \
  --cache-dir /mnt/nvme/qwen35-prefill-tp8-lnc1-s1

# Four compiled regions containing 10 layers each.
deploy/compile_prefill_trn2.sh \
  --layers 40 --splits 4 --tp 8 --lnc 1 \
  --cache-platform-target trn2 --scratchpad-page-size-mb 64 \
  --cache-dir /mnt/nvme/qwen35-prefill-tp8-lnc1-s4
```

A Trn2 NEFF cannot execute on the Trn1 compile host, so the driver can exit
nonzero after successful code generation when the benchmark tries to load it.
Use the per-rank logs, `neff_cache`, and `qwen35_compile_metadata.env` to
distinguish that expected load failure from a compiler failure. Restore the
complete cache root on Trn2 and use the same scratchpad page size at runtime.

The full-depth experiment used a `trn1.32xlarge` in `us-east-2`, compiler
2.25.1280.0, and the DLC recorded in the cache metadata. Both the one-region
and four-region graphs compiled successfully. The one-region compile peaked at
216 GiB host RAM; four-region variants peaked at 205-206 GiB. The four-region
cache contains all 32 large model-region NEFFs (four regions across eight
ranks). No compiler OOM or 16M descriptor-materialization error occurred.

Hardware DGE remained enabled: the command does not pass `--disable-dge`, and
the Trn2 Walrus invocations contain `--dge-levels` for I/O, spill/reload,
transpose, reductions, and dynamic offsets. The driver does not force
`dge_mode=none`; precomputing every static descriptor would increase NEFF/HBM
footprint and is not a workaround for this descriptor cap.

Trn2 replay initially hit an HBM wall for TP=8/LNC=1 (superseded — see the
resolution below). With a **replicated** LM head, the base module loaded at 10.73
GB/core and lazy loading of the compiled prefill regions failed for every matched
compiler/runtime scratchpad page size:

| Regions | Page size | HBM at failure | Decisive allocation failure |
|---:|---:|---:|---|
| 1 | 64 MiB | 12.989 GB | next 200 MiB scratchpad allocation |
| 4 | 64 MiB | 12.123 GB | next 64 MiB shared-scratchpad page |
| 4 | 128 MiB | 12.198 GB | 128/200 MiB scratchpad or 13.964 MiB model code |
| 4 | 256 MiB | 12.014 GB | next 256 MiB shared-scratchpad page |
| 4 | 512 MiB | 11.352 GB | second aligned 512 MiB page |

With LNC=1, pairs of ranks share each 24 GiB HBM bank, so these per-rank
allocations exhaust the bank while the module is replicated.

#### Resolution (2026-07-29): TP=8/LNC=1 fits and is now the fastest prefill

The wall was resident HBM, not the kernel. Two `[V, H]` tensors are replicated at
~1.02 GB/rank each (vocab 248,320 × hidden 2048 × BF16): the LM head **and** the
token embedding. Sharding both across TP at **load** time
(`PREFILL_SHARDED_LM_HEAD=1 PREFILL_SHARDED_EMBED=1`) drops the module from 10.73
→ 8.95 GB/core; combined with `--scratchpad-page-size-mb 256` (the sweet spot: 5×
256 MiB pool + tiny overflow, TOTAL ≈ 11.785 GB at the last failing step) the 40L
graph loads with headroom. Full progression:

| Step | Lever | Module | Overflow SP | TOTAL | Outcome |
|---|---|---:|---:|---:|---|
| a | replicated, pg64 | 10.73 | 175 MB | 12.146 | OOM |
| b | lm_head sharded, pg64 | 9.84 | 687 MB | 11.988 | OOM |
| c | + pg512 | 9.84 | 1.5 MB | 11.947 | OOM |
| d | + pg256 | 9.84 | 201 MB | 11.785 | OOM (200 MB intermediate) |
| e | **+ embed sharded, pg256** | **8.95** | — | — | **loads → 3,456.8 tok/s** |

Both shards reconstruct the replicated result exactly (disjoint contiguous vocab
ranges, zero-padded and sum-all-reduced — no extra rounding; proven bit-identical
on CPU across 8 simulated ranks before any compile). The LM-head all-reduce is
~2 MB (prefill needs logits for one token position); the embedding adds one
`[B, chunk, H]` all-reduce per chunk and does not change the step count. Sharding
only one tensor is insufficient — the freed space is reabsorbed by scratchpad
growth. Result: **3,456.8 aggregate prompt tok/s** (TIMED 11.572 s),
`sum=-3.17835375e+05 norm=1.20818213e+03`, warm≡timed and all-finite — **+24.5%**
over the TP=4/LNC=2 record below. The `moe_cte` shard-on-I kernel needed only a
~30-line additive LNC=1 patch (see `patches/nkilib-lnc1-moe-cte.patch`); with the
patch applied the TP=4/LNC=2 fingerprint stays **bit-identical**
(`sum=-3.12377031e+05 norm=1.20273230e+03 top5=[517,607,261,290,294]`), proving
it inert at 2 shards.

The `trn2-3xl-bs4-c16-s20000-tp4-b512-fused-direct512` artifact was validated
as a complete 3.4 GiB cache root (664 files, 66 NEFFs). A separately restored
copy ran the matching BS=4 S=20,000 graph without backend codegen, retained the
same finite fingerprint, and measured 39,788.3 ms / 2,010.6 aggregate prompt
tok/s.

### Optimization levers (environment flags)

| Flag | Effect |
|---|---|
| `DN_NKI=1` | DeltaNet NKI kernel — **required past ~20 layers** (the pure-torch recurrence trips a compiler tiling assertion) |
| `MOE_SPARSE=1` | True-sparse MoE dispatch (gathers only the top-8 experts) — ~2× at BS=1 |
| `MOE_DECODE_TP=1` | BF16 BS=1 decode only: shard each expert's intermediate width across TP ranks, avoiding dummy non-local expert reads |
| `GQATAIL=1` | Fused GQA attention-tail kernel |
| `DNBATCHED_V2=1` | DMA-coalesced batched DeltaNet decode |
| `DN_DIRECT_STATE_OUT=1` | Full-graph decode: write BF16 DeltaNet/conv state directly to disjoint output buffers |
| `DN_PERLAYER_STATE=1` | Full-graph decode: one distinct DeltaNet state buffer per layer (kills the shared-base writeback WAW serialization) — **+96% decode**, bit-identical; requires `DN_DIRECT_STATE_OUT=1` |
| `GQA_STATEFUL_KV=1` | Full-graph decode: keep BF16 K/V caches as aliased module state and append only the current rows |
| `MOE_CTE=1` | Long-token nkilib context-encoding MoE kernel for prefill |
| `GQA_CTE_PREFILL=1` | Prefix-aware nkilib CTE attention; requires `GQA_DYNAMIC_ROPE_KV=1` |
| `DN_CHUNK_NKI=1`, `CHUNK_SIZE=16` | Stable long-context DeltaNet prefill kernel |
| `MOE_FUSED_W8=fp8|int8` | Experimental high-batch full-graph decode using official block-scaled FP8 experts |
| `MOE_FP8=1` | Older per-row FP8 MoE path |
| `MOE_SHARED_ONLY`, `NOREDUCE`, `DN_PASSTHROUGH` | Diagnostics (default off) |

## Performance summary

All numbers are `trn2.3xlarge`, TP=4, LNC=2 unless stated otherwise, measured
with a `torch.neuron`-synchronized 30-50-iter timer. "PyTorch Native" = this
repo's `static_decode_35b.py` (one compiled decode NEFF or four coarse prefill
NEFFs). The "XLA" reference is the
[NxDI](https://github.com/aws-neuron/neuronx-distributed-inference) implementation
of the same model (PR #60) on the torch-xla stack.

### Decode — BS=1 optimization progression (seq=2048)

| Config | Framework | TPOT (ms) | tok/s |
|---|---|---|---|
| masked-dense MoE (start) | PyTorch Native | 66.2 | 15.1 |
| + true-sparse MoE (`MOE_SPARSE=1`) | PyTorch Native | 33.4 | 30.0 |
| + DeltaNet micro-opt | PyTorch Native | 32.8 | 30.5 |
| + `GQATAIL=1` | PyTorch Native | 24.4 | 40.9 |
| + `DNBATCHED_V2=1` | PyTorch Native | 23.2 | 43.2 |
| **+ TP within routed experts (`MOE_DECODE_TP=1`)** | PyTorch Native | **20.46** | **48.9** |
| NxDI reference (PR #60) | XLA | 18.4 | 54.3 |

3.24× total from these levers. True-sparse MoE gives ~2× (not ~8×) because the MoE
expert GEMMs are only about half the step — DeltaNet / GQA / projections / norms /
all-reduces are the rest. The NxDI (XLA) reference is faster at BS=1; it is the
validated oracle (100% token-match vs CPU) but its MoE uses a non-portable NxDI
library module.

The TP-expert layout keeps all 256 expert ids on every rank but stores one quarter
of each expert's intermediate width. Resident weights remain 19.09 GB/core, while
each rank gathers eight quarter-experts instead of eight full experts with roughly
six clamped dummy routes. The existing TP all-reduce reconstructs the full down
projection. A matched S=16, 10-layer Explorer replay estimated 758→380 MB HBM
reads, 754→377 MB software-dynamic DMA, and 270k→181k dynamic DMA packets per
rank; trace time fell from 2.95-3.14 to 2.258-2.261 ms. These traffic values are
directional because the profile has missing dynamic-DMA metadata. A production
S=2048, 40-layer replay was consistent across ranks at about 1.77 GB estimated
HBM reads.

`MOE_DECODE_TP` is deliberately restricted to BF16 and one decode token. For
higher batches, leave it disabled and use masked-dense MoE: once many experts
are active, dense grouped GEMMs amortize better than per-route gathers.

### Decode — batch sweep (seq=256, masked-dense MoE + `DN_NKI+GQATAIL+DNBATCHED_V2`)

| BS | Framework | TPOT (ms) | tok/s | scale |
|--|--|--|--|--|
| 1 | PyTorch Native | 54.9 | 18.2 | 1.0× |
| 4 | PyTorch Native | 70.1 | 57.0 | 3.1× |
| 8 | PyTorch Native | 84.3 | 94.9 | 5.2× |
| 16 | PyTorch Native | 120.4 | 132.9 | 7.3× |
| 32 | PyTorch Native | 188.3 | 170.0 | 9.3× |

Near-linear throughput scaling to BS=32 with no OOM (weights fixed ~19 GB/core; KV +
DeltaNet state are tiny at seq=256). Throughput-optimal is high-BS masked-dense;
latency-optimal is BS=1 true-sparse (sparse only wins at BS≤4, since it gathers
`T·K` experts and `T·K ≥ 64` once batch grows).

### Decode - BS=32 full graph on TP=8, LNC=1 (seq=256)

For high-batch decode, compiling embedding, all layers, state updates, the LM
head, and exact greedy token selection into one graph removes eager boundaries.
The LM head is vocab-sharded across the eight TP ranks; two all-reduces select
the exact global top-1 token, including lowest-id tie breaking, without
materializing full-vocabulary logits on every rank.

| Layers | Path | TPOT (ms) | aggregate tok/s |
|--|--|--:|--:|
| 2 | segmented, replicated LM head | 56.01 | 571.3 |
| 2 | full graph, replicated LM head | 10.26 | 3,119.4 |
| 2 | full graph, vocab-sharded LM head | **8.03** | **3,986.7** |
| 40 | full graph, vocab-sharded LM head | 108.86 | 293.9 |
| 40 | + direct recurrent-state output | **105.31** | **303.9** |
| 40 | + stateful K/V cache | **99.80** | **320.6** |

BS=64 is above the BF16 full-graph HBM ceiling even at sequence length 256.
All eight BS=64 NEFFs compiled, and the module occupied 10.89 GB/rank, but
execution could not allocate the next 240 MB recurrent-state tensor. Each rank
was already at 11.852 GB and pairs of LNC=1 ranks share one 24 GB HBM bank.
An independent Trn2-targeted cross-compile on `trn1.32xlarge` produced the
eight ~43 MB rank NEFFs in 13 minutes. Restoring that cache on `trn2.3xlarge`
loaded those full-graph artifacts without recompiling them, then reproduced
the same 240 MB allocation failure at 11.852 GB/rank. Compiler host RAM is not
the limiting resource. BS=128 and BS=256 were therefore not compiled; BS=32
remains the largest loadable full-depth batch.

The 40-layer results are cache-hot, synchronized 30-iteration runs with masked-
dense MoE. `DN_DIRECT_STATE_OUT=1` keeps recurrent inputs read-only and has the
DeltaNet NKI kernel convert its final FP32 tiles directly into separate BF16
output buffers. This removes the whole-state input clone and the per-layer FP32
state output followed by a BF16 copy. Two real-weight decode steps matched the
control's greedy IDs and local logits on all ranks; DeltaNet and convolution
state were bit-identical. The earlier matched DGE profile showed that sharding
the LM head reduced HBM reads from 1500.1 to 610.1 MB, software DMA from 1507.5
to 603.6 MB, and device execution from 10.266 to 6.324 ms.

On a matched two-layer profile, direct output reduced device execution from
6.324 to 6.105 ms, estimated HBM reads from 610.1 to 589.2 MB, HBM writes from
46.3 to 29.5 MB, and combined dynamic DMA from 648.1 to 623.0 MB. These no-DGE
traffic estimates are directional because the profile reports missing dynamic
DMA metadata. The full graph has roughly 968,000 instructions and uses about
10.72 GB HBM per rank. A full-depth inline capture cannot allocate its trace
buffers beside the model, so traffic was measured on the two-layer graph only.

`GQA_STATEFUL_KV=1` removes K/V from the compiled step's inputs and outputs.
Each GQA call reads the prior BF16 cache, includes the current K/V row in FP32
attention math, then appends that row to aliased module buffers after attention.
This avoids cloning and returning the full `batch * sequence` cache while
preserving one graph and the established arithmetic path. It requires
`GQATAIL=1 DECODE_FULLGRAPH=1` and one local KV head per rank.

A paired 100-step four-layer run (one GQA layer) measured 12.79 to 12.68 ms.
Two real-weight steps matched every greedy ID on all eight ranks; DeltaNet,
convolution, K, and V state were bit-identical. A matched no-DGE replay reduced
device execution from 10.236 to 10.096 ms, estimated HBM reads from 1079.7 to
1046.2 MB, HBM writes from 77.5 to 44.0 MB, and combined dynamic DMA from
1131.9 to 1094.2 MB. Treat those traffic values as directional because Explorer
reports missing dynamic-DMA metadata. The production 40-layer run improved
105.31 to 99.80 ms/token, or 303.9 to 320.6 aggregate tok/s.

Use:

```bash
DN_NKI=1 DN_K_HEADS=2 DN_V_HEADS=4 \
GQATAIL=1 GQA_Q_HEADS=2 DNBATCHED_V2=1 \
DN_DIRECT_STATE_OUT=1 GQA_STATEFUL_KV=1 \
DECODE_FULLGRAPH=1 DECODE_SHARDED_LM_HEAD=1 \
NEURON_LOGICAL_NC_CONFIG=1 NEURON_CC_FLAGS="--target trn2 --lnc 1" \
  torchrun --nproc-per-node=8 static_decode_35b.py \
    --model-path <weights> --max-seq-len 256 --num-layers 40 \
    --graph-splits 1 --batch-size 32 --num-tokens 2 \
    --bench --bench-iters 30
```

### Decode — long-context batch sweep (seq=10000 and 20000, masked-dense MoE + `DN_NKI+GQATAIL`)

At long context the KV cache is no longer negligible, so the batch ceiling is set by
**device HBM** (24 GB/core), not by throughput scaling. Weights are a fixed
~19.1 GB/core; each sequence's KV cache grows with batch × seq, and past a point the
NEFF fails to load (`NRT_RESOURCE: Failed to allocate resource`).

| Seq | BS | TPOT (ms) | tok/s | scale | notes |
|--|--|--|--|--|--|
| 10000 | 1 | 62.2 | 16.1 | 1.0× | 19.1 GB/core |
| 10000 | 4 | 131.2 | 30.5 | 1.9× | |
| 10000 | 8 | 164.3 | 48.7 | **3.0×** | **peak that fits** |
| 10000 | 16 | — | — | — | OOM on device load |
| 20000 | 1 | 122.5 | 8.2 | 1.0× | 19.1 GB/core |
| 20000 | 4 | — | — | — | OOM on device load |

At seq=10000 the throughput knee is **BS=8 (48.7 tok/s)**; BS=16 exceeds HBM and
fails to load. At seq=20000 the per-sequence KV cache is 2× larger, so the ceiling
drops to **BS=1** — even BS=4 fails to load. This is a memory ceiling, not a compute
one, and it is exactly where FP8 experts help (see below): halving the expert weights
(~19→11 GB/core) frees the headroom to push the long-context batch ceiling higher
(e.g. BS=16 at 10k, or BS>1 at 20k, both of which OOM in bf16 today).

### Legacy FP8 experts — a memory/capacity lever, not a decode-latency win

FP8 (e4m3, per-output-channel row scales) on the MoE experts is wired in behind
`MOE_FP8=1`. It is **CPU-validated coherent** (fp8-vs-bf16 cosine 0.9991) and delivers
its headline benefit as a **memory saving: expert weights 16→8 GB/core, total
~19→11 GB/core**.

However, across three independent attempts it did **not** improve decode latency:

| Path | Result |
|---|---|
| Hand-rolled FP8 grouped-matvec (`MOE_FP8=1`) | BS=1 72.3 ms vs 32.8 ms bf16 — **2.2× slower** |
| `nkilib` fused MoE, bf16 (`MOE_NKILIB=1`) | BS=1 **28.2 ms (best bf16)**; FP8 path blocked |
| `nkilib` fused MoE, FP8-row | compile/dtype wall on this toolchain (legacy-e4m3 vs torch e4m3fn) — not reachable |

The reason is the BS=1 GEMV regime: FP8 replaces wide fused GEMMs with many tiny
per-expert matvecs (moving-free=1), and the per-instruction dispatch overhead dwarfs
the bandwidth saved. FP8's real value here is **capacity** — the ~8 GB/core it frees is
what would let the long-context batch ceiling above go higher (e.g. BS=16 at 10k, which
currently OOMs in bf16). Making FP8 also win latency would need the dequant fused into
one wide kernel, or the BS≫1 regime where matvecs become GEMMs. All FP8 paths are
default-off; **bf16 is the recommended decode default.**

The newer `MOE_FUSED_W8` path above is separate: it preserves the official
128x128 block scaling and fuses all local experts into one NKI call per layer
for the high-batch GEMM regime. Its two-layer device correctness and traffic
gates passed, but exact tile-local E4M3FN decoding regressed synchronized TPOT
by 4.1x. After correcting direct-state propagation, the 40-layer numerical
comparison passed continuous recurrence gates but regressed BS=32 TPOT by
6.20x. Higher batches were skipped after that throughput failure. The
historical results in this section must not be attributed to it.

### Prefill (prompt throughput)

#### FP8 CTE MoE prefill (2026-08-21/22) — fastest measured, but at N=10,000

**Every row in this sub-table is `N=10,000`, not the `N=20,000` used by the main table
below.** They are therefore not directly comparable to it. Prompt-length sensitivity was
measured separately and is small (~0.5% from 10k to 20k, see "Context-length sweep"), but
an FP8 run at N=20,000 has **not** been done. All rows: 40 layers, TP=8/LNC=1, bucket
1024 unless noted, O1, beta-5, `MOE_CTE_FP8=1`, **5.08 GB/core** (BF16 CTE is 8.94).

| Config | BS | tok/MoE call | Latency | Prompt tok/s |
|---|---:|---:|---:|---:|
| **Output-block scale grid + PSUM hoist + uncapped packer** — re-measured 2026-09-02 on a new box | **6** | **6,144** | **13.728 s** | **4,370.8 aggregate** |
| ↳ the same config as originally published, on the now-terminated predecessor box | 6 | 6,144 | 13.686 s | 4,383.9 aggregate |
| same levers, BS=4 | 4 | 4,096 | 9.326 s | 4,289.2 aggregate |
| same levers, BS=8 — **worse**, tokens/call is not monotone | 8 | 8,192 | — | 4,271.9 aggregate |
| same levers, bucket 2048 — **worse**, intra-chunk work grows as `S·chunk/2` | 3 | 6,144 | — | 4,187.5 aggregate |
| same levers, bucket 256 | 16 | 4,096 | 40.721 s | 3,929.2 aggregate |
| output-block grid + hoist only, capped packer | 16 | 2,048 | 41.333 s | 3,871.0 aggregate |
| **BF16 CTE reference at the only config where both were run** | 2 | 2,048 | — | **4,227.7 aggregate** |
| FP8 *before* the output-block hoist, same config as that BF16 row | 2 | 2,048 | — | 3,920.3 aggregate |
| pre-lever FP8 baseline | 16 | 2,048 | 43.536 s | 3,675.1 aggregate |

Clean per-lever attribution, all three at BS=4/bucket 1024 so only the lever changes:

| Variant | Prompt tok/s | Delta |
|---|---:|---|
| capped at 2,048 tok/call, no output-block grid | 3,846.2 | — |
| + output-block grid + PSUM hoist + batched packer shift | 4,150.9 | **+7.9%** |
| + uncapped (6,144 tok/call) | 4,289.2 | **+3.3%** |

`1.079 × 1.033 = 1.115`, matching the +11.5% end-to-end exactly, so the two levers compose
cleanly and the output-block hoist is the larger one.

##### Reproduction A/B (2026-09-02): unconditional block-metadata tiling cost 3.0%

The headline was re-measured on a fresh trn2.3xlarge (host driver 2.30.2 / runtime 2.34.10)
after the original box was terminated. Source, nkilib and container image are
md5-identical across all rows; the fingerprint (`sum=-6.51782125e+05 norm=2.05050854e+03
top5=[220,13,197,198,62]`, 5.08 GB/core) is **identical in every row**, so nothing here changes
arithmetic.

| variant | box | wall | agg tok/s |
|---|---|---:|---:|
| untiled block metadata | predecessor, terminated | 13,686.3 ms | **4,383.9** (as published) |
| untiled block metadata | current | 13,727.5 ms | 4,370.8 |
| **conditional tiling (the fix, shipping)** | **current** | **13,728.2 ms** | **4,370.6** |
| tiling always emitted (`af4c000`), run 1 | current | 14,132.5 ms | 4,245.5 |
| tiling always emitted, run 2 | current | 14,136.2 ms | 4,244.4 |

The fix lands **0.005%** off the untiled path — ~5× below the 0.026% run-to-run spread — so
the fast path is fully recovered, and `+3.0%` over the always-tiled build is confirmed at the
shipping config.

Two separate effects, and they were initially conflated:

1. **Our code, −3.0%, fixed.** `af4c000` tiled the route packer's partition-major block-metadata
   section to lift a 128-partition `iota` ceiling that blocked BS=8. It landed *after* the
   record was set. At BS=6, `max_blocks = 128`, so the tiling loop runs exactly one iteration
   and is bit-identical — and it still cost 3.0%, because it adds an SBUF alloc scope, extends
   `condition_row`'s live range across the whole section, and renames every buffer (changing
   allocation order). The tiling is now emitted only when `max_blocks > 128`, keeping both the
   lifted ceiling and the fast path. Do not "simplify" it back to unconditional.
2. **Host stack, −0.30%, recorded not chased.** What remains between 4,383.9 and 4,370.8 is
   ~11× the 0.026% run-to-run spread, so it is real, but the old box's neuron versions were
   never recorded and the box is gone — there is no target to A/B against. The process fix is
   the durable one: **record the host neuron stack with every benchmark number.**

The lesson worth generalizing: **a restructuring that is bit-identical is not therefore
performance-neutral.** This workload has shown ±1–2% sensitivity to SBUF allocation order
several times (SBUF-resident intermediates +0.7%, transpose-once +1.4%); a "no-op" loop that
runs once is still a different allocation problem for the compiler.

A second, methodological one: **NEFF bytes cannot gate this.** Comparing the fix's NEFFs
against the untiled build's was attempted as a stronger-than-timing equivalence proof and is
invalid — `neuronx-cc` is not byte-reproducible. Only **8 of 108** NEFFs matched, including two
files of *identical* size (10,493 B) with different md5s for a helper kernel neither change
touches. The gates that do hold are the output fingerprint, device memory, and timing against
a measured spread.

**What this does and does not show.** At the one config where BF16 and FP8 were both
measured (BS=2, 2,048 tok/call) **BF16 was 7.8% faster** — the FP8 dequant tax. The
output-block hoist removed that tax, and the tokens-per-call tuning then pushed FP8 past
the BF16 *configuration*. But **BF16 CTE was never re-run at BS=6/bucket 1024**, so the
FP8-versus-BF16 question at the tuned operating point is open, and the headline gap should
not be attributed to FP8 alone. The memory result needs no such qualification: 5.08 vs
8.94 GB/core.

Two caveats on the numerics:

- Capping is **not** bit-identical to not capping at 40L (`sum=-4.87432719e+05` capped vs
  `-4.93343469e+05` uncapped; norm within 0.12%, top5 identical). Two calls of 2,048
  versus one of 6,144 group the expert blocks differently, so the FP reduction order
  changes. Expected reassociation, not a defect.
- The output-block hoist moved the 4-layer logits norm by −0.35%, outside a pre-registered
  0.2% gate. **The gate was mis-specified, not the code wrong:** it assumed the hoist only
  reorders same-precision arithmetic, but the down-projection accumulator `block_new_lst`
  is bf16, so the baseline rounds the partial sum to bf16 *between* contraction blocks
  while the hoist accumulates both in fp32 PSUM and rounds once. bf16's relative quantum is
  ~2⁻⁹ ≈ 0.2%, so a 0.35% shift is the expected size — and the hoisted path has **one fewer
  rounding**, i.e. it is the more accurate of the two. Tokens are identical either way.

#### BF16 prefill at the targeted N=20,000 long-context regime

| Test | Framework | Config | Latency | Prompt tok/s |
|---|---|---|---|---|
| **beta-5 container, per-GQA-layer rope-KV, same source (compiler-only win, bit-identical, reproduced 9.768 s / 4,095.2)** | PyTorch Native | BS=2, N=20000 each, bucket=1024, pg256, splits=4 | **9.761 s** | **4,097.9 aggregate** |
| **beta-4 + in-place rope-KV write, one buffer per GQA layer (CORRECT: cache 100% populated)** | PyTorch Native | BS=2, N=20000 each, bucket=1024, pg256, splits=4 | **9.995 s** | **4,002.1 aggregate** |
| **beta-4 container, unchanged source (compiler-only win, numerics bit-identical)** | PyTorch Native | BS=2, N=20000 each, bucket=1024, pg256 | **10.974 s** | **3,645.0 aggregate** |
| ~~beta-4 + in-place rope-KV write~~ — **INVALID**, drops ~60% of KV writes | PyTorch Native | BS=2, N=20000 each, bucket=1024, pg256 | ~~10.284 s~~ | ~~3,889.4 aggregate~~ |
| ~~In-place rope-KV write, old container~~ — **INVALID**, same defect | PyTorch Native | BS=2, N=20000 each, bucket=1024, pg64 | ~~10.913 s~~ | ~~3,665.5 aggregate~~ |
| **Packed DeltaNet C32 n=4 + vocab-sharded head/embed, TP=8/LNC=1** | PyTorch Native | BS=2, N=20000 each, bucket=1024, pg256 | **11.572 s** | **3,456.8 aggregate** |
| Packed DeltaNet C32 n=4 (SBUF-resident + transpose-once), TP=4/LNC=2 | PyTorch Native | BS=2, N=20000 each, bucket=1024 | 14.411 s | 2,775.6 aggregate |
| **Batched compiled CTE-GQA + fused NKI-routed CTE-MoE + stable DeltaNet C32 (block-diagonal)** | PyTorch Native | BS=2, N=20000 each, bucket=1024 | **17.568 s** | **2,276.9 aggregate** |
| Batched compiled CTE-GQA + fused NKI-routed CTE-MoE + paired DeltaNet C16 | PyTorch Native | BS=2, N=20000 each, bucket=1024 | 19.141 s | 2,089.7 aggregate |
| Batched compiled CTE-GQA + fused NKI-routed CTE-MoE + paired DeltaNet C16 | PyTorch Native | BS=4, N=20000 each, bucket=512 | 39.788 s | 2010.6 aggregate |
| **Compiled CTE-GQA + fused NKI-routed CTE-MoE + DeltaNet C16** | PyTorch Native | BS=1, N=20000, bucket=1024 | **13.488 s** | **1482.8** |
| Compiled CTE-GQA + Torch-routed CTE-MoE + DeltaNet C16 | PyTorch Native | BS=1, N=20000, bucket=1024 | 17.374 s | 1151.1 |
| Batched compiled CTE-GQA + Torch-routed CTE-MoE + DeltaNet C16 | PyTorch Native | BS=2, N=20000 each, bucket=1024 | 41.069 s | 974.0 aggregate |
| Compiled CTE-GQA + CTE-MoE + DeltaNet C16 | PyTorch Native | N=20000, bucket=512 | 17.855 s | 1120.1 |
| Compiled CTE-GQA + CTE-MoE + DeltaNet C16 | PyTorch Native | N=20000, bucket=2048 | 20.632 s | 969.4 |
| Compiled flash-GQA + CTE-MoE + DeltaNet C16 | PyTorch Native | N=20000, bucket=512 | 20.886 s | 957.6 |
| Bucketed prefill, flash-GQA + DeltaNet-chunk kernels | PyTorch Native | N=20000 | 77.2 s (warm) | **259.2** |
| Eager prefill (pre-kernelization) | PyTorch Native | N=4000 | 146.7 s | 27.3 |
| Eager prefill (pre-kernelization) | PyTorch Native | N=2000 | 68.4 s | 29.3 |

The highest validated aggregate throughput uses BS=2, 1024-token buckets, four
compiled 10-layer segments, runtime bucket offsets/valid lengths, fused
NKI-routed CTE MoE, and **stable DeltaNet C32** (`DN_STABLE_C32=0 CHUNK_SIZE=32`;
opt-in). It measured 17.568 seconds / 2,276.9 aggregate prompt tok/s — **+8.5%**
over the paired-C16 baseline (19.141 s / 2,089.7), with identical finite warm and
timed fingerprints. C32 halves the DeltaNet chunk count; the win required a
numerically stable inverse — `_tri_inverse_blockdiag` splits the 32×32 chunk
matrix into two 16×16 diagonal blocks plus a coupling term and inverts the blocks
by doubling (the naive full-32 doubling overflows on near-1-decay streams, and a
Horner series is stable but ~4× costlier at 2,037.5 tok/s). C32 correctness was
gated on all four checks: finite warm≡timed fingerprint; final-token top-5
matching the C16 baseline; **all-rank capture-replay vs the CPU reference at deep
context (cosine ≈ 1.0, max_diff ~1e-6 on all four TP ranks)**; and real-prompt
coherence identical to C16 (bit-identical greedy continuation via iterative
prefill). C16 remains the default; C32 is the opt-in faster path.

The paired-C16 configuration (same shapes) measured 19.1411 seconds / 2,089.7
aggregate prompt tok/s with identical finite warm and timed fingerprints. BS=4
also fits with 512-token buckets and measured 39.7883 seconds / 2,010.6 aggregate
prompt tok/s after restoring its compiler cache, matching the original 2,012.2
tok/s median.

The fastest validated single-prompt path uses 1024-token buckets, the fused
NKI-routed CTE MoE kernel (`MOE_CTE_NKI_PACK=1`), and `CHUNK_SIZE=16`.
The nkilib CTE attention kernel only visits the used KV prefix; it measured
0.77-0.81 ms per production-shape GQA call versus 11.66-11.69 ms for the local
fixed-KMAX flash kernel. At full depth this improves 957.6 to 1120.1 tok/s and
preserves the validated real-prompt continuation. The warm and timed synthetic
fingerprints were identical.

CTE also removes the descriptor ceiling at bucket 2048: all four segments
compiled and loaded. That configuration is nevertheless slower (969.4 tok/s).
The CTE attention kernel itself scales efficiently from active-query sizes 512
to 1024 to 2048 (about 0.79, 1.28, and 2.11 ms), so the regression is in the
larger surrounding compiled graph. The matched bucket-1024 Torch-route run
measured 1151.1 tok/s.

The fused route path replaces the compiled `one_hot().cumsum()` metadata packer
with an NKI stable compaction inside the existing CTE custom call. It passed 96
exact metadata cases across both production token counts, both block sizes, and
all four TP expert ranges. Distributed fused CTE output matched the
precomputed-metadata path exactly, and the four-layer BS=2 isolation test still
matched independent BS=1 executions.

Three synchronized cache-hot BS=1 runs measured 13.4834, 13.4967, and
13.4878 seconds; the 13.4878-second median is 1482.8 tok/s. A matched
8-DeltaNet/2-GQA segment improved from 188.89 to 122.26 ms, total HBM traffic
fell from 31.96 to 3.30 GB, route HBM traffic fell from 24.94 GB to 45.9 MB,
and the old `reduce-window` instruction disappeared. Standalone route packing
measured 2.420 ms for 8,192 assignments and 3.777 ms for 16,384 assignments
(1.56x scaling). The fallback remains available with
`MOE_CTE_NKI_PACK=0`; the validated BS=2 and BS=4 throughput paths use the
fused packer.

Homogeneous batching is implemented with independent DeltaNet, convolution, and
KV state per prompt while retaining one custom call per layer. A four-layer
BS=2 isolation test with distinct prompts and a partial final bucket matched
independent BS=1 runs with cosine >=0.999936 across logits and all carried
states. Full S=20000 BS=2 loaded successfully and all returned states were
finite on the Torch-route baseline, but latency increased 2.36x for 2x the
tokens: 41.069 s / 974.0 aggregate tok/s. This is 15.4% below its matched
BS=1 baseline. Fused NKI route packing plus paired-C16 DeltaNet changed the
BS=2 result to 19.141 s / 2,089.7 aggregate tok/s. The later BS=4 run remained
finite and repeatable at 39.788 s / 2,010.6 aggregate tok/s.

Isolated production-shape profiles explain part of the scaling limit. The C16
DeltaNet call increased from 11.57-11.93 ms at B=1 to 22.58-22.92 ms at B=2
because its independent recurrent streams execute sequentially. The opaque CTE
expert kernel itself stayed near 6.7-7.0 ms for 1024 versus 2048 flattened
tokens.

Full 10-layer segment traces locate the superlinear regression in the routing
wrapper around that expert kernel. For the matched 8-DeltaNet/2-GQA segment,
BS=1 to BS=2 increased from 188.9 to 471.0 ms. The
`pack_local_routes()` prefix scan at `moe_cte_adapter.py:50` increased from
61.3 to 258.3 ms and its attributed HBM traffic increased from 24.9 to
105.7 GB. Neuron lowers the `group_hot.cumsum(dim=0)` to HLO
`reduce-window` backed by TensorE MATMUL/LDWEIGHTS. DeltaNet increased from
82.4 to 165.6 ms. Those two changes explain 99% of the matched segment
regression. The alternating 7-DeltaNet/3-GQA segment showed the same result:
485.7 ms total, 263.7 ms in the route scan, and 145.6 ms in DeltaNet.
The fused NKI route path removes this scan and is used by the validated BS=2
and BS=4 throughput configurations above.

`GQA_CTE_PREFILL=1` needs a recent nkilib with `attention_cte` support for
head-dim 256 and runtime `prior_used_len`; the nkilib bundled in the current DLC
rejects head dimensions above 128. Point `PYTHONPATH` at a compatible
`nki-library/src/nkilib_src`.

The original 259.2 tok/s path used eager sequence bucketing
(`--bucket-chunk 2048 --bucket-compile 0`) with local flash GQA and chunked
DeltaNet. Two kernel bugs had to be fixed before compilation was trustworthy:
pad-token DeltaNet-state corruption and an L2-norm epsilon-semantics mismatch on
near-zero rows (see `kernels/tests/`). DeltaNet C32 is faster but becomes
non-finite at long context, so C16 remains a correctness requirement.

**20k context:** BS=1 memory fits at ~19.1 GB/core; the optimized compiled
prefill path loads through BS=4. BS=2 uses about 0.42 GB/core of persistent
K/V cache. Preserve
`NEURON_COMPILE_CACHE_URL` on NVMe: the first CTE-GQA run, including four segment
compiles and the 20k warm pass, took 861.6 seconds; cached execution is 17.9
seconds.

### Prefill — context-length scaling (current best build)

Measured on the current fastest prefill build — packed DeltaNet **C32 n=4**
(`DN_PACK_C32=1 DN_PACK_N=4`) with SBUF-resident intermediates + transpose-once
finish (Levers 1+2) — BS=2, 1024-token buckets, four compiled 10-layer segments,
TP=4/LNC=2, runtime bucket offsets/valid lengths. This is the 2,775.6 tok/s @20k
configuration (the +21.9% successor to the 2,276.9 C32-block-diagonal row above).

| Context length | Latency (BS=2) | Aggregate prompt tok/s |
|---|---|---|
| 5,000 | 3.567 s | **2,803.6** |
| 10,000 | 7.167 s | **2,790.6** |
| 20,000 (customer p95) | 14.411 s | **2,775.6** |

**Throughput is essentially flat across the customer's whole context range**, in
fact ~1% *higher* at shorter lengths (2,803.6 @5k vs 2,775.6 @20k). Because the
prefill tiles the prompt into fixed 1024-token chunks driven by a single compiled
graph set (runtime `q_base`/`valid_len` scalars), per-token cost is nearly
constant; shorter prompts accumulate less DeltaNet/KV history, so the per-chunk
GQA is marginally cheaper. **Wall-clock latency scales ~linearly** with length
(3.6 → 7.2 → 14.4 s), which is the number that sets time-to-first-token at each
size. All three lengths returned finite logits and carried state. No recompile is
needed to change context length — the one 1024-chunk graph set serves any prompt
length (10k ran 10 chunks, 5k ran 5) off the same compiled cache.

**Bucket (chunk) size re-validation (2026-07-28).** The earlier chunk-size sweep
picked bucket=1024 on pre-packing code; re-running it on the current packn4 build
(BS=2, N=20000) confirms 1024 is still optimal:

| Bucket | Latency (BS=2, N=20000) | Aggregate prompt tok/s |
|---|---|---|
| 512 | 18.251 s | 2,191.7 |
| **1024** | **14.411 s** | **2,775.6** |
| 2048 | — | runtime `NRT_TIMEOUT` at BS=2 (not viable) |

512-token buckets are −21% vs 1024 (more per-chunk fixed overhead / more chunks).
2048-token buckets compiled but the warm execution hit the 30 s per-NEFF runtime
watchdog on every core (`FATAL-RT-UNDEFINED-STATE`) at BS=2 — the doubled
per-chunk working set is not viable at this batch (historically 2048 only ran at
BS=1, 969.4 tok/s). **1024 remains the recommended prefill bucket.**

### beta-4 container (2026-08-03): prefill faster, and numerically bit-identical

> **Correction (2026-08-04).** An earlier revision of this section reported
> "**+12.5%** prefill for free, no source change" and attributed a **+1.80% norm**
> fingerprint drift to beta-4's compiler, complete with a transpose-retiling
> mechanism derived from the two on-disk compile caches. **Both claims were wrong,
> and for the same reason:** the two runs being compared did *not* share source. The
> `gqa_rope_kv_35b` kernel, its op wrapper, and the `static_decode_35b.py` call site
> were rewritten on **2026-07-30 00:44–00:45 UTC** — after the old cache was built
> (07-29 20:07) and before beta-4's (08-03 17:18). The full-tree `diff -rq` between
> the two on-host snapshots (`src-old` vs `src`) confirms exactly three runtime files
> differ (65 changed lines); everything else is docs, `experiments/`, and tests that
> the run does not import. The transpose-retiling analysis compared two *different
> programs* and has been withdrawn — the `[2,1024,256]` bf16 class it leaned on is
> better read as the GQA K/V tensor (256 = 2 KV heads × 128) for a 1024-token chunk,
> i.e. the very path that was rewritten, than as the DeltaNet packed layout.

The four recorded runs at this config, in order, give a clean attribution:

| # | date | compiler | source | page | TIMED | agg prompt tok/s | fingerprint `sum` / `norm` |
|---|---|---|---|---:|---:|---:|---|
| 1 | 07-29 20:30 | pre-beta-4 | pre-07-30 | 256 | 11,571.5 ms | 3,456.8 | `-3.17835375e+05` / `1.20818213e+03` |
| 2 | 07-30 01:30 | pre-beta-4 | pre-07-30 (`src-old` replay) | 256 | 11,573.7 ms | 3,456.1 | `-3.17835375e+05` / `1.20818213e+03` |
| 3 | 07-30 01:12 | pre-beta-4 | rope-KV rev2 | 64 | 10,912.5 ms | 3,665.5 | **`-3.29478219e+05`** / **`1.22990430e+03`** |
| 4 | 08-03 17:37 | **beta-4** | rope-KV rev2 | 256 | **10,284.3 ms** | **3,889.4** | **`-3.29478219e+05`** / **`1.22990430e+03`** |

**Numerics: beta-4 introduces no drift at all.** Runs 3 and 4 are **bit-identical** —
same `sum`, same `norm` to all printed digits, same `top5=[517,607,15089,258,261]` —
across both a compiler change *and* a page-size change. Runs 1 and 2 likewise
reproduce each other to the bit (3,456.8 vs 3,456.1 tok/s is 0.02% timing noise),
which makes run 2 a validated control rather than a lucky repeat. So the
`-3.17835375e+05 → -3.29478219e+05` shift (norm +1.80%, sum +3.66%, `294`→`258`,
`15089` rank 5→3) belongs **entirely to the 2026-07-30 rope-KV source change**, and
the earlier framing of a "beta-4 correctness issue" was a misattribution against a
stale baseline. Module resident is 8.95 GB/core on both sides.

**Throughput: the +12.5% is real but is not all the compiler's.** It spans two
changes plus a page-size difference, and the one cell that would separate them —
old compiler + rope-KV rev2 + page 256 — was never run, and can no longer be run
because the host now holds only the beta-4 image (the swap was a re-pull of the *same*
mutable tag, so the pre-beta-4 image was replaced, not co-installed; no dangling layers,
and ECR `DescribeImages` is denied to this host's role). Digests for each image label are
in the gitignored `.env`. What the data does support:
the rope-KV rewrite alone is worth **+6.0%** at page 64 (3,456 → 3,665.5), and
beta-4 plus the page-64→256 change together add a further **+6.1%**
(3,665.5 → 3,889.4). Attributing the whole +12.5% to "compiler scheduling" is not
supported.

#### The remaining open question: is rope-KV rev2 correct?

The rewrite stops returning `kv_key`/`kv_value` (they were re-exported as graph
outputs, 1.74 GB of HBM traffic per 10-layer region to publish 1024 changed rows —
the #1 prefill hotspot at 25% of instruction time) and instead mutates the cache in
place, passing the *whole* flattened cache and selecting the layer with a static
`group_index`. The addressing is provably unchanged: the new
`(group_index * batch_size + batch) * kmax` with `kmax = rows/(B*num_groups)` lands
on the same rows as the old `batch * kmax` within `kv_k[gi, :, 0]`. The consumer
changed from the kernel's returned output to `k_filled = kv_k[gi, :, 0]`, a read of
the mutated buffer ordered only by functionalization.

That read-after-write hazard is the obvious suspect, and it is **already refuted on
device**: `test_gqa_rope_kv_alias_probe.py` checks the write is observable through
both the `base_slice` pattern the prefill caller uses and the flat view passed to the
op, exactly (`torch.equal`, so a lost write reads back as zeros), at production
geometry (`NUM_GQA=10, max_seq_len=20480, gi=9`, flat row 409,471 / byte offset
2.1e8); `test_gqa_rope_kv_multicall_probe.py` confirms ordering across 3 in-place
calls and 2 chunks. Both passed (`logs/ropekv_probe.log`,
`logs/ropekv_multicall.log`, 2026-07-30 02:03/02:06).

So: identical arithmetic, identical addressing, both paths individually
device-validated — yet a 1.8% norm difference at 20k context. Something is
unaccounted for, and it cannot be settled by reading code. Note also that rev1 of
this rewrite was *grossly* wrong (`sum=-3.98127219e+05`, norm `1.39189014e+03`,
+15%) before rev2 brought it to within 1.8%, which is not reassuring about rev2
being the fixpoint.

#### Localised (2026-08-05): the divergence is seeded by the cross-chunk KV readback

The matched A/B ran. Both arms are one source tree apart (the rope-KV hunk and
nothing else — `diff` is 24 ± lines), under **one** container (beta-4) at **one**
page size (256), with identical `PREFILL_TRACE_FINITE=1` instrumentation. The
instrumentation is empirically HLO-neutral: the rev2 arm hit its existing beta-4
cache with **102/102 NEFFs unchanged**.

**First: beta-4 is exonerated for the third and final time.** The `src-old` arm under
beta-4 reproduced the pre-07-30 fingerprint *exactly* —
`sum=-3.17835375e+05 norm=1.20818213e+03 top5=[517, 607, 261, 294, 15089]`. That is
the missing cell of the 2×2: old source now has a datapoint under the new compiler,
so source and compiler are fully separated. beta-4 changes no numerics on either
path.

**Second: chunk 0 is bit-identical, and chunk 1 is not.** All four segments × three
tensors × two metrics are exactly `0.000000%` apart at chunk 0. Every cell from
chunk 1 on differs:

| | `conv.norm` | `dn.norm` | `hidden.norm` |
|---|---:|---:|---:|
| chunk 0, all segments | 0.000000% | 0.000000% | 0.000000% |
| chunk 1, segment 0 | 0.005216% | 0.021286% | 0.482412% |
| worst cell (chunk 19, segment 3) | 1.454% | 0.524% | **13.660%** |

This is the KV-write signature, not compounding-from-the-start. It also explains why
chunk 0 matches: `cte_prefill` takes `k_active` (this chunk's keys) *and* `k_filled` +
`q_base` separately, so at chunk 0 the prefix `[0:q_base]` is empty and `k_filled` is
never read. **Chunk 1 is the first invocation that reads rows written by a previous
invocation** — and it is the first that differs.

`hidden.norm` per segment (segment 0 = layers 0–9, segment 3 = layers 30–39):

```
chunk    seg0      seg1      seg2      seg3
    0   0.000%    0.000%    0.000%    0.000%
    1   0.482%    1.067%    0.251%    1.267%
    5   0.160%    2.126%    0.136%    6.958%
   10   0.410%    1.730%    0.139%   10.114%
   19   0.487%    1.842%    1.768%   13.660%
```

Segment 0 stays **bounded** at ~0.1–0.5% with no growth in chunk index — so the
per-chunk seed is a roughly constant-magnitude perturbation, and the recurrent state
is not diverging exponentially. The growth is in *depth*: by layer 39 the deep-layer
divergence reaches 13.7%, which is far too large to be fp reassociation. This is a
semantic difference, not noise, and the final-logits 1.80% is the attenuated tail of
it.

**Third: the shape hypothesis is dead.** `kv_k` is
`[NUM_GQA, B, nkv, max_seq_len, GQA_HEAD_DIM]` (`static_decode_35b.py:2799`), so
`kv_k[gi, :, 0]` is `[B, max_seq_len, HD]` — identical to the old
`k_filled.reshape(B, max_seq_len, HD)`, and identical to what the static
`index_copy_` branch already uses. No silent rank/stride change.

**What this leaves, and it inverts the question.** The old path passed a *sliced*
view (`kv_k[gi, :, 0].reshape(...)`) as `cache_k` and consumed the kernel's
**returned** `k_filled`. Per the rewrite's own comment, a sliced view is exactly the
case where the in-place write is **dropped** unless the tensor is also returned. If
that is right, the old path never persisted anything into `kv_k` across chunk
invocations, and its cross-chunk prefix was whatever the kernel handed back — in
which case **rev2 is the fix and the old, "gated" fingerprint is the wrong one.**
The alias/multicall probes cannot adjudicate this: they assert the *new* pattern
works, never that the *old* one did.

That is directly measurable, and it was measured. The prefill already returns
`kv_k`/`kv_v` as `outputs[3:5]` and already materialises them for the `finite[...]`
check, so adding eager `sum`/`norm`/`nz` per output is HLO-neutral (both arms keep
their caches). Both arms, one container, page 256:

| arm | `kv_k` nz / numel | `kv_k` norm | `kv_v` norm | tok/s |
|---|---:|---:|---:|---:|
| `src-old` (pre-07-30) | **104,857,600 / 104,857,600 = 100%** | 1.54342100e+04 | 6.88771240e+03 | 3,645.0 |
| `src` (rope-KV rev2) | **41,943,040 / 104,857,600 = 40%** | 9.63778613e+03 | 2.98420020e+03 | 3,898.4 |

The old path populates the cache **completely**. rev2 populates **40%** of it, and
`(9637.79 / 15434.21)² = 0.39` — rev2's cache is the old cache with ~60% of it
**zeroed**, not filled with different values. The hypothesis above is refuted: the old
path was fine; rev2 is losing writes.

#### Root cause (2026-08-05): only the FIRST in-place cache mutation per graph is carried back

The occupancy map (`PREFILL_KV_MAP=1`, also eager) localises it exactly:

```
kvmap shape=(10, 2, 1, 20480, 256) numel=104857600
per_group_nz         = [10485760, 0, 10485759, 0, 0, 10485760, 0, 10485760, 0, 0]
per_rowblock1024_nz  = [2097152 × 20]        <- every chunk writes
```

Row blocks are uniformly full, so nothing is wrong with the chunk loop or the runtime
`q_base` offset. The loss is entirely in the **group** dimension: groups **0, 2, 5, 7**
are populated and the other six are bit-zero. GQA sits on every 4th layer
(`gi = 0…9` ↔ layers 3, 7, …, 39) and the run used `--prefill-splits 4`:

| segment | layers | GQA `gi` in segment | populated |
|---|---|---|---|
| 0 | 0–9 | 0, 1 | **0** |
| 1 | 10–19 | 2, 3, 4 | **2** |
| 2 | 20–29 | 5, 6 | **5** |
| 3 | 30–39 | 7, 8, 9 | **7** |

Exactly one per segment — the *first* one. **Only the first in-place mutation of a
given buffer inside a single traced graph is carried back to the caller; every
subsequent mutation of that buffer in the same graph is silently dropped.** No error,
no warning, and the output stays finite and superficially plausible.

This is also why the two probes passed and proved nothing about the production path:
`test_gqa_rope_kv_alias_probe.py` issues one mutation, and
`test_gqa_rope_kv_multicall_probe.py`'s three calls do not sit inside one compiled
segment the way ten GQA layers do. **A probe that validates an aliasing pattern must
reproduce the number of mutations per graph, not just the addressing.**

Consequences:

- **rev2 must not be landed as-is.** Its fingerprint
  (`sum=-3.29478219e+05 norm=1.22990430e+03`) is the signature of a model attending
  over a 60%-empty KV cache. The reference fingerprint is the `src-old` one,
  `sum=-3.17835375e+05 norm=1.20818213e+03`.
- **The +6% was partly work not done.** Six of ten GQA layers skipped their cache
  DMA entirely. The honest beta-4 prefill number is the `src-old` arm's
  **3,645.0 tok/s** (+5.4% over 3,456.8 at identical source and page size).
- **The optimisation itself is still valid and still the top prefill lever** —
  re-exporting the cache is genuinely 25% of instruction time and 21.7% of HBM
  traffic. It just needs a form where each graph mutates each buffer at most once,
  e.g. **one cache tensor per GQA layer** (each segment then touches ≤3 *distinct*
  buffers, one write each) rather than one shared `[NUM_GQA, …]` slab.
- **Full-graph decode uses the identical pattern and is now suspect.**
  `gqa35b::tail_stateful` takes the whole flattened cache plus a static
  `layer_index` and mutates in place (`static_decode_35b.py:1821`), and
  `DECODE_FULLGRAPH=1` puts all ten GQA layers in **one** graph — the worst case for
  this defect. Every decode validation to date compared two runs that would share it,
  so a token-identical `gen hash` cannot rule it out. Checked directly with an eager
  `DECODE_KV_MAP=1` occupancy read of `mod.decode_kv_k` (see below).

The greedy coherence continuation **passed token-identically** on both arms —
prompt `760,6511,314,9338,369` →
`[11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, …]`, matching the
reference in `PREFILL_RECIPE.md`. That is worth recording as a **negative result about
the gate itself**: greedy argmax over a cyclic reference prompt survived a 60%-empty
KV cache and a 13.7% deep-layer perturbation. The coherence gate is not sensitive
enough to catch a defect of this size, and must not be treated as sufficient. The
cheap eager state fingerprint (`nz` + `norm` per state tensor) catches it instantly
and should run alongside every fingerprint from now on.

**Decode on beta-4 was blocked by host OOM, not by the device — now resolved with
`--optlevel 1`; see below.** Two full-graph decode compiles died after ~48 min each with
`[F137] neuronx-cc was forcibly killed` (exit 70) — 2026-08-03T18:29:52Z (rank6)
and 2026-08-03T20:40:57Z (rank5). **159 GB of swap across three swapfiles was
insufficient**; the compiler is killed while the graph is still one region.
Reducing rank count is *not* an option (the LNC=1 fit depends on 8 shards).

> **`--graph-splits 2` is NOT the fallback — it is a no-op here.** An earlier
> revision of this entry recommended it. Under `DECODE_FULLGRAPH=1`,
> `static_decode_35b.py:2804` hardcodes `setup_segments(1, compile_each=False)` and
> compiles the whole decode step with one `torch.compile`; `args.graph_splits` is
> read **only** in the `else` branch. Passing `--graph-splits 2` changes nothing —
> the log still prints `compiled as one graph ...: [(0, 40)]`. Splitting is also not
> a drop-in: `DN_DIRECT_STATE_OUT=1` and `GQA_STATEFUL_KV=1` both *require*
> `DECODE_FULLGRAPH=1` (`:470-473`), so leaving full-graph mode means giving up the
> max-throughput decode config, not just resegmenting it. A run launched with
> `--graph-splits 2` was confirmed to be a bit-for-bit repeat of the OOM config and
> was killed rather than re-paid.

#### Resolved: `--optlevel 1` compiles it (2026-08-04)

The lever was **compiler optlevel**. Decode had been running with
`NEURON_CC_FLAGS="--target trn2 --lnc 1"` — no `--optlevel`, i.e. the neuronx-cc
default — while prefill compiled successfully at an explicit **O1** on the same host.
Adding `--optlevel 1` (`run_decode_beta4_o1.sh`, one variable changed, no extra swap)
**compiled and ran on the first attempt**:

| | beta-4 + O1 | baseline (old container, default optlevel) |
|---|---:|---:|
| compile outcome | `DOCKER_EXIT=0`, **24m21s** total wall | ok |
| weights load+shard | 420.0 s | — |
| first decode step (incl. compile) | 997.5 s | — |
| module resident | **7.09 GB/core** | — |
| TPOT (BS=128, seq=256, synced, 20 iter) | **395.11 ms/tok** | 289.53 ms/tok |
| throughput | **324.0 tok/s** | 442.1 tok/s |
| `gen hash` row0 / row127 | **`0cc59fb25112`** / `0cc59fb25112` | `0cc59fb25112` |

**Numerics: token-identical.** The generated-token hash matches the baseline exactly,
on both the first and last row of the batch. Taken with the prefill result above
(bit-identical fingerprints on matched source), beta-4 has now produced **no
numerical change on either path**. The fp8 expert conversion also reports its usual
`cosine=0.9996470, normalized RMSE=2.66934%`.

**Throughput: −26.7%, and not yet attributable.** 395.11 vs 289.53 ms/tok is a real
regression, but it confounds O1-vs-default with old-vs-beta-4, and **neither
confound can be lifted on this host**: beta-4 at the default optlevel is the
configuration that OOMs, and the old image is gone (only beta-4 remains, no
dangling layers, ECR `DescribeImages` denied to this role). The honest statement is
that the only configuration in which beta-4 full-graph decode compiles is O1, and
that configuration is 26.7% slower than the recorded default-optlevel baseline.
O1 cutting optimization passes is the likely cause and is the cheaper hypothesis;
testing it needs an old-container O1 run, i.e. the image by digest.

**Swap was never touched** — peak host use was 64/124 GB with 0 B of the 63 GB
swapfile consumed, which retrospectively explains why 159 GB of swap did not save
the default-optlevel attempts: the compiler was killed on a fast allocation spike
rather than gradually exhausting RAM, so more swap was never the fix.

#### CONFIRMED (2026-08-05): full-graph decode had the same defect, and worse

The `DECODE_KV_MAP=1` occupancy read predicted above was run on exactly the O1
configuration in the table (40 layers, BS=128, seq=256, `GQA_STATEFUL_KV=1
DECODE_FULLGRAPH=1`, three generated tokens):

```
DECODE kvmap shape=(10, 128, 1, 256, 256) numel=83886080 total_nz=98304
DECODE kvmap per_group_nz           = [98304, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DECODE kvmap per_group_nonzero_rows = [384,   0, 0, 0, 0, 0, 0, 0, 0, 0]
```

384 = 128 batch rows × 3 tokens, so **GQA group 0 is written completely and correctly
and groups 1–9 never receive a single write.** Full-graph decode is the worst case for
this defect: all ten GQA layers share one traced graph, so nine of ten GQA layers
attend over an all-zero cache.

Three things to record about how this went undetected:

- **It is not the 07-30 rewrite's bug.** `git blame` puts the pattern at `ad342fbf`,
  **2026-07-17** (`static_decode_35b.py:1821`). It predates the prefill rewrite by
  two weeks; the prefill rewrite copied it *because* decode appeared to validate.
- **The original gate could not have caught it.** The validation recorded above —
  *"A paired 100-step four-layer run (one GQA layer) … K and V state were
  bit-identical"* — used **four layers**, which is **exactly one GQA layer**, i.e.
  exactly one mutation per graph. Same failure mode as the two prefill alias probes:
  the geometry that exposes the defect was never exercised.
- **`gen hash` cannot detect it.** Both flags are opt-in and every recorded decode
  comparison had them on both sides, so the two runs shared the defect and agreed.

Scope of invalidated numbers: **anything measured with `GQA_STATEFUL_KV=1` and
`DECODE_FULLGRAPH=1` together**, which is the configuration in `DECODE_RECIPE.md` and
in `deploy/compile_decode_fp8_trn2.sh`. That includes the `+5.5%` (105.31 → 99.80
ms/token, 303.9 → 320.6 tok/s) attributed to stateful KV above, and the BS=128 O1
rows in the table. The `GQA_STATEFUL_KV=0` path returns the cache as a graph output
and is unaffected.

#### The fix: one cache buffer per GQA layer (both paths)

Landed in `static_decode_35b.py`, no new flag — it is a bug fix, not a variant:

| | before | after |
|---|---|---|
| decode cache | one `decode_kv_k` `[NUM_GQA,B,NKV,S,HD]` buffer | `decode_kv_k{0..9}`, each `[B,NKV,S,HD]` |
| decode call | `kv_k.reshape(NUM_GQA*B*S,HD)`, `layer_index=gi` | `kv_k[gi].reshape(B*S,HD)`, `layer_index=0` |
| prefill cache | `kv_cache_k.clone()` | `[kv_cache_k[gi].clone() for gi in …]` |
| prefill call | `kv_k.reshape(NUM_GQA*B*S,HD)`, `gi`, `NUM_GQA` | `kv_k[gi].reshape(B*S,HD)`, `0`, `1` |

The kernels need no change: `nki_gqa_tail`'s row base is
`(layer_index * B + b) * S` and `nki_gqa_rope_kv_dynamic`'s is
`(group_index * B + b) * kmax` with `kmax = shape[0] // (B * num_groups)`, so a
per-layer buffer is just the `layer_index = 0, num_groups = 1` case. Each buffer is
still passed **whole**, and total
KV memory is unchanged — it is the same bytes, differently owned. Correctness is now
independent of `--prefill-splits`, which matters because the cheap
`--prefill-splits 10` confirmation (4-layer segments, one GQA layer each) is not
actually available: dynamo keys its cache **per code object**, and all N `segment`
closures in `prefill_bucketed` share one, so N > 8 hits
`FailOnRecompileLimitHit` under `fullgraph=True`. That is now raised to
`max(16, 2*splits+4)` at the top of `prefill_bucketed`.

Gates, both now in-source rather than sed patches:
`PREFILL_FINGERPRINT=1` prints `sum`/`norm`/`nz`/`numel` per carried state tensor,
`PREFILL_KV_MAP=1` adds `per_group_nz`, and `DECODE_KV_MAP=1` prints decode's
`per_group_nz` + `per_group_nonzero_rows`. A correct 40-layer BS=2 @20480 prefill has
`kv_k` **100% non-zero** (104,857,600) at `norm=1.54342100e+04`; a correct decode has
every one of the ten groups populated.

#### Gated 2026-08-05: the fixed in-place write is a real +9.8% — NEW PREFILL RECORD

40 layers, BS=2, N=20000, bucket=1024, `--prefill-splits 4`, pg256, beta-4 container.
Both variants of "one buffer per GQA layer" were run; the second is what landed.

| variant | `kv_k` per-group nz | TIMED | tok/s | logits fingerprint |
|---|---|---:|---:|---|
| ten views of one base (`[gi].contiguous()`) | 10,485,760 × 10 | 10,021.2 ms | 3,991.6 | exact |
| **ten owned buffers (`[gi].clone()`)** — landed | 10,485,760 × 10 | **9,994.6 ms** | **4,002.1** | **exact** |
| baseline: beta-4, cache returned as graph output | 10,485,760 × 10 | 10,974.1 ms | 3,645.0 | exact |
| ~~defective in-place write~~ | [10485760,0,10485759,0,0,10485760,0,10485760,0,0] | ~~10,284.3 ms~~ | ~~3,889.4~~ | mismatched |

Reading these:

- **The win is real and it is bigger than the defective one claimed.** 4,002.1 vs
  3,645.0 = **+9.8%**, against the withdrawn 3,889.4's +6.7%. Not returning a 1.74 GB
  cache per 10-layer region pays for itself even when the writes actually happen; the
  earlier number was *both* inflated by skipped work *and* an underestimate of the
  lever. HBM is unchanged at **8.95 GB/core** — same bytes, differently owned.
- **`nz` is 10,485,760 for every group** bar 3 elements total across 209 MB
  (`[…,10485759,…,10485758]`), which are genuine bf16 zeros in the data, not gaps.
  `kv_k norm=1.54342275e+04` vs the old path's `1.54342100e+04` — agreement to 6
  significant figures, a 1.1e-5 relative difference from the kernel's accumulation
  order. The **logits fingerprint is exact**: `sum=-3.17835375e+05
  norm=1.20818213e+03 top5=[517,607,261,294,15089]`.
- **`clone()`, not `contiguous()`.** A dim-0 slice of a contiguous tensor is already
  contiguous, so `[gi].contiguous()` returns the view unchanged and all ten "buffers"
  still share one storage. That happened to work — so the dropped-write behaviour is
  keyed on graph-input *tensor* identity, not on storage — but it makes correctness
  rest on the compiler never canonicalising ten views of one base into one input. The
  clone variant has no such dependency and measured 0.26% faster, i.e. the same.
- **Input aliasing structure is part of the graph key.** Swapping views for clones
  changed nothing about the traced ops, yet forced a full cold recompile (8
  `neuronx-cc` processes against a cache already holding the view-variant NEFFs).
  Worth knowing before assuming an allocation-only change will be cache-hot.

#### Gated 2026-08-05: decode is correct now, and it was never the fast path it claimed

Same fix, re-gated at 40 layers, BS=128, seq=256, `--optlevel 1`, beta-4,
`GQA_STATEFUL_KV=1 DECODE_FULLGRAPH=1 DECODE_KV_MAP=1`:

| | `per_group_nz` | `per_group_nonzero_rows` | TPOT | tok/s |
|---|---|---|---:|---:|
| before (one shared buffer) | `[98304, 0 × 9]` | `[384, 0 × 9]` | 395.95 ms | 323.3 |
| **after (per-layer buffers)** | **`[98304 × 9, 98176]`** | **`[384] × 10`** | **393.56 ms** | **325.2** |

All ten GQA layers now write, `384 = 128 batch × 3 tokens` uniformly (the one 98,176
is 128 genuine bf16 zeros in the data, not missing rows).

**Unlike prefill, this fix buys no throughput — and that is the expected result.**
393.56 vs 395.95 ms/token is 0.6%, i.e. noise. The attention kernel does dense masked
work over the whole `S=256` buffer regardless of what is *in* it, so nine caches full
of zeros cost exactly what nine correct caches cost. The defect was pure lost
correctness with no compensating speed, which is also why no profile ever flagged it.

**`gen hash` was identical on both sides** (`0cc59fb25112`, rows 0 and 127) — before
and after a fix that changed nine of ten attention inputs from zeros to real keys and
values. Under `--skip-prefill` only 2–3 tokens are generated from a seed token at
position 0, and greedy argmax simply does not resolve the difference. This is the
same insensitivity that let the defect live for two and a half weeks, now measured
from the other direction: **a matching `gen hash` is not evidence of equivalence.**

What this does *not* do is restore the withdrawn `+5.5%`. That claim (105.31 → 99.80
ms/token) was measured at BS=32 on the older container, and has **not** been
re-measured on a correct path; it should be treated as unknown, not as recovered. The
datum above is the first honest number for stateful-KV decode.

#### Measured 2026-08-05: what the aliasing rule actually is (two earlier claims retracted)

The fix above is correct and gated, but the *explanation* attached to it was wrong in
two places. Both were inherited hypotheses that had never been measured; writing the
probes to actually assert on device produced the following, on the beta-4 container at
`--optlevel 1`, one rank, tiny standalone graphs (probe files listed per row):

| form | writes landed | file |
|---|---:|---|
| 1 mutation, whole-tensor reshape | 1/1 | `test_gqa_rope_kv_alias_probe.py` |
| 1 mutation, **sliced view** | **1/1** | `test_gqa_rope_kv_alias_probe.py` |
| N mutations, N separate tensors (**shipped**) | 3/3, 10/10 | multicall, tail_stateful |
| N mutations, **N distinct views of one base** | **0/3, 0/10** | multicall, tail_stateful |
| N mutations, *identical* whole-tensor view | 3/3, 10/10 | multicall, tail_stateful |
| in-graph read, emitted **after** the op's return value | sees the write | multicall |
| in-graph read, emitted **before** the op's return value | **sees zeros** | multicall |

**Retraction 1: "a sliced view does not alias, so its write is lost" is false.** A
single sliced mutation lands, and always did — the pre-2026-07-30 prefill path mutated
`kv_k[gi, :, 0].reshape(…)` and filled the cache 100% (table above). The hazard is
**sharing one base across several mutations**, not slicing. Note this is *harsher* than
the coarse rule, not milder: three distinct slices of one base lost **all three**
writes, not "all but the first".

**Retraction 2: "only the FIRST mutation per buffer per graph is carried back" is the
right instinct with the wrong mechanism**, and it over-promises. N mutations through the
*identical* whole-tensor view land at probe scale — which is exactly the form decode
shipped, and decode still lost 9 of 10 at 40 layers. So the probes cannot reproduce the
decode defect, and a green run on them is a floor rather than clearance; the gate remains
a `DECODE_KV_MAP=1` run. This is stated in the probe's own output so it cannot be
misread as a pass.

**New finding, and the one to be careful with: an in-graph read of a mutated buffer is
not ordered against the mutation.** Two byte-identical bodies differing *only* in graph
output order disagree — digest emitted before the op's own return value reads
pre-mutation zeros, emitted after it reads the write. Prefill's
`k_filled = kv_k[gi][:, 0]` is in the working order, and its logits are bit-exact
against the path that consumed the kernel's returned cache, so it is correct today; but
nothing in the source enforces it. Do not copy that read pattern without an occupancy
gate.

The practical rule is unchanged and now has a measured basis: **one distinct tensor per
mutation**, and gate on `PREFILL_KV_MAP=1` / `DECODE_KV_MAP=1` occupancy.

#### Root cause, and the structural fix (2026-08-05): the mutation was never in the graph

Everything above characterises a behaviour without explaining it. The mechanism, traced
through the installed `torch_neuronx` and then measured:

`dconfig.operand_output_aliases` (`nki_kernel.py:172-178`) is inverted from the NKI
compiler's `result.input_output_aliases`, and **NKI populates that map only for inputs
the kernel RETURNS**. Our kernels deliberately did not return the caches, so the map was
empty, so the `ctx.replace` / `mark_mutation_hidden_from_autograd` / `commit_update` /
`sync` block at `nki_hop.py:391-398` never ran. `mutates_args` on `@nki_op` shapes only
the *PyTorch-level* schema; it never reaches the emitted custom call. So the in-place
write had **no representation in the emitted graph at all** — writes landing was the
backend happening not to reorder or DCE them, never a contract. That is the whole
explanation for "sometimes lands, sometimes doesn't, scale-dependent".

Isolated in `kernels/tests/probe_kv_alias_f4.py` — two micro-kernels with byte-identical
write bodies, differing only in whether the kernel returns its mutated input:

| kernel | `operand_output_aliases` | writes landed | HBM | wall |
|---|---|---:|---:|---:|
| does not return the cache | `{}` | **0/3** | 0.002 GB | 1.00× |
| returns the cache | `{0: 1}` | **3/3** | 0.002 GB | 0.94× |

**An aliased return does not materialize the buffer** — it *is* the input buffer, so
there is no allocation and no copy. The "1.74 GB per region" cost the 2026-07 comments
warned about applies to an *unaliased* output, and those comments are now corrected in
place. The secondary finding also has its mechanism: reading the cache back through the
op's *returned* value is ordered against the write, whereas reading the caller's own view
is not — the prefill probe's read-before-return-value arm moved 0/3 → 3/3.

Both ops now return their mutated caches (`gqa_rope_kv_35b.py`, `gqa_tail_35b.py`), and
prefill reads the cache back through those returns. Gated on device at full depth:

| | prefill (BS=2, N=20000, bucket 1024) | decode (BS=128, seq=256) |
|---|---|---|
| before | 9991.0 ms / **4003.6 tok/s** | TPOT 392.87 ms / **325.8 tok/s** |
| aliased returns | 10140.9 ms / **3944.4 tok/s** (−1.5%) | TPOT 397.14 ms / **322.3 tok/s** (−1.1%) |
| numerics | fingerprint bit-identical | `gen hash 0cc59fb25112` identical |
| KV occupancy | all 10 groups full | `per_group_nonzero_rows=[384]×10` |
| HBM | 8.95 GB/core (unchanged) | 7.09 GB/core (unchanged) |

So the correctness contract costs **~1%**, at the edge of the run-to-run spread on these
configs (prefill has landed at 3991.0 / 4002.1 / 4003.6 across identical builds), and
costs **nothing** in memory — which is the point: it buys a modelled mutation in place of
a lucky one. Note what this does *not* do: it does not make sharing one base across N
mutations safe. The probes' shared-base arms now all land (0/10 → 10/10 on the decode
one), so those files no longer discriminate, and the occupancy gate is the only remaining
check. The per-layer buffers stay.

#### Gated 2026-08-06: per-layer DeltaNet state buffers (`DN_PERLAYER_STATE=1`) — +96% decode, and it beats the published O2

The GQA per-layer-buffer fix above (one distinct buffer per mutation) was applied to
the **DeltaNet recurrent-state writeback** as well, behind `DN_PERLAYER_STATE=1`
(requires `DN_DIRECT_STATE_OUT=1`). The old path wrote every DeltaNet layer's new state
into one shared `[NUM_DELTANET, B, …]` base tensor via a self-copy at
`static_decode_35b.py:~1855`; with ~30 DeltaNet layers in one full-graph decode step
that is ~30 write-after-write dependencies serialized through a single buffer. The fix
gives each layer its own `torch.empty_like` buffer (held in a Python list, mutated
once, `torch.stack`ed at return) — the `layer_index=0, num_groups=1` analogue of the
GQA fix, but here the payoff is **throughput** (the WAW chain is gone), not correctness.

40 layers, BS=128, seq=256, `--optlevel 1`, beta-4, `GQA_STATEFUL_KV=1
DECODE_FULLGRAPH=1 DN_DIRECT_STATE_OUT=1`, TP=8/LNC=1, FP8 `block_ob_coalesced` MoE +
tiled DeltaNet conv — the same flag set as the O1 baseline, only `DN_PERLAYER_STATE`
added:

| | TPOT | tok/s | gen hash (row0 / row127) |
|---|---:|---:|---|
| baseline (shared-base state) | 397.45 ms | 322.1 | `7f4b446344cf` |
| **+ `DN_PERLAYER_STATE=1`** | **202.54 ms** | **632.0** | `7f4b446344cf` |

**+96.3%** throughput at O1, reproduced across two independent runs (a separate
2-token bench measured 202.11 ms / 633.3 tok/s). This does not merely recover the
O1→O2 penalty — at O1 it is **+43% over the published O2 decode headline (442.1
tok/s)**, so shipping beta-4 at O1 + `DN_PERLAYER_STATE` publishes a decode
*improvement*, not the −26.7% O1 regression. The trn1 O2 cross-compile (below) becomes
optional upside — O2 stacked on this lever could go higher — rather than the critical
path to a non-regressing decode headline.

**Numerics.** The 8-token A/B (7 recurrent-state feedbacks — a stronger gate than the
2–3 token skip-prefill runs that the aliasing sections above showed are insensitive)
was **bit-identical**, `gen hash 7f4b446344cf` on rows 0 and 127 for both arms. The
DeltaNet-state occupancy gate (`DN_OCCUPANCY_CHECK=1`, asserts each per-layer state has
≥1 non-zero row) passed on **both** arms, so — unlike the GQA cache — the shared-base
DeltaNet writeback was *not* dropping writes; this is a pure ownership/serialization
change with identical arithmetic, which is why bit-identical is the expected and
observed result. Per the discipline established above, greedy `gen hash` is treated as
a floor, not proof; the occupancy gates plus the pure-ownership nature of the refactor
are what carry the equivalence claim. KV occupancy was `[384]×10`.

### beta-5 container (2026-08-10): both paths faster, numerics bit-identical

The **beta-5** DLC (built 2026-08-05T10:02Z — **6 days newer** than beta-4; both
digests are in the gitignored `.env`) was pulled
onto the native trn2.3xlarge and both published configs re-run against it with the
**same source** (per-GQA-layer rope-KV for prefill; O1 + `DN_PERLAYER_STATE=1` for
decode). Both are clean compiler-only wins with no numerical change. `ecr:DescribeImages`
is denied to the host instance role, so `:latest` was confirmed newer via
`docker inspect … .Created`; the pull deduped almost entirely against beta-4's layers.

| Path | beta-4 | **beta-5** | Δ | correctness |
|---|---:|---:|---:|---|
| Prefill TP=8/LNC=1 pg256, 40L BS=2 N=20000 | 4,002.1 tok/s (9994.6 ms) | **4,097.9 tok/s** (9761.1 ms) | **+2.4%** | fingerprint `sum=-3.17835375e+05 norm=1.20818213e+03 top5=[517,607,261,294,15089]` **exact**, warm≡timed, `kv_k` 104857597/104857600 |
| Decode BS=128 O1 `DN_PERLAYER_STATE=1`, seq=256 | 632.0 tok/s (202.54 ms) | **681.3 tok/s** (187.87 ms/tok) | **+7.8%** | gen hash `0cc59fb25112` row0/row127 (== beta-4 2-token ref), `DOCKER_EXIT=0` |
| Decode BS=128, seq=1024 (generated 1024 tokens) | — | **582.3 tok/s** (219.83 ms/tok) | — | `DOCKER_EXIT=0`; module 8.10 GB/core |

**Prefill reproduced** across two runs (4,097.9 / 4,095.2, within 0.07%), fingerprint
bit-identical both times.

**Two caveats.**
1. **Prefill aborts on teardown** with glibc `corrupted size vs. prev_size` → SIGABRT on
   one rank (`DOCKER_EXIT=1`) **after** both warmup and timed results + fingerprints have
   printed. Reproduced on both prefill runs; decode never crashed (`DOCKER_EXIT=0`). The
   measurement is valid — the abort is a beta-5 shutdown/heap issue, not a benchmark
   failure. Read the `PREFILL TIMED` line, not the exit code.
2. **Decode compiled fast** on beta-5: "first decode step (incl compile): 65–86 s" vs
   beta-4's O1 24m21s, with the NEFF cache still not persisting (`ccache` stayed empty —
   see the cache-transplant notes). A large apparent compiler speedup, but observed only
   as a side effect of these two runs (with an unexplained ~10-min gap before the BENCH
   print) — not yet measured in isolation, so treat it as an observation, not a claim.

**seq-length cost.** Going seq=256 → 1024 costs −14.5% throughput / +17% TPOT because GQA
reads the full `max_seq_len` KV cache each decode step, so per-step cost scales with
`max_seq_len` (not with tokens generated so far); the module also grows 7.09 → 8.10
GB/core. The seq=1024 run generated 1024 tokens; over that many greedy steps the row0 and
row127 gen hashes diverge (`6c3b3a1909bd` / `554aa924beb3`), which is expected for long
from-zeros greedy decode (per-row FP differences in the fp8/batched GEMMs compound into
different argmax picks) and not a regression — the seq=256/2-token rows stay identical.

## Reference

The validated NxDI implementation
(`aws-neuron/neuronx-distributed-inference` PR #60,
`jimburtoft:contrib/qwen3.5-35b-a3b`, torch-xla) is the correctness oracle: 100%
token-match vs CPU, BS=1 54.3 tok/s / 18.4 ms/tok on the same hardware. Its MoE uses
an NxDI library module and is not portable, which is why this implementation carries
its own MoE kernels and CPU oracle.

### vLLM-Neuron port (different regime — not a ranking)

A separate vLLM-Neuron port of this model exists and was validated by its authors
(a local `vllm-neuron` checkout, `origin/add-qwen36-moe` @ `65ef8b7`). Its measured
numbers, for reference:

| Metric | vLLM-Neuron port | Config |
|---|---:|---|
| GSM8K-CoT exact-match (flexible) | 93.0% ± 2.6% | BS=1, 100 q, 4-shot |
| Output throughput @ concurrency 1 | 22.57 tok/s | 256-in / 128-out |
| Output throughput @ concurrency 4 | 74.37 tok/s | 256-in / 128-out |
| Output throughput @ concurrency 8 | 123.78 tok/s | 256-in / 128-out |

**These rows are not comparable to the numbers above and must not be read as a
ranking.** They were measured on a **trn2.48xlarge** (16 chips / TP8/EP8) versus a
single-chip trn2.3xlarge here; at `max_model_len=1024` with ≤512-token GDN prefill
segments versus our 20,000-token prefill; and under continuous batching versus our
fixed batch. In particular our prefill headline (3,456.8 agg prompt tok/s at 20k
tokens) has **no counterpart** in that port, whose context is capped at 1,024.

Full analysis — file inventory, reproduction blockers, GDN kernel comparison, seam
analysis, and a numerical finding in their draft chunked prefill kernel — is in
**[VLLM_NEURON_ASSESSMENT.md](VLLM_NEURON_ASSESSMENT.md)**.
