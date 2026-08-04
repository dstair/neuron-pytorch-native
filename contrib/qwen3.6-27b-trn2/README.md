# Qwen3.6-27B on Trainium2 (PyTorch Native)

A PyTorch-Native inference implementation of the dense **Qwen3.6-27B** model on a
single Trainium2 device (`trn2.3xlarge`, TP=4, LNC=2, bf16). The whole decode step
compiles to a single NEFF via `torch.compile(fullgraph=True, backend="neuron")`,
with custom NKI kernels for the DeltaNet recurrence and the GQA attention tail.

## Architecture

```
64 layers = [DeltaNet × 3, GQA × 1] × 16   (full-attention every 4th layer)
hidden 5120, intermediate 17408, vocab 248320
RMSNorm eps 1e-6 with (1 + weight) scaling; RoPE partial-64 @ theta 1e7

DeltaNet (48 layers): 16 key heads, 48 value heads, k/v dim 128
  in_proj_qkv [5120,10240] = q(2048)||k(2048)||v(6144); depthwise conv1d k=4
  in_proj_z [5120,6144]; in_proj_a/b [5120,48]; A_log/dt_bias [48]; out_proj [6144,5120]
  recurrence: state *= exp(g); state += outer(k, delta); out = state.T @ q

GQA (16 layers): 24 Q-heads (+ output gate), 4 KV-heads, head_dim 256
  q_proj [5120,12288] (query||gate); k/v_proj [5120,1024]; o_proj [5120,6144]
  per-head q_norm / k_norm (RMSNorm)

MLP (all 64 layers): SwiGLU, gate/up [5120,17408], down [17408,5120]
```

Sharding is manual and per-core (colwise for q/k/v/gate/up, rowwise for the
output/down projections) with an all-reduce after each `out_proj` and `down_proj`.
See the sharding helpers in `static_decode.py`.

**TP=4 and TP=8 are both supported** (`TP_DEGREE`, default 4). TP=8 is what LNC=1
requires — see [TP=8 / LNC=1](#tp8--lnc1-8-logical-cores-of-12-gb) below. Every
per-core dimension is derived from `TP_DEGREE` rather than hardcoded:

| | TP=4 (LNC=2) | TP=8 (LNC=1) |
|---|---|---|
| DeltaNet key / value heads | 4 / 12 | 2 / 6 |
| DeltaNet per-core QKV dim | 2560 (= 20×128) | 1280 (= 10×128) |
| GQA query heads | 6 | 3 |
| GQA KV heads | 1 | 1 (**replicated ×2**) |
| MLP per-core dim | 4352 | 2176 |

The one dimension that does not divide by 8 is `num_key_value_heads = 4`, so past
TP=4 whole KV heads are **replicated** across rank groups instead of split
(`shard_linear_colwise_replicated`) — splitting a 256-dim head across two ranks
would break attention, since `q·k` contracts over all 256 dims and `k_norm` is
per-head. The row-wise `o_proj` all-reduce stays exact under replication because the
*query* heads remain disjoint: the reduce sums over a partition of q heads, so a
replicated KV head cannot be double-counted.

## Layout

```
static_decode.py        the static decode/prefill forward (all 64 layers), the compile
                        harness, manual TP sharding, and the benchmark entry point
chunked_prefill.py      Neuron-compatible chunked DeltaNet prefill reference (matches HF)
kernels/                NKI kernels + their torch.ops registrations (*_ops.py)
  deltanet_full*        fused DeltaNet inner block (conv + gates + recurrence + gated norm)
  deltanet_chunked_v2*  chunked-prefill DeltaNet kernel (makes prefill compilable)
  deltanet_full_batched*  batched-decode DeltaNet (BS>1 throughput; v2 = DMA-coalesced)
  gqa_tail*             fused GQA attention tail (norm + RoPE + scores + softmax + gate)
  fp8_matmul*           FP8 W8A16 matmul kernel
  rms_norm_nki*         RMSNorm kernel
  tests/                CPU / simulator correctness oracles + device smoke tests
deploy/                 vLLM-Neuron serving scripts (baseline, chunked-prefill, EAGLE3)
```

## Running

The decode/prefill harness runs inside the PyTorch-Native container on a Trainium2
box. It is launched with `torchrun` across the 4 logical cores:

> **The container does not ship `transformers`.** Unlike the 35B harness (which reads
> safetensors directly), this one loads the checkpoint via HF
> `Qwen3_5ForConditionalGeneration.from_pretrained` and then shards it. The Neuron
> DLC has `torch`, `torch-neuronx`, `neuronx-cc`, and `safetensors` but **no
> `transformers`**, so a bare run dies with `ModuleNotFoundError: No module named
> 'transformers'`. Install it once into a host directory and mount it on
> `PYTHONPATH` rather than baking a new image:
>
> ```bash
> docker run --rm -v /mnt/nvme/tmp/pysite:/pysite "$QWEN27_VLLM_IMAGE" \
>   python3 -m pip install --target /pysite transformers    # 5.14.1 validated
> # then add to every run:  -v /mnt/nvme/tmp/pysite:/pysite:ro -e PYTHONPATH=/pysite
> ```
>
> `transformers` **5.14.1** has `Qwen3_5ForConditionalGeneration` and parses this
> checkpoint's `model_type: qwen3_5` config (the checkpoint itself records
> `transformers_version: 4.57.1`). It also needs the neuron devices visible at
> import time (`--privileged` + `-v /opt/aws/neuron:/opt/aws/neuron:ro` +
> `LD_LIBRARY_PATH=/opt/aws/neuron/lib`), because `torch` auto-loads the
> `torch_neuronx` backend extension on import and hard-fails without a device.

```bash
# Decode benchmark, full 64 layers, BS=1
torchrun --nproc-per-node=4 static_decode.py --num-layers 64 --max-seq-len 2048 --batch-size 1

# BS=8 throughput with the banked-best decode flags
DNBATCHED_V2=1 GQATAIL=1 \
  torchrun --nproc-per-node=4 static_decode.py --num-layers 64 --batch-size 8

# Compiled chunked prefill. --tiny is required: --prompt-len is only read in tiny
# mode (static_decode.py:1448), so without it you silently prefill the 5-token real
# prompt and --compile-prefill rejects it (S must be a multiple of CHUNK_SIZE=64).
# Tiny mode keeps full width and --num-layers layers but randomizes weights, which
# is sound here — prefill TTFT is structural, independent of weight values.
CHUNKEDPREFILL=1 \
  torchrun --nproc-per-node=4 static_decode.py --num-layers 64 --tiny --prompt-len 128 --compile-prefill

# TP=8 / LNC=1 (8 logical cores of 12 GB). T1b is mandatory here, not optional:
# a replicated vocab needs ~11.1 GB of the 12 GB ceiling.
NEURON_LOGICAL_NC_CONFIG=1 TP_DEGREE=8 SHARDED_EMBED=1 SHARDED_LM_HEAD=1 \
  DNBATCHED_V2=1 GQATAIL=1 \
  torchrun --nproc-per-node=8 static_decode.py --num-layers 64 --batch-size 8
```

`TP_DEGREE` must match `--nproc-per-node`; the harness raises at startup if it does
not, rather than sharding for one world size and running on another.

Useful flags: `--tiny` (small config for quick iteration), `--num-layers N`,
`--skip-prefill`, `--fp8-weights`, `--batch-size N`, `PROFILE_STEPS=N` (emit a
device profile over N decode steps).

### Optimization levers (environment flags)

All are **default-off**; the baseline path is byte-identical without them.

| Flag | Effect |
|---|---|
| `CHUNKEDPREFILL=1` | Compiled chunked-DeltaNet prefill (~82× faster prefill compile) |
| `DNBATCHED_V2=1` | DMA-coalesced batched-DeltaNet decode kernel (BS>1) |
| `DN_TILED_CONV=1` | Tile-partition-major DeltaNet conv-state layout (**BS=8 any TP; BS=16 only at TP=8** — regresses 37% at BS=16/TP=4, see T1a; requires `DNBATCHED_V2=1` and `--batch-size >= 2`) |
| `DN_CONV_GROUP=G` | 128-aligned group-tiled Phase-1 conv, `G` items per pass (**prototype, ungated** — requires `DN_TILED_CONV=1`) |
| `GQATAIL=1` | Fused GQA attention-tail mega-kernel |
| `NORMFUSE=1` | NKI RMSNorm |
| `SHARDED_EMBED=1` | Row-shard the replicated `embed` table across TP (HBM capacity) |
| `SHARDED_LM_HEAD=1` | Row-shard the replicated `lm_head` weight across TP (HBM capacity) |
| `LEANKV=1`, `NOREDUCE=1`, `DNF32STATE=1` | Reference levers, ruled out (kept for A/B) |

`SHARDED_EMBED` / `SHARDED_LM_HEAD` are the **T1b** transfer from the 35B
(`PREFILL_SHARDED_*` there). `embed` and `lm_head` are the only weights TP leaves
replicated — 2.54 GB bf16 each at V=248320, H=5120 — so together they are ~5.1 GB of
the ~17.3 GB/core module. Each rank keeps a disjoint contiguous vocab range; the
replicated result is reconstructed **exactly** (masked gather + sum-all-reduce for
embed; local matmul + zero-pad + sum-all-reduce for the head), so this is a pure
*capacity* lever with no numerical cost. `kernels/tests/test_decode_sharded_lm_cpu.py`
proves the reduction bitwise-equal at ws=1/2/4/8 in fp32 and bf16.

One asymmetry worth knowing: `SHARDED_LM_HEAD`'s pre-reduce padded tensor is
`[B, 1, V]` and grows linearly with batch (0.5 / 4.0 / 7.9 MB per rank at BS=1/8/16),
while `SHARDED_EMBED` has no padded intermediate at all. If only one is wanted on a
high-batch decode, take `SHARDED_EMBED`.

## Performance summary

All numbers are `trn2.3xlarge`, TP=4, LNC=2, bf16 — **except the
[TP=8 / LNC=1](#tp8--lnc1-8-logical-cores-of-12-gb) section, which is the current
best decode configuration** — measured with a
`torch.neuron`-synchronized timer (async-enqueue artifacts excluded). "PyTorch
Native" = this repo's `static_decode.py` (single compiled NEFF). "XLA" reference
points are the [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)
implementation of the same model on the torch-xla stack, for comparison.

### Decode (TPOT / throughput)

| Test | Framework | BS | TPOT (ms, synced) | Throughput (tok/s) |
|---|---|---|---|---|
| Decode-only, `static_decode.py` | PyTorch Native | 1 | 35.9 | 28.2 |
| Decode-only, `DNBATCHED_V2` + `GQATAIL` | PyTorch Native | 8 | **127.6** | **62.7** |
| Decode-only, `DNBATCHED_V2` + `GQATAIL` | PyTorch Native | 16 | 253.5 | 63.1 |
| Decode, NxDI (single-stream, published) | XLA | 1 | 54.2 | 18.5 |
| Decode, NxDI offline `llm.generate` | XLA | 8 | 361 | 22.2 |
| Decode, NxDI offline `llm.generate` | XLA | 16 | 525 | 30.5 |

- BS=1 synced steady-state TPOT is **35.9 ms**; the realistic host-synchronized
  greedy loop (a `.item()` D2H per token) is ~43.6 ms.
- BS=8 improved 55.6 → 58.8 → 62.7 tok/s across the batched-DeltaNet
  (`DNBATCHED_V2`) and GQA-tail (`GQATAIL`) kernels.
- Throughput plateaus ~63 tok/s: 8→16 adds only ~0.4 tok/s (per-step work grows
  nearly linearly with batch; see the barrier-bound analysis below).
  **⚠ Retracted-pending-remeasurement:** both figures come from the `synchronize()`
  timer, which is demonstrably wrong at BS=16 (see "The `synchronize()` timer is
  unreliable at BS=16"). On beta-4 the two agreeing end-to-end loops show 8→16
  scaling +32%, not flat.
- The XLA/NxDI rows are not a strict head-to-head (different serving overhead and
  the offline path carries per-step Python scheduler + sampling cost) — treat them
  as an independent reference, not an A/B against the Native numbers.

#### Re-baselined on the beta-4 container (2026-08-04)

The table above was measured on the earlier PyTorch-Native DLC. Re-running the same
three configurations unchanged on the **beta-4** image
(`concourse-release-0461d3b:latest`, digest `sha256:ad7f7bbcd468…` — the one that gave
the 35B **+12.5%** prefill for free) shows the container is **not** a uniform win here:

| BS | TPOT (ms, synced) | tok/s synced | vs table above | tok/s host-sync |
|---:|---:|---:|---|---:|
| 1 | 39.8 | 25.1 | **−11.0%** | 19.8 |
| 8 | **100.1** | **79.9** | **+27.4%** | 70.1 |
| 16 | 483.0 ⚠ | 33.1 ⚠ | timer artifact — see below | **92.4** |

- **BS=8 is the headline: 62.7 → 79.9 tok/s (+27.4%)**, TPOT 127.6 → 100.1 ms, from a
  compiler change alone. (Since superseded at BS=8 by `DN_TILED_CONV=1` — see T1a below.)
- **BS=1 regresses 28.2 → 25.1 tok/s (−11.0%).** The container helps the batched,
  barrier-bound step and hurts the latency-bound one — consistent with the
  barrier-bound analysis below, but it means "upgrade the container" is a
  per-configuration decision, not a global one.
- ⚠ **The BS=16 synced figure is a defect in the synced timer, not in the model.**
  Repeated three times (see below); use **92.4 tok/s** for BS=16.
- Numerics are unchanged across every run: all runs emit the identical
  greedy continuation for `'The meaning of life is'` →
  `' to find your gift. The purpose of life is to give it away.'`
- Module resident is **17,264 MB/core** replicated — the figure T1b is measured against.

#### ⚠ The `synchronize()` timer is unreliable at BS=16 (resolved 2026-08-04)

The harness reports four timings per run. Three of them agree everywhere; the fourth
breaks at BS=16. Values in ms/token, `DNBATCHED_V2=1 GQATAIL=1`, beta-4:

| Method | BS=1 | BS=8 | BS=16 (×3 runs) | C2 ÷ B |
|---|---:|---:|---:|---:|
| **A** per-token host sync (`.item()` D2H) | 47.93 | 114.06 | 172.86 / 173.51 / 172.97 | — |
| **B** on-device greedy, no per-step sync | 48.21 | 121.32 | 189.63 / 190.81 / 189.99 | — |
| **C1** no synchronize (old bench style) | 1.77 | 1.86 | 3.54 / 1.74 / 3.91 | async-enqueue artifact |
| **C2** + `torch.neuron.synchronize()` ← *the "synced TPOT" column* | 37.47 | 100.15 | 473.66 / 486.38 / 488.91 | **2.50 / 2.55 / 2.57** |

A and B are independent end-to-end greedy loops; they agree with each other to within
10% at every batch size and scale **smoothly and sublinearly** (A: 47.9 → 114.1 → 173.1,
i.e. 8→16 costs only +52% for 2× the tokens). C2 tracks them at BS=1 and BS=8 at a
stable 0.78–0.83× of B — the expected offset, since C2 skips the argmax/token-update
work. At BS=16 C2 alone jumps to **2.5× B**, deterministically: three runs landed at
473.7 / 486.4 / 488.9 ms (±1.6%), so this is **not** throttling and not a one-off.

**Verdict: C2 is the broken measurement at BS=16, and BS=16 decode is ~92.4 tok/s**
(mean of A over three runs; B gives the more conservative 84.1 tok/s). Root cause of
the C2 blow-up is not yet identified — the mechanism is in the timing loop, not the
step, since the same NEFF measured by two other loops is fine.

Two consequences worth flagging:

- **The "throughput plateaus ~63 tok/s" finding below is suspect.** It rests entirely
  on the C2 column (62.7 at BS=8 vs 63.1 at BS=16). Measured by A, beta-4 BS=8→16 scales
  **70.1 → 92.4 tok/s (+32%)** — no plateau. Note the old-container BS=16 C2 figure
  (253.5 ms) was already 1.34× B, so C2 was mildly off at BS=16 there too and got worse;
  but there are no A/B numbers from the old container, so whether the *old* stack
  plateaued is genuinely unknown. Re-measure with A/B before repeating the claim.
- **Any BS=16 A/B (e.g. T1a) must use method A or B**, not the "synced TPOT" line.

#### T1b: vocab-row sharding (measured, gate passed)

`SHARDED_EMBED` / `SHARDED_LM_HEAD` at BS=1, 64 layers, same beta-4 container:

| Config | Module (MB/core) | Freed | TPOT (ms) | tok/s | Continuation |
|---|---:|---:|---:|---:|---|
| replicated (baseline) | 17,264 | — | 39.8 | 25.1 | reference |
| `SHARDED_EMBED=1` | 15,357 | 1,907 (−11.0%) | 39.9 | 25.1 | **identical** |
| `SHARDED_EMBED=1 SHARDED_LM_HEAD=1` | **13,450** | **3,814 (−22.1%)** | **37.5** | **26.7** | **identical** |

- **The freed bytes match the arithmetic exactly.** Each `[248320, 5120]` bf16 tensor
  is 2,542.8 MB; TP=4 leaves 1/4 resident, so 1,907.1 MB per tensor should go. Measured
  deltas are 1,907 MB (embed only) and 3,814 MB (both). No hidden copy survives.
- **Correctness gate passes:** both sharded configurations emit a token-identical
  continuation to the replicated baseline, as the exactness argument requires.
- TPOT does not regress; with both flags it measured **37.5 ms vs 39.8** (+6.4%
  throughput). That is a single measurement, not a claimed latency win — the honest
  statement is "no regression, plausibly a small win from reduced HBM pressure on the
  head." Note the win comes from the `lm_head` half; `SHARDED_EMBED` alone is
  TPOT-neutral (39.9 vs 39.8).
- Both flags stay **default-off**. The capacity they return (~3.8 GB/core out of a
  24 GB budget) is the point: it is headroom for larger batch or longer context.

#### T1a: tiled conv-state layout (measured — the sign tracks `B × NT`, not batch size)

`DN_TILED_CONV=1` stores DeltaNet `conv_states` tile-partition-major
(`[NUM_DN, B, 128, NT, 3]`, element `[b, p, t, j]` = tap `j` of channel `t*128 + p`,
`NT = 2560/128 = 20`) instead of channel-major (`[NUM_DN, B, 2560, 3]`). That lets the
kernel move each batch item's whole 3-tap history in **one** DMA instead of one per
128-channel tile, collapsing ~48 conv-state DMAs per item to ~3. Ported from the 35B,
where it was ~+15% at BS=32.

**It does not transfer cleanly. The sign flips with batch size** — measured on beta-4,
64 layers, `DNBATCHED_V2=1 GQATAIL=1`, methods A/B (the "synced TPOT" line is unusable
at BS=16, see above):

| BS | `DN_TILED_CONV` | A (ms/tok) | B (ms/tok) | tok/s (host-sync) | Δ |
|---:|---|---:|---:|---:|---|
| 8 | off | 113.80 | 121.42 | 70.3 | — |
| 8 | **on** | **97.46** | **105.00** | **82.1** | **+16.8%** |
| 16 | off | 172.91 / 171.88 | 189.87 / 188.88 | 92.5 / 93.1 | — |
| 16 | **on** | 275.14 / 275.19 | 290.51 / 291.23 | 58.2 / 58.1 | **−37.3%** |

- **BS=8 is a clean win, agreed by all three timers** (A −14.4%, B −13.5%, C2 −17.0%).
  82.1 tok/s host-sync is the best measured 27B BS=8 decode.
- **BS=16 is a large regression, and it is real, not an artifact.** The BS=16 pair was
  run twice in *opposite* order (off→on, then on→off) specifically to separate the flag
  from thermal/ordering drift after ~45 min of continuous device load. The regression
  follows the **flag**, not the position, and both configurations reproduce to within
  0.6% across the two passes.
- **Numerics are unaffected in every leg.** All four legs — and both BS=16 repeats —
  emit the identical greedy continuation
  (`' to find your gift. The purpose of life is to give it away.'`). This is a
  performance cliff, not a correctness failure.

**Gates that passed.** The device kernel gate runs the same `cpu_reference` against both
layouts (feeding the tiled buffer, un-tiling the result), and the two layouts produce
**identical max-diffs to the last digit** at both batch sizes — B=8: state `0.000039`,
conv_state `0.000000`, output `0.000977`; B=16: `0.000053` / `0.000000` / `0.007812` —
which is strong evidence the layout change is bit-identical, as designed.
`kernels/tests/test_tiled_conv_layout_cpu.py` gates the
layout algebra device-free (16 checks: round-trip bitwise exactness, the
`channel = t*128 + p` mapping over the full grid, a negative check that the transposed
mapping is genuinely different, and that the q/k/v slot boundaries all land on tile
boundaries). Both flag-off legs reproduced their baselines exactly, so the edits left
the default path untouched.

**The single-call kernel microbenchmark cannot explain the E2E result — don't use it.**
`test_deltanet_full_batched.py` reports 0.110 → 0.101 ms/call at B=8 (−8%) and
0.106 → 0.112 at B=16 (+6%): the right *signs*, but the magnitudes are nowhere near the
E2E −14%/+59%, and B=8 and B=16 cost the *same* ~0.11 ms, which is only possible if the
loop is bound by per-call dispatch rather than kernel work. Treat this test as the
numerics gate it is, not as a timer. (An earlier run of it reported 0.174 → 0.115 at
B=8; that −34% was launch-noise and should not be cited.)

**The sign tracks `B × NT`, not batch size alone.** The TP=8 legs settled this. `NT` is
the per-core 128-channel tile count (`QKV_DIM / 128`), so it is set by TP: 20 at TP=4,
10 at TP=8. Every measurement to date, with `B × NT` as the ordering key:

| `B × NT` | Model / config | Δ |
|---:|---|---|
| 80 | 27B, B=8, NT=10 (TP=8) | +12.1% |
| 160 | 27B, B=8, NT=20 (TP=4) | +16.8% |
| 160 | 27B, B=16, NT=10 (TP=8) | **+17.1%** |
| 256 | 35B, B=32, NT=8 | +15–19% |
| 320 | 27B, B=16, NT=20 (TP=4) | **−37.3%** |

So the cliff sits between **256 and 320**, and `B=16` is not the trigger — `B=16` at
`NT=10` is a clean win. Halving `NT` flipped the sign at fixed batch size, which is the
controlled experiment the earlier TP=4-only data could not provide.

*Two corrections to what this section previously claimed.* The 35B row is a **weaker**
counterexample than first treated: its `DN_TILED_CONV` numbers were measured with
`MOE_FUSED_W8=fp8` (`../qwen3.6-35b-a3b/DECODE_RECIPE.md` §2), so an FP8-MoE model's
SBUF/PSUM budget is not this bf16 kernel's. And it is `NT=8` there, not 16, because that
config runs `DN_K_HEADS=2 DN_V_HEADS=4`. Second, the SBUF-spill hypothesis above is
**not supported by the arithmetic**: the tiled branch's per-item fp32 tiles come to
roughly `56 × NT` bytes per partition, so the *losing* config holds ~18 KB of each
partition's 224 KiB — about 8% of SBUF. Whatever the mechanism is, plain capacity
exhaustion is not it. Both statements are retracted; the `B × NT` ordering is what the
data supports, and the mechanism remains open (task #13).

**Two NKI tiling-guidance violations, which is the actionable lever.** Checked against
`~/dev/NKIBootcamp/neuron-agentic-development/skills/neuron-nki-writing/SKILL.md`, whose
guidance is to use the full 128 partition dimension and keep Vector/Scalar free dims
≥64:

1. **Phase-1 free dim is `NT × 3` = 30 at TP=8** (60 at TP=4) — below the ≥64 minimum,
   so the wide `tensor_tensor`/`tensor_reduce`/`activation` calls run at a fraction of
   the vector engines' efficient width.
2. **`mq_raw` is `[NT, PMAX]`** (`deltanet_full_batched_v2.py:169`) = `[10, 128]` at
   TP=8, so its `nc_transpose` feeds **10 of 128 partitions** — 92% of the partition
   dimension idle on that op.

Neither is a batch-size effect, which is why they are worth fixing independently of #13.
`DN_CONV_GROUP` (below) is the prototype that addresses both.

**Status: default-off. Enable at BS=8 (any TP), and at BS=16 only at TP=8** — where it
is the largest single decode win measured on this model (+32.7%). Do not enable at
BS=16/TP=4.

#### T1a follow-up: 128-aligned group-tiled conv (`DN_CONV_GROUP`) — prototype, ungated

`DN_CONV_GROUP=G` (requires `DN_TILED_CONV=1`) runs Phase-1 across `G` batch items on
the free axis instead of one item at a time, so the conv tiles become `[128, G×NT, 3]`
and the `mq` transpose becomes `[G×NT, 128] → [128, G×NT]`. That targets both violations
above directly: free dim 30 → 120, transpose partitions 10 → 120.

`nc_transpose` caps moving partitions at 128 and `B` must divide by `G`, so the kernel
picks `G` at trace time as the largest divisor of `B` with `G×NT ≤ 128` (G=12 at
BS=16/NT=10 → 120; G=8 at BS=8/NT=10 → 80), falling back to the per-item tiled path if
no divisor ≥2 exists. Phases 2 and 3 are untouched — they already run full 128-partition
column ops per head, and simply slice this group's `qkv_act`. The HBM buffer layout is
unchanged from `DN_TILED_CONV=1`, so there is no harness plumbing change.

**Unmeasured.** No numerics gate and no device A/B yet (task #15); the design, the `G`
table, and the gating plan are in `PLAN_TIER1_TRANSFER.md`. The open question is whether
Phase-1 width was ever the bound — if the `B × NT` cliff is DMA pipelining rather than
vector width, this will be flat, and that null result is itself the answer to #13.

#### TP=8 / LNC=1: 8 logical cores of 12 GB {#tp8--lnc1-8-logical-cores-of-12-gb}

Everything above this section runs at **TP=4 / LNC=2** — 4 logical cores of 24 GB.
LNC=1 splits the same chip into 8 cores of 12 GB, which forces TP=8. Measured
2026-08-04, beta-4 container, 64 layers, `--max-seq-len 2048`.

**Both sides of every A/B carry `SHARDED_EMBED=1 SHARDED_LM_HEAD=1` (T1b).** That is
not a preference: TP=4 needs **13,450 MB/core** *with* T1b already, so it cannot run
under a 12 GB ceiling at all, and TP=8 without T1b needs ~11.1 GB of 12 GB. Putting
T1b on the controls is what makes this a measurement of TP/LNC rather than of T1b.

| Config | HBM/core | A) host-sync ms/tok | B) on-device ms/tok | tok/s (host-sync) | tok/s (synced) |
|---|---|---|---|---|---|
| BS=1, TP=4/LNC=2 | 13,450 MB | 47.61 | 48.09 | 21.0 | 26.7 |
| BS=1, TP=8/LNC=1 | **6,768 MB** | **44.06** | **43.62** | **22.7** (+8.1%) | **29.7** (+11.2%) |
| BS=8, TP=4/LNC=2 | 13,450 MB | 110.61 | 117.43 | 72.3 | 82.5 |
| BS=8, TP=8/LNC=1 | **6,768 MB** | 97.51 | 105.75 | **82.0** (+13.4%) | **94.2** (+14.2%) |
| BS=8, TP=8/LNC=1 **+ T1a** | **6,768 MB** | **86.94** | **95.74** | **92.0** (+27.2%) | **107.0** (+29.7%) |
| BS=16, TP=4/LNC=2 | 13,450 MB | 169.69 | 185.76 | 94.3 | 59.4 ⚠ |
| BS=16, TP=8/LNC=1 | — | — | — | **OOM at NEFF load** | — |

`DNBATCHED_V2=1 GQATAIL=1` on all BS=8 and BS=16 rows. Every leg that ran emits the
identical greedy continuation, so none of this moves numerics. ⚠ the BS=16 "synced"
figure is the known-broken C2 timer — it reports 59.4 tok/s *below* the same run's
94.3 host-sync figure, which is impossible; read BS=16 off method A.

**BS=16 does run at TP=8 once `--max-seq-len` is halved to 1024**, which buys back the
headroom the replicated KV cache consumes. That makes BS=16 an apples-to-apples A/B at
both TP values, and it also rescues the T1a hypothesis test, since SBUF tile pressure
depends on `B` and `NT` and not on `max_seq_len`:

| Config (BS=16, `--max-seq-len 1024`) | HBM/core | A) host-sync ms/tok | B) on-device ms/tok | tok/s (host-sync) | Δ |
|---|---|---|---|---|---|
| TP=4 / LNC=2 | 13,450 MB | 159.48 | 174.77 | 100.3 | — |
| TP=8 / LNC=1 | **6,768 MB** | 140.68 | 155.10 | **113.7** | **+13.4%** |
| TP=8 / LNC=1 **+ T1a** | **6,768 MB** | **120.20** | **134.36** | **133.1** | **+32.7%** |

Two things follow. First, **LNC=1's decode win is remarkably stable at +13.4%** — the
identical figure at BS=8 (72.3 → 82.0) and BS=16 (100.3 → 113.7). Second, and more
important: **`DN_TILED_CONV` (T1a) *wins* at BS=16 when `NT` is 10, having lost 37.3% at
BS=16 when `NT` was 20.** 133.1 tok/s is the best 27B decode number measured to date at
any batch size or configuration.

That is the finding that reframes T1a — see the correction below.

**LNC=1 wins, and the prediction that it would not was wrong.** Going in, the stated
risk was that TP=8 doubles the participants in every `out_proj`/`down_proj`
all-reduce, and the barrier-bound analysis below says ~58% of a BS=8 step is
GLUE+SYNC — so more ranks on a collective-dominated step looked like a losing trade.
It is not, at either batch size. The reason appears to be that decode here is
weight-read bound before it is collective bound: TP=8 halves the per-core weight
traffic per step (13,450 → 6,768 MB resident, and the HBM halves almost exactly,
6,768×2 = 13,536), and that saving is larger than the added collective cost. The
lesson is the same one T1a and beta-4 already taught — *predicting the sign of a
Trainium lever from a profile attribution is unreliable; measure it.*

**T1a stacks, and the combination is the best 27B decode configuration measured:
133.1 tok/s host-sync at BS=16/TP=8** (`--max-seq-len 1024`), with 92.0 at BS=8/TP=8
versus 82.1 previously (T1a at TP=4, no T1b) and 72.3 for the TP=4 control. T1a's own
delta is +12.1% at BS=8/TP=8 against +16.8% at BS=8/TP=4 and +17.1% at BS=16/TP=8 —
i.e. the delta grows with `B × NT` right up to the cliff, which is consistent with the
lever being about conv-state DMA volume but *not* with `NT` alone setting the size of
the win.

**At `--max-seq-len 2048`, TP=8 cannot reach BS=16 at all.** It dies allocating a
62.5 MB `model code` block
(`TDRV:dmem_alloc_internal ... ret=-12`). The allocator breakdown, captured with
`NEURON_RT_ENABLE_DEVICE_MEMORY_OOM_LOG=1`:

```
ND 0 HBM 1   TOTAL 11.873GB | Model Code 35.6MB | Model Constants 5.2MB
             | Tensors 10.609GB | Shared Scratchpad 1.000GB | Runtime 26.1MB
             | DMA Rings (IO/Spill/Collectives) 7.8 / 135.4 / 0.3 MB
LNC size is 1. Neuron Cores using this HBM: NC 2 and NC 3
```

11.873 of 12 GB — genuinely full, and the runtime notes that "aligned allocations may
fail even if HBM has space … due to running out of alignment boundaries," so the
proximate failure is fragmentation at ~99% occupancy rather than a clean shortfall.

**`Tensors` is 10.609 GB against 6.6 GB of weights** — i.e. ~4 GB is batch-scaling
state, and TP=8 shrinks only part of it. DeltaNet states do shard with TP; the **KV
cache does not**. `torch.zeros(NUM_GQA, B, max_seq_len, GQA_HEAD_DIM)`
(`static_decode.py:1429`) holds one whole KV head per rank at *both* TP values, because
`num_key_value_heads = 4` forces replication past TP=4. Add the functional state
threading — `kv_cache_k.clone()` at lines 704/748 double-buffers every state tensor in
the graph — plus a fixed 1.0 GB shared scratchpad, and the headroom is gone. So the
cost of the KV-replication approach lands precisely where batch scaling needs room.
Worth noting too that at LNC=1 two NeuronCores share one HBM stack.

**LNC=1 is the better configuration on every axis measured, once you also pick the
context length.** It wins TPOT at every batch size, wins prefill at every sequence
length, uses half the resident HBM, and with T1a it holds the throughput record
(133.1 tok/s at BS=16). The one thing LNC=2 still buys is **BS=16 at a 2048 context**,
which LNC=1 cannot fit — so the trade is not TP against TP, it is *batch × context*
against *per-core headroom*, and the KV cache is what sets the exchange rate. If a
deployment needs both BS=16 and long contexts, that is the case for LNC=2 (or for
sharding the replicated KV head, which is still unexplored).

**One flag trap found while measuring this.** `--prompt-len` is read only in tiny mode
(`static_decode.py:1448`), so the prefill command previously documented in this README
silently prefilled the 5-token real prompt and then tripped `--compile-prefill`'s
multiple-of-64 guard. It failed identically at TP=4 and TP=8, which is how it was
identified as a flag error and not a port defect. The command above now includes
`--tiny`.

**What changed in the code.** Per-core dimensions are derived from `TP_DEGREE`
instead of hardcoded (see the table under [Architecture](#architecture)), the derived
head counts are exported to the NKI kernels through the environment *before* they are
imported — NKI needs them as Python constants for loop bounds and tile shapes, so
kwargs will not do — and GQA KV heads are replicated rather than split past TP=4.
`kernels/tests/test_tp8_sharding_cpu.py` gates all of it without a device (60 checks,
including a negative check that a plain column shard at TP=8 really would hand a rank
half a KV head, and a frozen assertion that the TP=4 constants are bit-for-bit the
ones every published number above was measured with).

### Prefill (TTFT / prompt throughput)

| Test | Framework | Config | Latency | Prompt tok/s |
|---|---|---|---|---|
| Prefill, compiled chunked-DeltaNet kernel | PyTorch Native | S=128, BS=1 | **83.7 ms** (warm) | **1530** |
| Prefill, same, **TP=8 / LNC=1** | PyTorch Native | S=128, BS=1 | **65.3 ms** (warm) | **1961** |
| Prefill, compiled, TP=4 / LNC=2 | PyTorch Native | S=512, BS=1 | 248.9 ms (warm) | 2057 |
| Prefill, compiled, **TP=8 / LNC=1** | PyTorch Native | S=512, BS=1 | **177.8 ms** (warm) | **2880** |
| Prefill, eager fallback (unoptimized) | PyTorch Native | S=128, BS=1 | 6847 ms | 19 |

Wiring the validated chunked-DeltaNet NKI kernel makes the prefill graph
compilable (one custom call/layer instead of an un-compilable data-dependent torch
chunk-loop), giving a **~82×** speedup over the eager fallback. Cold compile of the
prefill graph is ~391 s (one-time; NEFF cached after). Enabling chunked prefill
does not regress decode (BS=1 TPOT stays 36.7 ms).

**LNC=1 is worth +27.4% on prefill at S=128, and the win grows with sequence length —
+40.0% at S=512** (2,057 → 2,880 prompt tok/s; 248.9 → 177.8 ms), at half the resident
footprint (12,200 → 6,143 MB/core). Measured 2026-08-04 on beta-4 with both T1b flags
on both sides. That the advantage widens with S is the expected shape for a
compute-bound phase getting twice the cores, and it means the +27.4% at S=128 is a
floor, not the headline — the 35B measured its LNC=1 win at 20k context.

The TP=4 control landing at 1,539 reproduces the
1,530 in the table above almost exactly, which is worth stating for two reasons: it
confirms the tiny-mode basis is the same one that figure was measured on, and it means
beta-4 did **not** move 27B prefill at TP=4 (unlike the 35B, where beta-4 alone was
+12.5%). Cold compile is longer at TP=8 (683 s vs 527 s).

This is the one 35B lever whose sign has transferred cleanly — LNC=1 was +24.5% on
35B prefill and is +27.4% here. Contrast T1a (sign flips with batch size) and beta-4
(+27% at BS=8, −11% at BS=1).

### The core finding: BS=8 decode is barrier-bound, not compute-bound

A critical-path decomposition of the BS=8 decode NEFF found GEMM-exclusive time at
**0.7%** of the step; the step is dominated by engines waiting on each other at
inter-op sync barriers (~58% GLUE+SYNC not behind GEMM). So the lever that moves
TPOT is the **number of serialized sync-points per token**, not the cost of any
single op — which is why the GQA-tail kernel (it removes ~12 barriers/GQA-layer)
wins while optimizations that only cut overlapped per-op cost were flat.

Levers that were tried and **ruled out** at BS=1 latency: MLP/GQA projection
fusion (hand-rolled tiled+transpose GEMM can't beat `F.linear`), FP8 W8A16 /
W8A8 weights (per-Linear op boundaries + B=1 GEMV mapping regress), and DMA
coalescing v3/v4 (targeted DMAs are off the critical path). FP8's value is memory
headroom for larger batch, not per-step TPOT.

## Serving (vLLM-Neuron)

`deploy/` contains scripts for serving the model through a vLLM-Neuron fork in
three modes — baseline bf16, chunked-prefill, and EAGLE3 speculative decode. They
load the repository's ignored `.env` automatically; configure the image, model,
plugin, and compile-cache locations using the `QWEN27_*` variables documented in
`.env.example`. The serving-side model plugin lives in a separate vLLM-Neuron
fork and is not included here.

> The performance numbers above are all from the PyTorch-Native `static_decode.py`
> harness. vLLM-Neuron serving throughput numbers are intentionally omitted.
