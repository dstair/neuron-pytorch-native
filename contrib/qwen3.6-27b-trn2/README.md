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

TP=4 sharding is manual and per-core (colwise for q/k/v/gate/up, rowwise for the
output/down projections) with an all-reduce after each `out_proj` and `down_proj`.
See the sharding helpers in `static_decode.py`.

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

# Compiled chunked prefill
CHUNKEDPREFILL=1 \
  torchrun --nproc-per-node=4 static_decode.py --num-layers 64 --prompt-len 128 --compile-prefill
```

Useful flags: `--tiny` (small config for quick iteration), `--num-layers N`,
`--skip-prefill`, `--fp8-weights`, `--batch-size N`, `PROFILE_STEPS=N` (emit a
device profile over N decode steps).

### Optimization levers (environment flags)

All are **default-off**; the baseline path is byte-identical without them.

| Flag | Effect |
|---|---|
| `CHUNKEDPREFILL=1` | Compiled chunked-DeltaNet prefill (~82× faster prefill compile) |
| `DNBATCHED_V2=1` | DMA-coalesced batched-DeltaNet decode kernel (BS>1) |
| `DN_TILED_CONV=1` | Tile-partition-major DeltaNet conv-state layout (**BS=8 only** — regresses at BS=16; requires `DNBATCHED_V2=1` and `--batch-size >= 2`) |
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

All numbers are `trn2.3xlarge`, TP=4, LNC=2, bf16, measured with a
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

#### T1a: tiled conv-state layout (measured — a BS=8 win *and* a BS=16 regression)

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

**Why BS=16 regresses is not yet established.** Two hypotheses, untested: (1) SBUF
pressure — the tiled branch holds `[128, 20, 3]` fp32 tiles (`hist`, `new_hist`,
`prod3`, ~30 KB each) per batch item where the untiled path holds `[128, 4]` (~2 KB),
so at B=16 the working set may spill; (2) the single wide DMA loses the
many-small-DMA pipelining that hid latency at the larger batch. Note the mechanism
T1a targets is DMA *count*, and the barrier-bound analysis below says BS=8 is where
that cost dominates — so a lever that helps at BS=8 and hurts at BS=16 is consistent
with the profile, just not with the 35B's BS=32 result.

**Status: default-off and scoped to BS=8.** Do not enable at BS=16.

### Prefill (TTFT / prompt throughput)

| Test | Framework | Config | Latency | Prompt tok/s |
|---|---|---|---|---|
| Prefill, compiled chunked-DeltaNet kernel | PyTorch Native | S=128, BS=1 | **83.7 ms** (warm) | **1530** |
| Prefill, eager fallback (unoptimized) | PyTorch Native | S=128, BS=1 | 6847 ms | 19 |

Wiring the validated chunked-DeltaNet NKI kernel makes the prefill graph
compilable (one custom call/layer instead of an un-compilable data-dependent torch
chunk-loop), giving a **~82×** speedup over the eager fallback. Cold compile of the
prefill graph is ~391 s (one-time; NEFF cached after). Enabling chunked prefill
does not regress decode (BS=1 TPOT stays 36.7 ms).

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
