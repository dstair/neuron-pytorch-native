---
license: apache-2.0
tags:
  - neuron
  - nki
  - trainium
  - qwen3.5
  - qwen3.6
  - moe
  - mixture-of-experts
  - fp8
  - quantization
  - decode
  - token-generation
---

# Qwen3.5-MoE fused block-scaled FP8 MoE kernel (Trainium2)

A high-batch **decode** kernel for the 256-expert / top-8 MoE of Qwen3.5-MoE
(published as [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
arch `Qwen3_5MoeForConditionalGeneration`). It fuses, per local expert:

```
gate/up GEMM  ->  SwiGLU  ->  down GEMM  ->  affinity scale  ->  expert-sum
```

converting official **E4M3FN** expert weights to BF16 tile-by-tile inside the
kernel (TensorE cannot take E4M3FN as an `nc_matmul` stationary dtype), and
post-scaling once per 128-output-row block ("Reduction B1" / `_ob`).

## Why FP8 here: capacity, then throughput — not per-token latency

On Trainium, FP8 W8A16 on the experts is a **capacity** lever first: it halves
the resident expert weights (~16 → 8 GB/core), and *that* is what lets high
batch sizes fit on one device. Measured on `trn2.3xlarge`, TP=8/LNC=1, 40
layers, seq=256 (beta-5 DLC):

| Path | Best batch that loads | Throughput |
|---|---|---|
| **FP8 `block_ob_coalesced`** (this kernel) | **BS=128** | **681.3 tok/s** |
| BF16 full-graph | BS=32 (BS≥64 OOMs on load) | 320.6 tok/s |

BF16 is faster *per token at low batch*; FP8 wins **aggregate throughput** by
unlocking a batch bf16 cannot reach. Do not expect FP8 to speed up BS=1 decode —
it will not (at BS=1 the expert GEMMs become GEMVs and dispatch dwarfs the
bandwidth saved). This kernel is for the **high-batch serving regime**.

## What ships

| File | Role |
|---|---|
| `nki_kernels/moe_fused_w8.py` | The NKI kernel (self-contained; no nki-library dependency). Entry: `nki_moe_fused_w8_fp8_block_coalesced_ob`. |
| `ops.py` | Registers `torch.ops.moe_w8.fused_fp8_block_coalesced_ob` via `nki_op` so it drops into a `torch.compile(fullgraph=True, backend="neuron")` graph. |
| `weight_prep.py` | CPU-side, offline: decode official E4M3FN + block scales, requantize to per-output-block legacy-E4M3, and pack the coalesced operands. Pure torch. |

## Kernel interface

`torch.ops.moe_w8.fused_fp8_block_coalesced_ob(hidden, gate_up, down, gate_up_scales, down_scales, affinities) -> [B, H] f32`

Shapes for Qwen3.5-MoE at **TP=8** (H=2048, I=512, 256 experts → 32 local/rank):

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `hidden` | `[B, 2048]` | bf16 | B ∈ {32, 64, 128} |
| `gate_up` | `[32, 2048, 2, 512]` = `[E, H, 2, I]` | int8 (E4M3 bytes) | I **must** == 512 |
| `down` | `[32, 512, 2048]` = `[E, I, H]` | int8 (E4M3 bytes) | |
| `gate_up_scales` | `[32, 128, 128]` = `[E, 128, 2·(H/128)·(I/128)]` | bf16 | 128-row-repeated |
| `down_scales` | `[32, 128, 64]` = `[E, 128, (I/128)·(H/128)]` | bf16 | |
| `affinities` | `[B, 32]` = `[B, E_local]` | f32 | per-token routing weight; 0 for non-local experts |
| **output** | `[B, 2048]` | f32 | local-expert routed sum, **pre** all-reduce |

Constraints: `I == 512`, `H % 512 == 0`, `B ∈ {32,64,128}`. The `_ob` PSUM-post-scale
path activates at **B=128** (`column_packing==1`); B=32/64 use the standard
per-block scale-add (same op, same result).

## Integration contract (serving path)

This is **not** a drop-in `KernelConfig` layer — it computes only the local-rank
routed experts (expert-parallel) and needs the router + shared expert in the
surrounding graph. See [`INTEGRATION.md`](INTEGRATION.md) for the full code
sketch. In brief, per MoE layer:

1. **Router** (bf16/f32, in-graph): `softmax(x @ router.T)` → `topk(8)` →
   `norm_topk_prob` (divide by top-8 sum).
2. **Affinities**: `weight_prep.build_local_affinities(top_ids, top_weights, expert_start, local_experts)` → `[B, E_local]`.
3. **Kernel**: the op above, with the offline-prepped `gate_up/down/*_scales` for this rank's experts.
4. **All-reduce** the `[B, H]` output across the TP group (sum) — reconstructs the full top-8 routed sum.
5. **Shared expert** (bf16, in-graph, replicated): `sigmoid(x @ shared_gate.T) * SwiGLU_shared(x)`, added **after** the all-reduce (no second all-reduce).

## Weight prep (offline, CPU, once)

The routed experts come from the **official FP8 checkpoint**; router, shared
expert, attention, DeltaNet, embeddings, and LM head stay BF16 (a second
checkpoint). `weight_prep` reads each expert's `{gate,up,down}_proj.weight`
(F8_E4M3 bytes) + `weight_scale_inv` (bf16 block scales) and:

- `dequantize_official_fp8(raw, scales)` → full-precision block, then
- `requantize_official_fp8_output_block(...)` → one absmax scale per **128-output-row block spanning the full input width** (this input-independence is what makes the `_ob` PSUM post-scale exact), then
- stack to `gate_up=[E,H,2,I]`, `down=[E,I,H]` and `pack_coalesced_block_scales(...)` → the 128-row-repeated `[E,128,N]` scale tables.

All on CPU at load; no device ops. See `weight_prep.py` docstrings and
`requantize_official_fp8_output_block` / `pack_coalesced_block_scales`.

## Requirements & constraints

- **Hardware/topology**: trn2, **TP=8/LNC=1** for the production config
  (`NEURON_LOGICAL_NC_CONFIG=1`, `NEURON_CC_FLAGS="--target trn2 --lnc 1"`).
- **Decode-only**: no prefill path (run prefill in bf16 / the model's prefill kernels).
- **Fullgraph**: `torch.compile(fullgraph=True, backend="neuron")`; one static
  compile per batch shape (32/64/128). Use `--optlevel 1` (default optlevel OOMs
  the compiler at full depth).
- **Two checkpoints**: BF16 base + official FP8 experts.
- Numerics: exact greedy IDs can flip on BF16 near-ties; gate on logits
  cosine / state rel-L2 (the source repo reports cosine ≥ 0.9999 at two layers),
  not bit-exact token match.

## Validation

The packaged kernel was re-validated on `trn2.3xlarge` (DLC
`concourse-release-0461d3b`, 2026-08-12) with the self-contained synthetic device
test — output vs the quantized CPU reference:

```
BS=32  E=2: cosine=1.0000004  nrmse=0.00011%
BS=64  E=2: cosine=1.0000021  nrmse=0.00114%
BS=128 E=2: cosine=1.0000051  nrmse=0.00136%
PASS  (gates: cosine ≥ 0.9999, nrmse ≤ 1%)
```

Reproduce (synthetic, no checkpoint needed) from the source tree:

```bash
python3 kernels/tests/test_moe_fused_w8_device.py --mode fp8 \
  --fp8-impl block_ob_coalesced --batch-sizes 32,64,128 \
  --local-experts 2 --hidden-size 2048 --intermediate-size 512
```

Add `--expert-model-path <official-fp8-weights> --expert-count 32` to validate
against real weights.

## Provenance / license

Original work from the neuron-pytorch-native Qwen3.6-35B-A3B port. Apache-2.0.
Reference architecture: Qwen3.5-MoE (canonical Qwen3-MoE routing).
