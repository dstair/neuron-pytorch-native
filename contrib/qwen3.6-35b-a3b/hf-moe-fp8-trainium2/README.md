---
license: apache-2.0
tags:
  - aws-trainium
  - aws-neuron
  - nki
  - fp8
  - mixture-of-experts
  - quantization
  - qwen
base_model:
  - Qwen/Qwen3.5-35B-A3B-FP8
---

# Qwen3.5-35B-A3B FP8 on AWS Trainium2 — loader + MoE kernel

Loader and [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
kernel code for running the **official** Qwen3.5-35B-A3B FP8 checkpoint's routed experts on
AWS Trainium2 via [PyTorch Native](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html).

**This repository does not contain model weights.** Download them directly from Qwen:

- Weights (FP8): https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8
- Base model (BF16): https://huggingface.co/Qwen/Qwen3.5-35B-A3B

Both are Apache-2.0.

## Why this exists

Qwen's official FP8 checkpoint stores expert weights as **OCP E4M3FN** bytes with one BF16
scale per 128×128 block (`weight_scale_inv`). Two things make it non-trivial to run on
Trainium2:

1. **Trn2's Tensor Engine does not accept E4M3FN**, and neuronx-cc rejects
   `torch.float8_e4m3fn` on Trn2 (`NCC_EVRF051`). Trn2 wants **legacy** `float8_e4m3`
   (finite through 240), which torch does not expose as a dtype.
2. The Native DLC image ships **neither `safetensors` nor `transformers`**, so the FP8
   bytes and their block scales must be read and decoded by hand.

This code closes both gaps, losslessly by default.

## Contents

| File | What it is |
|---|---|
| `moe_w8.py` | CPU-side weight prep. `OfficialFP8ExpertReader.load_layer(...)` turns one layer's experts into kernel operands; `build_local_affinities` turns router top-k into a dense `[B, E]` affinity matrix; `fused_moe_reference` is a CPU fp32 reference on the same operands. Also the E4M3FN / legacy-E4M3 codecs. Pure torch. |
| `st_reader.py` | Dependency-free `.safetensors` reader. Keeps `F8_E4M3` tensors as raw bytes. |
| `kernels/moe_fused_w8_35b.py` | Fused all-local-expert MoE NKI kernels (gate/up GEMM → SwiGLU → down GEMM → affinity scale → expert sum) that never materialize `[experts, batch, hidden]`. |
| `kernels/moe_fused_w8_35b_ops.py` | `torch.ops.moe_w8.*` registration, so the kernels sit in a `torch.compile(fullgraph=True, backend="neuron")` graph without a graph break. |
| `tests/test_moe_fp8.py` | CPU checks, kernel parity on Trn2, and end-to-end accuracy against the official checkpoint. |

## Two FP8 modes

| `fp8_impl` | op | scales | accuracy vs official weights | speed |
|---|---|---|---|---|
| **`block_pow2_coalesced`** (default) | `moe_w8::fused_fp8_block_coalesced` | Qwen's 128×128 blocks | **lossless** (conversion cos 1.0000000, NRMSE 0.000%) | baseline |
| `block_ob_coalesced` (opt-in) | `moe_w8::fused_fp8_block_coalesced_ob` | one per 128 output rows | ~4% rel-L2 per MoE layer | ~2× faster at B=128 only; same at B=32/64 |

**Default (pow2).** E4M3FN is finite through 448 and legacy E4M3 through 240, so any block
that uses the E4M3FN-only codes is halved and its scale doubled (an exact exponent shift);
every other block keeps its bytes and scale bit for bit. Only the smallest-normal /
subnormal edge can round.

**Opt-in (output-block).** Re-quantizes with one scale per 128 output rows, independent of
the input block, so the B=128 kernel can accumulate the whole contraction in PSUM and scale
once instead of once per 128×128 block. That is the speed-up, and the coarser scale is the
accuracy cost. Using the `_ob` op on pow2 weights would be silently wrong, so `load_layer`
checks the scale grid (`assert_output_block_scale_grid`).

Pass the same `fp8_impl` to the weight prep and pick the matching op.

## Usage

```python
import sys
sys.path[:0] = ["/path/to/this/repo", "/path/to/this/repo/kernels"]
import torch
import moe_w8
import moe_fused_w8_35b_ops  # registers torch.ops.moe_w8.*

# 1) Offline (CPU): one layer, this rank's experts (TP=8 -> 32 of 256 experts per rank).
reader = moe_w8.OfficialFP8ExpertReader("/path/to/Qwen3.5-35B-A3B-FP8")
w, stats = reader.load_layer(layer=0, expert_start=0, expert_end=32)  # fp8_impl="block_pow2_coalesced"
ops = [w[n].to("neuron") for n in ("w8_gate_up", "w8_down", "w8_gate_up_scale", "w8_down_scale")]

# 2) Per decode step: router top-k -> dense local affinities [B, 32].
affinities = moe_w8.build_local_affinities(top_ids, top_weights, 0, 32).to("neuron")

# 3) This rank's routed sum [B, H] f32; all-reduce across TP, then add the shared expert.
routed = torch.ops.moe_w8.fused_fp8_block_coalesced(hidden_bf16, *ops, affinities)
```

Shape contract: `B in {32, 64, 128}`, `moe_intermediate_size == 512`, `hidden % 512 == 0`.
Under `torch.compile`, use `dynamic=False` (or `torch._dynamo.mark_static` on the batch
dim): NKI kernels reject symbolic shapes, so a second batch size otherwise fails.

The kernels compute only this rank's local experts (expert-parallel). The router and shared
expert stay in the surrounding graph.

## Accuracy (trn2.3xlarge, 2026-10-21)

`tests/test_moe_fp8.py --experts 32 --checkpoint ...`, real `Qwen/Qwen3.5-35B-A3B-FP8`
weights, experts 0-31, top-8 routing that includes tokens with no local expert and duplicate
routes. "vs official" is an fp32 MoE computed straight from the checkpoint (E4M3FN decode ×
`weight_scale_inv`), independent of this repo's packing.

| layer | `fp8_impl` | kernel vs same-operand reference | **kernel vs official**, B=32 / 64 / 128 |
|---|---|---|---|
| 0 | pow2 (default) | cos 0.9999987-0.9999988 | **cos 0.9999987 / 0.9999988 / 0.9999988**, rel-L2 1.6e-3 |
| 20 | pow2 (default) | cos 0.9999989 | **cos 0.9999989 / 0.9999989 / 0.9999989**, rel-L2 1.5e-3 |
| 39 | pow2 (default) | cos 0.9999986-0.9999987 | **cos 0.9999987 / 0.9999986 / 0.9999986**, rel-L2 1.6e-3 |
| 0 | output-block | cos 0.9999988 | cos 0.99909 / 0.99910 / 0.99908, rel-L2 4.3% |
| 20 | output-block | cos 0.9999989 | cos 0.99918 / 0.99918 / 0.99920, rel-L2 4.1% |
| 39 | output-block | cos 0.9999986-0.9999987 | cos 0.99905 / 0.99903 / 0.99903, rel-L2 4.4-4.5% |

For pow2, the error against the official weights equals the kernel's own error (bf16
activations into the kernel), so the conversion adds nothing measurable. Measured on torch
2.14 / torch-neuronx 2.14.3 / nki 0.7.0b1 (PyTorch Native dev build).

```bash
python3 tests/test_moe_fp8.py --cpu-only                     # any host
python3 tests/test_moe_fp8.py --experts 32                   # trn2, synthetic weights
python3 tests/test_moe_fp8.py --experts 32 --checkpoint /path/to/Qwen3.5-35B-A3B-FP8 --layer 0
```

## Performance and scope

> ⚠️ FP8 here is a **memory / capacity lever**, not a single-stream decode-latency win. It
> roughly halves resident expert bytes, which is what lets the batch grow.

In the source implementation (below; TP=8 / LNC=1, seq=256, beta-5 PyTorch Native
container), Qwen3.5-35B-A3B decode with the output-block kernel reached 681.3 tok/s at
BS=128, where bf16 experts ran out of HBM at BS=64. That figure was measured with that
implementation and stack, not re-run with this repository, and predates the lossless
default. A reviewer's per-call measurement at E=32 / H=2048 / I=512 (eager, includes launch
overhead, torch 2.13 / torch-neuronx 2.13.3 / nki 0.7.0b1) gave B=32 3.4 ms and B=64
3.6 ms for both ops, and B=128 2.0 ms output-block vs 4.0 ms pow2.

## Requirements

AWS **Trainium2** and a pre-GA PyTorch Native build for the kernels. The CPU-side code in
`moe_w8.py` / `st_reader.py` runs anywhere with PyTorch.

## Source

Developed as part of a full Qwen3.5-35B-A3B reference implementation:
https://github.com/dstair/neuron-pytorch-native (`contrib/qwen3.6-35b-a3b`).

## License

Apache-2.0. See `LICENSE` and `NOTICE`. Original code only; no model weights are
redistributed. Model weights are © the Qwen team, released under Apache-2.0 at the links
above.
