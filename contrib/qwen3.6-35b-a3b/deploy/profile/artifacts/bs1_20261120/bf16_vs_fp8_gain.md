# BF16 versus row-FP8 MoE replacement gain

## Replacement boundary

This comparison isolates the local routed-expert operator invoked as
`torch.ops.moe_w8.tkg_row_fp8` by `static_decode_35b.py`. The BF16 control is
the same nkilib `moe_tkg` call with BF16 expert operands. It excludes the
router, shared expert, expert-parallel all-reduce, and the rest of the model.

The BF16 baseline and FP8 candidate both use nkilib `moe_tkg` with identical
model shapes and execution controls:

- batch size 1
- 32 local experts, top-k 8
- hidden size 2048, intermediate size 512
- all-expert scheduling
- post-affinity scaling and SiLU activation
- BF16 input, affinities, and output
- LNC=1 on one NeuronCore

The only material operand difference is BF16 expert weights versus legacy-E4M3
expert weights quantized directly from BF16, plus the FP8 row scales.

## Measured gain

| Metric | BF16 `moe_tkg` | Row-FP8 `moe_tkg` | Gain |
|---|---:|---:|---:|
| Profiled device time | 888.772 us | 451.914 us | **1.967x; -49.15%** |
| 20-iteration host-observed time | 894.617 us | 468.139 us | **1.911x; -47.67%** |
| HBM/DMA transfer | 201.335 MB | 101.065 MB | **-49.80%** |
| DMA active time | 767.478 us | 377.579 us | **-50.80%** |
| Tensor Engine active time | 297.856 us | 258.939 us | **-13.07%** |
| Spill/reload excess | 0 / 0 | 0 / 0 | unchanged |

The measured device saving is **436.859 us per replaced MoE invocation**.
The host benchmark independently measures a **426.479 us** saving.

The gain is memory-side: traffic falls by 100.270 MB while achieved DMA
bandwidth remains essentially unchanged (262.3 versus 267.7 GB/s). Both
profiles are DMA-bound and neither spills, so the 1.967x speedup tracks the
1.992x reduction in transferred bytes closely.

## Scope

This is an isolated-operator result, not an end-to-end token-latency claim.
If the same saving applied independently to all 40 MoE layers, its arithmetic
upper contribution would be 17.474 ms per decoded token. Actual model gain
must be measured end-to-end because routing, shared experts, collectives,
attention, launch scheduling, and overlap are outside these profiles.

At batch size 1 this is the production row-scaled FP8 layout. It is not the
coalesced per-128-output-row-block kernel, which supports batch sizes
32, 64, and 128.

Both captured NEFFs deliberately use the all-expert scheduler because the
row-FP8 custom op currently fixes `is_all_expert=True`. Thus the 1.967x result
is the apples-to-apples gain over the matching all-expert BF16 operator. It
does not compare against the separate selective BF16 `moe_tkg` configuration
available for sub-16-token calls.
