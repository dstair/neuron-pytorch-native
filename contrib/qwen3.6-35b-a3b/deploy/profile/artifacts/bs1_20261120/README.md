# Qwen 35B-A3B MoE BS=1 BF16 vs FP8 profile

Captured on 2026-11-20 on a `trn2.3xlarge` using the public PyTorch Native
host environment (no container).

## Test configuration

- Layer 0 weights loaded directly from a BF16 checkpoint
- One NeuronCore, LNC=1
- Batch size 1
- 32 local experts, top-k 8
- Hidden size 2048, intermediate size 512
- BF16 and FP8 use the same all-expert `moe_tkg` scheduling
- FP8 weights are quantized directly from the BF16 tensors using the
  [`row` layout](../../../../hf-moe-fp8-trainium2/bf16_weight_prep.py)
- Device capture profiles execution 2 of 2 with DGE notifications enabled

At BS=1 the profile uses row-scaled FP8 (`QuantizationType.row`). The
coalesced per-128-output-row-block kernel is restricted to batch sizes
32/64/128, so this is not a profile of that high-batch layout.

## Results

| Metric | BF16 | FP8 | Change |
|---|---:|---:|---:|
| Device profile time | 888.772 us | 451.914 us | 1.967x faster |
| 20-iteration host benchmark | 894.617 us | 468.139 us | 1.911x faster |
| HBM/DMA bytes | 201.335 MB | 101.065 MB | -49.80% |
| DMA active time | 767.478 us | 377.579 us | -50.80% |
| Achieved DMA bandwidth | 262.3 GB/s | 267.7 GB/s | 60.3% / 61.5% of peak |
| Tensor Engine active time | 297.856 us | 258.939 us | -13.07% |
| Spill bytes | 0 | 0 | unchanged |
| Reload excess | 0 | 0 | unchanged |

Both kernels are DMA-bound. FP8 retains essentially the same DMA efficiency
while halving weight traffic, which accounts for the measured gain.

The isolated local routed-expert operator therefore saves **436.859 us per
invocation** on device. See `bf16_vs_fp8_gain.md` for the exact production
replacement boundary and scope.

### Performance bounds

All times are device times.

| Bound | BF16 | FP8 |
|---|---:|---:|
| Total | 888.772 us | 451.914 us |
| Perfect pipeline | 767.478 us (DMA) | 377.579 us (DMA) |
| Memory bound | 767.478 us | 377.579 us |
| Memory ideal at 435 GB/s | 462.839 us | 232.333 us |
| Memory ideal, no reloads | 462.839 us | 232.333 us |
| Compute bound | 297.856 us | 258.939 us |
| Compute ideal hardware FLOPs | 2.567 us | 2.567 us |
| Compute ideal useful FLOPs | 2.560 us | 2.560 us |

DMA attribution coverage is 100%. BF16 weight transfers use 4 KiB DMA
descriptors. FP8 weight traffic is half-sized; its scale traffic is small.
There are no redundant weight loads or SBUF spills in either kernel.
Transpose MATMULs are 0.262% of adjusted FLOPs.

## Numerical checks

The parity check here is CPU reference versus Trainium device output, using
the same supplied weights.

| Metric | BF16 | FP8 vs quantized CPU reference |
|---|---:|---:|
| Cosine similarity | 0.99999058 | 0.99999058 |
| Relative L2 | 0.43796% | 0.44013% |
| Max absolute error | 1.8567e-4 | 1.8224e-4 |

Direct BF16-to-FP8 per-projection weight relative L2 error:

- down: 2.5966%
- gate: 2.6408%
- up: 2.6422%

### CPU reference versus NKI simulator

The layer-0, BS=1 simulator test uses the production per-rank MoE shapes:
32 local experts, top-k 8, hidden size 2048, and intermediate size 512.

| Comparison | Cosine | Relative L2 | Max absolute | Allclose (`atol=2e-4`, `rtol=2e-2`) |
|---|---:|---:|---:|---:|
| BF16 simulator vs BF16 CPU | 0.99999064 | 0.43757% | 1.8567e-4 | PASS |
| FP8 simulator vs quantized CPU | 0.99999058 | 0.44013% | 1.8224e-4 | PASS |
| Quantized FP8 CPU vs BF16 CPU | 0.99918514 | 4.03783% | 1.0564e-3 | expected fail |
| FP8 simulator vs BF16 CPU | 0.99917436 | 4.06573% | 1.0823e-3 | expected fail |

The same-operand comparisons isolate kernel/simulator correctness and pass.
The approximately 4.04% output difference from the BF16 reference is the
effect of weight quantization on this sampled MoE invocation; it should not
be confused with the 2.60-2.64% per-projection weight error.

## Neuron Explorer

Load the externally stored NEFF/NTFF pairs into Neuron Explorer to inspect
the BF16 and FP8 captures. Host addresses, SSH configuration, and Explorer
database paths are intentionally not recorded in source control.

## Artifacts

The binary profile bundle is stored outside Git at
`s3://dstair-neuron-models/Qwen3.5-35B-A3B/fp8-profiling/`. Its contents are:

- `bf16_bs1.neff`
- `bf16_bs1_capture_exec_2.ntff`
- `bf16_run.json`
- `bf16_vs_fp8_gain.md`
- `fp8_bs1.neff`
- `fp8_bs1_capture_exec_2.ntff`
- `fp8_run.json`
- `moe_bs1_simulator_parity.json`
- `moe_bs1_simulator_parity.log`

SHA-256:

```text
fd104c00a2925d86de54b8206c4a8e9add29864588ce4aa6a1cbf7200385dc3d  bf16_bs1.neff
d2376ed6bfcdd619b1a42008f039d3431e37ca7599de27329dfbef50e4513f19  bf16_bs1_capture_exec_2.ntff
4d2d90ceb3db3ed0169e9d1f00ee57b9d9b2ffec8511440d0d4ce0ae55a986d7  bf16_run.json
d24138678658100a59a97462ec871a7d3496db6ebc0b9d83304e72779eb9c3b7  fp8_bs1.neff
685dd8258cb7436c065af8ce170725da8c54949ed51973d2a321e50d77ea2951  fp8_bs1_capture_exec_2.ntff
ea0c3c665625c0e40b3ffeedda37ec95e230a6f07b32a9eb1c05406b4a74aada  fp8_run.json
```
