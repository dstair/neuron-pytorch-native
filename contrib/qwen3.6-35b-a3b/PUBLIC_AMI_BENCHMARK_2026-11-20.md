# Public DLAMI throughput remeasurement — 2026-11-20

This remeasurement checks the two headline `trn2.3xlarge` configurations from
the README on the public PyTorch Native Neuron DLAMI released 2026-11-12. The
benchmark source was pinned to commit
`39cbb09f5771bd16047f6d44f482b4335585f56a`, which produced the shipping
4,370.6 prefill result, so source changes are not mixed with the stack change.

## Environment

- Instance type: `trn2.3xlarge`
- DLAMI: PyTorch Native Neuron SDK 2.33.1, released 2026-11-12
- PyTorch: `2.14.0+cpu`
- torch-neuronx: `2.14.3.0.1`
- neuronx-cc: `2.28.7631.0`
- nkilib source: `1ee625782cb1bf91b40bccab741a82c726445080`, with the
  repository's LNC=1, `fori_loop`, hybrid block-quant, and output-block patches
- Topology/compiler: TP=8, LNC=1, `--optlevel 1`

No instance address, hostname, checkpoint location, or compiler-cache location
is required to interpret or reproduce the measurements below.

## Results

| Phase | Exact shape | Previous headline | Run 1 | Run 2 | Mean | Change |
|---|---|---:|---:|---:|---:|---:|
| FP8 CTE prefill | BS=6, N=10,000, bucket=1024, four 10-layer regions | 4,370.6 aggregate tok/s | 4,404.1 | 4,404.2 | **4,404.2** | **+0.77%** |
| FP8 full-graph decode | BS=128, seq=256, 40 layers, 20 synced iterations | 681.3 tok/s | 526.5 | 527.1 | **526.8** | **-22.68%** |

Prefill timed execution was 13,623.7 and 13,623.5 ms, versus 13,728.2 ms
for the previous shipping measurement. Decode TPOT was 243.13 and 242.82
ms/token, averaging 242.98 ms/token versus the previous 187.87 ms/token.

The two-run spread was 0.002% for prefill and 0.114% for decode.

## Matched configuration

The prefill runs used the headline FP8 CTE path:

- output-block expert scales and PSUM hoisting
- uncapped route packer (`MOE_CTE_MAX_TOKENS=0`)
- packed C32 DeltaNet with four-stream packing
- sharded embedding and LM head
- 5.08 GB/core resident module

The decode runs used the headline FP8 full-graph path:

- `block_ob_coalesced` MoE
- tiled DeltaNet convolution state
- direct, per-layer DeltaNet recurrent-state outputs
- stateful, per-GQA-layer KV buffers
- sharded LM head
- 7.09 GB/core resident module

## Correctness gates

Both prefill runs reproduced:

```text
top5=[220,13,197,198,62]
finite[logits=True,deltanet=True,conv=True,kv_k=True,kv_v=True]
```

Both decode runs reproduced `0cc59fb25112` for rows 0 and 127. All ten KV
groups were populated, with 384 nonzero rows in every group.

## Conclusion

Prefill did not regress on the public release; it improved by 0.77% and the two
runs were effectively identical. Decode regressed by 22.68%, despite matching
source, weights, shape, topology, compiler level, module footprint, output
hashes, and KV-state occupancy. This isolates the observed decode change to the
public software stack's generated full graph or runtime execution rather than
workload or source drift.

The public compiler recommended a 512 MiB scratchpad page during prefill. The
remeasurement deliberately retained the record's 256 MiB setting for an
apples-to-apples comparison; a 512 MiB run would be a separate optimization
experiment.
