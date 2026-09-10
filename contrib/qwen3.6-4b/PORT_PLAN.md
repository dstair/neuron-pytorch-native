# Qwen3.5-4B Trainium2 optimization plan

The target is the text backbone of `Qwen/Qwen3.5-4B` (the Qwen 3.6 release
family): 32 layers arranged as `[DeltaNet ×3, GQA ×1] ×8`, hidden size 2560,
dense SwiGLU width 9216, and a 248,320-token tied vocabulary.

## Transfer map

| 35B optimization | 4B application | Rationale / gate |
|---|---|---|
| One full decode graph | Compile embedding, all 32 layers, state updates, vocab head, and greedy selection together | Removes Python and per-layer dispatch. Verify one NEFF and synchronized timing. |
| Batched DeltaNet v2 | Reuse the exact TP=4 kernel geometry: 4 local K heads and 8 local V heads | The head geometry is identical to 35B. Gate every recurrent-state layer on nonzero occupancy. |
| Tiled DeltaNet convolution state | Store each layer as `[B,128,16,3]` | Coalesces history reads/writes and matches the qualified 35B layout. |
| Direct per-layer state output | Give every DeltaNet layer distinct state and convolution buffers, then stack once | Avoids the shared-base write-after-write chain that nearly halved 35B throughput. |
| Stateful GQA tail | Give each of the eight GQA layers its own KV buffers and append inside the fused attention kernel | Avoids cache copies and preserves all writes. The 4B TP=4 topology has one local KV head, exactly matching the kernel. |
| Vocab-sharded LM head | Load and execute one quarter of the vocabulary per rank; select global greedy top-1 with collectives | Cuts the largest decode matmul and FP8 weight footprint by 4× while preserving exact argmax. |
| FP8 W8A16 projections | Quantize every dense projection per output channel and execute with native legacy-E4M3 TensorE kernels | Halves projection weight traffic. Gate against the same optimized BF16 graph with top-1 agreement and throughput. |

## Measurement progression

1. Compile and run FP8 at batch 128, sequence bucket 256. This is the maximum
   batch supported by the current FP8 kernel and the expected throughput peak.
2. Run the identical graph in BF16 at batch 128 to establish whether FP8 is a
   throughput win rather than only a memory win.
3. If batch 128 is not optimal, sweep 64, 32, and 16 with separate compile
   caches. Report synchronized aggregate decode tokens/second and TPOT.
4. Compare deterministic first-step token IDs between BF16 and FP8. Require
   high top-1 agreement and nonzero recurrent state in all 24 DeltaNet layers.
5. Keep the best measured configuration as the documented recipe. Profile only
   if FP8 loses, focusing on FP8 custom-call boundaries, collectives, and the
   vocab projection.

## Follow-up work

The current milestone is decode throughput. A production port should next add
the 35B chunked DeltaNet and CTE/flash GQA prefill paths, then integrate the
qualified decode graph with the serving stack.

The initial TP=4 qualification found that generic per-linear FP8 is a capacity
optimization, not a throughput win: 573.3 tok/s all-FP8 and 658.5 tok/s with
only the vocabulary head in FP8, versus 679.6 tok/s in BF16.

The LNC=1 follow-up is now qualified. Eight independent TP=1 replicas at BS=32,
using FP8 for the large MLP and vocabulary projections and BF16 for the smaller
DeltaNet/GQA projections, measured 748.1 aggregate tok/s on the longer repeat.
This removes TP collectives and 152 small FP8 custom calls while fitting in
6.57 GiB of module buffers per core.

The LNC=2 data-parallel follow-up is also qualified. Four complete TP=1 models
at BS=64, one per LNC=2 logical core, measured 513.0 aggregate tok/s. Although
one replica reaches 128.2 tok/s, combining each pair of physical NeuronCores
reduces total device throughput by 31.4% versus eight LNC=1 replicas. Further
work remains the same: fuse the MLP projections and fuse vocabulary projection
with argmax. Detailed measurements and gates are in [RESULTS.md](RESULTS.md).
