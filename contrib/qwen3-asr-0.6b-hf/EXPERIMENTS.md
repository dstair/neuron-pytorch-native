
### Setup + eager HF generate() on Neuron (2026-10-15 19:16 UTC, porter)
**Model**: Qwen/Qwen3-ASR-0.6B-hf (transformers 5.17.0, `Qwen3ASRForConditionalGeneration`). Audio encoder: 3x stride-2 Conv2d over 100-frame mel chunks (13 tokens/chunk), 18 layer Whisper-like encoder with windowed attention (windows of 104 post-CNN tokens via `cu_seqlens`, packing with `nonzero`/`index_select`), MLP projector 896->1024; text decoder Qwen3 28 layers, GQA 16/8, head_dim 128, q/k RMSNorm, tied lm_head (vocab 151936).
Prompt = 9 template tokens + N audio placeholders + 9 tokens (`...assistant\nlanguage English<asr_text>`).

**CPU fp32 reference** (`cpu_ref.py` -> `ref/ref.json`): librispeech_asr_dummy clean validation, first 10 samples, `apply_transcription_request(language="English")`, greedy `generate(max_new_tokens=256)`. WER 0.0433, median CPU time 1.43 s/sample.

**Eager on Neuron** (`eager_test.py`, bf16, HF generate unchanged, max_new_tokens=64, sample 0): **works**, token-exact vs CPU ref (23 tokens). First call 176.6 s (per-op compilation), second call 3.65 s. The eager-index-op failure seen in other ports did not occur here.


### Static-shape port validated on CPU fp32: 10/10 token-exact (2026-10-15 19:18 UTC, porter)
`qasr_port.py` (Encoder / Prefill / Decode modules reusing HF weights) + `run.py`.
Replacements vs HF: encoder `nonzero`/`index_select` packing + per-window `torch.split` loop -> padded [C*13] layout (valid frames are a flat prefix at batch 1) with block-diagonal mask (window = p//104, key < n_valid); audio placeholder `masked_scatter` -> `torch.where(ids==audio_tok, audio shifted by fixed template offset 9)`; DynamicCache -> static K/V [28,8,S=768,128] returned functionally, decode slot write `torch.where(arange==pos)`; in-graph argmax. Buckets: encoder chunks {8,16,32}, prefill {128,256,512}, decode T=1.
`python run.py --device cpu --no-compile --dtype fp32`: **10/10 token-exact**, WER 0.0433 (= reference).


### Compiled baseline v0: correct but lm_head lowered to 1.3M DMAs (655 ms/decode step) (2026-10-15 20:51 UTC, porter)
First compiled run (`run.py -n 10`, bf16, buckets enc {8,16,32}, prefill {128,256,512}, S=768): **9/10 token-exact**, WER 0.0394 (CPU 0.0433), but decode step **655 ms**, prefill(128/256) **640–720 ms**, prefill(512) 14 ms, encoder 11–19 ms. Cold compile ≈ 10–11 min per large graph (7 graphs, ~58 min total run).
Isolated timing (`diag_time.py`): dispatch ~1.7 ms, device sync 652 ms/decode step; prefill128 719 ms; prefill512 11 ms. NEFF sizes: slow graphs ~36 MB, prefill512 4.9 MB.
Profile of the decode NEFF (`profiles/dec0`, neuron-explorer capture + ingest): total 658 ms; 1.23M `Sync DMA_DIRECT2D` instructions (979 ms instruction time), 133M hardware-DGE DMA packets, 1.28 GB spill/reload. **1.32M instructions / 857 ms attributed to `qasr_port.py:153` = `self.lm_head(h[0])` (`%dot.253 = dot(reshape, transpose)`)** — the tied lm_head (nn.Linear over the [151936,1024] embedding weight) is lowered as a per-call DMA transpose of the weight. Whatever layout decision the compiler made for P=512 avoided it.
Fix under test: pre-transposed contiguous buffer `lm_w_t [1024, V]`, logits = `h @ lm_w_t` (switch `QASR_LMHEAD=linear` restores HF path).


### Pre-transposed lm_head: 85x faster prefill, but exposed an in-graph argmax miscompile (fixed) (2026-10-15 21:12 UTC, porter)
After switching lm_head to `h @ lm_w_t` (pre-transposed [1024,V] buffer), prefill-128 device time fell **719 ms -> 8.5 ms** (`diag_time.py`), but the in-graph `torch.argmax` over the [1,151936] bf16 logits returned **INT_MAX** (2147483647) in both prefill and decode, so the next decode step's embedding gather went out of bounds (`NRT status 1006 ... indirect memcopies out of bound`, source `Embedding[embed_tokens]`). The logits themselves were correct (`dbg_decode.py`: host argmax of the returned logits = 12275 = reference; no NaN; weight checksum matched).
A manual argmax (`max` then `min` over `where(lg==max, arange, V)`) on the bf16 logits returned **V** (no element equal to the max). My hypothesis, which I did not confirm: the compiler fuses the reduction onto fp32 PSUM values and compares them against the bf16-rounded logits.
**Fix**: upcast logits to fp32 before the manual argmax (`greedy_argmax` in `qasr_port.py`; `QASR_ARGMAX=torch` restores torch.argmax). Device token = host argmax for prefill and decode. Kept.


### Compiled baseline v1 (lm_head fix + fp32 argmax): 7.68 ms/decode step, 9/10 exact, WER = CPU (2026-10-15 21:22 UTC, porter)
`run.py -n 10` (bf16, compiled, buckets enc {8,16,32} / prefill {128,256,512}, S=768, 3 warm repeats per sample):
**9/10 token-exact** vs CPU fp32 greedy; **WER 0.0433 = CPU 0.0433**. Encoder 10.6–18.6 ms (C=8/16: ~11 ms, C=32: ~18.5 ms), prefill 10.8–16.1 ms (P=128 ~11 ms, P=256/512 ~15.5 ms), **decode step 7.68 ms** median (includes 1-int D2H per step for EOS check), e2e median **209.8 ms**/utterance vs CPU fp32 1425 ms (6.8x).
Before this fix: decode step 655 ms, prefill 640–720 ms.
Mismatch analysis, sample 4 (30 s clip): Neuron diverges at token 8 (drops ',' before ' up'); HF's own bf16 `generate()` on CPU diverges at token 11 (' on'->' and'). Teacher-forced CPU fp32 top-2 margins (`margin_check.py 4`): **token 8 margin 0.004, token 11 margin 0.015** (median margin 8.1). These are fp32 near-ties, so bf16 flips are expected and the port is not at fault. The CPU fp32 port is 10/10 exact.


### Final baseline with cold compile timing (2026-10-15 21:34 UTC, porter)
Fresh NEFF cache (`TORCH_NEURONX_NEFF_CACHE_DIR=neff-cache-cold`), `run.py -n 10`, clean flags (no debug env), 3 warm repeats per sample, device sync via token D2H / `torch.neuron.synchronize()`:
- Cold compile, 8 graphs: sample0 (enc C=8 + prefill P=128 + decode) 250 s; enc16 + prefill256 155.6 s; enc32 + prefill512 104.1 s; prefill256 with 416-row audio input 101.7 s. Whole 10-sample run including compiles: **640 s wall**.
- Warm medians across samples: encoder **10.76 ms** (C=32: 18.6 ms), prefill **15.16 ms** (P=128: ~11 ms), decode step **7.53 ms**, e2e **206.1 ms**/utterance (CPU fp32: 1425 ms).
- Accuracy: 9/10 token-exact, WER 0.0433 (CPU 0.0433). Sample 4 diverges at an fp32 near-tie (margin 0.004).
Host stack versions: `host_neuron_versions.txt`.
Notes for the profiler/improver: the decode step returns the full K/V caches functionally (2×[28,8,768,128] bf16 ≈ 88 MB copied per step), so in-place/aliased cache updates are the obvious next target. Prefill graphs take the audio embeddings at encoder-bucket shape, so each (C,P) pair is a separate compile (e.g. prefill256 compiled twice).


### Profile round 0: decode graph (T=1) is DMA-bound; lm_head matmul + unattributed cache/GQA traffic dominate HBM bytes (2026-10-15 21:50 UTC, profiler)
**What**: profiled the decode graph (the hot graph — it runs once per generated token, 13–86 times/utterance, and the port's own timings show it is 82–95% of end-to-end wall time per sample). Compiled `Decode` alone into a fresh cache (`neff-cache-debug-r0`) with `XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 NEURON_FRAMEWORK_DEBUG=1` so the NEFF carries `Instruction.layer` / `bir_debug_info_source_location`. Ran 4 warm decode steps (`profiles/r0/gen_debug_neff.py`), found one 4.58 MB real NEFF (`31125b1b…neff`) in the debug cache (the only large graph, since only `Decode` was compiled here — unambiguous). Captured with `neuron-explorer capture --profile-nth-exec=2 --ignore-exec-errors` (no DGE notifications, single core), ingested to parquet as display name `dec_r0`, queried with pandas (`profiles/r0/analyze*.py`).

**Reconciliation**: captured `total_time` = 6.10 ms for one decode-step execution. The port's clean-flag measured decode step (median over 10 samples, 3 warm repeats) is 7.53 ms, which includes python dispatch + a 1-int D2H for the EOS check that the capture does not. 6.10/7.53 = 81%, consistent with ~1.4 ms of host-side overhead already documented in `artifacts/port.json`. Treated as reconciled.

**Bottleneck**: DMA. `dma_active_time_percent` = 90.9% of `total_time` (5.55 ms of 6.10 ms) — per the profiling skill's rule, an engine instruction stream near 100% of total_time *is* the critical path. `tensor_engine_active_time_percent`=62.5% (3.82ms), `gpsimd`=57.3% (3.50ms), `scalar`=48.7% (2.97ms), `vector`=34.7% (2.12ms) all overlap with DMA. `perfect_pipeline` = max(engine active) = 5.55 ms (DMA); `serialization_gap` = 6.10 − 5.55 = 0.55 ms (9%) — the step is well pipelined, so cutting DMA bytes should convert close to 1:1 into wall time.
`mfu_estimated_percent` = 0.14%, `mbu_estimated_percent` = 47.8% → compute is essentially idle, confirming memory-bound, as expected for a single-token (M=1) decode step.

**Key finding 1 — the final lm_head/logits matmul (`qasr_port.py:146`, `logits = h @ lm_w_t`) is responsible for a large, directly-attributed, and unexpectedly duplicated chunk of HBM traffic**, despite the port already having fixed the original "nn.Linear transposes the weight every call" bug:
- `qasr_port.py:146` (HLO node `_dot.5284`): 311.2 MB `hbm_read` (reads the full `[1024, 151936]` bf16 weight — unavoidable minimum for one full-vocab matmul) **plus** 155.6 MB `spill_reload`.
- `neuronxcc/private_nkl/transpose.py:641` (same op, compiler-internal retiling): **another** 311.2 MB `hbm_read` — i.e. the weight is read from HBM *twice* in one decode step.
- `neuronxcc/private_nkl/transpose.py:853`: 302.0 MB `hbm_write` + 151.0 MB `spill_save`.
- Summing these: ≈ 900 MB–1.2 GB of this step's 2.09 GB total HBM traffic (1.644 GB read + 0.444 GB write) is attributable to this single matmul — 43–57% of all step bytes — even though the minimum necessary read (the weight, once) is 311 MB.
- Tensor-engine level evidence of the same waste: `hardware_flops` = 182.08 GFLOP and `adjusted_hardware_flops` = 182.22 GFLOP for the step, but `model_flops` (real useful matmul work) = only **1.37 GFLOP**. `transpose_flops` = 180.7 GFLOP = **99.2%** of the hardware flop budget. Grouping `Instruction` by `tensor_instruction_type` on the Tensor engine: `TRANSPOSE` = 88,800 instructions / 17.36 ms instruction-time / 180.8 GFLOP vs `REGULAR` (actual MATMUL) = 83,539 instructions / 10.99 ms / 1.37 GFLOP. The Tensor engine spends almost all its flop budget transposing/retiling, not computing — spread across the lm_head matmul and the many degenerate (M=1) per-layer Q/K/V/O/MLP matmuls.

**Key finding 2 — 72% of instructions (124,272 of 172,401) carry no module/source attribution (`layer`="Unknown", empty `bir_debug_info_source_location`/`hlo_name`), and this bucket holds 1.33 GB of the step's 1.64 GB total `hbm_read` (81%) and 0.40 GB of the 0.44 GB `hbm_write` (90%).** Breaking the "Unknown" bucket down by (engine, opcode): `GpSimd DMA_DIRECT2D` (6,904 instr) reads 1.28 GB / writes 44 MB; `Scalar DMA_DIRECT2D` (1,132 instr) writes 343 MB. Average packet size for the GpSimd group is ~185 KB, close to one `[768,128]` bf16 K/V head-slice (196 KB) — consistent with (but not proven to be) the per-layer functional K/V-cache update (`torch.where(sel, k[0], kc[i])` over the *whole* `[8,768,128]` cache, 28 layers) and/or the GQA head-duplication (`kn[:,:,None].expand(1,8,2,768,128).reshape(1,16,768,128)`, 28 layers) materializing duplicate K/V data every step — exactly the mechanism `artifacts/port.json` already flagged as "the obvious next target" (there it estimated ~88 MB/step from the two cache round trips alone; the measured unattributed traffic is ~15× that, suggesting the GQA-expand materialization and/or per-layer full-cache rewrite is far more expensive than the naive estimate). This is a hypothesis from byte/shape matching, not confirmed by source attribution — flagged as needing one more profiling pass (R4) before committing engineering time.

**Bounds (ms, this one decode-step capture)**:
- `total_time` (measured) = 6.10
- `memory_bound` (DMA active, i.e. the real floor given current bytes) = 5.55
- `memory_ideal` (unique bytes 2.09 GB / inferred peak BW) = 2.65 (peak BW backed out from `mbu_estimated_percent`=47.8% against achieved 376 GB/s while DMA was active → ≈787 GB/s)
- `no_reloads_ideal` (in+out+weights 1.68 GB / 787 GB/s, i.e. without the extra lm_head re-read/spill) = 2.14
- `compute_bound` (182.2 GFLOP hardware flops incl. transpose tax / 167 TFLOP/s LNC2 bf16 peak) = 1.09
- `perfect_pipeline` (max engine active = DMA) = 5.55
- `serialization_gap` = 0.55 (9%)
- clean-flag `wall_ms` for the same unit (median decode step, port.json) = 7.53

**Verdict**: decode is clearly memory/DMA-bound with low serialization loss, so the lever is bytes moved, not instruction count or raw flops. Recommendations below rank by estimated bytes removed.


### Round 1: split lm_head, grouped GQA, pre-transposed+fused layer weights (kept); in-place KV cache (rejected) (2026-10-15 22:31 UTC, improver)
**Starting point**: profile_r0 showed the decode step (T=1, hot graph) is DMA-bound (DMA active 91% of 6.1 ms). The lm_head matmul `h @ lm_w_t` ([1024,151936]) read its weight twice and spilled it (~0.9–1.2 GB of 2.09 GB/step). 99% of Tensor-engine flops were TRANSPOSE (weight retiling for the nn.Linear `x @ W.T` matmuls). There was also unattributed traffic that looked like the GQA expand and the full-cache rewrite. Snapshot of the pre-round code: `snapshots/before_r1/`.

**Tools**: `r1/mb_lmhead.py` microbenchmarks the lm_head+argmax variants on synthetic data. `r1/bench_decode.py` runs the real-weight Decode graph alone for 60 steps on a synthetic cache and reports the median of steps 10–59, a token list and a cache occupancy/norm fingerprint. Every timing uses clean flags with `.cpu()` sync.

### lm_head microbench (one graph: logits + greedy argmax, median of 25 calls, including dispatch)
| variant | ms |
|---|---|
| A: `h @ Wt[1024,V]` (current) | 2.67 |
| B: `W[V,1024] @ h.T` | 2.36 |
| C512: blocked [297,1024,512] bmm | 2.66 |
| D: VectorE multiply+sum | 2.37 |
| E2 / E4 / **E8** / E16 / E32 / E64: V split into k contiguous [1024,V/k] buffers, k matmuls + cat | 1.17 / 0.92 / **0.95** / 0.96 / 1.00 / 1.12 |

All variants gave the same argmax as fp32, with cosine 0.999999. One big matmul makes the compiler re-tile and spill the weight; several medium ones do not.

### Changes (all behind env switches in `qasr_port.py`)
- **R1** `QASR_LMSPLIT=8` (default; `1` gives the old path): the lm_head weight is stored as 8 contiguous column blocks, with one matmul each and then a cat.
- **R2a** `QASR_GQA=group` (default; `expand` gives the old path): q is reshaped to [1,nkv,nrep*T,hd] so K/V are never duplicated. Head order matches HF repeat_kv; the prefill mask is repeated nrep times.
- **R3** `QASR_WT=tf` (default; `t` keeps separate pre-transposed weights; `linear` gives the HF modules): every decoder-layer weight is pre-transposed to [in,out], q|k|v are fused into [1024,4096] and gate|up into [1024,6144]. This applies to both prefill and decode.
- **R2b** `QASR_KV=inplace` (NOT default; `where` is the default): decode attends over the old cache plus the new k/v as an extra column, then `kc.index_copy_(2,pos,k)` (graph-input mutation) and returns only the token. A toy test confirmed that the mutation is correct on neuron.

CPU fp32 validation (`run.py --device cpu --no-compile`, 3 samples): 3/3 token-exact for split+group, wt=t, wt=tf and kv=inplace.

### Decode-only bench (real weights, median ms/step)
| config | ms |
|---|---|
| baseline (split=1, expand, linear) | 7.74 |
| split=8 | 6.11 |
| group only | 7.46 |
| split=8 + group | 6.04 (rerun 6.04) |
| split=8 + group + wt=t | 5.40 |
| **split=8 + group + wt=tf (new default)** | **4.50** |
| ablation, tf base, GQA expand | 4.85 (group is worth 0.35 ms) |
| ablation, tf base, split=1 | 6.44 (split is worth 1.9 ms) |
| tf base, split=16 | 4.85 |
| tf base + kv=inplace | **6.98 (+2.5 ms, worse)** |

Every variant had full cache occupancy (13440/13440 slots written, 0 beyond).

### Full 10-sample run (`run.py -n 10`, 3 warm repeats, medians), same session
| | baseline (`QASR_LMSPLIT=1 QASR_GQA=expand QASR_WT=linear`) | new defaults |
|---|---|---|
| token-exact vs CPU fp32 | 9/10 | **10/10** |
| WER | 0.0433 | 0.0433 |
| decode step ms | 7.67 | **4.91** (−36%) |
| prefill ms | 15.35 | **11.22** |
| encoder ms | 11.21 | 11.20 |
| e2e ms (median utterance) | 210.0 | **140.2** (−33%) |

The near-tie sample 1272-128104-0004 now matches the reference too, because the different accumulation order happens to land on the right side of the 0.004 margin. Treat that as luck, not as a precision gain.

**Verdicts**: R1 split lm_head is **kept** (−1.6 to −1.9 ms/step). R3 pre-transposed+fused weights is **kept** (−1.5 ms/step on top of R1+R2a; fusion alone is −0.9 ms vs unfused `t`). R2a grouped GQA is **kept** (−0.35 ms on the tf base, about 7%). R2b in-place KV is **rejected**: graph-input mutation through torch.compile/neuron costs +2.5 ms/step, probably because the mutated inputs are copied back as whole tensors and the attention is split. It stays behind `QASR_KV=inplace` for reference. Results are in `artifacts/run_r1_new.json` and `artifacts/run_r1_baseline.json`.


### Profile round 1 (post improve_r1): Decode step profiled with op attribution, DMA still bottleneck but weight re-reads/transpose tax from r0 are gone (2026-10-15 22:40 UTC, profiler)
**What**: Re-profiled the Decode graph (T=1, 28-layer Qwen3 text decoder) after round-1 improvements
(`QASR_LMSPLIT=8, QASR_GQA=group, QASR_WT=tf, QASR_KV=where` — the kept defaults from `improve_r1`).
Compiled fresh into `neff-cache-debug-r1` with `XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 NEURON_FRAMEWORK_DEBUG=1`,
captured with `neuron-explorer capture --profile-nth-exec=2` (DGE notifications off, single core), ingested
to parquet as display name `dec_r1`.

**NEFF**: `profiles/r1/neffs/dec_r1.neff` (2.58MB — down from r0's 4.58MB, consistent with the weight-fusion/
pre-transpose changes). Only one large NEFF in the fresh cache, so identification is unambiguous.

**Reconciliation**: captured `total_time` = 3.942 ms/step vs. clean-flag median wall (`run_r1_new.json`,
10-sample synced-every-call "step" field) = 4.908 ms → 80.3%, same ratio as r0's 81% (gap is host dispatch +
the 1-int D2H EOS check, as before — not a mis-identified unit).

**Bounds table (ms, this execution)**:
| metric | ms |
|---|---|
| total_time (measured) | 3.942 |
| memory_bound (dma_active, 88.9%) | 3.503 |
| perfect_pipeline (max engine active = DMA) | 3.503 |
| serialization_gap | 0.439 (11%) |
| compute_bound, hw flops incl. transpose (29.77 GFLOP / 167 TFLOP/s) | 0.178 |
| compute_bound, useful model flops only (1.37 GFLOP) | 0.008 |
| wall_ms (clean-flag median, 10 samples) | 4.908 |

Engine active (absolute, from active_time_percent × total_time): DMA 3.503ms (88.9%), GpSimd 2.096ms (53.2%),
Scalar 1.195ms (30.3%), Tensor 1.661ms (42.1%), Vector 1.108ms (28.1%). mfu_estimated=0.22%,
mbu_estimated=54.6% → implied peak bandwidth ≈ 806 GB/s (achieved 1.541GB/3.503ms = 440 GB/s), consistent
with r0's ~787 GB/s estimate on the same box.

**What round 1 fixed (confirmed by this profile, not just inferred)**:
- `has_op_attribution`: **99.75%** of instructions carry a `layer`/source location now (vs. 72% Unknown in
  r0). The pre-transposed, fused weight matmuls (`qasr_port.py:146` wqkv, `:206` wo, `:209` wgu, `:213` wd,
  `:221` lm_head split-cat) are each individually named and each reads its weight **exactly once per step,
  with zero duplicate reads and zero spill**: wgu 352.3MB, lm_head 311.2MB, wqkv 234.9MB, wd 176.2MB, wo
  117.4MB = 1192.0MB, matching the weights' true byte size (e.g. wgu = 1024×6144×2B×28 = 352.3MB exactly).
  r0 had found the lm_head weight read **twice** (622MB) plus 155.6MB spill from a compiler retile; that is
  gone — R1 from round 0 (the lm_head split) incidentally fixed the double-read, even though the agent noted
  it did no quantization.
- Tensor-engine TRANSPOSE instruction count fell from 88,800 (r0) to 14,952; `adjusted_flops` fell from
  182.2 GFLOP to 29.8 GFLOP (−84%). Tensor engine active time fell from 62.5% to 42.1% and is no longer close
  to DMA's share.

**What is still costing bytes (the remaining 1.541GB of DMA, now almost all attributed)**:
1. Weight reads, 1192.0MB (77.3% of DMA bytes) — now at the floor for bf16 (read-once, no reload/spill).
2. KV-cache functional full-tensor rewrite (`qasr_port.py:175-180`, `torch.where(sel, k, kc[i])` per layer):
   GpSimd DMA_DIRECT2D, 672 instr, reads 88.08MB + writes 88.08MB = 176.2MB (11.4%) — exactly
   `28 layers × 8 heads × 768 × 128 × 2B × 2(k,v) × 2(read+write)`. This is the unavoidable cost of returning
   the whole cache tensor every step in a functional graph; round 1 already tried the alternative
   (`QASR_KV=inplace`, `index_copy_` graph-input mutation) and it was **slower** (6.98 vs 4.50 ms/step) despite
   moving far fewer bytes — graph-input mutation overhead dominates. Not re-recommending that exact fix.
3. An **unattributed spill round-trip**: `spill_save_bytes`=78.64MB + `spill_reload_bytes`=81.92MB = 160.6MB
   (10.6% of DMA bytes), all on Scalar/Sync `DMA_DIRECT2D`, all inside the "Unknown"-layer bucket (no
   module/source tag — same blind spot r0 flagged for a different, now-mostly-fixed, bucket). Packet pattern
   (many ~32KB chunks, one set per layer-ish semaphore range) suggests a per-layer SBUF-overflow spill, most
   likely from the GQA head-fold reshape (`qg = q.reshape(1,nkv,nrep*T,hd)`) or the mask broadcast
   (`mg = mask.repeat(1,1,nrep,1)`), but this is evidence by elimination (what's left once weights+cache are
   accounted for), not direct attribution — flagged as needing confirmation before investing engineering time.

**Byte-budget reconciliation**: 1192.0 (weights) + 176.2 (cache) + 160.6 (spill) + 12.1 (misc) = 1540.9MB ≈
measured `dma_transfer_total_bytes` 1540.8MB. At the achieved rate (440 GB/s), these convert to 2.71ms /
0.40ms / 0.37ms / ~0ms respectively — summing to the measured 3.50ms DMA-active time almost exactly, so the
byte attribution is internally consistent and a safe basis for ranking levers.

**Verdict**: kept as the round-1 profile baseline for round-2 improvement. DMA is still the bottleneck (88.9%
active, 11% serialization gap — same well-pipelined shape as r0), but the composition changed completely: r0's
top finding (lm_head double-read + 99.2%-transpose hardware flops) is now fixed. The new top lever is
precision (weight bytes, which are already at the bf16 floor with zero waste) rather than redundant traffic.


### R2-round2 / R2: remove per-layer KV rewrite + stack ("late" single where) -- REJECTED (2026-10-15 23:05 UTC, improver)
**Why we tried it**: profile_r1 has 160.6 MB of unattributed spill (save 78.6 MB + reload 81.9 MB), in per-layer groups, plus a 176 MB KV round trip. In decode (T=1) the GQA mask is not repeated (`mask.shape[2]==1`), so the profiler's mask/reshape hypothesis does not apply to decode. The remaining per-layer suspect was the full-cache `torch.where` per layer followed by `torch.stack` of 28 full-size layer caches. That is a likely place for an intermediate copy.

**Change** (`QASR_KV=late`): each layer attends over the OLD cache (mask `ar < pos`) plus the new k/v as one extra score column, the same math as r1's rejected inplace path. Then a single `torch.where(ar==pos, k_new[28,8,1,128], kc)` runs over the stacked cache at the end of the graph. Graph-input mutation is not used.

**Result** (`r1/bench_decode.py`, decode-only, 60 steps, median of steps 10..59):
- where (current): 4.61 ms
- late: **11.08 ms** (2.4× slower)

Both have occupancy 13440/13440 with 0 writes beyond pos. CPU fp32 port with late: 3/3 token-exact.

**Verdict: REJECTED** (much slower). Like r1's inplace attempt, the split-softmax / extra-column structure is expensive on this backend. The code stays behind `QASR_KV=late`; the default stays `where`. The spill source is still not localized; a profile capture of the late graph was not taken.


### R2-round2 / R1: weight-only quantization of decode weights (int8, fp8) -- REJECTED (2026-10-15 23:05 UTC, improver)
**Why we tried it**: profile_r1 found that the 5 heavy weight reads (wqkv, wgu, wd, wo, lm_head) are 1192 MB per decode step, 77% of DMA bytes, already read once each. DMA is the bottleneck (88.9% active). Halving the bytes looked like it would save about 1 ms per step.

**Microbench** (`r2/mb_quant.py`): 28 chained MLP blocks ([1024,6144] gate|up, [3072,1024] down), M=1. The weight is dequantized in the graph (`x @ w.to(bf16)`) with a per-out-channel scale applied to the output. Median of 20 synced calls:

| weight dtype | med ms |
|---|---|
| bf16 | 1.98 |
| int8 (convert + scale) | **2.39** (slower; the in-graph int8→bf16 convert costs more than the bytes it saves) |
| float8_e4m3fn | compile error `NCC_EVRF051 Data type F8E4M3FN is not supported on TRN1/TRN2`; the suggested `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` flag is rejected by this neuronx-cc (`NCC_EARG002`) |
| float8_e4m3fnuz | runtime error `Unknown MLIR element type: f8E4M3FNUZ` |
| float8_e5m2 | 1.59 (−20%) |

**Accuracy of e5m2** (the only variant that was faster): implemented in `qasr_port.py` behind `QASR_Q=e5m2 QASR_QW=<lm,qkv,o,gu,d>`, default off. Tested with the CPU fp32 port on the 10-sample gate, token-exact vs the CPU fp32 HF reference. The CPU numbers are an upper bound; device bf16 can only add error.
- lm_head only: 9/10 (sample 0 diverges at token 0)
- gu,d only: 8/10
- all 5 groups: 6/10, WER 0.0472 vs 0.0433

**Verdict: REJECTED.** int8 is slower; e4m3 does not compile or run on this stack; e5m2 has only 2 mantissa bits and breaks token-exactness even in fp32 on CPU. The code stays behind `QASR_Q` (default `none`).


### R2-round2: lagged EOS check / device-resident pos tensors in decode loop -- REJECTED (no gain) (2026-10-15 23:05 UTC, improver)
**Why we tried it**: the device time per step is 3.94 ms but the wall time is 4.91 ms, so about 1 ms per token goes to host dispatch, the H2D copy of `pos`, and the D2H token read. Idea: dispatch step n+1 (input = the device token of step n) before the blocking read of token n (`QASR_LAG=k`), and build the `pos` tensors on device once (`QASR_POSDEV=1`).

**Result**: full `run.py -n 10`, 3 warm repeats. `step_amort` = decode_total/(ntok-1), median over samples. All runs 10/10 token-exact, WER 0.0433.

| config | ms/token | e2e median ms |
|---|---|---|
| LAG=0 POSDEV=0 (r1) | 4.93 | 139.5 |
| LAG=0 POSDEV=1 | 4.94 | 139.7 |
| LAG=1 | 5.00 | 138.6 |
| LAG=2 | 5.21 | 144.9 |

**Verdict: REJECTED.** No gain, which suggests the compiled-graph call is effectively synchronous, so host overhead cannot overlap the device. Both switches stay in run.py with default off. This finding motivated the next trial: run K tokens per graph call.


### R2-round2 / R3: K decode steps per graph call (DecodeK, QASR_MULTI) -- KEPT (2026-10-15 23:18 UTC, improver)
**Why we tried it**: per profile_r1, one decode step is 3.94 ms on device but 4.91 ms wall. The lagged-dispatch trial showed the ~1 ms per-call host/runtime overhead cannot be overlapped, so the only way to cut it is to make fewer calls. profile_r1's R3 (amortize work across tokens) suggested this direction. A draft model is not needed: greedy decoding can feed the argmax token straight back in-graph.

**Change**: `qasr_port.DecodeK(hf, S, K)` unrolls K greedy decode steps in one compiled graph. Each step is the unchanged r1 `Decode.step` (embedding gather of the in-graph argmax token, functional where-cache). It returns `(toks[K], last_tok, kc, vc)`. In run.py the switch is `QASR_MULTI=K` (default now 4; `QASR_MULTI=1` restores one token per call). The host reads K tokens per call and truncates at EOS/max_new. Up to K-1 steps past EOS are wasted, and cache writes at pos >= S are no-ops.

**Result**: full `run.py -n 10`, 3 warm repeats per sample. `step_amort` = decode_total/(ntok-1), so wasted overshoot steps are included; median over samples.

| config | ms/token | e2e median ms | token-exact |
|---|---|---|---|
| K=1 (r1 path), session A | 4.93 | 139.5 | 10/10 |
| K=2 | 4.42 | 129.7 | 10/10 |
| K=4 | 4.21 | 127.6 | 10/10 |
| final confirm, K=4 default | **4.23** | **129.4** | 10/10, WER 0.0433 |
| final confirm, K=1 same session | 4.79 | 134.8 | 10/10, WER 0.0433 |

That is −11.7% per token against the same session, and −14% against the r1 baseline of 4.91 ms. Longer utterances gain the most (88-token sample: 4.44 → 4.08 ms/token at K=4). Short ones gain less because of overshoot (14-token sample: 5.07 ms/token). Encoder and prefill are unchanged.

**Cost**: the K=4 decode graph takes about 305 s to compile cold (about 145 s for K=2). The NEFF cache absorbs it.

**Verdict: KEPT** (K=4 default).


### Profile round 2: DecodeK (K=4) hot-graph attribution (2026-10-15 23:32 UTC, profiler)
**What**: Profiled the round-2 default decode graph, `DecodeK` with `QASR_MULTI=4` (4 greedy steps unrolled
in one compiled call, kept from improve_r2's R3). Compiled fresh into `neff-cache-debug-r2` with
`XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 NEURON_FRAMEWORK_DEBUG=1` (script `profiles/r2/gen_debug_neff_r2.py`), found one
large NEFF (9.39MB, next biggest 11.9KB → unambiguous), captured with
`neuron-explorer capture --profile-nth-exec=2 --ignore-exec-errors` (DGE off, single core), ingested as display
name `dec_r2` → `profiles/ne-data/profiles/global/dec_r2@latest`. Never timed from this run (debug flags).

**Why**: round 2 changed the hot unit from a single decode step (profiled in round 1 as `dec_r1`, 2.58MB NEFF,
3.942ms device/step) to a 4-token unrolled call; round-1's profile numbers no longer describe the graph that
actually runs, so re-attribution was needed before trusting any further recommendation.

**Result** (per one DecodeK call = 4 tokens, Summary.parquet `total_time` and `*_active_time_percent`, NOT the
raw overlapping-engine Instruction-sum which inflates by 3–6x same as it did in round 1):
- total_time = 14.497ms (3.624ms/token) vs r1's 3.942ms/token device → **-8.1%** device-level improvement
  that's independent of host-overhead amortization (bonus beyond what R3 in improve_r2 claimed).
- dma_active_time = 12.862ms (88.72%, same magnitude as r1's 88.9%) → still DMA-critical-path, serialization_gap
  11.3% (1.635ms/call) vs r1's 11% — equally well pipelined, byte cuts should still convert ~1:1.
- Byte breakdown of the 5.689GB/call moved: **weight reads 4762.4MB (83.7%, all on one source line
  `qasr_port.py:155` = `return x @ w` inside the shared `W()` helper, confirming the bf16 read-once floor holds
  even unrolled — 1190.6MB/token, within 0.1% of r1's 1192.0MB/step)**; spill save+reload 714.58MB (12.6%,
  all in the "Unknown" layer bucket exactly matching Summary's spill_save/spill_reload fields, 178.6MB/token vs
  r1's 160.6MB/step, **+11% worse per token**); KV-cache + misc 212.15MB (3.7%, only 53MB/token vs r1's
  176.2MB/step, **-70%**, apparently the unrolled graph schedules the per-layer cache rewrite far more cheaply).
  At the achieved 442.5GB/s (matches r1's ~440GB/s, same box/bandwidth): weight=2.691ms/token, spill=0.404ms/token,
  kv+misc=0.120ms/token, summing to the measured 3.215ms/token DMA-active (vs r1's 2.71+0.40+0.365=3.503ms/token).
- Tensor engine 34.0% active (down from r1's 42.1%, DMA grew its share), GpSimd 58.2% (up from 53.2%) — both still
  well below DMA's 88.7%, so neither is the critical path (next-engine-floor, same conclusion as r1).
  TRANSPOSE is 45.98/51.47 GFLOP = 89.3% of hardware flops (r1: 95%) — not worth chasing while Tensor < DMA.
  Model useful flops 5.49GFLOP/call = 1.373GFLOP/token, identical to r1 (same real workload). mfu≈0.23%,
  mbu≈54.8% (peak used: profiler backs out ~807GB/s bandwidth, 167TFLOP/s LNC2 bf16 dense peak for mfu — both
  consistent with r1 on the same box).
- Reconciliation: device 14.497ms vs wall 16.936ms (= 4 x measured clean-flag step_amort median 4.2339ms/token
  from `artifacts/run_r2_final.json`, 10-sample synced-every-call harness) = 85.6%, comparable to (better than)
  r1's 80.3%.

**Verdict**: DMA remains the bottleneck and weight reads remain the dominant, already-at-floor cost (83.7% of
bytes, even higher share than r1 since other costs shrank). The K=4 unroll (kept from improve_r2) delivered a
real device-side efficiency gain beyond the host-overhead amortization it was built for (KV-cache round trip
dropped 70%), which is new information for round 3: pushing K higher is now better motivated than improve_r2's
own notes suggested ("Larger K was not tried"). The spill-roundtrip bucket grew proportionally worse per token
(+11%) and is now the second-largest controllable chunk after weights (12.6% of bytes) — still unattributed to
a specific source line (same caveat as r1's R2, which was never fixed and whose one attempted fix made things
2.4x slower). Full details and the 5-line scripts used are in `profiles/r2/`.


### r3 R1: larger DecodeK unroll (K=6, K=8) — rejected (2026-10-16 00:08 UTC, improver)
**Motivation (profile_r2 R1):** K=1→4 cut per-token device time 8% (KV round-trip bytes −70%), so a larger K might keep paying.
**Change:** none in code; `QASR_MULTI=6` / `8` (existing switch). Same-session full 10-sample run (`r3/sweep_k.sh`, clean flags, 3 warm repeats/sample, median across samples).

| K | token-exact | WER | step_amort ms/token (median) | e2e median ms | cold compile of DecodeK |
|---|---|---|---|---|---|
| 4 (baseline) | 10/10 | 0.0433 | **4.26** | 128.2 | ~369 s |
| 6 | 10/10 | 0.0433 | 4.36 | 127.0 | ~521 s |
| 8 | 10/10 | 0.0433 | 4.27 | 130.2 | ~773 s |

Per-sample: K=8 does help the long utterance (88 tok: 4.16→3.90 ms/token) and some short ones, but the wasted post-EOS steps (up to K-1 full steps) cancel it on the 14–31-token utterances (e.g. 31 tok: 4.48→4.06, but 27 tok: 4.36→4.74). Net change is zero on the median, and compile time doubles.
**Verdict:** rejected (<2% change, and it goes the wrong way for K=6). Default stays K=4.


### r3 R2a: per-layer K/V cache lists instead of stacked cache (QASR_KVLIST=1) — rejected (2026-10-16 00:18 UTC, improver)
**Motivation (profile_r2 R2):** the unattributed spill is 357.29 MB saved + 357.29 MB reloaded per DecodeK call, i.e. 89 MB/token. That is almost exactly one full K+V cache (28 layers × 2 × 8 × 768 × 128 × 2 B = 88.1 MB). Hypothesis: `torch.stack(ks)` at the end of `_layers` makes every layer's rewritten cache stay live until the stack, so it gets spilled.
**Change:** `QASR_KVLIST=1` makes `_layers` return tuples of 28 per-layer `[8,768,128]` tensors (prefill and decode), so there is no stack and the graph has 2×28 cache inputs/outputs.
**CPU fp32:** 3/3 token-exact.
**Device (same session as the R1 sweep, K=4, full 10-sample run):** 10/10 token-exact, WER 0.0433. step_amort **4.43 ms/token vs 4.26 baseline (+4% slower)**, e2e median 132.3 vs 128.2 ms. Prefill got slightly faster (11.05 vs 11.66 ms), but that runs once per utterance and does not make up for the decode loss. The extra per-call I/O (112 cache tensors instead of 4) probably costs more than the stack does.
**Verdict:** rejected. Left behind the switch, default off. No profile was captured, so it is still unconfirmed whether the spill moved.


### r3 R2b: write the K/V cache once per DecodeK call (QASR_KV=lateK) — rejected (2026-10-16 00:30 UTC, improver)
**Motivation (profile_r2 R2):** the spill is 89 MB/token, about one full K+V cache. In DecodeK every inner step rewrites the whole `[28,8,768,128]` cache with `torch.where` and passes it on to the next inner step. The idea was to write the cache once per K=4 call instead of four times.
**Change:** `QASR_KV=lateK` → new `DecodeKLate` class (qasr_port.py, opt-in; run.py picks it only when the env var is set). Inner step j attends over the unmodified input cache (mask: slots < pos0) plus the j+1 new k/v columns, which are kept in a small per-layer list. It uses a two-part softmax (shared max, exp, joint denominator, normalized before the bf16 cast; `QASR_LATENORM=0` instead divides after the PV matmul). After the 4 steps the new columns are written into the cache once, with an index_select gather and a where. This is unlike r2's `QASR_KV=late`, which concatenated scores (S+1 columns) and had K=1.
**CPU fp32:** 10/10 token-exact, WER 0.0433.
**Device (same session, full 10-sample run, clean flags):**

| config | token-exact | step_amort ms/token | e2e median ms |
|---|---|---|---|
| K=4 baseline (where per inner step) | 10/10 | **4.21** | 127.3 |
| K=4 lateK | 10/10 | 4.97 (+18%) | 146.1 |

**Verdict:** rejected; the switch stays off by default. This is the second time (with r2's `late`, 2.4× slower) that "attend over old cache + separate new columns" has lost to the plain functional where-rewrite on this neuronx-cc. The likely cause is the second small matmul/softmax branch per layer per step, plus the gather, costing more than the cache bytes saved. Unless someone profiles the variant, the per-step where-rewrite should be treated as the best form available at the torch level.


### Documentation stage: batch-size sweep (BS 1/2/4/8/16/32) — flat throughput, as expected for a batch=1-only port (2026-10-16, docs)
**Why**: the final-config code (K=4 `DecodeK`, `QASR_LMSPLIT=8 QASR_GQA=group QASR_WT=tf QASR_KV=where`, all defaults) had never been run past 10 samples. The repo convention asks for a batch-size sweep as part of documentation.

**Caveat established first**: `qasr_port.py`'s three compiled graphs (`Encoder`, `Prefill`, `DecodeK`) all hardcode a leading batch dim of 1 (e.g. `x.view(1, nmel, C, 100)`, `ids_p = torch.full((1, P), ...)`, caches `[28, 8, S, 128]` with no batch axis). Unlike the `whisper-large-v3` contribution in this repo (which decodes B utterances in lockstep with a real batch dimension), this port was never extended to a batch dim > 1 — the audio-encoder windowing and the prefill audio-placeholder packing were both built assuming "batch 1 ⇒ valid frames are a flat prefix", which doesn't hold once multiple independent-length utterances share a batch axis. So "batch size B" in this sweep means **B independent utterances transcribed sequentially, in one process, through the same (already-compiled) batch=1 graphs** — it measures dataset-level sequential throughput, not a batching speedup. Extending the port to real batching is listed as a known limitation/next step.

**Setup**: extended the CPU fp32 reference from 10 to 32 `librispeech_asr_dummy` samples (`python cpu_ref.py 32`; first 10 rows reproduce byte-for-byte, WER over the 32 is 0.0440 vs 0.0433 over the first 10). Ran `QASR_RUNTAG=bs<N> python run.py -n <N> --repeats 3` for N in {1,2,4,8,16,32}, one process per N, against the already-warm NEFF cache from the improvement rounds (no new bucket shapes are hit — the 73-sample dataset's utterances all land in the existing encoder {8,16,32} / prefill {128,256,512} buckets, so each run's "cold" first-sample time is dynamo retracing + NEFF-cache lookup, not a neuronx-cc compile: ~33 s for the first (8,128) pair, ~9 s for (16,256), ~8 s for (32,512)/(32,256), ~0.1-0.2 s for repeats of an already-hit bucket in the same process).

**Result** (medians over the N utterances, 3 warm repeats each; throughput = N / (sum of each utterance's median e2e latency), i.e. true sequential throughput on one NeuronCore):

| N (sequential utterances) | token-exact | WER | median e2e / utt (ms) | median step_amort (ms/tok) | median encoder (ms) | median prefill (ms) | utt/s | tok/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 1/1 | 0.0588 | 117.0 | 4.47 | 10.70 | 8.02 | 8.55 | 196.7 |
| 2 | 2/2 | 0.0370 | 98.9 | 4.66 | 10.65 | 7.91 | 10.11 | 187.1 |
| 4 | 4/4 | 0.0120 | 136.9 | 4.46 | 11.25 | 9.84 | 7.55 | 198.1 |
| 8 | 8/8 | 0.0350 | 127.3 | 4.36 | 11.20 | 11.56 | 6.37 | 206.2 |
| 16 | 15/16 | 0.0388 | 123.3 | 4.37 | 11.11 | 10.00 | 6.82 | 200.9 |
| 32 | 30/32 | 0.0423 | 99.6 | 4.34 | 10.85 | 8.16 | 8.46 | 198.1 |

(Per-N WER is computed over that N-sample prefix against the dataset's ground-truth text, so it moves around with which utterances are included — e.g. N=4's 0.0120 is luck of the draw, not a real improvement. The 10-sample WER reported elsewhere, 0.0433, is the one used as the accuracy gate.)

**Verdict**: as expected given the batch=1-only graphs, throughput does **not** scale with N — it stays in a 6.4-10.1 utt/s / 187-206 tok/s band regardless of N, because every utterance still goes through the same single-sequence Encoder→Prefill→DecodeK pipeline one at a time; "batch size" only changes which/how many utterances are summed over, and the per-utterance latency depends mostly on that utterance's own length (bucket C/P and generated-token count), not on N. The small WER/latency swings between rows are sampling noise from which utterances fall into each prefix, not a batch-size effect. This is a real, measured limitation, not a bug: true batching would need the Encoder/Prefill/DecodeK graphs rewritten with a real batch axis (as `whisper-large-v3`'s `transcribe_batch()` does), which was not attempted in this run. Numbers are in `artifacts/run_bs{1,2,4,8,16,32}.json`.
