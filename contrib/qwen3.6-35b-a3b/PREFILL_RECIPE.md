# Max Prefill Throughput Recipe — Qwen3.6-35B-A3B (packed C32)

Reproduces the fastest validated **prefill** configuration. The current best is
**≈3,457 aggregate prompt tok/s** at **TP=8/LNC=1** (§3c) — **+23.8%** over the
TP=4/LNC=2 configuration. Both use the identical packed-C32 DeltaNet kernel; LNC=1
just gives twice the tensor engines. LNC=1 additionally requires both replicated
`[V, H]` vocab tensors to be **load-time vocab-sharded** to fit the ~12 GB/rank
budget, and the ~30-line `moe_cte` LNC=1 patch (§3c). The LNC=1 number was measured
before the hoisted packed transposes landed, so it has not yet been re-baselined
with them.

The TP=4/LNC=2 configuration (§3a) is **≈2,792 aggregate prompt tok/s** (BS=2,
N=20000 tokens, query bucket 1024, DeltaNet **stable C32** block-diagonal inverse
with **4-stream block-diagonal packing**, **SBUF-resident intermediates**, and
hoisted packed transposes, fused NKI route packer, optlevel-1) — **+22.6%** over
unpacked C32 (≈2,277 tok/s) and **+33.6%** over the paired-C16 baseline (≈2,090).
It remains the simplest to reproduce (no vocab sharding, no nkilib patch) and is
the reliable fallback.

The packing (`DN_PACK_C32=1 DN_PACK_N=4`) folds four independent DeltaNet streams
(4 of the 8 per-core V-heads) into one P=128 block-diagonal chunk tile, so the
tiny 32×32 intra-chunk matmuls/transposes run once on a P=128 tile instead of four
times on P=32 — far fewer instructions and the full PE partition dimension used.
It is validated **bit-identical** to unpacked C32 (see §6). C16 remains the
compiled-in default and reliable fallback; **packed C32 is the opt-in fastest
path**. All configs use identical shapes, topology, and the fused NKI route packer.

All host/path values live in `.env` — copy `.env.example` to `.env` and fill in.
This recipe references: `QWEN35_NATIVE_IMAGE`, `QWEN35_MODEL_DIR` (BF16 — prefill
is BF16, no FP8 experts needed), `QWEN35_NKILIB_DIR`, `QWEN35_COMPILER_CACHE_DIR`,
`QWEN35_RUN_HOST`/`QWEN35_RUN_REGION` (Trn2). Nothing is hard-coded here.

---

## 1. Host requirements — compile **and** run on the SAME Trn2

Unlike decode, prefill's best topology is **TP=4 / LNC=2**, which **cannot be
cross-compiled on Trn1** (LNC=2 fails at `dist.init_process_group("neuron")` on
Trn1 — Trn1 cross-compile is TP=8/LNC=1 only). So prefill compiles **natively on
the Trn2** and benches in the same run. No device-to-device transfer step.

| Item | Requirement |
|---|---|
| Host | **Trn2.3xlarge** (`QWEN35_RUN_HOST`), native LNC=2 (4 logical cores) |
| **Swap** | **~11 GB swap required** — Trn2 has **none by default**; add a swapfile first (§2) or the compile OOMs |
| optlevel | **O1 only.** O2/O3 are compile-cost-prohibitive (walrus >51 min on a single 10-layer segment, OOM-risks the box) |
| Cold compile time | ~27 min at O1 (compiles all regions **and** benches + fingerprints in one run) |
| nkilib | validated at revision `1ee625782`; point `QWEN35_NKILIB_DIR` at that checkout |
| Container | internal Neuron DLC (`QWEN35_NATIVE_IMAGE`) with host Neuron lib available |

The compile is split into 4 regions of 10 layers (`--splits 4`) to keep per-region
peak RAM manageable. The stable C32 inverse (`_tri_inverse_blockdiag`) and the
4-stream packer (`_chunk_pack4`) are in `kernels/deltanet_chunked_prefill_35b.py`,
and `compile_prefill_trn2.sh` now **defaults to packed C32**
(`CHUNK_SIZE=32 DN_STABLE_C32=0 DN_PAIRED_BATCH=0 DN_PACK_C32=1 DN_PACK_N=4`);
the unpacked-C32 and paired-C16 fallbacks are one env override each (§3b). The
fused NKI route packer (`MOE_CTE_NKI_PACK=1`) is set inside the compile script.

---

## 2. Add swap (one-time, on the Trn2)

```bash
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile \
  && sudo mkswap /swapfile && sudo swapon /swapfile
swapon --show   # confirm ~16G available
```

---

## 3. Compile + bench (one command, on the Trn2)

`deploy/compile_prefill_trn2.sh` compiles the prefill graph (`--bucket-compile 1`)
**and** runs the throughput benchmark (`--prefill-bench 20000`) + fingerprints, in
a single native run.

### 3a. Fastest — packed C32 (≈2,776 tok/s, +21.9%) — default

The script defaults to packed C32 (`DN_PACK_C32=1 DN_PACK_N=4`); no flags or edits
needed:

```bash
source .env
deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR/c32pack4"
```

### 3b. Fallbacks — unpacked C32 (≈2,277), 2-stream pack (≈2,561), or paired C16 (≈2,090)

Override via environment; each config changes the traced graph, so give each its own
`--cache-dir`:

```bash
source .env
# unpacked C32
DN_PACK_C32=0 deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR/c32"
# 2-stream pack (P=64)
DN_PACK_N=2 deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR/c32pack2"
# paired C16
CHUNK_SIZE=16 DN_STABLE_C32=1 DN_PAIRED_BATCH=1 DN_PACK_C32=0 \
deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR/c16"
```

`CHUNK_SIZE` / `DN_PACK_*` change the traced graph, so **each config needs its own
`--cache-dir`** — configs will not (and must not) cache-hit each other's cache root,
and the script's metadata guard refuses to mix them.

- Both paths need `QWEN35_NATIVE_IMAGE`, `QWEN35_MODEL_DIR` (BF16), `QWEN35_NKILIB_DIR`.
- The script pins `DN_CHUNK_NKI=1`, `MOE_CTE=1 MOE_CTE_NKI_PACK=1`, batch-size 2,
  `--max-seq-len 20480`, `--prefill-bench 20000`, and
  `NEURON_CC_FLAGS="--target trn2 --lnc 2 --optlevel 1 --hbm-scratchpad-page-size 64"`.

**Kick off headless** (~27-min compile — don't hold the terminal):
```bash
nohup bash -c 'deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR/c32pack4"' \
  > /mnt/nvme/runlog/prefill_bench.log 2>&1 &
# poll: grep -E 'tok/s|prompt|throughput|compiled|Error' /mnt/nvme/runlog/prefill_bench.log
```
Re-runs are cache-hot from the matching `--cache-dir` (skip §2/compile).

### 3c. Fastest — TP=8/LNC=1 (≈3,457 tok/s, +24.5%)

Same packed-C32 kernel, but at TP=8/LNC=1 (8 logical cores → twice the tensor
engines). Two extra requirements:

1. **The `moe_cte` LNC=1 patch.** `deploy/compile_prefill_trn2.sh` applies
   `patches/nkilib-lnc1-moe-cte.patch` to the mounted nkilib host-side, idempotently
   (`git apply --check` / `--reverse --check`), and fails loudly if it cannot. The
   patch is ~30 additive lines that relax a `NUM_SHARDS == 2` shard-count guard (the
   old `[NCC_INKI016] ... only work on TRN2` error was a mislabeled shard guard, not
   a hardware one) and skip the peer `sendrecv` at one shard — all inert at LNC=2
   (proven: the LNC=2 fingerprint is bit-identical with the patch applied).
2. **Vocab-sharded head + embedding**, ON by default at `--lnc 1`
   (`PREFILL_SHARDED_LM_HEAD=1 PREFILL_SHARDED_EMBED=1`). The two replicated `[V, H]`
   tensors are ~1.02 GB/rank each; sharded across TP they drop the module 10.73 →
   8.95 GB/core, which is what makes 40L fit the ~12 GB/rank budget. Both reconstruct
   the replicated result exactly (disjoint contiguous ranges, zero-pad + sum-reduce).
3. **`--scratchpad-page-size-mb 256`** — the sweet spot (5× 256 MiB pool + tiny
   overflow); pg64 and pg512 both OOM at 40L.

```bash
source .env
deploy/compile_prefill_trn2.sh \
  --tp 8 --lnc 1 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --scratchpad-page-size-mb 256 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR/tp8-lnc1-c32pack4"
```

Measured **3,456.8 aggregate prompt tok/s** (TIMED 11.572 s), warm≡timed
fingerprint `sum=-3.17835375e+05 norm=1.20818213e+03 top5=[517,607,261,294,15089]`,
all-finite. The fingerprint is **not** bit-identical to LNC=2 (different TP
reduction order — norm 0.45% / sum 1.7% apart, top-1/top-3 identical, 4/5 top-5
shared, all finite + sane magnitude), but the greedy coherence continuation is
**token-identical** to the LNC=2 baseline: prompt `760,6511,314,9338,369` → first
tokens `[11751, 13, 198, 760, 6511, 314, ...]`, matching the C32/C16 baseline.

Its own `--cache-dir` is mandatory — topology + the sharding flags change the
traced graph, and the metadata guard refuses to mix it with the LNC=2 cache.

---

## 4. Reading the result

The bench reports **aggregate prompt tok/s** (total prompt tokens across the BS=2
batch ÷ prefill wall-time). References for the two configs:

| Config | Wall time | Aggregate prompt tok/s |
|---|---:|---:|
| **Packed C32 ×4, TP=8/LNC=1** (§3c, vocab-sharded head+embed, pg256; pre-hoisted-transpose) | **11.572 s** | **3,456.8** |
| Packed C32 ×4, SBUF-resident + hoisted-transpose finish, TP=4/LNC=2 (`DN_PACK_C32=1 DN_PACK_N=4`) — LNC=2 default | 14.328 s | 2,791.7 |
| Packed C32 ×4, SBUF-resident + transpose-once finish, TP=4/LNC=2 | 14.411 s | 2,775.6 |
| Packed C32 ×2 (`DN_PACK_N=2`), TP=4/LNC=2 | 15.620 s | 2,560.9 |
| Unpacked C32 (`DN_PACK_C32=0`), TP=4/LNC=2 | 17.568 s | 2,276.9 |
| Paired C16, TP=4/LNC=2 | 19.141 s | 2,089.7 |

A per-run token-ID/state fingerprint is printed for correctness; the warm and
timed fingerprints must be identical and finite. Compare across builds to confirm
identical output.

---

## 5. Notes / levers

- **BS=4** also fits at 512-token buckets and measured 39.788 s / 2,010.6
  aggregate tok/s (paired C16). BS=1 single-prompt best is 1,482.8 tok/s.
- **FP8 prefill** is a future lever (nkilib `moe_cte` MX variant), not yet
  integrated — this recipe is BF16.
- Do **not** raise optlevel; O2/O3 do not finish in reasonable time/RAM here, and
  O3 does not extract any extra prefill throughput (measured flat at 2,273.5 tok/s
  after a ~3h45m compile, and it perturbs numerics slightly).
- **`DN_STREAM_WINDOW`** (software-pipelining the independent-stream loop) was tried
  and is a **negative lever** — flat at O1 and O3 (the compiler declines to overlap
  the unrolled streams). Left in, default `1`. The win came from *packing* (fewer,
  larger matmuls) rather than *overlap*.

---

## 6. Why packed C32 is safe (and why the packing helps)

**Stable C32 inverse.** C32 halves the DeltaNet chunk count vs C16, but it requires a
**numerically stable chunk-matrix inverse**. `_tri_inverse_blockdiag` splits the 32×32
chunk matrix into two 16×16 diagonal blocks plus a coupling term and inverts the blocks
by doubling:

- the naïve **full-32 doubling** overflows on near-1-decay streams → NaN at bs2
  (root-caused to layer 18 / near-1 `T` entries); **unusable**;
- a **Horner series** is stable but ~4× costlier → −2.9%, so it is not used;
- the shipped **block-diagonal doubling** is both stable and cheap → 2,276.9 tok/s.

**4-stream packing.** The prefill wall was ~50% a serialization gap: the DeltaNet
intra-chunk inverse issues a huge number of *tiny* 32×32 matmuls/transposes (P=32, so
1/4 of the PE partition dim) whose per-instruction + weight-load overhead — not FLOPs —
dominates (MFU ~3.5%). `_chunk_pack4` folds four independent streams (4 of the 8 per-core
V-heads) into one **P=128 block-diagonal** tile: block-diagonal masks
(`pack_m_incl`/`pack_m_strict`/`pack_eye`) zero every cross-stream term, so each 32-row
bank is an independent solve, and the packed inverse uses `_tri_inverse_blockdiag` with a
block-diagonal lower-left mask (16-wide sub-blocks). The intra-chunk matmuls thus run
**once on P=128 instead of four times on P=32** → ~4× fewer tiny-matmul instructions,
full partition dim, ~19.4% faster. (Off-stream blocks are exactly zero, so each stream
gets the identical baseline result.)

**SBUF-resident intermediates.** The packed intra-chunk results
(`k_c/g_cum/v_corr/k_cumdecay/intra/qSe`) stay in SBUF and are handed to the per-stream
finish directly (extracted via on-chip `[row:row+C]` copy) instead of round-tripping
through HBM scratch. Removes 6 HBM scratch buffers + their writes and cuts region HBM
traffic ~0.5 GB; +0.7% (2,718.9 → 2,738.0), bit-identical. (The remaining prefill wall is
Tensor-floor + MoE bound, not DMA — the scratch DMAs were largely overlapped.)

**Transpose-once finish.** The per-stream finish transposed `k_cumdecay`/`qSe` 4× on
`[C,K_DIM]` tiles (P=32, 1/4 of the PE array). `_finish_pack4_bank` instead transposes
the packed `[P,K_DIM]` tile ONCE (→`[K_DIM,P]`) in `_chunk_pack4`; each stream free-slices
`[0:K_DIM, row:row+C]` as its matmul stationary (the transpose of that stream's block, so
results are identical). Removes 2 transposes + 2 DMA reads per stream. +1.4%
(2,738.0 → 2,775.6), bit-identical.

**Constant/intra transpose hoists.** The packed inclusive-mask transpose is constant,
so it is now built once outside the recurrent chunk loop. The packed block-diagonal
`intra` tile is also transposed once at P=128; each finish copies its diagonal C32 block
to partition zero instead of issuing four C32 transposes. Existing one/zero tiles are
reused as well. A repeated production-shape profile improved 2.5156 → 2.4435 ms
(+2.87%) and reduced matmul/transpose instructions 6,792 → 6,537. Confirmed with a
matched full 40-layer same-host A/B on the reference Trn2 (BS=2, N=20000, bucket
1024, cache-hot): baseline **2,774.1** → candidate **2,791.7** aggregate tok/s
(**+0.63%**, 14.419 s → 14.328 s), warm≡timed fingerprints **bit-identical** to the
reference below (`sum=-3.12377031e+05 norm=1.20273230e+03 top5=[517,607,261,290,294]`).
The table headline now reflects the candidate (2,791.7). An earlier A/B on a
persistently slower host reproduced the same direction and magnitude (cold-after-
compile 2,427.4 → 2,447.1, +0.81%; cache-hot 2,432.7 → 2,450.4, +0.73%) — a
same-host A/B cancels host speed, so those deltas hold, but their absolutes are
host-specific and are not the headline.

**Correctness gate.** Both pack widths (n=2 and n=4) produce a **bit-identical** N=20000
fingerprint to unpacked C32 — `sum=-3.12377031e+05 norm=1.20273230e+03
top5=[517,607,261,290,294]`, all finite, warm≡timed. Two independent pack widths landing
on identical logits + carried state is strong evidence the packing is numerically exact.
(The unpacked C32 path itself was previously gated against the C16 baseline on all four
checks — finite warm≡timed fingerprint, final-token top-5, all-rank capture-replay cosine
≈ 1.0 / max_diff ~1e-6, and real-prompt coherence — so packed C32 inherits that lineage.)

Real-prompt coherence (greedy generation, `--num-tokens`) also passes: packed C32 ×4
produces a **token-identical** continuation to unpacked C32 — e.g. for prompt ids
`760,6511,314,9338,369`, both emit `[11751, 369, 264, 3177, 314, 1880, 11, 10829, 11,
321, 7431, 13, 1049, 369, 1048, 264, 3177]`. (This required a separate fix: `_moe`
unconditionally used the context-encoding MoE kernel — a prefill-only kernel — during
decode too, which graph-broke at T=1; `_moe` is now phase-aware and routes decode
through `moe_tkg` on the same packed expert weights. That fix is independent of the
DeltaNet packing.)

**NKI implementation note.** The packer passes chunk base offsets as **named scalar
arguments** (`base0..base3`), not a Python list — the NKI index specializer rejects a
slice offset sourced from a list of symints ("unsupported expression"). Hence the
explicit 4-way `_chunk_pack4` rather than a list-driven loop.
