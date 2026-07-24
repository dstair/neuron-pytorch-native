# Max Prefill Throughput Recipe — Qwen3.6-35B-A3B (stable C32, TP=4/LNC=2)

Reproduces the fastest validated **prefill** configuration:
**≈2,277 aggregate prompt tok/s** (BS=2, N=20000 tokens, query bucket 1024,
TP=4 / LNC=2, DeltaNet **stable C32** block-diagonal inverse, fused NKI route
packer, optlevel-1) — **+8.5%** over the paired-C16 baseline (≈2,090 tok/s).

C16 remains the compiled-in default and the reliable fallback; **C32 is the
opt-in faster path** (`DN_STABLE_C32=0 CHUNK_SIZE=32`) and is validated as
numerically equivalent to C16 (see §6). Both configs use identical shapes,
topology, and the fused NKI route packer.

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
peak RAM manageable. The stable C32 inverse (`_tri_inverse_blockdiag`) is already
in `kernels/deltanet_chunked_prefill_35b.py`; it is selected at runtime by
`DN_STABLE_C32=0 CHUNK_SIZE=32` (§3). The fused NKI route packer
(`MOE_CTE_NKI_PACK=1`) is set inside the compile script.

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

### 3a. Fastest — stable C32 (≈2,277 tok/s, +8.5%)

The committed script hard-pins the DeltaNet chunking (`CHUNK_SIZE=16`,
`DN_PAIRED_BATCH=1`) and does not pass `DN_STABLE_C32`, so **two lines must be
changed** to build the C32 graph. In `deploy/compile_prefill_trn2.sh`:

- the exported-env line (~L183): `export CHUNK_SIZE=16` → `export CHUNK_SIZE=32`
- the container `-e` line (~L247): `-e CHUNK_SIZE=16 -e DN_PAIRED_BATCH=1` →
  `-e CHUNK_SIZE=32 -e DN_STABLE_C32=0` (drop `DN_PAIRED_BATCH`; C32 does not pair)

Use a **separate `--cache-dir`** from any C16 build — `CHUNK_SIZE` changes the
traced graph, so C32 will not (and must not) cache-hit a C16 cache root. Then:

```bash
source .env
deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR/c32"
```

### 3b. Reliable default — paired C16 (≈2,090 tok/s)

The committed script builds C16 as-is (no edits):

```bash
source .env
DN_PAIRED_BATCH=1 deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR/c16"
```

- Both paths need `QWEN35_NATIVE_IMAGE`, `QWEN35_MODEL_DIR` (BF16), `QWEN35_NKILIB_DIR`.
- The script pins `DN_CHUNK_NKI=1`, `MOE_CTE=1 MOE_CTE_NKI_PACK=1`, batch-size 2,
  `--max-seq-len 20480`, `--prefill-bench 20000`, and
  `NEURON_CC_FLAGS="--target trn2 --lnc 2 --optlevel 1 --hbm-scratchpad-page-size 64"`.

**Kick off headless** (~27-min compile — don't hold the terminal):
```bash
nohup bash -c 'deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR/c32"' \
  > /mnt/nvme/runlog/prefill_bench.log 2>&1 &
# poll: grep -E 'tok/s|prompt|throughput|compiled|Error' /mnt/nvme/runlog/prefill_bench.log
```
Re-runs are cache-hot from the matching `--cache-dir` (skip §2/compile).

---

## 4. Reading the result

The bench reports **aggregate prompt tok/s** (total prompt tokens across the BS=2
batch ÷ prefill wall-time). References for the two configs:

| Config | Wall time | Aggregate prompt tok/s |
|---|---:|---:|
| **Stable C32** (`DN_STABLE_C32=0 CHUNK_SIZE=32`) | **17.568 s** | **2,276.9** |
| Paired C16 (default) | 19.141 s | 2,089.7 |

A per-run token-ID/state fingerprint is printed for correctness; the warm and
timed fingerprints must be identical and finite. Compare across builds to confirm
identical output.

---

## 5. Notes / levers

- **BS=4** also fits at 512-token buckets and measured 39.788 s / 2,010.6
  aggregate tok/s (paired C16). BS=1 single-prompt best is 1,482.8 tok/s.
- **FP8 prefill** is a future lever (nkilib `moe_cte` MX variant), not yet
  integrated — this recipe is BF16.
- Do **not** raise optlevel; O2/O3 do not finish in reasonable time/RAM here.

---

## 6. Why C32 is safe (and why not the naïve C32)

C32 halves the DeltaNet chunk count, which is where the +8.5% comes from, but it
required a **numerically stable chunk-matrix inverse**. `_tri_inverse_blockdiag`
splits the 32×32 chunk matrix into two 16×16 diagonal blocks plus a coupling term
and inverts the blocks by doubling:

- the naïve **full-32 doubling** overflows on near-1-decay streams → NaN at bs2
  (root-caused to layer 18 / near-1 `T` entries); **unusable** (~2,408 tok/s if it
  were finite);
- a **Horner series** is stable but ~4× costlier → 2,037.5 tok/s (**−2.9%**), so
  it is not used;
- the shipped **block-diagonal doubling** is both stable and cheap → 2,276.9 tok/s
  (**+8.5%**).

C32 correctness was gated on all four checks and passed: finite warm≡timed
fingerprint; final-token top-5 matching the C16 baseline; **all-rank
capture-replay vs the CPU reference at deep context (cosine ≈ 1.0, max_diff ~1e-6
on all four TP ranks)**; and real-prompt coherence identical to C16 (bit-identical
greedy continuation via iterative prefill). A cheaper stable inverse that closes
the remaining gap toward the theoretical +14.7% (~2,408 tok/s) is a future lever.
