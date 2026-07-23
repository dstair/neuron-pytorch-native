# Max Prefill Throughput Recipe — Qwen3.6-35B-A3B (C16-paired, TP=4/LNC=2)

Reproduces the fastest validated **prefill** configuration:
**≈2,090 aggregate prompt tok/s** (BS=2, N=20000 tokens, query bucket 1024,
TP=4 / LNC=2, DeltaNet **C16 paired-batch**, fused NKI route packer, optlevel-1).

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
peak RAM manageable; C16 paired-batch (`CHUNK_SIZE=16`, `DN_PAIRED_BATCH=1`) and
the fused NKI route packer are set inside the compile script.

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
a single native run:

```bash
source .env
DN_PAIRED_BATCH=1 deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR"
```

- Needs `QWEN35_NATIVE_IMAGE`, `QWEN35_MODEL_DIR` (BF16), `QWEN35_NKILIB_DIR` set.
- The script pins `CHUNK_SIZE=16`, `DN_PAIRED_BATCH=1`, `DN_CHUNK_NKI=1`,
  `NEURON_CC_FLAGS="--target trn2 --lnc 2 --optlevel 1 --hbm-scratchpad-page-size 64"`,
  batch-size 2, `--max-seq-len 20480`, `--prefill-bench 20000`.

**Kick off headless** (27-min compile — don't hold the terminal):
```bash
nohup bash -c 'DN_PAIRED_BATCH=1 deploy/compile_prefill_trn2.sh \
  --tp 4 --lnc 2 --layers 40 --splits 4 --bucket 1024 --optlevel 1 \
  --cache-dir "$QWEN35_COMPILER_CACHE_DIR"' \
  > /mnt/nvme/runlog/prefill_bench.log 2>&1 &
# poll: grep -E 'tok/s|prompt|throughput|compiled|Error' /mnt/nvme/runlog/prefill_bench.log
```
Re-runs are cache-hot from `QWEN35_COMPILER_CACHE_DIR` (skip §2/compile).

---

## 4. Reading the result

The bench reports **aggregate prompt tok/s** (total prompt tokens across the BS=2
batch ÷ prefill wall-time). Reference for this config: **≈2,090 tok/s** (2,089.7;
reproduced at 2,098.7 = noise). A per-run token-ID/state fingerprint is printed for
correctness — compare across builds to confirm identical output.

---

## 5. Notes / levers (not landed)

- **C32 chunking is worth ~+14.7% (≈2,408 tok/s)** but is **NOT production-ready**:
  the correct in-isolation `DN_STABLE_C32=0` path NaNs at **bs2** (a C=32 32-partition
  tile inverse-method bug, root-caused to layer 18 / near-1 `T` entries). A
  CPU-validated block-diagonal fix exists but is blocked on a C=32 SBUF
  allocation-scheduling conflict. Leave C16 (default) for the reliable number.
- **FP8 prefill** is a future lever (nkilib `moe_cte` MX variant), not yet
  integrated — this recipe is BF16.
- Do **not** raise optlevel; O2/O3 do not finish in reasonable time/RAM here.
