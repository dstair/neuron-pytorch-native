# Max Decode Throughput Recipe — Qwen3.6-35B-A3B (BS=128 FP8, tiled)

Reproduces the fastest validated **decode** configuration:
**442.1 tok/s at BS=128** (289.5 ms/token, TP=8 / LNC=1, FP8 MoE
`block_ob_coalesced` + tiled DeltaNet conv, optlevel-2). This is **Reduction B1**
(coarse per-128-output-block FP8 scale + PSUM-accumulate) stacked on the tiled
DeltaNet conv: **+28.7%** over `block_pow2_coalesced` (343.6 tok/s, 372.5 ms/tok),
with **bit-identical output** (gen hash `0cc59fb25112`). The tiled conv layout is
itself ~+15% over untiled; B1 adds ~−83 ms/tok on top by removing the
per-128x128-block MoE Vector scale-adds. `block_ob_coalesced` re-quantizes the
routed experts to coarse per-output-block scales at load time (no separate
checkpoint — just the env below); it needs the same official FP8 checkpoint.

All host/path values live in `.env` — copy `.env.example` to `.env` and fill in.
This recipe references: `QWEN35_NATIVE_IMAGE`, `QWEN35_MODEL_DIR` (BF16),
`QWEN35_FP8_MODEL_DIR` (FP8 experts), `QWEN35_COMPILER_CACHE_DIR`,
`QWEN35_RUN_HOST`/`QWEN35_RUN_REGION` (Trn2). No values are hard-coded here.

---

## 1. Host requirements

| Step | Host | Requirement |
|---|---|---|
| Compile **and** run (required for tiled, self-contained) | **Trn2.3xlarge** (`QWEN35_RUN_HOST`) | Native TP=8/LNC=1 (8 cores via `NEURON_LOGICAL_NC_CONFIG=1`). Compiles on the first decode step (~40–45 min cold for `block_ob_coalesced`+tiled) then benches in the same process. ~64 GB host RAM is plenty; the fullgraph NEFF is ~60 MB. |
| Compile only (untiled validation, faster) | **Trn1.32xlarge** (`QWEN35_COMPILE_HOST`) | 128 vCPU / **512 GB RAM** for concurrency-8 parallel `neuronx-cc`. Cross-compiles a Trn2 NEFF; can't execute it. **Untiled only** — see §4 caveat. |

The FP8 MoE decode graph is Trn2 TP=8/LNC=1. **optlevel-2 is optimal** — optlevel-3
gives no gain, optlevel-1 is ~13% slower. Don't override optlevel.

Prereqs on the run host: the internal Neuron DLC (`QWEN35_NATIVE_IMAGE`, has
`torch_neuronx` + `nki_op`), BF16 + FP8 weights on fast local storage, and the
host Neuron lib mounted into the container (fixes the DLC-runtime/host-driver
mismatch — without it NRT init fails with `ucode_ll_create error 6`).

---

## 2. Recommended path — compile + bench on Trn2 (one command)

`static_decode_35b.py --bench` compiles the decode fullgraph on the first step
and then benchmarks it. Run inside the DLC on the Trn2 (`QWEN35_RUN_HOST`):

```bash
source .env   # QWEN35_* from your environment
docker run --rm --privileged --device=/dev/neuron0 \
  -v /opt/aws/neuron/lib:/host_neuron_lib:ro \
  -e MOE_FUSED_W8=fp8 -e MOE_FUSED_W8_FP8_IMPL=block_ob_coalesced \
  -e DN_NKI=1 -e DNBATCHED_V2=1 -e DN_DIRECT_STATE_OUT=1 -e DN_TILED_CONV=1 \
  -e DN_K_HEADS=2 -e DN_V_HEADS=4 -e GQA_Q_HEADS=2 \
  -e GQATAIL=1 -e GQA_STATEFUL_KV=1 -e DECODE_FULLGRAPH=1 -e DECODE_SHARDED_LM_HEAD=1 \
  -e NEURON_LOGICAL_NC_CONFIG=1 -e NEURON_CC_FLAGS="--target trn2 --lnc 1" \
  -e NEURON_COMPILE_CACHE_URL=/ccache \
  -v "$QWEN35_MODEL_DIR":/models/Qwen3.5-35B-A3B:ro \
  -v "$QWEN35_FP8_MODEL_DIR":/models/Qwen3.5-35B-A3B-FP8:ro \
  -v "$QWEN35_COMPILER_CACHE_DIR":/ccache \
  -v "$PWD/contrib/qwen3.6-35b-a3b":/work -w /work \
  "$QWEN35_NATIVE_IMAGE" bash -lc '
    source /opt/torch-neuronx/.venv/bin/activate
    export LD_LIBRARY_PATH=/host_neuron_lib:$LD_LIBRARY_PATH
    torchrun --nproc-per-node=8 static_decode_35b.py \
      --model-path /models/Qwen3.5-35B-A3B \
      --expert-model-path /models/Qwen3.5-35B-A3B-FP8 --skip-prefill \
      --max-seq-len 256 --num-layers 40 --graph-splits 1 --batch-size 128 \
      --num-tokens 2 --bench --bench-iters 20 2>&1'
```

The compiled NEFF persists in `QWEN35_COMPILER_CACHE_DIR` (mounted `/ccache`), so
re-runs are cache-hot (seconds, no recompile).

**Kicking it off headless** (compile is long; don't hold an interactive
session): wrap the above in a script, `nohup` it, and write to a log:
```bash
nohup bash run_decode_bench.sh > /mnt/nvme/runlog/decode_bench.log 2>&1 &
# then poll: grep -E 'loaded|first decode|TPOT|tok/s|gen hash' /mnt/nvme/runlog/decode_bench.log
```

---

## 3. Reading the result

Success line (last, after ~40–45 min cold / seconds cache-hot):
```
BENCH BS=128 seq=256: TPOT 289.53 ms/tok (synced, 20 iter) | throughput 442.1 tok/s
gen hash(row0): 0cc59fb25112
gen hash(row127): 0cc59fb25112
```
- **throughput tok/s** is the headline decode number (batch × 1/TPOT).
- **gen hash** is a bit-exactness fingerprint; `0cc59fb25112` is the reference — it
  is the SAME hash as `block_pow2_coalesced`, i.e. B1's coarse quant produces
  numerically identical generated tokens (zero quality change).
- The prior `block_pow2_coalesced` config gives 372.51 ms/tok / 343.6 tok/s with
  the same hash — use it as the A/B baseline.

---

## 4. Optional — fast cross-compile on Trn1, then run on Trn2

To avoid the ~38 min on-Trn2 compile, cross-compile the NEFF on the 512 GB Trn1:
```bash
DN_TILED_CONV=1 deploy/compile_decode_fp8_trn2.sh --mode fp8 \
  --fp8-impl block_pow2_coalesced --layers 40 --batch-size 128 \
  --direct-state-output on --compile-concurrency 8 --cache-dir "$QWEN35_COMPILER_CACHE_DIR"
```
(Uses `test_decode_fullgraph_device.py` compile-only — needed because it exits
cleanly on the expected cross-host "Invalid NEFF" load error, so the fullgraph
NEFF persists; `static_decode` hangs there and loses it.)

**Transfer** the cache Trn1 → Trn2 via S3 (`QWEN35_COMPILER_CACHE_S3_URI` /
`_S3_REGION`): push from Trn1, pull into `QWEN35_COMPILER_CACHE_DIR` on Trn2,
then run §2.

**Caveat:** a cache built with `test_decode_fullgraph_device.py` has a different
traced graph than `static_decode --bench`, so it **won't cache-hit** the §2
bench. Use §4 to validate/profile the graph quickly; use §2 (Trn2 native) for the
canonical throughput number. Match optlevel across any A/B (default O2).

**Tiled is §2-only.** `DN_TILED_CONV=1` does **not** trace under
`test_decode_fullgraph_device.py` (it fails with a DeltaNet `cv_states` reshape
error, "expand: too few dimensions"), so §4 cannot produce a tiled cache that
runs — and `static_decode --bench` hangs on the Trn1 cross-compile host. There is
therefore **no fast cross-compile path for the tiled (max-throughput) config**:
the 442.1 tok/s number above must be compiled via §2 on Trn2 (~40–45 min cold).
§4 is only for untiled (`DN_TILED_CONV=0`) fullgraph validation/profiling.
