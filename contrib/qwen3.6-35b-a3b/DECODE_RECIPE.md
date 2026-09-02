# Max Decode Throughput Recipe — Qwen3.6-35B-A3B (BS=128 FP8, tiled, per-layer state)

Reproduces the fastest validated **decode** configuration:
**681.3 tok/s at BS=128, seq=256** (187.87 ms/token, TP=8 / LNC=1, **beta-5
container**, **`--optlevel 1`**), from the FP8 MoE `block_ob_coalesced` + tiled
DeltaNet conv stack with **per-layer DeltaNet state buffers** (`DN_PERLAYER_STATE=1`).
Each DeltaNet layer gets its own recurrent-state buffer (mutated once, `torch.stack`
at the return boundary) instead of scattering ~30 layers' writebacks into one shared
`[NUM_DN,B,…]` base; killing that write-after-write chain is **+96.3%** over the same
O1 stack with the shared-base writeback (on beta-4: 202.54 vs 397.45 ms/tok, 632.0 vs
322.1 tok/s). The `DN_PERLAYER_STATE` change is a pure ownership/serialization change
— no arithmetic change — **bit-identical** to the shared-base baseline (8-token
numerics gate `gen hash 7f4b446344cf`, both arms) and passing the DeltaNet-state
occupancy gate on both arms. Requires `DN_DIRECT_STATE_OUT=1`.

**beta-5 (2026-08-10): 681.3 tok/s, +7.8% over beta-4's 632.0**, gen hash
`0cc59fb25112` (== the beta-4 2-token reference) — a clean compiler-only win on the
newer DLC (beta-5, built 2026-08-05), bit-identical. At **seq=1024**
the same stack decodes **582.3 tok/s** (219.83 ms/tok) — −14.5% vs seq=256, the cost
of the 4× larger KV cache (GQA reads the full `max_seq_len` cache each step; module
grows 7.09 → 8.10 GB/core). See §3 and the seq-length note there. All numbers below
supersede the beta-4 / pre-beta-4 O2 (442.1 tok/s) headlines.

The underlying FP8 MoE is **Reduction B1** (coarse per-128-output-block FP8 scale +
PSUM-accumulate) stacked on the tiled DeltaNet conv: **+28.7%** over
`block_pow2_coalesced` (343.6 tok/s, 372.5 ms/tok) with bit-identical output, the
tiled conv layout itself ~+15% over untiled. `block_ob_coalesced` re-quantizes the
routed experts to coarse per-output-block scales at load time (no separate
checkpoint — just the env below); it needs the same official FP8 checkpoint.

> **Supersedes the 442.1 tok/s O2 headline.** That number was measured on the
> **pre-beta-4** container at optlevel-2. On beta-4, O2 full-graph decode no longer
> compiles (host-OOMs, `[F137]`), so O1 is the only path that compiles — and with
> `DN_PERLAYER_STATE=1` it is faster than the old O2 anyway. See `BENCHMARK.md`.

All host/path values live in `.env` — copy `.env.example` to `.env` and fill in.
This recipe references: `QWEN35_NATIVE_IMAGE`, `QWEN35_MODEL_DIR` (BF16),
`QWEN35_FP8_MODEL_DIR` (FP8 experts), `QWEN35_COMPILER_CACHE_DIR`,
`QWEN35_RUN_HOST`/`QWEN35_RUN_REGION` (Trn2). No values are hard-coded here.

---

## 1. Host requirements

| Step | Host | Requirement |
|---|---|---|
| Compile **and** run (required for tiled, self-contained) | **Trn2.3xlarge** (`QWEN35_RUN_HOST`) | Native TP=8/LNC=1 (8 cores via `NEURON_LOGICAL_NC_CONFIG=1`). Compiles on the first decode step then benches in the same process (~24 min cold at O1 on beta-4; **beta-5 compiled the first step in ~65–90 s** — a large compiler speedup, observed on two runs but not yet re-measured in isolation). ~64 GB host RAM is plenty; the fullgraph NEFF is ~60 MB. |
| Compile only (untiled validation, faster) | **Trn1.32xlarge** (`QWEN35_COMPILE_HOST`) | 128 vCPU / **512 GB RAM** for concurrency-8 parallel `neuronx-cc`. Cross-compiles a Trn2 NEFF; can't execute it. **Untiled only** — see §4 caveat. |

The FP8 MoE decode graph is Trn2 TP=8/LNC=1. **On beta-4/beta-5, use `--optlevel 1`.**
O2 full-graph decode no longer compiles on beta-4 (host-OOMs after ~48 min, `[F137]`;
`--graph-splits 2` is a no-op under `DECODE_FULLGRAPH=1`), so O1 is the only path that
compiles. With `DN_PERLAYER_STATE=1` the O1 result (681.3 tok/s on beta-5, 632.0 on
beta-4) is well above the old pre-beta-4 O2 headline (442.1 tok/s), so O1 is both the
only viable and the fastest option — the old "O2 is optimal, don't override optlevel"
guidance applied to the pre-beta-4 container and is superseded.

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
  -e DN_NKI=1 -e DNBATCHED_V2=1 -e DN_DIRECT_STATE_OUT=1 -e DN_PERLAYER_STATE=1 -e DN_TILED_CONV=1 \
  -e DN_K_HEADS=2 -e DN_V_HEADS=4 -e GQA_Q_HEADS=2 \
  -e GQATAIL=1 -e GQA_STATEFUL_KV=1 -e DECODE_FULLGRAPH=1 -e DECODE_SHARDED_LM_HEAD=1 \
  -e NEURON_LOGICAL_NC_CONFIG=1 -e NEURON_CC_FLAGS="--target trn2 --lnc 1 --optlevel 1" \
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

> **Gate `GQA_STATEFUL_KV=1` on cache occupancy, not on `gen hash`.** Until
> 2026-08-05 this flag combination wrote **only GQA group 0** — the ten GQA layers
> share one traced graph under `DECODE_FULLGRAPH=1`, and mutating one shared cache
> tensor ten times inside that graph loses writes, so nine of ten layers attended
> over zeros (`per_group_nz=[98304,0,0,0,0,0,0,0,0,0]`) while producing finite,
> plausible output. Fixed by giving each GQA layer its own cache buffer — one
> distinct tensor per mutation.
>
> **This one does not reproduce at probe scale.** The pre-fix form (all ten calls
> handed the *identical* whole-tensor view, the kernel doing the per-layer offset)
> lands all ten writes in a small graph — `kernels/tests/test_gqa_tail_stateful_probe.py`
> says so out loud. Only the real 40-layer graph, with MoE, DeltaNet and collectives
> at BS=128, drops them. So the probe is a floor and the occupancy run is the gate. Add `-e DECODE_KV_MAP=1` and require **all ten groups non-zero**:
> ```
> DECODE kvmap per_group_nz = [98304] * 10        # correct: every group written
> ```
> Every decode number measured with this flag pair before 2026-08-05 is invalid,
> including the +5.5% originally attributed to stateful KV. See `BENCHMARK.md`.
>
> **Why it is unpredictable rather than simply broken:** the in-place write has no
> representation in the emitted graph — NKI only reports `input_output_aliases` for
> inputs the kernel *returns*, and ours deliberately doesn't, so `mutates_args` never
> reaches the custom call. Mechanism and line references in `PREFILL_RECIPE.md`
> ("Why the platform behaves this way").
>
> Post-fix reference at 40 layers / BS=128 / seq=256 / `--optlevel 1` / beta-4:
> `per_group_nonzero_rows=[384]×10`, **TPOT 393.56 ms/token, 325.2 tok/s**. The fix is
> correctness-only — 393.56 vs the broken 395.95 ms is noise, because masked attention
> costs the same over a zero-filled cache as over a real one. Do not gate on `gen
> hash`: it was **identical** across the fix (`0cc59fb25112`) despite nine of ten
> attention inputs changing from zeros to real K/V.

**Kicking it off headless** (compile is long; don't hold an interactive
session): wrap the above in a script, `nohup` it, and write to a log:
```bash
nohup bash run_decode_bench.sh > /mnt/nvme/runlog/decode_bench.log 2>&1 &
# then poll: grep -E 'loaded|first decode|TPOT|tok/s|gen hash' /mnt/nvme/runlog/decode_bench.log
```

---

## 3. Reading the result

Success line (last; beta-5 first-step compile ~65–90 s / seconds cache-hot):
```
BENCH BS=128 seq=256: TPOT 187.87 ms/tok (synced, 20 iter) | throughput 681.3 tok/s
gen hash(row0): 0cc59fb25112
gen hash(row127): 0cc59fb25112
```
- **throughput tok/s** is the headline decode number (batch × 1/TPOT). **681.3 tok/s
  on beta-5** (187.87 ms/tok), vs 632.0 on beta-4 (202.54 ms/tok) — a +7.8%
  compiler-only win, `DOCKER_EXIT=0`, gen hash unchanged.
- **Longer context:** the same command with `--max-seq-len 1024 --num-tokens 1023`
  decodes ~1024 tokens and measures **582.3 tok/s (219.83 ms/tok) at seq=1024** on
  beta-5 — −14.5% vs seq=256, the cost of the 4× larger KV cache (module 7.09 → 8.10
  GB/core; GQA reads the full `max_seq_len` cache every step). `--num-tokens 1023`
  (not 1024) keeps the post-gen synced-bench step, which runs at `position=num_tokens`,
  in-bounds of the 1024-slot cache. NOTE: over ~1024 greedy steps the row0 and row127
  gen hashes diverge — expected for long from-zeros greedy decode (per-row FP
  differences in the fp8/batched GEMMs compound into different argmax picks); it is not
  a regression, and short (2-token) rows stay identical.
- **gen hash** is a bit-exactness fingerprint. With `--num-tokens 2` the reference
  is `0cc59fb25112` (unchanged from the old baseline — `DN_PERLAYER_STATE` is a pure
  ownership/serialization change, so it is bit-identical). The deeper 8-token
  numerics gate (7 recurrent-state feedbacks) is `7f4b446344cf`, identical across
  the `DN_PERLAYER_STATE` A/B on rows 0 and 127.
- **A/B baseline:** re-run the exact command with `-e DN_PERLAYER_STATE=0`. That is
  the shared-base writeback — ~397.45 ms/tok / **322.1 tok/s**, the *same* gen hash.
  `DN_PERLAYER_STATE=1` is **+96.3%** over it, purely by removing the ~30-layer
  write-after-write serialization on the shared state base.
- **Gate on DeltaNet-state occupancy, not just `gen hash`** (see the `GQA_STATEFUL_KV`
  callout above for why): add `-e DN_OCCUPANCY_CHECK=1` (on by default) — the run
  asserts every per-layer DeltaNet state holds ≥1 non-zero row, catching a dropped
  per-layer writeback that finite, plausible, token-identical output would hide.

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
canonical throughput number. Match optlevel across any A/B — on beta-4 that is
**O1** (O2 no longer compiles here).

**Tiled is §2-only.** `DN_TILED_CONV=1` does **not** trace under
`test_decode_fullgraph_device.py` (it fails with a DeltaNet `cv_states` reshape
error, "expand: too few dimensions"), so §4 cannot produce a tiled cache that
runs — and `static_decode --bench` hangs on the Trn1 cross-compile host. There is
therefore **no fast cross-compile path for the tiled (max-throughput) config**:
the 442.1 tok/s number above must be compiled via §2 on Trn2 (~40–45 min cold).
§4 is only for untiled (`DN_TILED_CONV=0`) fullgraph validation/profiling.
