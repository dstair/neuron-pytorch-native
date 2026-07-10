# AGENT.md — Qwen3.6-35B-A3B (MoE) on Trainium2, PyTorch Native

Guidance for an AI coding agent (Claude or similar) bringing this model up on AWS
Trainium2 in a fresh environment. It captures the architecture, the design
conventions, the validated run recipes, and — most importantly — the hard-won
gotchas that are expensive to rediscover. Pair it with `README.md` (results) and
the code in this directory.

> Naming: published as `Qwen/Qwen3.6-35B-A3B` on Hugging Face; the architecture
> class is `Qwen3_5MoeForConditionalGeneration` (`model_type: qwen3_5_moe`). "3.5"
> = arch family, "3.6" = release. Same architecture; this code runs on the HF
> checkpoint (see the weights note below).

## What this is

A PyTorch-Native (`torch.compile(fullgraph=True, backend="neuron")`) inference
harness for a 35B sparse-MoE hybrid model: 40 layers = [DeltaNet ×3, GQA ×1] ×10,
256-expert top-8 MoE on every layer, hidden 2048, TP=4 on one `trn2.3xlarge`
(LNC=2, 96 GB device / ~24 GB per core). Verified dims live in `model_dims.py`
(single source of truth, config-driven). Routing math and full shapes are in
`README.md`.

## Design conventions (read before editing)

- **One static forward → one NEFF.** `static_decode_35b.py` expresses the whole
  40-layer decode (and prefill) as a fixed-shape function that compiles to a single
  NEFF. This is what avoids the per-op eager-dispatch overhead that dominates
  latency. Do not introduce data-dependent shapes into the compiled path.
- **Heavy ops must be opaque NKI kernels, not pure torch.** neuronx-cc's tiler
  cannot handle the DeltaNet recurrence or GQA attention when they're exposed as
  pure-torch einsums at scale — see the compile gotchas below. Custom kernels live
  in `kernels/`, each registered as a `torch.ops.*` custom op via a `*_ops.py`
  companion so it drops into the graph without a graph break.
- **Manual TP.** Weights are sharded per-core by hand (colwise for q/k/v/gate/up
  and experts, rowwise for output/down) with functional all-reduces at known
  boundaries. KV heads (2) don't divide TP=4 → each KV head is replicated across
  `world_size // 2` cores.
- **CPU oracle before device.** Validate kernel and routing math on a CPU reference
  first (`kernels/tests/test_moe_oracle_cpu.py` proves the masked-dense MoE ≡
  canonical `Qwen3MoeSparseMoeBlock` to ~3e-9). Cheap; catches most bugs before any
  device compile.

## Environment

- **You need the PyTorch-Native DLC image**, not a host XLA venv. Host
  `torch-neuronx` venvs on the DLAMI import `torch_xla` (the XLA lowering path);
  Native adds a `neuron` device via PrivateUse1 and is only in the Native DLC. The
  harness reads weights with a dependency-free safetensors parser (`st_reader.py`),
  so it does not need `transformers`/`safetensors` in the image.
- **⚠️ Runtime/driver version match.** The Neuron runtime bundled in the DLC image
  must be compatible with the host's kernel driver (`aws-neuronx-dkms`). A mismatch
  fails `nrt_init()` even single-core with `ucode_ll_create ... error 6` /
  `Copy from buffer to memory failed`. If the host driver is *older* than the
  image's runtime, bind-mount the host runtime over the image's:
  `-v /opt/aws/neuron:/opt/aws/neuron:ro -e LD_LIBRARY_PATH=/opt/aws/neuron/lib`.
  Diagnose by checking whether the host's own stack initializes the device (a tiny
  XLA `xm.xla_device()` tensor op) — if that works, the device is fine and it's
  purely a runtime-in-image mismatch.
- **Weights packaging.** This code's loader is validated against a 14-shard /
  1811-tensor packing of the checkpoint. The current HF repo may be re-sharded
  (e.g. 26 shards / 1169 tensors, with fused expert `gate_up`); the parameters are
  the same model but `st_reader.py` / `model_dims.py` expect the packing they were
  written for. If loading a differently-packed copy, update the reader accordingly
  (dims/routing/RoPE are unchanged — no perf difference).

## Run recipes (validated)

Decode benchmark, full 40 layers, TP=4, recommended flags:
```
DN_NKI=1 GQATAIL=1 \
  torchrun --nproc-per-node=4 static_decode_35b.py \
    --model-path <weights> --max-seq-len <S> --num-layers 40 \
    --graph-splits 1 --batch-size <BS> --bench --bench-iters 30
```
Coherence check (real prompt, greedy): add `--num-tokens 16 --prompt-ids
"760,6511,314,9338,369"` ("The capital of France is") and drop `--skip-prefill`;
expect it to echo the prompt back (the documented greedy loop).

CPU correctness (no device): `python3 kernels/tests/test_moe_oracle_cpu.py --tokens 8`.

### Flags that matter

| Flag | Effect |
|---|---|
| `DN_NKI=1` | DeltaNet NKI kernel — **required past ~20 layers** (see gotcha) |
| `GQATAIL=1` | Fused GQA attention-tail kernel — also collapses long-context compile |
| `MOE_SPARSE=1` | True-sparse MoE dispatch — ~2× at BS=1; do NOT use at BS≥16 (see gotcha) |
| `DNBATCHED_V2=1` | DMA-coalesced batched DeltaNet decode |
| `MOE_FP8=1` | FP8 experts — memory lever only (see FP8 gotcha) |
| `GQA_FLASH_PREFILL=1`, `DN_CHUNK_NKI=1` | Prefill kernels (see prefill recipe) |

## Hard-won gotchas (each cost real time)

- **Pure-torch DeltaNet decode does not compile past ~20 layers.** neuronx-cc trips
  a `PGTiling` assertion on the recurrent-state einsum. `DN_NKI=1` (opaque kernel)
  cures it — the full 40 layers then compile. Not TP, not MoE (both ruled out).
- **Long-context cold-compile: use `GQATAIL=1`.** A naïve 20k decode graph took
  ~2.7 hours to compile because pure-torch GQA attention is exposed to the tiler.
  `GQATAIL=1` (opaque attention-tail kernel) collapses it to ~10 min. Memory fits
  at ~19.1 GB/core at 20k regardless (KV is small; weights dominate).
- **Long-context batch is HBM-bound, not throughput-bound.** Fixed ~19.1 GB/core of
  weights leaves little of the 24 GB/core for KV cache. At seq=10k the batch ceiling
  is BS=8 (BS=16 OOMs at NEFF load: `NRT_RESOURCE: Failed to allocate resource`); at
  seq=20k only BS=1 fits. Short context (seq=256) scales to BS=32. So the lever to
  raise the long-context batch ceiling is *reducing weight bytes* (FP8), not tuning.
- **FP8 is a memory/capacity lever, NOT a decode-latency win.** It halves expert
  weights (~19→11 GB/core) and is CPU-coherent, but every latency attempt regressed
  (BS=1 grouped-matvec ~2.2× slower; a fused-MoE FP8 path hit a dtype/compile wall).
  Cause: at BS=1 FP8 turns wide fused GEMMs into many tiny per-expert matvecs and
  dispatch overhead dwarfs the bandwidth saved. Trn2's `nc_matmul` wants *legacy*
  `float8_e4m3`; torch only has `e4m3fn` — a real plumbing gap. Treat FP8 as the way
  to unlock higher batch at long context (capacity), not to speed up a single stream.
- **Sparse MoE does not scale to BS≥16.** `MOE_SPARSE=1` gathers `T·K` experts;
  the gathered graph explodes the *host* compiler memory at BS≥16 (F137 / OOM /
  host wedge). Use masked-dense MoE for BS≥16; sparse only wins BS≤4.
- **Prefill: eager sequence-bucketing is the working long-context path.**
  `--bucket-chunk 2048 --bucket-compile 0` with `GQA_FLASH_PREFILL=1 DN_CHUNK_NKI=1`
  plus pad-token masking gives coherent 20k prefill with no OOM and ~9 min compile
  (~259 prompt tok/s). Two kernel bugs had to be fixed for correctness: (1) pad
  tokens (id 0) corrupt the carried DeltaNet state — zero their `beta`/`g`; (2) the
  chunked-DeltaNet L2-norm used `x·rsqrt(‖x‖²+eps)` but must match
  `x/max(‖x‖,eps)` for near-zero rows. Random-input kernel tests missed both — test
  on real post-conv/proj distributions.
- **Measure synced TPOT, not enqueue time.** The Neuron eager backend dispatches
  async; a timing loop without `torch.neuron` synchronize measures enqueue, not
  execution (this produced bogus ~1 ms "TPOT" figures historically that were really
  ~35 ms). Always synchronize before timing.
- **Device-smoke at the real `max_seq`.** Some tiling limits only appear at large
  seq; a 512-token smoke can pass while 2048/20k fails.

## Device / ops hygiene

- **Free the device before a new run.** A live container holds all 4 cores →
  "Logical Neuron Core(s) not available". Kill stray containers first.
- **Cap per-config compile time** and watch host RAM on high-BS or long-seq
  compiles — a runaway neuronx-cc compile can exhaust host memory and wedge the box
  (recover by reboot). Instance-store NVMe is typically not in fstab → remount after
  any reboot (data survives reboot, lost only on stop/terminate).

## Correctness reference (oracle)

The validated NxDI implementation (`aws-neuron/neuronx-distributed-inference`
PR #60, torch-xla) is the correctness oracle: 100% token-match vs CPU, ~18.4 ms/tok
BS=1 on trn2.3xlarge TP=4. Its MoE uses a non-portable NxDI library module, which is
why this harness carries its own MoE kernels + CPU oracle. Use it to validate token
output, not as an architecture to copy.

## Reuse map

| Piece | Source |
|---|---|
| DeltaNet / GQA / RoPE / RMSNorm / TP / compile harness | the sibling `qwen3.6-27b` (dense) — retune head counts |
| MoE (masked-dense grouped-bmm, expert-parallel) | `kernels/` here + `test_moe_oracle_cpu.py` |
| Correctness oracle | this package's CPU oracle + NxDI PR #60 token match |
