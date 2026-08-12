# Implementation plan — Tier-1 transfers (35B-A3B → 27B)

Companion to `SPIKE_35B_TO_27B_TRANSFER.md`. Two independent workstreams, each on
its own branch. Both are default-off env flags so the baseline path stays
byte-identical. Line numbers are from the trees as read on 2026-08-03 and will
drift — treat them as anchors, re-grep before editing.

Reference implementations to copy from (all in `../qwen3.6-35b-a3b/`):
- **T1a** tiled conv: `kernels/deltanet_full_batched_v2_35b.py` +
  `static_decode_35b.py` lines ~55–74, ~1668–1720, ~2925.
- **T1b** vocab sharding: `static_decode_35b.py` `_embedding`/`_lm_head_logits`
  (~883–919), `get_embed`/`get_lm_head` (~2500–2524), flag defs (~443–468).

Key structural difference to keep in mind throughout: the **35B loads weights with
its own `st_reader`-based `load_sharded_weights`** (contiguous per-rank slices,
`self.rank` stored on the module), whereas the **27B loads via HF
`from_pretrained` → `shard_model` → `extract_weights`** and the module does *not*
currently store `self.rank`. So T1b can't be copied verbatim; the *math* transfers
but the plumbing attaches to a different loader.

---

## Workstream T1a — Tiled-conv batched DeltaNet decode (`DN_TILED_CONV`)

**Goal.** Coalesce DeltaNet conv-state DMA from ~48 → ~3 per batch item. On the 35B
this was ~+15% (BS=32) and **bit-identical**. Payoff on the 27B is its BS=8/16
decode throughput (the model's headline batched path, currently ~63 tok/s and
barrier-bound).

**Precondition (verified):** tiled conv needs per-core `QKV_DIM % 128 == 0`.
27B per-core `DN_QKV_DIM = 2560 = 20×128` ✓ (35B is 2048 = 16×128). So `_NT = 20`
tiles for the 27B vs 16 for the 35B — parameterized, no code change needed.

### Steps

1. **Bring the kernel up to the 35B version.** The 27B `deltanet_full_batched_v2.py`
   (354 L) is the pre-tiled ancestor of the 35B's (549 L, diff=307). Rather than
   hand-merge, port the 35B additions:
   - `USE_TILED_CONV`/`USE_WIDE_CONV` env reads + the `DN_TILED_CONV`↔`DN_WIDE_CONV`
     mutual-exclusion guard (35B lines ~53–74).
   - `_PMAX = 128`, `_NT = QKV_DIM // _PMAX` module constants.
   - The `USE_TILED_CONV` branch inside the batch loop: the tile-partition-major
     `hist`/`mq`/`new_hist` DMA path (35B ~181–260) and the tiled `new_conv_state`
     alloc (`(B*_PMAX, _NT*(CONV_KERNEL-1))`).
   - Keep head counts env-driven: `K_HEADS=int(getenv("DN_K_HEADS","4"))`,
     `V_HEADS=int(getenv("DN_V_HEADS","12"))`. **Note the 27B default V_HEADS is 12,
     not the 35B's 8** — this is the one value that must not be copied blindly.
   - The optional `state_out`/`conv_state_out` args are part of `DN_DIRECT_STATE_OUT`
     (a T2c concern). For T1a, port them too (they're inert unless passed) so the
     kernel signature matches the 35B and T2c later needs no re-touch.
2. **Ops shim.** `deltanet_full_batched_ops.py` already switches v1/v2 on
   `DNBATCHED_V2`. No change unless the new kwargs need registering in the
   `torch.library` custom-op schema — check whether the 35B's
   `deltanet_full_batched_35b_ops.py` widened the op signature and mirror only if so.
3. **Harness plumbing in `static_decode.py`:**
   - Add flag reads near the other DN flags: `USE_DN_TILED_CONV`, guard that it
     requires `DNBATCHED_V2=1` (the 27B has no `DN_NKI` flag — the batched kernel is
     always NKI here, so drop that half of the 35B's guard).
   - Add the `_to_tiled_conv(cs)` reshape/permute helper (35B ~68–74) and apply it
     **once** to the `conv_states` buffer at allocation (35B ~2925), gated on the flag.
   - At the batched call site (`_deltanet_layer`, ~830–842): when tiled, reshape
     `conv_in`/`new_conv_state` to `(B*_PMAX, _NT*(CONV_KERNEL-1))` instead of
     `(B*Qd, 3)`, and reshape the returned `new_conv_state` back to the tiled buffer
     layout (mirror 35B ~1700–1720).
   - The `conv_states` buffer shape in `forward`/`prefill` signatures (`[48,B,2560,3]`)
     stays as the *logical* shape; only the in-kernel view changes. Confirm the
     prefill path (which shares `conv_states`) still gets the untiled layout — tiled
     is decode-only on the 35B.

### Validation (device, when Trn2 frees)
- **Bit-identical gate.** Run BS=8 decode with and without `DN_TILED_CONV=1`; the
  generated-token hash must match exactly (the 35B calls this out as bit-identical).
- **Throughput A/B.** BS=8 and BS=16, `DNBATCHED_V2=1 GQATAIL=1` ± `DN_TILED_CONV=1`,
  synced TPOT. Expect a double-digit % gain if the 35B's ~+15% carries.
- **CPU pre-check.** If there's a CPU/sim oracle for `deltanet_full_batched`
  (`kernels/tests/test_deltanet_full_batched.py`), extend it to exercise the tiled
  layout before any device compile.

**Effort:** ~1–2 days. **Risk:** low (bit-identical reference, math unchanged).

---

## Workstream T1b — Load-time vocab sharding (`PREFILL_SHARDED_LM_HEAD`, `PREFILL_SHARDED_EMBED`)

**Goal.** Stop replicating the two [V,H] vocab tensors on every rank. On the 27B
each is **2.54 GB bf16** (V=248320, H=5120) — **~5.1 GB/rank replicated**. Sharding
both to ~0.6 GB/rank each frees ~4 GB/rank of HBM headroom, which the dense 64-layer
27B needs more than the 35B did (this is a *capacity* win: enables higher batch /
longer context, not a per-step latency win).

Bit-identical reference exists on the 35B: disjoint contiguous vocab ranges,
zero-pad + sum-all-reduce reconstructs the replicated result with no extra rounding.

### Steps

1. **Store the rank on the module.** The 27B `StaticDecodeModule.__init__` takes
   `world_size` but not `rank`; it only keeps `self.tp_group = list(range(ws))`.
   Add a `rank` parameter (thread it from `main()` where `dist.get_rank()` /
   the torchrun local rank is known) and store `self.rank`.
2. **Flag definitions** (module scope, near the top): `USE_PREFILL_SHARDED_LM_HEAD`,
   `USE_PREFILL_SHARDED_EMBED`. Port the 35B's mutual-exclusion guard vs any runtime
   sharded-LM-head flag — the 27B has none today, so just define the two flags.
3. **Embedding read.** The 27B does `F.embedding(input_id, self.embed)` inline in
   both `forward` (~518) and `prefill` (~563). Extract an `_embedding(ids)` method
   and port the 35B body (~883–900): clamp out-of-range local ids, gather, zero the
   masked rows, sum-all-reduce. Replace both inline call sites.
4. **LM-head read.** The 27B calls `self._lin("lm_head_w", …)` in `forward` (~546)
   and `prefill` (~582). Add `_lm_head_logits(x)` mirroring the 35B (~902–919):
   local `_lin`, then `F.pad` the rank's slice to full vocab and sum-all-reduce.
   Replace both call sites. **Watch the padded-tensor size:** the 35B note stresses
   this is only cheap because prefill reduces *one* token position (~2 MB). The 27B's
   `forward` (decode) produces `[B, V]` logits — for BS=8 that padded tensor is
   8×248320×2 ≈ 3.8 MB per rank pre-reduce, still fine, but **do not** enable this on
   a large-BS decode without checking; it is fundamentally a prefill/low-BS lever.
   Prefer sharding only `embed` on the hot decode path if BS is high, or keep the
   flag scoped to the prefill entry point.
5. **Load-time slicing.** The 27B shards weights in `shard_model` (HF module, ~1025)
   then `extract_weights`. `embed`/`lm_head` are currently *not* sharded there
   (they're replicated). Add contiguous rowwise (vocab-dim) slicing for `embed` and
   `lm_head` gated on the two flags, mirroring the 35B `get_embed`/`get_lm_head`
   (~2500–2524): `w[rank*per_rank:(rank+1)*per_rank].clone()`, with the
   `vocab % world_size == 0` assert (248320 / 4 = 62080 ✓). Do this in
   `extract_weights` (or wherever `weights["embed"]`/`["lm_head"]` are finalized) so
   the `reg_buf`/`reg_linear` calls in `__init__` register the already-sliced tensor.
6. **`tie_word_embeddings`.** Confirm the 27B does not tie embed/lm_head (the 35B
   has `TIE_WORD_EMBEDDINGS=False`; the 27B loads a distinct `lm_head`). If untied
   (expected), the two can be sharded independently. If a build ties them, sharding
   must be consistent — assert and handle.

### Validation (device, when Trn2 frees)
- **Bit-identical gate.** Prefill a fixed prompt with and without the flags; the
  logits fingerprint / greedy continuation must match the replicated path exactly
  (the reduce is exact by construction — disjoint ranges, additive zeros).
- **HBM accounting.** Capture resident GB/rank before/after (the 35B tracked this
  via the runtime allocation breakdown). Target ~4 GB/rank freed.
- **CPU pre-check.** `test_decode_sharded_lm_cpu.py` exists on the 35B side as a
  model for a pure-CPU equivalence test of the pad+reduce; write the 27B analog.

**Effort:** ~1–2 days. **Risk:** low-medium (exact reference; the only subtlety is
the decode-BS logits-tensor size in step 4).

---

## Sequencing & branches

- Branch `feat/27b-tiled-conv` → T1a. Branch `feat/27b-sharded-vocab` → T1b.
  Independent; either can land first. Both merge cleanly (disjoint files: T1a is
  DeltaNet kernel + its call site; T1b is vocab I/O + loader).
- Recommend **T1b first** — it's the bigger, safer capacity win and unblocks any
  higher-BS / longer-context experiments that later throughput work (T2c) wants room
  for.
- Do **not** enable either flag by default until its bit-identical gate passes on
  device. Keep the 35B's discipline: for anything touching DeltaNet numerics, verify
  *magnitude*, not just finiteness (see the 35B AGENT.md C32 saga).

## Open items needing the Trn2 (cannot resolve statically)
- Actual 27B resident HBM/rank (sets T1b's real payoff and whether step-4's
  decode-path caveat bites).
- Whether the 27B `deltanet_full_batched_35b_ops.py` op schema widened for the
  `state_out` kwargs (affects whether T1a step-2 needs a schema edit).
- The measured 27B tiled-conv delta (hypothesis: ~+15%, from the 35B).
