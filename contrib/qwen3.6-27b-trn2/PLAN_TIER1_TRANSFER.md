# Implementation plan — Tier-1 transfers (35B-A3B → 27B)

> **Status 2026-08-04: T1b is DONE and gated on device; T1a is not started.**
> T1b landed as `SHARDED_EMBED` / `SHARDED_LM_HEAD` (not `PREFILL_SHARDED_*` — on the
> 27B one module serves both `forward` and `prefill`, so a `PREFILL_` prefix would
> have been misleading). Measured at BS=1/64L on the beta-4 container: module resident
> **17,264 → 13,450 MB/core (−3,814 MB, −22.1%)** with a **token-identical** greedy
> continuation and no TPOT regression (39.8 → 37.5 ms). Freed bytes match the
> arithmetic to the megabyte. See `README.md` → "T1b: vocab-row sharding" for the full
> table and `kernels/tests/test_decode_sharded_lm_cpu.py` for the bitwise CPU gate.
>
> Two deviations from the step list below, both deliberate: the load-time slicing went
> into `apply_vocab_sharding` called from `main()` rather than into `extract_weights`
> (it then covers the `--tiny` random-weight path too, and runs after `del model` so
> the full `[V, H]` tensor actually frees), and the 35B method is named
> `_embed_tokens`, not `_embedding`.

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

### Validation — DONE (2026-08-04, beta-4, 64L, TP=4/LNC=2, `DNBATCHED_V2=1 GQATAIL=1`)

**Outcome: the lever's sign flips with batch size. +16.8% at BS=8, −37.3% at BS=16.**
Ships default-off and scoped to BS=8. Full write-up in README → "T1a: tiled conv-state
layout". Numbers (methods A/B; the harness's synced-TPOT line is unusable at BS=16):

| BS | flag | A (ms/tok) | B (ms/tok) | tok/s host-sync | Δ |
|---:|---|---:|---:|---:|---|
| 8 | off | 113.80 | 121.42 | 70.3 | — |
| 8 | on | 97.46 | 105.00 | **82.1** | **+16.8%** |
| 16 | off | 172.91 / 171.88 | 189.87 / 188.88 | 92.5 / 93.1 | — |
| 16 | on | 275.14 / 275.19 | 290.51 / 291.23 | 58.2 / 58.1 | **−37.3%** |

- **Bit-identical gate: PASSED.** All four legs plus both BS=16 repeats emit the identical
  greedy continuation. The device kernel gate compares both layouts to the same
  `cpu_reference` and gets **identical max-diffs to the last digit** at B=8 and B=16.
- **CPU pre-check: built and passing.** `kernels/tests/test_tiled_conv_layout_cpu.py`,
  16 checks, device-free. `test_deltanet_full_batched.py` gained a `DN_TILED_CONV=1`
  mode (tiled input, un-tiled output, same reference).
- **The BS=16 regression was confirmed, not assumed.** The pair was re-run in reverse
  order to rule out thermal/ordering drift; the regression follows the flag, and both
  configs reproduce within 0.6% across the two passes.
- **Do not cite the single-call kernel benchmark as a timer.** It is dispatch-bound
  (B=8 and B=16 both ~0.11 ms/call), so its −8%/+6% has the right signs but no
  attribution power.

**Scope deviations from the steps above** (all deliberate, none affecting the gates):
1. `state_out`/`conv_state_out` were **not** ported (step 1's last bullet). They are
   inert for T1a and T2c can add them; keeping the signature narrow kept the diff
   auditable.
2. Head counts were **not** made env-driven — they stay module constants at the 27B's
   correct `K_HEADS=4`/`V_HEADS=12`. The warning that mattered (don't inherit the 35B's
   `V_HEADS=8`) is satisfied either way.
3. `USE_WIDE_CONV` and its mutual-exclusion guard were **not** ported (no `DN_WIDE_CONV`
   on the 27B, so there is nothing to be mutually exclusive with).
4. `feat/27b-tiled-conv` is **stacked on** `feat/27b-sharded-vocab` rather than branched
   from `main` in parallel, so the T1b docs are in its history. Files remain disjoint.

Two guards were added because both failure modes are silent: `DN_TILED_CONV=1` without
`DNBATCHED_V2=1` exits (only v2 has the tiled path), and with `--batch-size < 2` exits
(BS=1 uses the untiled `deltanet.full` kernel and would silently ignore the layout).

**Open, not blocking:** why BS=16 regresses. Hypotheses are SBUF pressure from the
`[128, 20, 3]` fp32 tiles (~30 KB/item vs ~2 KB untiled) or loss of many-small-DMA
pipelining at the wider batch. A profile would settle it.

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

### Validation — DONE (2026-08-04, beta-4 container, BS=1, 64L, TP=4/LNC=2)
- ✅ **Identity gate.** All three configurations (replicated, embed-only, both) emit the
  same greedy continuation for `'The meaning of life is'` →
  `' to find your gift. The purpose of life is to give it away.'`
- ✅ **HBM accounting.** 17,264 → 15,357 MB/core (embed only, −1,907) → 13,450 MB/core
  (both, −3,814, −22.1%). Predicted exactly: 248320×5120×2 B = 2,542.8 MB per tensor,
  3/4 of which leaves each rank at TP=4 = 1,907.1 MB.
- ✅ **CPU pre-check.** `kernels/tests/test_decode_sharded_lm_cpu.py` written; asserts
  `rtol=atol=0` for embed, lm_head, and greedy argmax at ws=1/2/4/8 in fp32 and bf16,
  plus the out-of-range-id clamp. Passes.
- ➕ **No latency cost.** TPOT 39.8 (replicated) / 39.9 (embed) / 37.5 ms (both). The
  small improvement at "both" comes from the `lm_head` half; treat it as
  "no regression," not a claimed win, until repeated.

**Actual effort:** under a day. **Risk as realised:** low — the exactness argument held
and the padded-tensor concern of step 4 did not bite at BS≤16.

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
- ~~Actual 27B resident HBM/rank~~ — **resolved: 17,264 MB/core replicated**
  (13,450 with both T1b flags). Step-4's decode-path caveat does **not** bite at
  BS ≤ 16: the padded pre-reduce tensor is 0.5/4.0/7.9 MB per rank at BS=1/8/16.
- Whether the 27B `deltanet_full_batched_35b_ops.py` op schema widened for the
  `state_out` kwargs (affects whether T1a step-2 needs a schema edit).
- ~~The measured 27B tiled-conv delta (hypothesis: ~+15%, from the 35B).~~
  **Resolved, and the hypothesis was half right: +16.8% at BS=8, −37.3% at BS=16.**
  See T1a's Validation section — the lever's sign flips with batch size, so it ships
  default-off and scoped to BS=8.
- **New, and it changes T1a's baseline:** the beta-4 container already moved BS=8
  decode 62.7 → **79.9 tok/s (+27.4%)** with no source change, while *regressing*
  BS=1 by 11%. T1a must be A/B'd against the beta-4 BS=8 number, not the README's
  62.7 — some of the barrier-bound headroom T1a targets may already be spent.
- ~~**BS=16 needs a clean number.**~~ **Resolved: the `synchronize()` ("synced TPOT")
  timer is broken at BS=16; the model is fine.** Three runs gave C2 = 473.7 / 486.4 /
  488.9 ms (±1.6%, so not throttling) while the two independent end-to-end loops agreed
  with each other and scaled smoothly (A = 172.9 / 173.5 / 173.0 ms → **92.4 tok/s**;
  B = 189.6 / 190.8 / 190.0 ms → 84.1 tok/s). C2 sits at a stable 0.78–0.83× of B at
  BS=1/8 and only blows up to 2.5× at BS=16. **T1a's BS=16 A/B must use method A or B**
  — the "TRUE steady-state TPOT (synced)" line the harness prints is not usable there.
  Root cause of the C2 blow-up is unidentified and is its own (low-priority) bug.
- ~~**T1a's BS=16 headroom is larger than the README implied.**~~ **Written before T1a
  was measured, and the measurement contradicts it.** The premise was right — the
  "plateaus ~63 tok/s" claim *was* a C2 artifact, and real beta-4 scaling *is*
  70.1 → 92.4 tok/s (+32%) from 8→16 — but the inference that more headroom meant more
  for T1a to capture was wrong. T1a *loses* 37% at BS=16 while winning 17% at BS=8. The
  correct reading is the opposite: BS=16 is where the untiled path is already doing well,
  and where T1a's wide-DMA trade stops paying.

## Workstream T1c — TP=8 / LNC=1 (added 2026-08-04, out of the original Tier-1 scope)

Not in the original transfer plan; added because LNC=1 was the 35B's single biggest
prefill lever (+24.5%) and the 27B had never been run at 8 logical cores. LNC=1
exposes the chip as 8 × 12 GB instead of 4 × 24 GB, so it **forces TP=8**.

**Two blockers had to be cleared first**, both real:
1. Per-core head counts were hardcoded in 6 NKI kernels — the exact scope deviation
   T1a had skipped as having no consumer. It has one now.
2. `num_key_value_heads = 4` does not divide 8. Every other dimension does (q heads
   24, DeltaNet 16/48, intermediate 17408, vocab 248320, q+gate 12288). Fixed by
   replicating whole KV heads across rank groups (`shard_linear_colwise_replicated`)
   rather than splitting them: a 256-dim head cannot be halved across ranks because
   `q·k` contracts over all 256 dims and `k_norm` is per-head. The row-wise `o_proj`
   all-reduce stays exact because *query* heads remain disjoint — the reduce sums over
   a partition of q heads, so a replicated KV head cannot be double-counted.

T1b is a **hard prerequisite**, not a companion lever: TP=4 needs 13,450 MB/core with
T1b already, so it cannot run under a 12 GB ceiling at all, and TP=8 with a replicated
vocab needs ~11.1 GB of 12. Both sides of every A/B therefore carry both T1b flags.

### Validation — DONE (2026-08-04, beta-4, 64L, both T1b flags on both sides)

Predicted 6,813 MB/core; measured **6,768** (0.7% off). Decode, host-sync tok/s:

| BS | TP=4/LNC=2 | TP=8/LNC=1 | Δ |
|---|---|---|---|
| 1 | 21.0 | 22.7 | +8.1% |
| 8 | 72.3 | 82.0 | +13.4% |
| 8 + T1a | — | **92.0** | +27.2% vs TP=4 |
| 16 | **94.3** | OOM | — |

All passing legs emit the identical greedy continuation. CPU gate:
`kernels/tests/test_tp8_sharding_cpu.py`, 60 checks, no device needed.

**The stated risk did not materialize, and that is the finding.** Going in, the
concern was that TP=8 doubles the participants in every `out_proj`/`down_proj`
all-reduce while ~58% of a BS=8 step is already GLUE+SYNC. LNC=1 won anyway, at every
batch size it could reach: decode here is weight-read bound before it is collective
bound, and halving per-core weight traffic beats the added collective cost. **Third
consecutive case (with T1a and beta-4) where the sign of a lever was mispredicted from
a profile attribution.** Stop predicting signs.

**The replication has a cost, and it lands on batch scaling.** BS=16 at TP=8 OOMs
loading the decode NEFF — weights are fine (6,768 of 12,288 MB), a 65.5 MB instruction
staging buffer is not (`TDRV:dmem_alloc_internal ... ret=-12`). The KV cache
(`static_decode.py:1429`, `[NUM_GQA, B, max_seq, 256]`) holds one whole KV head per
rank at *both* TP values, so the one tensor that grows with batch is the one tensor
TP=8 does not shrink — now measured against 12 GB instead of 24. **So LNC=1 is the
latency and footprint configuration; LNC=2 is still required for BS=16.**

### Open under T1c
- ~~BS=16 at TP=8 is unmeasured.~~ **Measured** at `--max-seq-len 1024`: 100.3 (TP=4) →
  113.7 (TP=8, +13.4%) → **133.1** (TP=8 + T1a, +32.7%), the model's best decode figure.
  Allocator breakdown for the 2048-context OOM captured under
  `NEURON_RT_ENABLE_DEVICE_MEMORY_OOM_LOG=1` (see README): 11.873 of 12 GB, `Tensors`
  10.609 GB against 6.6 GB of weights, so ~4 GB is batch state and fragmentation at ~99%
  occupancy is the proximate failure.
- ~~Prefill at TP=8 vs TP=4 is unmeasured.~~ **Measured:** +27.4% at S=128, +40.0% at
  S=512 — the advantage grows with S, so S=128's figure is a floor.
- **`B × NT` is the ordering key for T1a's sign** (cliff between 256 and 320), which
  reopens #13 with a sharper question: the SBUF-capacity story is dead (~8% occupancy at
  the losing config) and the 35B counterexample is FP8-MoE, so the candidates left are
  DMA pipelining and vector-width/alignment. `DN_CONV_GROUP` (above) tests the latter.
- Whether sharding the *replicated* KV head across its rank group — storing half a
  head per rank and all-gathering before attention — buys back BS=16. Trades HBM for a
  collective on every GQA layer; unattractive given the collective already dominates.

### T1a follow-up — 128-aligned group-tiled conv (`DN_CONV_GROUP`, prototype)

The tiled path's Phase-1 tiles are `[PMAX=128, NT, 3]` — free dim `NT*3` = **30 at TP=8**
(60 at TP=4), below the ≥64 the NKI skill recommends, and the `mq_raw` transpose feeds
only `NT` of 128 partitions. Two direct violations at TP=8, on the exact code path we
care about. The group-tiled variant runs Phase-1 across `G` batch items on the free
axis, one wide vector pass per group, so the tile is `[PMAX, G*NT, 3]` and the transpose
is `[G*NT, PMAX] -> [PMAX, G*NT]`. `nc_transpose` caps moving partitions at 128, and `B`
must be divisible by `G`, so `G` is picked at trace time as the largest divisor of `B`
with `G*NT ≤ 128`:

`G` must **divide `B`**, and that constraint — not the 128 cap — is what actually binds:

| TP | NT | B=8 | B=12 | B=16 |
|---|---|---|---|---|
| 8 | 10 | G=8 → **80** | G=12 → **120** ✅ | G=8 → **80** |
| 4 | 20 | G=4 → **80** | G=6 → **120** ✅ | G=4 → **80** |

**Only `B` divisible by 12 (TP=8) or 6 (TP=4) reaches the 120 target.** B=16 cannot: the
largest divisor of 16 within the cap is 8, so it lands on 80. So the "128-aligned"
configuration is reachable at B=12/24 but *not* at the B=8/16 powers of two — which makes
B=12 the config that actually tests the alignment hypothesis. Lifting this would mean
allowing a ragged final group (`B % G` items handled by a remainder pass), which is the
obvious follow-up if alignment proves to be the lever.

Phase 2/3 stay per-item and unchanged — the head recurrence at `qkv_act[:, kh]` /
`[K_HEADS+kh]` / `[2*K_HEADS+h]` is already full 128-partition column ops. Group is a
Phase-1-only pre-pass that fills a persistent `qkv_act_all [PMAX, B*NT]` SBUF tile the
batch loop then slices.

**Runtime knobs.** `DN_CONV_GROUP=G` (defaults 0 = off, `G >= 2` activates). Requires
`DN_TILED_CONV=1` — the HBM buffer layout is unchanged, so no harness plumbing changes.
When `B % G != 0` or `G*NT > 128`, the kernel silently picks the largest working
divisor of `B` (and falls back to `G=1` = the per-item tiled path if none exists).

**Numeric gate (must pass before any device time).** The existing CPU/device oracle at
`kernels/tests/test_deltanet_full_batched.py` already covers the tiled layout end to
end; run at BS=8 and BS=12 with `DN_TILED_CONV=1 DN_CONV_GROUP=<G>` and expect the same
`state / conv_state / output` diffs as `DN_CONV_GROUP=0`, not the looser 1e-2/5e-2
tolerance the test asserts against. Any regression there means the group's channel↔
(tile, partition) mapping drifted from the per-item path.

**Prediction and its caveats.** The vector-side wins are the two skill hits directly:
free dim 30 → 120 (Phase-1 SiLU, reduce, and 4 `tensor_tensor`s at 4× the width per
call) and transpose partitions 10 → 120. The unknown is whether Phase-1 was the bound
in the first place — the T1a regression at BS=16 is `B*NT`-shaped (won at 160, lost at
320), and if the mechanism is DMA pipelining rather than SBUF or vector width, group
conv won't help. That is exactly why this stays a flagged experiment on top of T1a
rather than replacing it. Measure BS=8 first (where T1a already won at +27%) before
deciding whether to reach for BS=16.

**Gated 2026-08-04 — numerics exact, throughput flat. Prediction was wrong; leave off.**

*Numerics.* Four legs at the TP=4 shape (`deploy/grp_gate.sh`) all pass with
**digit-identical** diffs vs `DN_CONV_GROUP=0`: BS=8 → `0.000039 / 0.000000 / 0.000977`,
BS=12 → `0.000036 / 0.000000 / 0.007812`. The exact `0.0` on `conv_state` confirms the
channel↔(tile, partition) mapping and the shifted-history writeback.

*End-to-end* (`deploy/grp_e2e.sh`, 64L, TP=8/LNC=1, S=1024, beta-4, all legs on T1a):

| B | G → G×NT | TPOT A | TPOT B | Δ vs T1a |
|---|---|---|---|---|
| 12 | — control | 98.55 ms | 112.99 ms | — |
| 12 | 12 → **120** aligned | 99.61 ms | 114.12 ms | **−1.1%** |
| 8 | — control | 77.53 ms | 86.37 ms | — |

The 128-aligned leg is flat to 1% slower. **Phase-1 vector width was never the bound**,
which by elimination makes the `B × NT` cliff (#13) DMA pipelining rather than vector
width or alignment. The caveat above ("the unknown is whether Phase-1 was the bound")
resolved against the prediction.

*Retraction.* The kernel microbench had shown BS=12/GNT=120 **31% faster**. That was an
artifact of a dispatch-bound timer with a ~0.10 ms floor — it measures call overhead, not
kernel time. It bounded the sign only, never the magnitude, and should not have been read
as sizing the lever.

**Still open (low priority now).** Isolated Phase-1 latency at G=1 vs G=12 would confirm
the mechanism directly rather than by elimination, but the end-to-end null already
answers the question this experiment existed to ask.

### A flag trap, worth recording
`--prompt-len` is read only in tiny mode (`static_decode.py:1448`), so the prefill
command documented in the README silently prefilled the 5-token real prompt and then
tripped `--compile-prefill`'s multiple-of-64 guard. It failed **identically at TP=4 and
TP=8**, which is what identified it as a flag error rather than a port defect — the
value of always running the control. README fixed to include `--tiny`.

## Reproducing these runs
The beta-4 DLC has no `transformers`, which this harness needs (`from_pretrained`).
See README → "Running" for the `pip install --target /pysite` + `PYTHONPATH=/pysite`
recipe; transformers **5.14.1** is the validated version for `model_type: qwen3_5`.
Driver scripts used for the 2026-08-04 runs live on the Trn2 at
`/mnt/nvme/tmp/run27b.sh`, `rerun16.sh`, `t1a_kernel_gate.sh`, `t1a_e2e.sh`,
`t1a_bs16_rev.sh`, `t1a_kernel_bs16.sh`, and for T1c `tp8_gate.sh`, `tp8_legs.sh`,
`tp8_pf2.sh`, `tp8_t1a16.sh`, `tp8_bs16fix.sh`; logs in `/mnt/nvme/tmp/runlog/`
(instance store — copy off before any stop/start). Source snapshots:
`/mnt/nvme/code/q27b-t1a` (T1a), `/mnt/nvme/code/q27b-tp8` (T1c).
