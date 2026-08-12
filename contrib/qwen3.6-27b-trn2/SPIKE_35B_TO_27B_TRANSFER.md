# Research spike: which 35B-A3B improvements transfer to the 27B

**Question.** Since the initial commit, ~64 commits of optimization work landed on
`qwen3.6-35b-a3b` and *none* on `qwen3.6-27b-trn2`. Which of those wins are portable to
the 27B, how much work is each, and what is the expected payoff?

**Method.** Static reading only (Trn2 was in use). Compared the two trees
file-by-file, read the 35B `AGENT.md` / `BENCHMARK.md` / `DECODE_RECIPE.md` /
`PREFILL_RECIPE.md`, and traced git history. No device runs — every number below is
lifted from the 35B's recorded measurements and flagged as such; nothing here is a
measured 27B result.

---

## 1. How the two models relate (why transfer is even possible)

Same architecture *family*, different counts/widths. Both are
`[DeltaNet ×3, GQA ×1] × N` hybrids with a depthwise-conv DeltaNet recurrence, a
gated GQA tail with partial RoPE (dim 64, θ=1e7, head_dim 256), and (1+w) RMSNorm.

| | 27B (dense) | 35B-A3B (MoE) |
|---|---|---|
| Layers | 64 (`[DN×3,GQA×1]×16`) | 40 (`×10`) |
| Hidden | 5120 | 2048 |
| MLP | dense SwiGLU 17408 | 256-expert top-8 MoE + shared expert |
| DeltaNet heads (full) | 16 K / 48 V | 16 K / 32 V |
| GQA | 24 Q / 4 KV | 16 Q / 2 KV |
| Vocab / embed+lm_head each | 248320 → **2.54 GB** bf16 | 248320 → 1.02 GB bf16 |
| TP / topo | 4 / LNC=2 | 4 (also 8/LNC=1) |

The 35B `README` even names the 27B as the *source* of the shared backbone
("DeltaNet / GQA / RoPE / RMSNorm / TP / compile harness — the sibling qwen3.6-27b-trn2;
retune head counts"). So the flow direction has now reversed: the 35B carries the
matured versions of kernels the 27B seeded.

**Confirmed shared lineage (diffed):**
- `chunked_prefill.py` — **byte-identical** across the two trees.
- `kernels/deltanet_full.py` ≡ `deltanet_full_35b.py` — **byte-identical**.
- `deltanet_full_batched_v2` — 27B 354 L vs 35B 549 L; the 35B added tiled-conv +
  direct-state-out + env-driven head counts. Core math identical.
- `gqa_tail` — 27B 203 L vs 35B 385 L; 35B added the stateful-KV variant and made
  `Q_HEADS` env-overridable.

The 35B made the head counts **env-configurable** (`DN_K_HEADS`, `DN_V_HEADS`,
`GQA_Q_HEADS`) precisely so one kernel body serves both models. That is the single
most important enabler: most transfers are "port the newer kernel file + set the
head-count envs to the 27B's per-core values," not a rewrite.

---

## 2. The improvements, ranked by (payoff × portability)

### Tier 1 — high payoff, low risk, mostly mechanical

**T1a. Batched-decode tiled conv (`DN_TILED_CONV`).**
The 35B's `deltanet_full_batched_v2_35b.py` coalesces conv-state DMA from ~48 DMAs
to ~3 per batch item via a tile-partition-major layout. On the 35B this was
**~+15%** decode throughput and is **bit-identical** (memory note
`tiled-conv-and-dlc-recipe`: "+19% at BS=32/40L, bit-identical"). The 27B's v2
kernel predates it. The conv machinery is head-count-agnostic — the only 27B-specific
values are `K_HEADS=4/V_HEADS=12` (per-core) vs the 35B's 4/8, and those are now env
vars. **Port:** copy the tiled-conv additions into `deltanet_full_batched_v2.py`,
plumb the optional `state_out`/`conv_state_out` buffers and the `DN_TILED_CONV` flag
through `static_decode.py`. Medium-small effort; the DeltaNet dims differ only in
V-head count. Best single decode-throughput lever for the 27B's BS=8/16 path.

**T1b. Load-time sharding of `lm_head` and `embed`.**
The 35B's biggest prefill unlock (`PREFILL_SHARDED_LM_HEAD`/`PREFILL_SHARDED_EMBED`)
shards the two [V,H] vocab tensors across ranks at load time. On the 35B each tensor
is 1.02 GB; **on the 27B each is 2.54 GB** (hidden 5120 vs 2048), so the two tensors
are ~5.1 GB replicated per rank. Sharding both to ~0.6 GB/rank each frees ~4 GB/rank.
The 27B is dense with 64 layers — it is more likely to be HBM-tight than the 35B, so
this headroom is worth even more here (enables higher batch / longer context). The
35B ships a validated implementation: `_lm_head_logits()` zero-pads each rank's vocab
slice and sum-reduces, proven bit-identical to the replicated path. **Port:** lift
the sharded-vocab load + logit-reduce; retune shapes for H=5120. Low risk (there is a
bit-identical reference), medium effort.

**T1c. Fused route-metadata / MoE levers — N/A.**
Everything under `MOE_*` (CTE prefill MoE, sparse dispatch, fused-W8/FP8 experts,
route-packer) is MoE-specific and has **no counterpart** in the dense 27B. Explicitly
out of scope — do not spend time trying to port it. This is the single largest chunk
of the 35B's commit count and none of it applies.

### Tier 2 — real payoff, more integration work

**T2a. Prefill DeltaNet C32 + block-diagonal inverse + stream packing.**
The 35B's headline prefill work: `CHUNK_SIZE=32` with the numerically-stable
`_tri_inverse_blockdiag` inverse (+8.5%) and then C32 block-diagonal stream packing
`DN_PACK_C32=1 DN_PACK_N=4` (+19.4% on top, bit-identical). The 27B prefill still
runs the original `CHUNK_SIZE=64` `deltanet_chunked_v2.py` (273 L) vs the 35B's
matured `deltanet_chunked_prefill_35b.py` (1465 L, diff=1404 — this is where most of
the divergence lives). **Caveats that gate this:** the 35B hit a genuine numerical
wall at C=64/C=32 on *real* layer-0 distributions (NaNs, catastrophic-but-finite rank
errors) that only the block-diagonal inverse fixed, and that debugging cost most of
the prefill commit history. The 27B's C=64 kernel has **not** been validated against
the same near-1 strict-lower-T failure mode — it may already be silently wrong at
long context (RMSNorm masks it; the 35B learned to "check magnitude, not just
finiteness"). **Port:** substantial. Bring over the block-diagonal inverse and packed
C32, then re-run the 35B's exact gate (all-rank capture replay cosine≈1 + magnitude
check + iterative-prefill coherence) on the 27B's head counts. High payoff
(potentially ~+20–28% prefill) but this is the one item that needs real device time
and careful numerical validation, not a mechanical port.

**T2b. Prefix-aware CTE GQA prefill (`GQA_CTE_PREFILL` + `GQA_DYNAMIC_ROPE_KV`).**
The 35B replaced fixed-KMAX flash prefill with nkilib `attention_cte` (prefix-aware,
14–15× faster per GQA call at 20k) and a dynamic-offset RoPE/KV op so one compiled
graph serves all buckets. The 27B has only `gqa_tail` (decode) and no long-context
prefill attention story in-tree. **Caveat:** this depends on the pinned nkilib
(`1ee625782`, head-dim 256, needs the LNC1 patch for TP=8) and the `gqa_rope_kv`
dynamic op — both are 35B-only files today. head_dim 256 matches, so the kernel
should accept the 27B shapes. **Port:** medium-high; mainly wiring nkilib + the
dynamic-RoPE op into the 27B prefill path. Only worth it if the customer needs
long-context (multi-k) prefill; for short prompts the existing chunked path is fine.

**T2c. Full-graph decode with stateful KV + direct state out
(`DECODE_FULLGRAPH`, `GQA_STATEFUL_KV`, `DN_DIRECT_STATE_OUT`).**
On the 35B these compile embedding→layers→state-update→sharded-head→greedy into one
NEFF and persist aliased K/V + BF16 recurrent state as module buffers, cutting
per-step HBM traffic and barriers. The 35B recorded a clean progression
(108.86 → 105.31 → 99.80 ms/token at BS=32). The 27B already has a single-NEFF decode
forward but returns state each step rather than aliasing it. **Note:** the 27B's own
BS=8 analysis found decode is **barrier-bound, not compute-bound** — its `GQATAIL`
win came from removing sync-points. Stateful-KV/direct-state-out remove exactly that
kind of per-step barrier/traffic, so the mechanism is aligned with the 27B's known
bottleneck. **Port:** medium. The stateful-KV `gqa_tail` variant is already in the
35B kernel (env-gated); the harness plumbing (buffer aliasing, dropping K/V from the
graph signature) is the work.

### Tier 3 — reference only, do not port

- All FP8 MoE work (`block_pow2_coalesced`, `block_ob_coalesced`, fused-W8) — MoE
  only, and even on the 35B FP8 is a *capacity* lever, never a BS=1 latency win.
  The 27B already tried and ruled out FP8 W8A16/W8A8 for its own decode TPOT.
- `MOE_SPARSE`, `MOE_DECODE_TP`, `MOE_CTE*` — MoE only.
- Sharded-LM-head *runtime* slicing (`DECODE_SHARDED_LM_HEAD`) is separate from
  T1b's *load-time* sharding; the 35B notes it saves the `[B,V]` logits tensor + the
  all-reduce, not resident HBM. Minor; fold into T1b if doing the full-graph decode.

---

## 3. Recommended order if the customer greenlights the work

1. **T1b (shard vocab tensors at load).** Biggest, safest capacity win; the 27B's
   2.54 GB×2 tensors make it worth more here than on the 35B. Bit-identical reference
   exists. ~1–2 days.
2. **T1a (tiled conv batched decode).** Best pure decode-throughput lever, bit-identical,
   mostly a kernel copy + env plumbing. ~1–2 days.
3. **T2c (full-graph stateful decode).** Attacks the 27B's *known* barrier-bound
   bottleneck. ~3–5 days.
4. **T2a (C32 + block-diag + packed prefill).** Highest prefill payoff but the only
   item needing real numerical re-validation on device; treat as its own scoped effort
   with the 35B's gate checklist. ~1–2 weeks including validation.
5. **T2b (CTE prefill)** only if long-context prefill is a customer requirement.

Items 1–3 are the low-risk, high-confidence set and should be a separate feature
branch each. Item 4 must not be presented as done until it clears the 35B's
all-rank-replay + magnitude + coherence gate on the 27B — the 35B history is a loud
warning that "finite and coherent-looking" is not sufficient for the DeltaNet inverse.

---

## 4. Key risks / unknowns (all require the Trn2 to resolve)

- **27B C=64 prefill may already be silently wrong** at long context (never gated the
  way the 35B was). Worth a validation run regardless of whether T2a proceeds.
- **HBM budget for the 27B is unmeasured here.** Dense 64-layer + 5120 hidden could be
  tighter or looser than the 35B's per-rank picture; T1b's payoff depends on it.
- **nkilib pin / LNC1 patch** (needed for T2b and any TP=8 experiment) is carried only
  in the 35B tree (`patches/nkilib-lnc1-moe-cte.patch`); a 27B TP=8 path would need the
  same host-side patch wiring.
- Every throughput number in this doc is a **35B measurement**; the 27B deltas are
  hypotheses until benched.
