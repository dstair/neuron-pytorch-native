# Interpreting a neuron-explorer profile

## The metric hierarchy

From `summary-json` (per NEFF, keyed by a `n_<hash>` top-level key — unwrap with
`list(json.load(f).values())[0]`):

| field | meaning | how to use it |
|---|---|---|
| `total_time` | wall time of the captured execution | the denominator for everything |
| `<engine>_engine_active_time` | time the engine was executing | **overlaps** other engines; sums >100% |
| `<engine>_engine_instruction_time` | time occupied by that engine's instruction stream | `/ total_time` = share of the critical path |
| `dma_active_time` | DMA engine active | split into `static_` / `software_dynamic_` / `hardware_dynamic_` |
| `cc_op_active_time` | collectives | compare against `cc_op_count` for per-collective cost |
| `mfu_estimated_percent` | achieved / peak FLOPs | with `mfu_max_achievable_estimated_percent` |
| `mbu_estimated_percent` | achieved / peak bandwidth | low MFU **and** low MBU ⇒ latency-bound |
| `hbm_read_bytes` / `hbm_write_bytes` | aggregate traffic | trustworthy with DGE off, unlike `DmaPacket` |
| `throttle_avg_util_limit_nc<N>_percent` | power/util cap | see the naming trap below |

Derived:

```
perfect_pipeline  = max(per-engine active time)          # best case with perfect overlap
serialization_gap = total_time - perfect_pipeline        # dependency/latency cost
```

### Diagnosing what you're bound by

- One engine's `instruction_time / total_time` ≈ 100% → **that engine's stream is the critical
  path**. Optimizing anything else is wasted. (Seen at LNC=2: Tensor at 100.1%.)
- Max is ~65–70%, gap ~60%, MFU and MBU both <15% → **latency / instruction-overhead bound**.
  Neither compute nor bandwidth. Reducing an engine's work helps only where it shortens a
  serial dependency chain. Say so when you report; do not quote engine savings as throughput.
- Sum all `*SEMAPHORE` opcode durations. If that is ~75%+ of `total_time`, the machine is
  mostly waiting and your levers are dependency structure, not arithmetic.

### Throttle counters — naming trap

`throttle_*` fields are suffixed by **physical** NC index, not logical. At LNC=2 a logical core
emits `nc4`+`nc5`; at LNC=1 it emits `nc0`. Grep for the wrong suffix and throttling looks
absent. On our box it is ~87–95% of time at `avg_util_limit ≈ 0.5` in every configuration —
**pre-existing platform cap, not something you introduced and not a lever.** Do not report it
as a finding.

## Instruction-table attribution

`Instruction.parquet` has 58 columns. The useful ones:

`engine`, `opcode`, `duration_ns`, `nki_source_location`, `bir_instruction_name`,
`hbm_read_bytes`, `hbm_write_bytes`, `adjusted_flops`, `evt_wait_time_ns`, `psum_*`, `sbuf_*`,
`spill_save_bytes`, `spill_reload_bytes`, `weight_queue_bytes`, `tensor_instruction_type`.

### Attribution pitfalls

- **A large fraction is unattributed** (`nki_source_location` null — ~40% for us). Report it as
  its own row rather than silently renormalizing.
- **A hot "source line" may be a call site, not the work.** `static_decode_35b.py:2326` showed
  43 ms of `DMA_DIRECT2D`; the line is `torch.ops.deltanet35b.chunked_prefill(...)`, i.e.
  operand marshalling for a kernel whose internals are attributed elsewhere. Read the line
  before naming it a lever.
- **A hot source line may be a stall site, not a compute hog.** `private_nkl/transpose.py:622`
  totaled 84 ms, but 65 ms of that was `EVENT_SEMAPHORE` waiting and only ~9 ms was MATMUL.
  Always break a suspect source down by opcode before believing it.
- `bir_instruction_name` is mostly compiler-generated (`Block1_LoopBody_33-barrier0-SP0`,
  `I-133_inst__I-1056-0`). `name=` kwargs you pass to `nisa.*` in NKI do **not** reliably
  survive into it — do not rely on them for attribution. `Block1_*` names do usefully identify
  dynamic-loop barrier structures.

## Interactive UI with source lines (validated)

```
neuron-explorer view -n NEFF -s NTFF \
  --data-path /explorer_data --source-code-root /srcroot \
  --display-name NAME -p 3001
```

Run via `docker run -d --network host` (so `-p` binds straight to the host).

**The gotcha:** `--source-code-root` must point at an isolated directory containing ONLY the
real source trees, bind-mounted at the exact sub-paths the profile's recorded absolute
`nki_source_location` strings use. If the profile says `/work/kernels/foo.py` and
`/nki-library/...`, mount them at `/srcroot/work` and `/srcroot/nki-library`. Passing
`--source-code-root /` makes it walk the entire container root filesystem, hit `/proc`, and
fail with `lstat /proc/1/fd/17: no such file or directory` — logged as a single error line, the
server still starts, and source simply never renders.

Re-running against the same `--data-path` skips re-ingestion.

## Known tool quirks (version-bound: explorer 2.32 capture / 2.31 view)

- `--output-format summary-json` → stdout only; `--output-file` is rejected.
- `--output-format json` returns empty; `--output-format parquet` errors. Use `--ingest-only`
  and read the parquet directly.
- `--ingest-only` requires `--display-name`.
- `view` needs the **container** path for `-s`, not the host path.
- Runs on a ~1 GB NTFF take >2 minutes. Do not assume empty means failed.
- `neuron-explorer inspect` output is NTFF v129 → unparseable by 2.31/2.32 `view`.
- `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` on a real run does **not** enrich the OOB message
  (still "instruction index = unknown") — driver 2.29 limitation.
- A runtime OOB message's `device IP = 0x0000...` is a **DRAM address** (the access target),
  not an instruction PC. Chasing it as a PC is a dead end; resolving it to a tensor needs the
  runtime memory map, which is not in the static artifacts.

## Decompressing a `.colz` (last resort)

`neuronx-cc/<hash>/<sub>/<kernel>.colz` = header `COLZ` + version + u32 uncompressed size, then
a **zstd frame at byte offset 16**:

```python
import zstandard
zstandard.ZstdDecompressor().decompress(open(p,'rb').read()[16:], max_output_size=64<<20)
```

Yields a multi-MB klir JSON. Be warned: it is an **allocation manifest** — every buffer with
name and shape, but `blocks[].instructions` is empty. Good for proving a tensor is correctly
sized; useless for finding an instruction or a source line.
