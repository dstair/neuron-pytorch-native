# Failure taxonomy: what the log actually means

## Hang — `NRT status 5`

```
ERROR TDRV:exec_request_process_errors (FATAL-RT-UNDEFINED-STATE) [ND 0][NC 0]
      execution timeout (30000 ms) on model module.neff, waiting for execution completion notification
ERROR TDRV:exec_request_progress_one_step [LNC 3] Failed to progress Neuron Core Exec State, error: 5
```

Every rank reports it, ~30 s after the launch that never returned. No `PREFILL TIMED`, no
fingerprint.

The runtime sometimes adds:

> Suspected hang after successful execution of collectives operation 0 (ALLREDUCE) ...
> Likely NOT a collectives issue, check for hangs on **instructions BETWEEN collective operations.**

Read that literally: the all-reduce **completed**. The deadlock is in compute/DMA afterwards.
Do not spend time on the collective.

### Debugging a hang: what worked and what didn't

Eliminated cheaply, in this order (each was a full experiment):

1. **Control-flow primitive** — rewrote a runtime loop three ways (deprecated `nl.dynamic_range`,
   fully-static `sequential_range` unroll, new `nl.fori_loop`). All three hung identically. How
   the loop is *expressed* is almost never the cause; don't start here.
2. **Graph co-allocation** — more graph splits (1 layer per split) hung identically. Rules out
   "co-allocated with adjacent layers".
3. **Isolation** — the same kernel, standalone, at the same shapes with the *real* routing dumped
   from a live run, on one core **and** at full TP=8: clean. A bug that needs the fused
   pipeline is a fused-pipeline bug (layout, co-allocation, or cross-rank synchronization) —
   or is data-dependent in a way your harness doesn't reproduce.
4. **Component swap** — replace the suspect subsystem with a reference implementation that keeps
   downstream data *correct* (not zeroed). If the hang vanishes, it is in that subsystem. This
   is much stronger than skipping the subsystem, which changes every downstream input.

Then, and only then, look for a **threshold**: at what size/count does it start? Follow
`isolating-thresholds.md` in this directory — do not accept the first variable that correlates.

### Multi-rank hangs: check the barriers

A hang at TP=N with the collective completing is classically a **missing or asymmetric barrier**
in a multi-rank scatter/store, especially one that is conditional. Grep the suspect kernel for
barrier flags that are computed rather than constant:

```python
scatter_barrier = assignments <= SOME_LIMIT   # <-- a barrier that switches off with size
```

A barrier whose presence depends on problem size means "small works, large hangs" is a
*synchronization* story, not a capacity story — and the two are trivially confused because both
correlate with size.

## Device OOM at load — not a hang

```
ERROR TDRV:dmem_alloc_internal   Failed to allocate DEVICE memory (95079808 bytes): ret=-12
ERROR TDRV:ib_transfer_block_to_tdram  Failed to allocate aligned staging instr buffer
ERROR NRT:nrt_load_collectives   Failed to load collectives for model.
```

This is the **instruction/staging buffer**, not weights — the module has already loaded (you'll
see its `GB/core` line above). Fix by shrinking activations in flight (batch × sequence chunk)
or the graph, not by quantizing weights further.

## Host wedge

- `ssh` handshake times out while `/dev/tcp/<ip>/22` still connects (kernel accepts, userspace
  can't fork/schedule).
- `aws ssm describe-instance-information` → `PingStatus: ConnectionLost`, with a `LastPingDateTime`
  that pins the death to the minute.
- `aws ec2 get-console-output` returns the ORIGINAL boot log, weeks stale — it is not a live
  console and won't help.

Cause here was always host RAM during parallel compiles. Recovery = **reboot** (see the
instance-store rule in SKILL.md).

## Compile failures

The eager dispatch prints a terse `COMPILATION FAILED` and hides the diagnostic. To see it:

```
-e TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1  -v $PWD/nbackend:/tmp/neuron_backend
# then read: nbackend/*/neuronx-cc/<hash>/log-neuron-cc.txt
```

Real messages worth recognizing:

| message | meaning |
|---|---|
| `memset dst partition dimension 288 exceeds maximum 128` | a per-item SBUF buffer laid out with partition = item count; tile it to ≤128 |
| `error: expecting simple variable` | a specializer/codegen limit — usually a tuple loop variable (`enumerate`) or an arithmetic slice bound; precompute into a plain name |
| duplicate op names | reused `name=` across loop iterations; make them unique per chunk |
| `NCC_EVRF051` on a `float8_e4m3fn` operand | that dtype is rejected on TRN2; store `int8` and `.view()` — see `fp8-quantization-perf` |

Compile-chain bugs come in **cascades**: fixing one reveals the next. Budget for several rounds
when taking a path to a scale it has never run at, and keep iterating at 4 layers.
