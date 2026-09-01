---
name: trainium-run-discipline
description: |
  Launch, time, gate, and recover model runs on a Trainium box without producing invalid
  numbers or losing the machine. Use when running or benchmarking a model on trn1/trn2,
  when a run hung or failed, or when you see "NRT status 5", "execution timeout (30000 ms)",
  "Failed to allocate DEVICE memory", "NRT status 1204", "Software notification queue
  overflow", "Failed to load collectives", "nrt_load", "0 compile activity", a SIGABRT or
  SIGSEGV at teardown, a docker exit code that disagrees with the printed results, an
  unreachable box, or timings that changed ~15x for no reason. Also covers the
  instance-store rule: reboot the box, NEVER stop/start.
---

# Running on the device without fooling yourself

Five rules, each of which cost real hours to learn.

## 1. Never time with debug flags

`NEURON_LAUNCH_BLOCKING=1` and `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` together measured
**~15× slower** than clean flags on a 40-layer prefill. We published wrong numbers twice from
this — once assuming the cost was "~3×", once believing FP8 was 15× slower than BF16 when the
real tax was 7%.

Also strip `XLA_IR_DEBUG`, `XLA_HLO_DEBUG`, `NEURON_FRAMEWORK_DEBUG`,
`TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS`, and any `nbackend` artifact mount.

Keep **two scripts**, never one with a flag: a clean runner for every number you will quote,
and a debug runner for diagnosis. `scripts/run_template.sh` is the clean one; the header lists
the debug additions.

At 40 layers DGE notifications don't just slow things down, they **fail**: one host
notification per indirect DMA overflows the queue →
`NRT status 1204 ... NRT_EXEC_SW_NQ_OVERFLOW`. Benign at 4 layers, fatal at 40.

## 2. Sudo-clear the compile cache, or you will benchmark a stale NEFF

A `--privileged` container creates `cache/{neff,hlo}_cache` and artifact dirs as **root**. A
run script's `rm -rf` running as your user **fails silently** (stderr to `/dev/null`) and the
previous NEFF is served on every subsequent run.

This silently invalidated about six consecutive experiments. Symptoms: implausibly fast
compiles, results identical across changes that should differ.

```bash
sudo rm -rf "$CACHE" nbackend      # before EVERY launch; build it into the runner
grep -c "compile activity" "$LOG"  # "0 compile activity" == you are on a stale NEFF
```

## 3. Read the app's success marker, not the exit code

The beta-5 container SIGABRTs/SIGSEGVs during teardown **after** printing valid results, so
`DOCKER_EXIT=1` on a perfectly good run.

Gate on your harness's own marker (`PREFILL TIMED`, `TPOT`, a fingerprint line) plus a finite
check. Conversely a exit-0 run can be meaningless — always confirm the marker is present.

## 4. Know the failure taxonomy

| symptom | meaning | what to do |
|---|---|---|
| `NRT status 5`, `execution timeout (30000 ms)`, "Failed to progress Neuron Core Exec State, error: 5" | **hang** | see `references/failure-taxonomy.md`; note "Suspected hang after collectives op0 ALLREDUCE ... check instructions BETWEEN collective operations" means the collective completed — look at compute/DMA, and at multi-rank barriers |
| `Failed to allocate DEVICE memory (N bytes)` + `nrt_load_collectives` failure | **device OOM at NEFF load**, not a hang | reduce in-flight activations (batch × chunk), not weights — the module already loaded |
| `NRT status 1204` / notification queue overflow | debug DGE notifications at scale | strip them (rule 1) |
| compile error only as terse "COMPILATION FAILED" | the real neuronx-cc diagnostic is hidden | re-run with `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1` + an artifact mount, read `neuronx-cc/<hash>/log-neuron-cc.txt` |
| SSH handshake times out while TCP 22 still accepts; `SSM PingStatus=ConnectionLost` | **host wedged** (usually host-RAM exhaustion from parallel compiles) | rule 5 |

A hang and an OOM look similar in a log tail and have opposite fixes. Grep for both before
concluding.

## 5. `/mnt/nvme` is instance store — reboot, never stop

The Trainium box's `/mnt/nvme` holds model weights, source snapshots, every compile cache, and
all profile artifacts. **A stop/start destroys all of it** (and changes the public IP). A
**reboot preserves it.**

```bash
aws ec2 reboot-instances --region <region> --instance-ids <id>   # safe: instance store survives
# aws ec2 stop-instances                                        # DESTROYS /mnt/nvme
```

Verified after a real recovery: post-reboot, weights, FP8 checkpoint and all caches were intact.

Always rediscover the IP from the instance ID rather than trusting a noted address:

```bash
aws ec2 describe-instances --region <region> --instance-ids <id> \
  --query 'Reservations[0].Instances[0].{State:State.Name,IP:PublicIpAddress}'
```

Get explicit user confirmation before any power action; a wedged box is not an emergency that
justifies risking the volume.

### What wedges the host

Compiling **8 ranks of a large graph in parallel** exhausted 124 GB of host RAM. Watch
`free -g` during compiles; the danger window is when several ranks enter neuronx-cc together.
If host RAM is the constraint, cross-compile on a large-RAM host and transplant the cache
rather than growing the graph.

## Sizing rules learned empirically (calibrate per model)

- Cap **in-flight tokens** (`batch × chunk`) rather than batch. On the 35B at 40 layers,
  4,096 in-flight compiled and ran; 8,192 wedged the host; 16,384 OOM'd the device at load.
- Iterate at small layer counts (4 layers ≈ 5 min compile) — block/loop counts usually depend
  on batch × chunk and are **layer-independent**, so most scaling bugs reproduce at 4 layers.
- Warm caches turn a 40-layer compile from tens of minutes into minutes; keep per-config cache
  dirs keyed by every parameter that changes the graph, or you will silently collide.

## Correctness gating for performance work

Never accept a speedup without an equivalence check, and pick the right strength:

- **Bit-identical** for changes that must not alter arithmetic (re-tiling, packing, chunking a
  data-parallel loop, scheduling). Gate on the full fingerprint: top-k token ids **and**
  `sum`/`norm` **and** all-finite **and** buffer occupancy (nonzero counts).
- **Coherence + cosine** for changes that legitimately alter numerics (quantization layout,
  different accumulation order). Pre-register the threshold (e.g. cosine ≥ 0.999) *before*
  measuring.
- A matching generation hash is **not** an equivalence gate — ours agreed across a real defect
  *and* across the fix for it. Occupancy plus norms caught what the hash missed.
- Beware silent finite-but-wrong: mutating one tensor twice in a traced graph loses writes, and
  the output stays plausible. Check occupancy of every buffer you expect to be written.
