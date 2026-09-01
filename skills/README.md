# Trainium / PyTorch-Native performance skills

Six skills capturing hard-won practice from bringing up and optimizing large models
(Qwen3.6-35B-A3B, 27B, OpenVLA) on Trainium2 with the **PyTorch Native** beta stack.

They exist because the generic vendor guidance and the kernel-scoped NKI skills
(`NKIBootcamp/neuron-agentic-development/skills/neuron-nki-*`) stop where real work starts:
a 40-layer TP=8 model, a compile cache full of NEFFs, a 30-second hang with no stack trace,
and a profile whose numbers are lying to you.

## Version stamp — READ THIS FIRST

Several facts here are version-bound and **will** age:

| component | version these skills were written against |
|---|---|
| DLC image | `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest` |
| Neuron SDK | 2.32 toolchain (`neuronx-cc` 2.27, `nki` 0.6.0) |
| driver / runtime | driver 2.29, `nrt` 2.33 |
| hardware | trn2.3xlarge (`NEURON_LOGICAL_NC_CONFIG=1` → 8 cores, TP=8/LNC=1) |
| tools | `neuron-explorer` 2.32 capture, bundled `view` binary reports release-2.31 |

Anything marked **(version-bound)** in a skill should be re-verified after an SDK bump.

## Which skill do I want?

| I want to… | skill |
|---|---|
| profile a full multi-layer / TP=N model, find the right NEFF, attribute time to engines, ops, source lines | `neuron-model-profiling` |
| launch, time, gate, or recover a run on the device; a run hung / OOM'd / gave suspicious timings | `trainium-run-discipline` |
| turn a profile into a ranked list of fixes; general kernel-level perf mechanics; check whether a lever was already refuted | `nki-perf-patterns` |
| make FP8 / quantization faster or smaller; FP8 dtype and scale-layout problems | `fp8-quantization-perf` |
| make a Mixture-of-Experts layer faster; routing, blocking, packing, expert-parallel | `moe-kernel-perf` |
| run or change the Qwen3.6-35B-A3B model in this repo specifically | `qwen35-35b-ops` |

Profile a **single NKI kernel** (one core, no collectives)? Use the existing
`neuron-nki-profiling` / `neuron-nki-profile-querying` skills instead — these six do not
duplicate them. Writing or debugging NKI source? `neuron-nki-writing` / `neuron-nki-debugging`.

## Boundary rules (so these don't drift into each other)

- **Detection** techniques (how to *find* a problem in profile data) → `neuron-model-profiling`.
- **Remedies** (what to *do* about it) → `nki-perf-patterns` and its two specializations.
- dtype and scale layout → `fp8-quantization-perf`.
- routing, blocking, packing, expert parallelism → `moe-kernel-perf`.
- Anything true only of one model in one repo → `qwen35-35b-ops`.

The one topic that genuinely straddles FP8 and MoE — block-quant scale application inside a
MoE kernel — lives in `fp8-quantization-perf`, because it is a property of the quantization
layout and its fix is a quantization-layout change. `moe-kernel-perf` points at it.

## What is deliberately NOT here

Project state and findings-in-flux (current baselines, what we tried last week, open blockers)
belong in session memory, not in skills. Build and workspace instructions belong in `CLAUDE.md`.
Skills hold **repeatable procedure and durable empirical verdicts** only.
