# Pika-H3-Flex on Trainium2

Port of [Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex)
to AWS Trainium (trn2), on branch `pika-h3-flex`.

## Start here

- **[docs/RESUME_PLAN.md](docs/RESUME_PLAN.md)** — goal, corrections to initial
  assumptions, environment, lessons learned, phased port plan, first-actions
  checklist. Read this first, especially when resuming with fresh context.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — model component/architecture
  findings from the HF repo.

## Key facts (read before assuming anything)

- The DiT is the **same 50-layer / 33B MiniMax-H3 architecture** as the
  MiniMax-H3 already ported on this box — NOT a smaller model.
- "Fewer" is the **6-NFE sampling schedule** (vs 50 steps), ~8x fewer denoise
  forwards.
- It's **LoRA adapters** (`fl2va`, `ref2va`, rank 128/384) on the MiniMax-H3
  base; merge on host at load.
- License: **MiniMax H3 Community License** (same as MiniMax-H3).

## Layout

| Path | Contents |
|---|---|
| `docs/` | Resume plan and architecture findings |
| `deploy/` | Box/SSH/env/run helpers (from the MiniMax-H3 project) |

The prior MiniMax-H3 project (cache/FP8/sparsity work, benchmark harnesses) lives
on the `minimax-h3` branch under `contrib/minimax-h3/` — reference only.
