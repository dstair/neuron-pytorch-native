# neuron-pytorch-native

Community contributions for **PyTorch Native** on AWS Trainium — reference model
implementations, custom [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
kernels, and benchmarking harnesses.

> PyTorch Native is a PyTorch backend for AWS Neuron devices that adds a `neuron`
> device via PyTorch's [PrivateUse1](https://docs.pytorch.org/tutorials/advanced/privateuseone.html)
> mechanism, so models compile through `torch.compile(backend="neuron")` instead
> of the XLA lowering path. See the
> [PyTorch Native overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html).

## What's here

Everything lives under [`contrib/`](contrib/). Each subdirectory is a
self-contained reference implementation of a model on PyTorch Native, including
the custom NKI kernels it needs, correctness tests, and the scripts used to
compile, run, and benchmark it on Trainium2.

| Directory | Model | Highlights |
|---|---|---|
| [`contrib/qwen3.6-27b`](contrib/qwen3.6-27b) | Qwen3.6-27B (dense hybrid) | DeltaNet + GQA backbone, fused DeltaNet NKI kernels, chunked prefill, GQA-tail mega-kernel, FP8 W8A16, EAGLE3 speculative decode |
| [`contrib/qwen3.5-35b-a3b`](contrib/qwen3.5-35b-a3b) | Qwen3.5-35B-A3B (sparse MoE) | 256-expert top-8 MoE (masked-dense + true-sparse dispatch), DeltaNet + GQA backbone, MoE FP8, 20k-context config sweep |

The [ESM](https://github.com/facebookresearch/esm) protein-language model is
planned as a future addition.

## Status

These are **research / bring-up** implementations built against a pre-GA build of
PyTorch Native. They are shared so that they are ready to run when PyTorch Native
reaches general availability, and so that customers evaluating these models on
Trainium have a working starting point. Expect rough edges: some paths are gated
behind environment flags, and a few tuning levers are kept in the tree
(default-off) for reference even where they did not ultimately win.

Each model README documents the verified architecture, how to run decode /
prefill / benchmarks, the performance results we measured, and which optimization
levers help.

## Design conventions

The implementations share a common structure worth knowing before you read the code:

- **A single static forward.** Each model has a `static_decode*.py` that expresses
  the whole decode (and optionally prefill) step as one fixed-shape function that
  compiles to a single NEFF per graph via `torch.compile(fullgraph=True,
  backend="neuron")`. This eliminates the per-op eager dispatch overhead that
  dominates latency on the HuggingFace eager path.
- **NKI kernels as `torch.ops`.** Custom kernels live under `kernels/`. Each kernel
  file (e.g. `deltanet_full.py`) has a companion `*_ops.py` that registers it as a
  `torch.ops.*` custom op so it drops into the compiled graph without a graph break.
- **Manual tensor parallelism.** Weights are sharded per-core by hand (colwise /
  rowwise) with functional all-reduces at known TP boundaries — no framework TP layer.
- **CPU oracles first.** Kernel math is validated against a CPU reference (and,
  where relevant, against HuggingFace) before any device compile. Look for
  `kernels/tests/` and files ending in `_ref`.

## Running the scripts

The deploy / profiling scripts under each model's `deploy/` directory reference a
couple of environment-specific values (a container registry, a weights bucket)
through placeholders like `${ECR_REGISTRY}` and `${S3_MODEL_BUCKET}`. Copy
[`.env.example`](.env.example) to `.env`, fill in your own values, and `source`
it before running them:

```bash
cp .env.example .env
# edit .env with your registry / bucket
source .env
```

`.env` is gitignored.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
