#!/usr/bin/env python3
"""Compare eager-PJRT and torch-compiled decode layer outputs on trn1."""

import os
import sys
from pathlib import Path


sys.path.insert(0, "/opt/xc/pysite")
sys.path.insert(
    0,
    os.environ.get("Q27_REPO", os.path.dirname(os.path.abspath(__file__))),
)

_local_rank = os.environ.get("LOCAL_RANK", "0")
for _key in ("NEURON_COMPILE_CACHE_URL", "TORCH_NEURONX_NEFF_CACHE_DIR"):
    if os.environ.get(_key):
        os.environ[_key] = os.environ[_key].rstrip("/") + f"/r{_local_rank}"

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend  # noqa: F401

import static_decode as sd
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from dev_stepcompile import PROMPT


def _neff_count() -> int:
    cache = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR")
    if not cache:
        return -1
    return sum(1 for _ in Path(cache).rglob("*.neff"))


def _compare(rank: int, name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    expected_cpu = expected.detach().float().cpu()
    actual_cpu = actual.detach().float().cpu()
    diff = (expected_cpu - actual_cpu).abs()
    if rank == 0:
        print(
            f"DIFF {name} shape={tuple(expected_cpu.shape)} "
            f"max={diff.max().item():.7g} mean={diff.mean().item():.7g} "
            f"expected_absmax={expected_cpu.abs().max().item():.7g} "
            f"actual_absmax={actual_cpu.abs().max().item():.7g} "
            f"exact={torch.equal(expected_cpu, actual_cpu)}",
            flush=True,
        )


@torch.no_grad()
def main() -> None:
    dist.init_process_group(backend="xla")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = xm.xla_device()
    model_path = os.environ["QWEN27_MODEL_PATH"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)[-128:]
    position = torch.tensor(len(prompt_ids), dtype=torch.long, device=device)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    sd.shard_model(model, rank, world_size)
    weights = sd.extract_weights(model)
    sd.apply_vocab_sharding(weights, rank, world_size)
    module = sd.StaticDecodeModule(
        weights, 512, world_size, batch_size=1, rank=rank
    ).to(device).eval()
    del model, weights

    snapshot_template = os.environ.get("PREFILL_SNAPSHOT_PATH")
    if not snapshot_template:
        raise SystemExit(
            "Set PREFILL_SNAPSHOT_PATH in the gitignored .env file; "
            "{rank} is supported."
        )
    snapshot = torch.load(
        snapshot_template.format(rank=rank), map_location="cpu"
    )
    base_dn = snapshot["dn"].to(device)
    base_cv = snapshot["cv"].to(device)
    base_kk = snapshot["kk"].to(device)
    base_vv = snapshot["vv"].to(device)
    token = torch.tensor([int(snapshot["first"])], dtype=torch.long, device=device)

    hidden = module._embed_tokens(token).unsqueeze(1)
    cos = (
        module.rope_cos.squeeze(0)
        .squeeze(0)
        .index_select(0, position.unsqueeze(0))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sin = (
        module.rope_sin.squeeze(0)
        .squeeze(0)
        .index_select(0, position.unsqueeze(0))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    xm.mark_step()
    xm.wait_device_ops()
    dist.barrier()

    eager_dn = base_dn.clone()
    eager_cv = base_cv.clone()
    eager_kk = base_kk.clone()
    eager_vv = base_vv.clone()
    eager_hidden = module._run_decode_layers(
        0, 1, hidden, cos, sin, position,
        eager_dn, eager_cv, eager_kk, eager_vv,
    )
    xm.mark_step()
    xm.wait_device_ops()
    dist.barrier()

    def layer_hidden_only(h, c, s, p, dn, cv, kk, vv):
        return module._run_decode_layers(0, 1, h, c, s, p, dn, cv, kk, vv)

    compiled_hidden_only = torch.compile(
        layer_hidden_only,
        backend=sd._COMPILE_BACKEND,
        fullgraph=True,
        dynamic=False,
    )
    compiled_dn = base_dn.clone()
    compiled_cv = base_cv.clone()
    compiled_kk = base_kk.clone()
    compiled_vv = base_vv.clone()
    before = _neff_count()
    compiled_hidden = compiled_hidden_only(
        hidden, cos, sin, position,
        compiled_dn, compiled_cv, compiled_kk, compiled_vv,
    )
    xm.wait_device_ops()
    after = _neff_count()
    dist.barrier()

    if rank == 0:
        print(f"hidden_only neff_delta={after - before}", flush=True)
    _compare(rank, "hidden_only.hidden", eager_hidden, compiled_hidden)
    _compare(rank, "hidden_only.dn0", eager_dn[0], compiled_dn[0])
    _compare(rank, "hidden_only.cv0", eager_cv[0], compiled_cv[0])

    def layer_explicit_state(h, c, s, p, dn, cv, kk, vv):
        out = module._run_decode_layers(0, 1, h, c, s, p, dn, cv, kk, vv)
        return out, dn, cv, kk, vv

    compiled_explicit = torch.compile(
        layer_explicit_state,
        backend=sd._COMPILE_BACKEND,
        fullgraph=True,
        dynamic=False,
    )
    explicit_dn = base_dn.clone()
    explicit_cv = base_cv.clone()
    explicit_kk = base_kk.clone()
    explicit_vv = base_vv.clone()
    before = _neff_count()
    (
        explicit_hidden,
        explicit_dn,
        explicit_cv,
        explicit_kk,
        explicit_vv,
    ) = compiled_explicit(
        hidden, cos, sin, position,
        explicit_dn, explicit_cv, explicit_kk, explicit_vv,
    )
    xm.wait_device_ops()
    after = _neff_count()
    dist.barrier()

    if rank == 0:
        print(f"explicit_state neff_delta={after - before}", flush=True)
    _compare(rank, "explicit.hidden", eager_hidden, explicit_hidden)
    _compare(rank, "explicit.dn0", eager_dn[0], explicit_dn[0])
    _compare(rank, "explicit.cv0", eager_cv[0], explicit_cv[0])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
