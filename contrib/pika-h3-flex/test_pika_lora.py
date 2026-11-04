# SPDX-License-Identifier: Apache-2.0
"""CPU tests for Pika LoRA application (delta math + module mapping, tp=1)."""

import pytest
import torch
from safetensors.torch import save_file
from test.unit.test_minimax_h3_single_device import _model, single_rank_groups  # noqa: F401
from vllm_omni_neuron.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Config
from vllm_omni_neuron.diffusion.models.minimax_h3.pika_lora import apply_pika_lora

# TINY config: heads=2 x 128, hidden 256, ffn 512, tp=1 -> inner 256 / ffn 512
H, F = 256, 512
MODULES = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"]
OUT = {"attn.to_q": H, "attn.to_k": H, "attn.to_v": H, "attn.to_out.0": H, "ff.net.0.proj": 2 * F, "ff.net.2": H}
IN = {"attn.to_q": H, "attn.to_k": H, "attn.to_v": H, "attn.to_out.0": H, "ff.net.0.proj": H, "ff.net.2": F}
BLOCKS = ["transformer_blocks.0", "transformer_blocks.1", "token_refiner.refiner_blocks.0"]


def make_adapter(path):
    torch.manual_seed(7)
    tensors = {}
    for b in BLOCKS:
        for m in MODULES:
            tensors[f"{b}.{m}.lora_A.default.weight"] = torch.randn(128, IN[m], dtype=torch.bfloat16)
            tensors[f"{b}.{m}.lora_B.default.weight"] = torch.randn(OUT[m], 128, dtype=torch.bfloat16)
    save_file(tensors, str(path))
    return tensors


def full_delta(tensors, b, m):
    a = tensors[f"{b}.{m}.lora_A.default.weight"].float()
    lb = tensors[f"{b}.{m}.lora_B.default.weight"].float()
    return lb @ a  # [out, in] in fp32; bf16 storage rounding expected in param


@pytest.mark.usefixtures("single_rank_groups")
def test_config_default_off_and_apply_updates_all_modules(tmp_path):
    assert MiniMaxH3Config().pika_lora_path == ""
    adapter = make_adapter(tmp_path / "adapter.safetensors")
    model = _model()
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    updated = apply_pika_lora(model, str(tmp_path / "adapter.safetensors"))
    assert updated == 3  # 2 DiT blocks + 1 refiner block

    def check(param, expected_t):
        assert tuple(param.shape) == tuple(expected_t.shape)
        torch.testing.assert_close(param, expected_t, rtol=1e-2, atol=0.2)

    for i in (0, 1):
        blk = model.transformer_blocks[i]
        A = f"transformer_blocks.{i}.attn"
        W = f"transformer_blocks.{i}.ff"
        B = f"transformer_blocks.{i}"
        # fused qkv [in, q|k|v] at tp=1
        check(blk.attn.qkv_proj_weight[:, 0:H], original[f"{A}.qkv_proj_weight"][:, 0:H] + full_delta(adapter, B, "attn.to_q").T)
        check(blk.attn.qkv_proj_weight[:, H : 2 * H], original[f"{A}.qkv_proj_weight"][:, H : 2 * H] + full_delta(adapter, B, "attn.to_k").T)
        check(blk.attn.qkv_proj_weight[:, 2 * H : 3 * H], original[f"{A}.qkv_proj_weight"][:, 2 * H : 3 * H] + full_delta(adapter, B, "attn.to_v").T)
        check(blk.attn.o_proj_weight, original[f"{A}.o_proj_weight"] + full_delta(adapter, B, "attn.to_out.0").T)
        check(blk.ff.up_proj_weight, original[f"{W}.up_proj_weight"] + full_delta(adapter, B, "ff.net.0.proj")[0:F].T)
        check(blk.ff.gate_proj_weight, original[f"{W}.gate_proj_weight"] + full_delta(adapter, B, "ff.net.0.proj")[F : 2 * F].T)
        check(blk.ff.down_proj_weight, original[f"{W}.down_proj_weight"] + full_delta(adapter, B, "ff.net.2").T)

    # refiner block updated too
    ref = model.token_refiner.refiner_blocks[0]
    check(ref.attn.o_proj_weight, original["token_refiner.refiner_blocks.0.attn.o_proj_weight"] + full_delta(adapter, "token_refiner.refiner_blocks.0", "attn.to_out.0").T)


@pytest.mark.usefixtures("single_rank_groups")
def test_forward_runs_after_lora(tmp_path):
    from test.unit.test_minimax_h3_single_device import _inputs, _run

    make_adapter(tmp_path / "adapter.safetensors")
    reference = _model()
    model = _model()
    model.load_state_dict(reference.state_dict())
    apply_pika_lora(model, str(tmp_path / "adapter.safetensors"))
    inputs = _inputs()
    with torch.no_grad():
        out_ref = _run(reference, inputs)
        out = _run(model, inputs)
    assert len(out) == len(out_ref)
    for a, r in zip(out, out_ref, strict=True):
        assert torch.isfinite(a).all()
        assert not torch.equal(a, r)  # LoRA changed the output
