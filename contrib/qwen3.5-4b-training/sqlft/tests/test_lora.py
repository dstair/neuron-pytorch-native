"""Correctness for the hand-written LoRA. CPU only -- no device needed.

    ./env/container.sh exec python3 -m pytest sqlft/tests/test_lora.py -q
"""

import torch
import torch.nn as nn
import pytest

from sqlft.lora import LoRALinear, apply_lora, lora_parameters, lora_state_dict


def test_identity_at_init():
    """B is zeros, so the adapted layer must match the base EXACTLY at step 0."""
    torch.manual_seed(0)
    base = nn.Linear(32, 16)
    x = torch.randn(4, 32)
    expected = base(x)
    lora = LoRALinear(base, r=4, alpha=8)
    assert torch.equal(lora(x), expected)


def test_nonzero_after_B_is_perturbed():
    torch.manual_seed(0)
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(4, 32)
    before = lora(x).clone()
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.02)
    assert not torch.allclose(lora(x), before)


def test_only_adapter_is_trainable():
    torch.manual_seed(0)
    base = nn.Linear(32, 16, bias=True)
    lora = LoRALinear(base, r=4)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}


def test_gradients_reach_only_the_adapter():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(16, 8), r=4)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)  # otherwise dL/dA is legitimately zero
    lora(torch.randn(3, 16)).sum().backward()
    assert lora.lora_A.grad is not None and lora.lora_A.grad.abs().sum() > 0
    assert lora.lora_B.grad is not None and lora.lora_B.grad.abs().sum() > 0
    assert lora.base.weight.grad is None


def test_gradcheck_fp64():
    """Analytic autograd vs numerical jacobian. Catches any scaling/transpose error."""
    torch.manual_seed(0)
    base = nn.Linear(6, 5, bias=False).double()
    lora = LoRALinear(base, r=3, alpha=6).double()
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.3)
    x = torch.randn(2, 6, dtype=torch.double, requires_grad=True)

    def f(x_, A, B):
        return base(x_) + (lora.alpha / lora.r) * (x_ @ A.t() @ B.t())

    assert torch.autograd.gradcheck(f, (x, lora.lora_A, lora.lora_B), eps=1e-6, atol=1e-8)


def test_merge_unmerge_roundtrip():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    x = torch.randn(3, 16)

    unmerged = lora(x).clone()
    w_before = lora.base.weight.clone()

    lora.merge()
    assert torch.allclose(lora(x), unmerged, atol=1e-6), "merged output differs from unmerged"

    lora.unmerge()
    assert torch.allclose(lora.base.weight, w_before, atol=1e-6), "unmerge did not restore W"
    assert torch.allclose(lora(x), unmerged, atol=1e-6)


def test_scaling_uses_alpha_over_r():
    torch.manual_seed(0)
    base = nn.Linear(8, 4, bias=False)
    a = LoRALinear(base, r=2, alpha=4)     # scaling 2.0
    b = LoRALinear(nn.Linear(8, 4, bias=False), r=2, alpha=8)  # scaling 4.0
    b.base.load_state_dict(base.state_dict())
    with torch.no_grad():
        a.lora_A.copy_(torch.ones_like(a.lora_A)); a.lora_B.copy_(torch.ones_like(a.lora_B))
        b.lora_A.copy_(torch.ones_like(b.lora_A)); b.lora_B.copy_(torch.ones_like(b.lora_B))
    x = torch.randn(2, 8)
    base_out = base(x)
    da = a(x) - base_out
    db = b(x) - base_out
    assert torch.allclose(db, 2 * da, atol=1e-6)


class _Hybrid(nn.Module):
    """Stand-in for Qwen3.5's hybrid stack: one linear_attention-ish and one full-attn-ish."""

    def __init__(self):
        super().__init__()
        self.layer0 = nn.ModuleDict({"in_proj_qkv": nn.Linear(8, 8), "out_proj": nn.Linear(8, 8)})
        self.layer1 = nn.ModuleDict({"q_proj": nn.Linear(8, 8), "o_proj": nn.Linear(8, 8)})
        self.head = nn.Linear(8, 4)  # not a target; must stay frozen and unwrapped


def test_apply_lora_intersects_with_present_modules():
    m = _Hybrid()
    summary = apply_lora(m, r=2, verbose=False)
    assert summary["wrapped"] == 4
    assert isinstance(m.layer0["in_proj_qkv"], LoRALinear)
    assert isinstance(m.layer1["q_proj"], LoRALinear)
    assert not isinstance(m.head, LoRALinear)
    assert m.head.weight.requires_grad is False
    assert {id(p) for p in lora_parameters(m)} == {
        id(p) for n, p in m.named_parameters() if "lora_" in n
    }


def test_apply_lora_tolerates_absent_targets():
    """--layers 2 truncation removes whole attention families; that must not raise."""
    only_linear_attn = nn.ModuleDict({"in_proj_qkv": nn.Linear(8, 8)})
    summary = apply_lora(only_linear_attn, r=2, verbose=False)
    assert summary["wrapped"] == 1
    assert summary["targets_absent"] > 0


def test_apply_lora_raises_when_nothing_matches():
    with pytest.raises(ValueError, match="none of the LoRA targets"):
        apply_lora(nn.ModuleDict({"weird_name": nn.Linear(4, 4)}), r=2, verbose=False)


def test_state_dict_is_adapter_only():
    m = _Hybrid()
    apply_lora(m, r=2, verbose=False)
    sd = lora_state_dict(m)
    assert sd and all("lora_A" in k or "lora_B" in k for k in sd)
    full = sum(v.numel() for v in m.state_dict().values())
    adapter = sum(v.numel() for v in sd.values())
    assert adapter < full / 2
