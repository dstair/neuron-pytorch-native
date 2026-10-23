#!/usr/bin/env python3
"""Weight-prep checks on CPU, kernel parity on trn2, and accuracy vs the official checkpoint.

    python3 tests/test_moe_fp8.py --cpu-only                 # weight prep only, any host
    python3 tests/test_moe_fp8.py                            # + kernel parity on trn2
    python3 tests/test_moe_fp8.py --experts 32               # one TP=8 rank's local experts
    python3 tests/test_moe_fp8.py --experts 32 --checkpoint /path/to/Qwen3.5-35B-A3B-FP8 --layer 0

Two references, two questions:

* `moe_w8.fused_moe_reference` (float32 CPU on the SAME FP8 operands) -- is the kernel right?
* an fp32 MoE computed straight from the official checkpoint (E4M3FN decode x 128x128
  `weight_scale_inv`, HF gate/up/down layout, independent of this repo's packing) -- what
  does the conversion cost end to end? Lossless pow2 is gated tight; the opt-in
  output-block path is checked against a loose gate, since its ~4% rel-L2 is by design.

Both fp8_impls run at every batch size, on routing that includes tokens with no local
expert at all (all-zero affinity rows) and duplicate routes to one expert.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "kernels"))

import moe_w8 as mf  # noqa: E402

KERNEL_GATE = (0.9999, 0.01)          # (min cos, max rel-L2) vs the same-operand reference
OFFICIAL_GATE = {                     # vs the official checkpoint
    "block_pow2_coalesced": (0.9999, 0.01),
    "block_ob_coalesced": (0.998, 0.06),
}
IMPLS = mf.FP8_IMPLS
OPS = {
    "block_pow2_coalesced": "fused_fp8_block_coalesced",
    "block_ob_coalesced": "fused_fp8_block_coalesced_ob",
}


def _cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def _rel_l2(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _gate(out, ref, gate):
    cos, rel = _cos(out, ref), _rel_l2(out, ref)
    good = cos >= gate[0] and rel <= gate[1] and bool(torch.isfinite(out).all())
    return good, f"cos={cos:.7f} rel_l2={rel:.2e} {'PASS' if good else 'FAIL'}"


def make_operands(experts, fp8_impl, hidden_size=2048, intermediate=512, seed=0):
    """Random expert weights -> kernel operands for one fp8_impl.

    pow2: one absmax scale per 128x128 block. ob: one per 128 output rows, repeated across
    input blocks (the shape `requantize_official_fp8_output_block` returns).
    """
    g = torch.Generator().manual_seed(seed)
    bs = mf.BLOCK_SIZE

    def quantize(w):  # w: [out, in] f32 -> (int8 [out,in], bf16 grid [out/128, in/128])
        out_dim, in_dim = w.shape
        q = torch.empty(out_dim, in_dim, dtype=torch.int8)
        grid = torch.empty(out_dim // bs, in_dim // bs, dtype=torch.bfloat16)
        for r in range(out_dim // bs):
            rows = slice(r * bs, (r + 1) * bs)
            if fp8_impl == "block_ob_coalesced":
                s = (w[rows].abs().amax().clamp_min(1e-12) / mf.LEGACY_E4M3_MAX).to(torch.bfloat16)
                q[rows] = mf.encode_legacy_e4m3(w[rows] / s.float())
                grid[r] = s
                continue
            for c in range(in_dim // bs):
                cols = slice(c * bs, (c + 1) * bs)
                blk = w[rows, cols]
                s = (blk.abs().amax().clamp_min(1e-12) / mf.LEGACY_E4M3_MAX).to(torch.bfloat16)
                q[rows, cols] = mf.encode_legacy_e4m3(blk / s.float())
                grid[r, c] = s
        return q, grid

    gu_w, gu_s, dn_w, dn_s = [], [], [], []
    for _ in range(experts):
        proj, proj_s = [], []
        for _p in range(2):  # gate, up: [I, H]
            q, s = quantize(torch.randn(intermediate, hidden_size, generator=g) * 0.02)
            proj.append(q.t().contiguous())
            proj_s.append(s)
        gu_w.append(torch.stack(proj, dim=1))          # [H, 2, I]
        gu_s.append(torch.stack(proj_s, dim=0))        # [2, I/128, H/128]
        q, s = quantize(torch.randn(hidden_size, intermediate, generator=g) * 0.02)
        dn_w.append(q.t().contiguous())                # [I, H]
        dn_s.append(s)                                 # [H/128, I/128]
    gu_grid, dn_grid = torch.stack(gu_s), torch.stack(dn_s)
    if fp8_impl == "block_ob_coalesced":
        mf.assert_output_block_scale_grid(gu_grid, dn_grid)
    gu_table, dn_table = mf.pack_coalesced_block_scales(gu_grid, dn_grid)
    return torch.stack(gu_w), torch.stack(dn_w), gu_table, dn_table


def make_affinities(batch, experts, total_experts=256, top_k=8, seed=1):
    """Top-k routing where every 4th token has NO local expert and every 5th has a duplicate."""
    g = torch.Generator().manual_seed(seed)
    probs = torch.softmax(torch.randn(batch, total_experts, generator=g), dim=-1)
    top_w, top_id = torch.topk(probs, top_k, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_id = top_id % experts                       # most tokens land on this rank
    top_id[::4] = experts + (top_id[::4] % (total_experts - experts))   # fully off-rank
    top_id[1::5, 1] = top_id[1::5, 0]               # duplicate route to one expert
    aff = mf.build_local_affinities(top_id, top_w, 0, experts)
    assert bool((aff[::4] == 0).all()) and bool((aff.sum(-1) > 0).any())
    return aff


def official_reference(reader, layer, experts, hidden, affinities):
    """fp32 MoE straight from the official checkpoint, independent of this repo's packing."""
    x = hidden.float()
    out = torch.zeros(x.shape[0], x.shape[1])
    prefix = f"{reader.prefix}layers.{layer}.mlp.experts."
    for e in range(experts):
        if not bool((affinities[:, e] != 0).any()):
            continue
        w = {}
        for p in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}{e}.{p}.weight"
            w[p] = mf.dequantize_official_fp8(
                reader._get(key, "F8_E4M3"), reader._get(key + "_scale_inv", "BF16")
            )
        act = F.silu(x @ w["gate_proj"].t()) * (x @ w["up_proj"].t())
        out += affinities[:, e : e + 1].float() * (act @ w["down_proj"].t())
    return out


def test_cpu():
    # Legacy E4M3 round-trip is exact on representable values and finite through 240.
    # Compare values, not bytes: -0 (0x80) canonicalises to +0 (0x00).
    codes = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    legal = codes[((codes.view(torch.uint8) >> 3) & 0x0F) != 0x0F]
    vals = mf.decode_legacy_e4m3(legal)
    assert torch.equal(mf.decode_legacy_e4m3(mf.encode_legacy_e4m3(vals)), vals), "E4M3 round-trip"
    assert float(vals.abs().max()) == mf.LEGACY_E4M3_MAX
    # Duplicate routes accumulate; off-rank routes are dropped.
    aff = mf.build_local_affinities(
        torch.tensor([[0, 0, 5]]), torch.tensor([[0.25, 0.25, 0.5]]), 0, 4
    )
    assert torch.allclose(aff, torch.tensor([[0.5, 0.0, 0.0, 0.0]]))
    # pow2 requant is exact on E4M3FN blocks, including ones using the >240 codes.
    raw = torch.randint(0, 256, (256, 256), dtype=torch.int16).to(torch.uint8)
    raw[(raw & 0x7F) == 0x7F] -= 1                   # E4M3FN's only NaN code -> largest finite
    raw[:128, :128] = (raw[:128, :128] & 0x3F)       # one block with no extended codes
    scales = torch.full((2, 2), 2.0 ** -8, dtype=torch.bfloat16)
    q, s, stats = mf.requantize_official_fp8_pow2(raw, scales)
    assert stats.shifted_block_count == 3 and stats.clipped_count == 0
    assert torch.equal(q[:128, :128].view(torch.uint8), raw[:128, :128])   # untouched block
    assert stats.normalized_rmse < 1e-3, stats.normalized_rmse
    # The ob kernel's precondition is enforced: a per-128x128 grid is refused.
    try:
        mf.assert_output_block_scale_grid(
            torch.rand(1, 2, 4, 16).bfloat16(), torch.rand(1, 16, 4).bfloat16()
        )
        raise AssertionError("per-block grid was accepted by the ob guard")
    except ValueError:
        pass
    # Same-operand reference runs for both impls.
    for impl in IMPLS:
        gu, dn, gu_t, dn_t = make_operands(2, impl, hidden_size=512, intermediate=512)
        ref = mf.fused_moe_reference(torch.randn(4, 512).bfloat16(), gu, dn, gu_t, dn_t, torch.rand(4, 2))
        assert ref.shape == (4, 512) and torch.isfinite(ref).all()
    print("CPU checks: PASS")


def run_device(fp8_impl, operands, batches, label, official=None):
    dev = torch.device("neuron")
    op = getattr(torch.ops.moe_w8, OPS[fp8_impl])
    gu, dn, gu_t, dn_t = operands
    hidden_size, experts = gu.shape[1], gu.shape[0]
    dev_ops = [t.to(dev) for t in operands]
    ok = True
    for b in batches:
        x = (torch.randn(b, hidden_size, generator=torch.Generator().manual_seed(b)) * 0.5).bfloat16()
        aff = make_affinities(b, experts)
        out = op(x.to(dev), *dev_ops, aff.to(dev)).cpu()
        good, msg = _gate(out, mf.fused_moe_reference(x, gu, dn, gu_t, dn_t, aff), KERNEL_GATE)
        line = f"[{label} {fp8_impl}] E={experts} B={b:3d} kernel: {msg}"
        ok &= good
        if official is not None:
            good_o, msg_o = _gate(out, official(x, aff), OFFICIAL_GATE[fp8_impl])
            line += f" | vs official: {msg_o}"
            ok &= good_o
        print(line)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--batches", default="32,64,128")
    ap.add_argument("--checkpoint", help="official Qwen3.5-35B-A3B-FP8 dir (optional)")
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    test_cpu()
    if args.cpu_only:
        return
    import moe_fused_w8_35b_ops  # noqa: F401  registers torch.ops.moe_w8.*

    batches = [int(b) for b in args.batches.split(",")]
    ok = True
    for impl in IMPLS:
        ok &= run_device(impl, make_operands(args.experts, impl), batches, "synthetic")
    if args.checkpoint:
        reader = mf.OfficialFP8ExpertReader(args.checkpoint)

        def official(x, aff):
            return official_reference(reader, args.layer, args.experts, x, aff)

        for impl in IMPLS:
            w, stats = reader.load_layer(args.layer, 0, args.experts, fp8_impl=impl)
            print(
                f"[layer{args.layer} {impl}] requant vs official: cosine={stats.cosine:.7f} "
                f"nrmse={stats.normalized_rmse:.3%} shifted_blocks={stats.shifted_block_fraction:.1%}"
            )
            ops = (w["w8_gate_up"], w["w8_down"], w["w8_gate_up_scale"], w["w8_down_scale"])
            ok &= run_device(impl, ops, batches, f"layer{args.layer}", official=official)
        reader.close()
    print("ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
