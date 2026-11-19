# Integrating the fused FP8 MoE kernel into a Qwen3.5-MoE decode graph

This kernel replaces the **routed-expert** compute of one MoE layer. The router
and shared expert stay in the surrounding `torch.compile(fullgraph=True,
backend="neuron")` graph, and the model runs **expert-parallel** across the TP
group (each rank owns `256 // world_size` experts).

## One-time, at load (CPU)

Per MoE layer, for this rank's expert range `[expert_start, expert_start + E_local)`,
build the kernel operands from the official FP8 checkpoint with `weight_prep`
(see its `load_layer(..., fp8_impl="block_ob_coalesced")` and the
`requantize_official_fp8_output_block` / `pack_coalesced_block_scales` helpers):

```python
# gate_up: [E_local, H, 2, I]  int8 (E4M3 bytes)
# down:    [E_local, I, H]     int8 (E4M3 bytes)
# gate_up_scales: [E_local, 128, 2*(H/128)*(I/128)] bf16
# down_scales:    [E_local, 128, (I/128)*(H/128)]    bf16
self.w8_gate_up, self.w8_down, self.w8_gate_up_scale, self.w8_down_scale = ...
```

Router weights, `shared_expert.{gate,up,down}`, and `shared_expert_gate` come
from the BF16 checkpoint, replicated on every rank.

## Per-layer forward (in the compiled decode graph)

```python
import torch, torch.nn.functional as F
from kernels import get_kernel
k = get_kernel("<org>/qwen35-moe-fp8-block-coalesced")

def moe_forward(x2d, layer):                 # x2d: [B, H] bf16, B in {32,64,128}
    xf = x2d.float()

    # 1) Router — softmax over all 256 experts, top-8, norm_topk_prob=True
    logits = F.linear(xf, layer.router.float())
    routing = F.softmax(logits, dim=1, dtype=torch.float)
    top_w, top_id = torch.topk(routing, 8, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)

    # 2) Local affinities  -> [B, E_local]  (0 for experts not on this rank)
    aff = k.weight_prep.build_local_affinities(
        top_id, top_w, layer.expert_start, layer.local_experts
    )

    # 3) Fused local-expert kernel  -> [B, H] f32 (pre all-reduce)
    local_routed = torch.ops.moe_w8.fused_fp8_block_coalesced_ob(
        x2d.to(torch.bfloat16),
        layer.w8_gate_up, layer.w8_down,
        layer.w8_gate_up_scale, layer.w8_down_scale,
        aff,
    )

    # 4) Expert-parallel all-reduce (sum) across the TP group
    global_routed = functional_all_reduce(local_routed, "sum", tp_group)

    # 5) Shared expert (replicated, bf16), added AFTER the all-reduce
    sh_gate = torch.sigmoid(F.linear(xf, layer.sh_sigmoid.float()))
    shared = F.linear(
        F.silu(F.linear(xf, layer.sh_gate.float())) * F.linear(xf, layer.sh_up.float()),
        layer.sh_down.float(),
    )
    return (global_routed + sh_gate * shared).to(x2d.dtype)
```

Notes:
- `functional_all_reduce` is your framework's functional collective; the sum
  reconstructs the full top-8 routed output from each rank's local partial.
- The shared expert is identical on every rank, so it is added post-reduce with
  no second collective.
- One compiled graph per batch shape (32/64/128). Decode-only; run prefill
  separately in bf16.
- The reference implementation this mirrors is `static_decode_35b.py`
  `_moe_fused_w8_parts` / `_moe_routing` / `_shared_expert`.
```
