#!/usr/bin/env python3
"""Batched bucketed-prefill isolation against independent BS=1 executions.

Exercises different prompt contents, a partial final bucket, logits, DeltaNet
and convolution carry state, and each batch row's GQA cache. Run with CTE GQA:

PYTHONPATH=<nki-library>/src/nkilib_src \
MOE_CTE=1 GQA_CTE_PREFILL=1 GQA_DYNAMIC_ROPE_KV=1 DN_CHUNK_NKI=1 CHUNK_SIZE=16 \
torchrun --nproc-per-node=4 kernels/tests/test_bucketed_prefill_batched_device.py \
  --model-path <weights> --batch-size 2 --num-layers 4 --seq 1152 --chunk 1024

For TP=8/LNC1, set QWEN35_LNC=1, DN_K_HEADS=2, DN_V_HEADS=4, and
GQA_Q_HEADS=2, then launch eight ranks.
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "kernels"))

import static_decode_35b as S
import model_dims as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/models/Qwen3.5-35B-A3B")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1152)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument(
        "--moe-only",
        action="store_true",
        help="run one real-weight MoE call and exit before bucketed prefill",
    )
    ap.add_argument(
        "--capture-routes-dir",
        help="save fused CTE route tensors before each call (diagnostic)",
    )
    ap.add_argument(
        "--capture-cte-inputs-dir",
        help="save fused CTE inputs before each call (diagnostic)",
    )
    ap.add_argument(
        "--capture-routes-only",
        action="store_true",
        help="capture fused CTE routes and return zero MoE output",
    )
    ap.add_argument(
        "--roundtrip-routes",
        action="store_true",
        help="replace fused CTE routes with a synchronized host-to-device copy",
    )
    ap.add_argument(
        "--roundtrip-cte-inputs",
        action="store_true",
        help="replace fused CTE runtime activations with host-to-device copies",
    )
    ap.add_argument(
        "--roundtrip-cte-weights",
        action="store_true",
        help="replace fused CTE expert weights with host-to-device copies",
    )
    ap.add_argument(
        "--synchronize-cte",
        action="store_true",
        help="synchronize the device immediately before each fused CTE call",
    )
    ap.add_argument(
        "--rewrap-cte-after-dist",
        action="store_true",
        help="recreate the fused CTE NKI wrapper after distributed initialization",
    )
    args = ap.parse_args()
    if args.seq <= args.chunk:
        ap.error("--seq must exceed --chunk to test a partial final bucket")
    if args.batch_size < 2:
        ap.error("--batch-size must be at least 2")
    if args.capture_routes_only and not (
        args.capture_routes_dir or args.capture_cte_inputs_dir
    ):
        ap.error(
            "--capture-routes-only requires --capture-routes-dir or "
            "--capture-cte-inputs-dir"
        )

    import torch_neuronx  # noqa: F401
    dist.init_process_group(backend="neuron")
    rank, ws = dist.get_rank(), dist.get_world_size()
    local_experts = D.NUM_EXPERTS // ws
    device = torch.neuron.current_device()
    if args.rewrap_cte_after_dist:
        if not S.USE_MOE_CTE_NKI_PACK:
            ap.error("--rewrap-cte-after-dist requires MOE_CTE_NKI_PACK=1")
        S._nkilib_moe_cte_hop = S.wrap_nki(
            S.nki_moe_cte_routed_35b
        )[S.LNC_DEGREE]
    if (
        args.capture_routes_dir
        or args.capture_cte_inputs_dir
        or args.roundtrip_routes
        or args.roundtrip_cte_inputs
        or args.roundtrip_cte_weights
        or args.synchronize_cte
    ):
        if not S.USE_MOE_CTE_NKI_PACK:
            ap.error("route diagnostics require MOE_CTE_NKI_PACK=1")
        if args.capture_routes_dir:
            os.makedirs(args.capture_routes_dir, exist_ok=True)
        if args.capture_cte_inputs_dir:
            os.makedirs(args.capture_cte_inputs_dir, exist_ok=True)
        original_cte = S._nkilib_moe_cte_hop
        route_call = 0

        def capture_routes(*cte_args):
            nonlocal route_call
            routes = None
            if (
                args.capture_routes_dir
                or args.roundtrip_routes
                or args.roundtrip_cte_inputs
            ):
                routes = cte_args[4].cpu()
            if args.capture_routes_dir:
                path = os.path.join(
                    args.capture_routes_dir,
                    f"routes_rank{rank}_call{route_call}.pt",
                )
                torch.save(routes, path)
                local = routes - rank * local_experts
                local_count = int(
                    ((local >= 0) & (local < local_experts)).sum()
                )
                print(
                    f"rank={rank} captured routes call={route_call} "
                    f"shape={tuple(routes.shape)} local={local_count} path={path}",
                    flush=True,
                )
            if args.capture_cte_inputs_dir:
                path = os.path.join(
                    args.capture_cte_inputs_dir,
                    f"inputs_rank{rank}_call{route_call}.pt",
                )
                torch.save(
                    {
                        "hidden": cte_args[0].cpu(),
                        "affinities": cte_args[1].cpu(),
                        "gate_up": cte_args[2].cpu(),
                        "down": cte_args[3].cpu(),
                        "routes": cte_args[4].cpu(),
                        "expert_lo": cte_args[5],
                        "block_size": cte_args[6],
                    },
                    path,
                )
                print(
                    f"rank={rank} captured CTE inputs call={route_call} path={path}",
                    flush=True,
                )
            route_call += 1
            if args.capture_routes_only:
                return torch.zeros_like(cte_args[0])
            if (
                args.roundtrip_routes
                or args.roundtrip_cte_inputs
                or args.roundtrip_cte_weights
            ):
                cte_args = list(cte_args)
            if args.roundtrip_routes or args.roundtrip_cte_inputs:
                cte_args[4] = routes.to(device)
            if args.roundtrip_cte_inputs:
                cte_args[0] = cte_args[0].cpu().to(device)
                cte_args[1] = cte_args[1].cpu().to(device)
            if args.roundtrip_cte_weights:
                cte_args[2] = cte_args[2].cpu().to(device)
                cte_args[3] = cte_args[3].cpu().to(device)
            if args.synchronize_cte:
                torch.neuron.synchronize()
            return original_cte(*cte_args)

        S._nkilib_moe_cte_hop = capture_routes

    D.load_from_config(os.path.join(args.model_path, "config.json"))
    D.NUM_LAYERS = args.num_layers
    D.NUM_GQA = sum(D.layer_type(i) == "gqa" for i in range(D.NUM_LAYERS))
    D.NUM_DELTANET = D.NUM_LAYERS - D.NUM_GQA
    weights = S.load_sharded_weights(args.model_path, rank, ws, num_layers=D.NUM_LAYERS)
    mod = S.StaticDecode35B(
        weights, args.max_seq_len, ws, batch_size=args.batch_size, rank=rank,
    ).to(device).eval()

    td = D.tp_dims(ws)
    vh, kd, vd = td["dn_v_heads"], D.DN_K_DIM, D.DN_V_DIM
    qkv = 2 * td["dn_k_heads"] * kd + vh * vd
    nkv = max(1, D.GQA_KV_HEADS // ws)

    def state(batch):
        return (
            torch.zeros(D.NUM_DELTANET, batch, vh * kd, vd, dtype=torch.bfloat16, device=device),
            torch.zeros(D.NUM_DELTANET, batch, qkv, D.DN_CONV_KERNEL - 1, dtype=torch.bfloat16, device=device),
            torch.zeros(D.NUM_GQA, batch, nkv, args.max_seq_len, D.GQA_HEAD_DIM, dtype=torch.bfloat16, device=device),
            torch.zeros(D.NUM_GQA, batch, nkv, args.max_seq_len, D.GQA_HEAD_DIM, dtype=torch.bfloat16, device=device),
        )

    prompt0 = torch.arange(args.seq, device=device) % D.VOCAB
    prompts = torch.stack(
        tuple((prompt0 * (17 + 2 * row) + 11 * row) % D.VOCAB for row in range(args.batch_size)),
    )
    if args.moe_only:
        moe_input = F.embedding(prompts[:, :args.chunk], mod.embed).reshape(
            args.batch_size * args.chunk, D.HIDDEN
        ).float()
        moe_output = mod._moe_cte(
            0,
            moe_input,
            (args.batch_size, args.chunk),
        ).cpu()
        if rank == 0:
            print(
                f"real-weight MoE only: PASS shape={tuple(moe_output.shape)} "
                f"finite={bool(torch.isfinite(moe_output).all())}",
                flush=True,
            )
        return
    batched = mod.prefill_bucketed(
        prompts, *state(args.batch_size), chunk=args.chunk, compile_chunk=args.compile,
    )

    for row in range(args.batch_size):
        single = mod.prefill_bucketed(
            prompts[row], *state(1), chunk=args.chunk, compile_chunk=args.compile,
        )
        log_b, dn_b, cv_b, kk_b, vv_b = batched
        log_s, dn_s, cv_s, kk_s, vv_s = single
        pairs = (
            ("logits", log_b[row], log_s[0]),
            ("deltanet", dn_b[:, row], dn_s[:, 0]),
            ("conv", cv_b[:, row], cv_s[:, 0]),
            ("kv_k", kk_b[:, row], kk_s[:, 0]),
            ("kv_v", vv_b[:, row], vv_s[:, 0]),
        )
        for name, actual, expected in pairs:
            actual, expected = actual.float(), expected.float()
            if actual.numel() == 0:
                assert actual.shape == expected.shape
                if rank == 0:
                    print(f"row={row} {name}: skipped empty state")
                continue
            cosine = F.cosine_similarity(actual.reshape(-1), expected.reshape(-1), dim=0)
            max_diff = (actual - expected).abs().max()
            if rank == 0:
                print(f"row={row} {name}: cosine={float(cosine):.7f} max_diff={float(max_diff):.4e}")
            assert float(cosine) > 0.999, f"row {row} {name} diverged"
    if rank == 0:
        print("batched bucketed prefill isolation: PASS")


if __name__ == "__main__":
    main()
