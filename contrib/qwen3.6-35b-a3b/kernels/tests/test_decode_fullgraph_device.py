#!/usr/bin/env python3
"""Compare segmented and full-graph vocab-sharded real-weight decode."""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


NAMES = ("logits", "next_id", "deltanet", "conv", "kv_k", "kv_v")


def compare(reference_dir, candidate_dir, world_size):
    for rank in range(world_size):
        reference = torch.load(
            os.path.join(reference_dir, f"rank{rank}.pt"),
            map_location="cpu",
            weights_only=True,
        )
        candidate = torch.load(
            os.path.join(candidate_dir, f"rank{rank}.pt"),
            map_location="cpu",
            weights_only=True,
        )

        full_logits = reference["logits"].float()
        local_logits = candidate["logits"].float()
        shard = full_logits.shape[-1] // world_size
        expected_logits = full_logits[:, rank * shard:(rank + 1) * shard]
        torch.testing.assert_close(
            local_logits, expected_logits, rtol=2e-2, atol=5e-2
        )
        assert torch.equal(candidate["next_id"], reference["next_id"])

        for name in NAMES[2:]:
            expected = reference[name].float()
            actual = candidate[name].float()
            diff = actual - expected
            cosine = F.cosine_similarity(
                actual.reshape(-1), expected.reshape(-1), dim=0
            ).item()
            rel_l2 = (
                torch.linalg.vector_norm(diff)
                / torch.linalg.vector_norm(expected).clamp_min(1e-12)
            ).item()
            print(
                f"rank={rank} {name}: cosine={cosine:.8f} "
                f"max_abs={diff.abs().max().item():.6e} rel_l2={rel_l2:.6e}"
            )
            assert cosine > 0.999, (rank, name, cosine)
            assert rel_l2 < 0.02, (rank, name, rel_l2)

    print("PASS: local logits, greedy IDs, and all carried states match")


def capture(args):
    import torch_neuronx  # noqa: F401

    import model_dims as D
    import static_decode_35b as S

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.neuron.current_device()

    D.load_from_config(os.path.join(args.model_path, "config.json"))
    D.NUM_LAYERS = args.num_layers
    D.NUM_GQA = sum(
        1 for layer in range(D.NUM_LAYERS) if D.layer_type(layer) == "gqa"
    )
    D.NUM_DELTANET = D.NUM_LAYERS - D.NUM_GQA
    weights = S.load_sharded_weights(
        args.model_path, rank, world_size, num_layers=D.NUM_LAYERS
    )
    model = S.StaticDecode35B(
        weights,
        max_seq_len=args.max_seq_len,
        world_size=world_size,
        batch_size=args.batch_size,
        rank=rank,
    ).to(device).eval()
    del weights

    if args.mode == "segmented":
        model.setup_segments(1, compile_each=True)

        def step(token, position, *state):
            outputs = model(token, position, *state)
            return outputs[0], outputs[0].argmax(-1), *outputs[1:]
    else:
        assert S.USE_DECODE_FULLGRAPH
        assert S.USE_DECODE_SHARDED_LM_HEAD
        model.setup_segments(1, compile_each=False)
        step = torch.compile(
            model.decode_step, backend="neuron", fullgraph=True, dynamic=False
        )

    td = D.tp_dims(world_size)
    vh = td["dn_v_heads"]
    qkv_dim = 2 * td["dn_k_heads"] * D.DN_K_DIM + vh * D.DN_V_DIM
    state_cpu = (
        torch.zeros(
            D.NUM_DELTANET,
            args.batch_size,
            vh * D.DN_K_DIM,
            D.DN_V_DIM,
            dtype=torch.bfloat16,
        ),
        torch.zeros(
            D.NUM_DELTANET,
            args.batch_size,
            qkv_dim,
            D.DN_CONV_KERNEL - 1,
            dtype=torch.bfloat16,
        ),
        torch.zeros(
            D.NUM_GQA,
            args.batch_size,
            max(1, D.GQA_KV_HEADS // world_size),
            args.max_seq_len,
            D.GQA_HEAD_DIM,
            dtype=torch.bfloat16,
        ),
    )
    state_cpu = (*state_cpu, torch.zeros_like(state_cpu[-1]))
    state = tuple(tensor.to(device) for tensor in state_cpu)
    token = (
        torch.arange(args.batch_size, dtype=torch.long) * 7919
        + 100
    ).remainder(D.VOCAB).to(device)

    with torch.no_grad():
        outputs = None
        try:
            for position_value in (5, 6):
                position = torch.tensor(
                    position_value, dtype=torch.long
                ).to(device)
                outputs = step(token, position, *state)
                token = outputs[1]
                state = outputs[2:]
            torch.neuron.synchronize()
        except RuntimeError as exc:
            expected_cross_target_failure = (
                os.environ.get("CROSS_TARGET_COMPILE_ONLY", "0") == "1"
                and "Invalid NEFF" in str(exc)
            )
            if not expected_cross_target_failure:
                raise
            marker_dir = os.environ.get(
                "CROSS_TARGET_MARKER_DIR", "/tmp/cross-target-compile"
            )
            os.makedirs(marker_dir, exist_ok=True)
            open(os.path.join(marker_dir, f"rank{rank}.done"), "w").close()
            deadline = time.time() + 900
            while len(os.listdir(marker_dir)) < world_size:
                if time.time() >= deadline:
                    raise TimeoutError("timed out waiting for rank compile markers")
                time.sleep(1)
            print(f"rank={rank} compiled; skipped incompatible target load")
            return

    captured = {
        name: tensor.cpu() for name, tensor in zip(NAMES, outputs)
    }
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(captured, os.path.join(args.output_dir, f"rank{rank}.pt"))
    dist.barrier()
    if rank == 0:
        print(
            f"captured mode={args.mode} next_ids="
            f"{captured['next_id'][:8].tolist()}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", default="/models/Qwen3.5-35B-A3B"
    )
    parser.add_argument("--mode", choices=("segmented", "sharded"))
    parser.add_argument("--output-dir")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--compare", nargs=2, metavar=("REFERENCE", "CANDIDATE"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare, args.world_size)
        return
    if not args.mode or not args.output_dir:
        parser.error("--mode and --output-dir are required for capture")
    capture(args)


if __name__ == "__main__":
    main()
