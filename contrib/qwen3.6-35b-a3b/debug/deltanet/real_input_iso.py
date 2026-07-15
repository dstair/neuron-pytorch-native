#!/usr/bin/env python3
"""Diagnose DeltaNet NKI and Torch recurrence on real layer-0 activations."""

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
os.environ.setdefault("DN_CHUNK_NKI", "1")

import model_dims as D
import static_decode_35b as S
from chunked_prefill import neuron_chunk_gated_delta_rule
from chunk_compile_iso import prepare_kernel_inputs


def stats(name, value):
    value = value.float().cpu()
    print(
        f"  {name}: min={value.min().item():.5e} "
        f"max={value.max().item():.5e} mean={value.mean().item():.5e}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seq", type=int, default=128)
    args = parser.parse_args()

    import torch_neuronx  # noqa: F401

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.neuron.current_device()

    D.load_from_config(os.path.join(args.model_path, "config.json"))
    D.NUM_LAYERS = 1
    D.NUM_GQA = 0
    D.NUM_DELTANET = 1
    weights = S.load_sharded_weights(
        args.model_path, rank, world_size, num_layers=1
    )
    mod = S.StaticDecode35B(
        weights, args.seq, world_size, batch_size=1, rank=rank
    ).to(device).eval()

    td = D.tp_dims(world_size)
    vh = td["dn_v_heads"]
    kh = td["dn_k_heads"]
    qkv_dim = 2 * kh * D.DN_K_DIM + vh * D.DN_V_DIM
    dn = torch.zeros(
        1, 1, vh * D.DN_K_DIM, D.DN_V_DIM,
        dtype=torch.bfloat16, device=device,
    )
    cv = torch.zeros(
        1, 1, qkv_dim, D.DN_CONV_KERNEL - 1,
        dtype=torch.bfloat16, device=device,
    )
    ids = (torch.arange(args.seq, device=device) * 7 + 3) % D.VOCAB
    x = S.rms_norm(
        F.embedding(ids, mod.embed).unsqueeze(0).float(),
        getattr(mod, "l0_input_norm"),
    )
    inputs = prepare_kernel_inputs(mod, x, dn, cv)
    state, q_hm, k_hm, v_hm, g_hm, beta_hm, m_incl, m_strict, eye = inputs

    nki_out, nki_state = torch.ops.deltanet35b.chunked_prefill(*inputs)

    seq = args.seq
    q = q_hm.reshape(vh, seq, D.DN_K_DIM).transpose(0, 1)
    k = k_hm.reshape(vh, seq, D.DN_K_DIM).transpose(0, 1)
    v = v_hm.reshape(vh, seq, D.DN_V_DIM).transpose(0, 1)
    g = g_hm.reshape(vh, seq).transpose(0, 1)
    beta = beta_hm.reshape(vh, seq).transpose(0, 1)
    torch_out, torch_state = neuron_chunk_gated_delta_rule(
        F.normalize(q, p=2, dim=-1, eps=1e-6).unsqueeze(0),
        F.normalize(k, p=2, dim=-1, eps=1e-6).unsqueeze(0),
        v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        chunk_size=64,
        initial_state=state.reshape(1, vh, D.DN_K_DIM, D.DN_V_DIM),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    torch_hm = torch_out.squeeze(0).transpose(0, 1).reshape_as(nki_out)
    torch_state = torch_state.reshape_as(nki_state)

    if rank == 0:
        out_cos = F.cosine_similarity(
            torch_hm.float().cpu().reshape(-1),
            nki_out.float().cpu().reshape(-1),
            dim=0,
        ).item()
        state_cos = F.cosine_similarity(
            torch_state.float().cpu().reshape(-1),
            nki_state.float().cpu().reshape(-1),
            dim=0,
        ).item()
        print(
            f"[dn-real] seq={seq}: out_cos={out_cos:.8f} "
            f"out_maxdiff={(torch_hm.float().cpu() - nki_out.float().cpu()).abs().max().item():.5e} "
            f"state_cos={state_cos:.8f} "
            f"state_maxdiff={(torch_state.float().cpu() - nki_state.float().cpu()).abs().max().item():.5e}",
            flush=True,
        )
        stats("g", g)
        stats("beta", beta)
        stats("q_norm", torch.linalg.vector_norm(q.float(), dim=-1))
        stats("k_norm", torch.linalg.vector_norm(k.float(), dim=-1))


if __name__ == "__main__":
    main()
