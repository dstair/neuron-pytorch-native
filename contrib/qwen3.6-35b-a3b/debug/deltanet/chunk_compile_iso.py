#!/usr/bin/env python3
"""Diagnose the DeltaNet prefill custom op eager vs inside a compiled graph.

The inputs are prepared from real layer-0 weights and activations in eager mode,
then held fixed for both calls. This isolates the opaque NKI call from the
projection/conv/gate preparation and the post-kernel norm/output projection.
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
os.environ.setdefault("DN_CHUNK_NKI", "1")

import model_dims as D
import static_decode_35b as S


def prepare_kernel_inputs(mod, x, dn_states, cv_states):
    """Mirror the pre-kernel portion of layer-0 _deltanet_prefill."""
    i = 0
    di = D.deltanet_index(i)
    td = mod.td
    kh, vh = td["dn_k_heads"], td["dn_v_heads"]
    kd, vd = D.DN_K_DIM, D.DN_V_DIM
    key_dim = kh * kd
    val_dim = vh * vd
    qkv_dim = 2 * key_dim + val_dim
    seq = x.shape[1]
    x_2d = x.squeeze(0)

    mixed_qkv = mod._lin(f"l{i}_dn_qkv", x_2d)
    mqf = mixed_qkv.t().float()
    conv_input = torch.cat([cv_states[di, 0].float(), mqf], dim=-1)
    conv_w = getattr(mod, f"l{i}_dn_conv_w")
    conv_out = F.conv1d(
        conv_input.unsqueeze(0), conv_w.float(), groups=qkv_dim
    )
    mixed_qkv = F.silu(conv_out.squeeze(0).t())

    q = mixed_qkv[:, :key_dim].reshape(seq, kh, kd)
    k = mixed_qkv[:, key_dim : 2 * key_dim].reshape(seq, kh, kd)
    v = mixed_qkv[:, 2 * key_dim :].reshape(seq, vh, vd)
    group = vh // kh
    q = q.repeat_interleave(group, dim=1)
    k = k.repeat_interleave(group, dim=1)

    a_out = mod._lin(f"l{i}_dn_a", x_2d).float()
    beta = mod._lin(f"l{i}_dn_b", x_2d).sigmoid()
    a_log = getattr(mod, f"l{i}_dn_A_log").float()
    dt_bias = getattr(mod, f"l{i}_dn_dt_bias").float()
    g = -a_log.exp() * F.softplus(a_out + dt_bias)

    return (
        dn_states[di].float(),
        q.float().transpose(0, 1).reshape(vh * seq, kd).contiguous(),
        k.float().transpose(0, 1).reshape(vh * seq, kd).contiguous(),
        v.float().transpose(0, 1).reshape(vh * seq, vd).contiguous(),
        g.transpose(0, 1).reshape(vh * seq, 1).contiguous(),
        beta.float().transpose(0, 1).reshape(vh * seq, 1).contiguous(),
        mod.chunk_m_incl,
        mod.chunk_m_strict,
        mod.chunk_eye,
    )


def kernel_call(state, q, k, v, g, beta, m_incl, m_strict, eye):
    return torch.ops.deltanet35b.chunked_prefill(
        state, q, k, v, g, beta, m_incl, m_strict, eye
    )


def compare(name, eager, compiled):
    eager = eager.float().cpu()
    compiled = compiled.float().cpu()
    cosine = F.cosine_similarity(
        eager.reshape(-1), compiled.reshape(-1), dim=0
    ).item()
    max_diff = (eager - compiled).abs().max().item()
    if dist.get_rank() == 0:
        print(
            f"[dn-prepare] {name}: cos={cosine:.8f} "
            f"maxdiff={max_diff:.4e}",
            flush=True,
        )


def compare_projection_and_conv(mod, x, cv):
    """Localize preparation drift to linear projections or grouped conv."""
    x_2d = x.squeeze(0)

    def projections(x_arg):
        return (
            mod._lin("l0_dn_qkv", x_arg),
            mod._lin("l0_dn_a", x_arg).float(),
            mod._lin("l0_dn_b", x_arg),
        )

    eager_projections = projections(x_2d)
    compiled_projections = torch.compile(
        projections, backend="neuron", fullgraph=True, dynamic=False
    )(x_2d)
    for name, eager_value, compiled_value in zip(
        ("qkv_linear", "a_linear", "b_linear"),
        eager_projections,
        compiled_projections,
    ):
        compare(name, eager_value, compiled_value)

    qkv_dim = eager_projections[0].shape[-1]
    conv_w = getattr(mod, "l0_dn_conv_w")

    def conv_activation(mixed_qkv, cv_arg):
        mqf = mixed_qkv.t().float()
        conv_input = torch.cat([cv_arg[0, 0].float(), mqf], dim=-1)
        conv_out = F.conv1d(
            conv_input.unsqueeze(0), conv_w.float(), groups=qkv_dim
        )
        return conv_out.squeeze(0).t(), F.silu(conv_out.squeeze(0).t())

    eager_conv = conv_activation(eager_projections[0], cv)
    compiled_conv = torch.compile(
        conv_activation, backend="neuron", fullgraph=True, dynamic=False
    )(eager_projections[0], cv)
    compare("conv", eager_conv[0], compiled_conv[0])
    compare("conv_silu", eager_conv[1], compiled_conv[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", default="/models/Qwen3.5-35B-A3B"
    )
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
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
        weights, args.max_seq_len, world_size, batch_size=1, rank=rank
    ).to(device).eval()

    td = D.tp_dims(world_size)
    vh = td["dn_v_heads"]
    qkv_dim = 2 * td["dn_k_heads"] * D.DN_K_DIM + vh * D.DN_V_DIM
    dn = torch.zeros(
        1, 1, vh * D.DN_K_DIM, D.DN_V_DIM, device=device
    )
    cv = torch.zeros(
        1, 1, qkv_dim, D.DN_CONV_KERNEL - 1, device=device
    )
    ids = (torch.arange(args.seq, device=device) * 7 + 3) % D.VOCAB
    x = S.rms_norm(
        F.embedding(ids, mod.embed).unsqueeze(0).float(),
        getattr(mod, "l0_input_norm"),
    )
    inputs = prepare_kernel_inputs(mod, x, dn, cv)

    compare_projection_and_conv(mod, x, cv)

    def prepare(x_arg, dn_arg, cv_arg):
        return prepare_kernel_inputs(mod, x_arg, dn_arg, cv_arg)

    compiled_prepare = torch.compile(
        prepare, backend="neuron", fullgraph=True, dynamic=False
    )
    compiled_inputs = compiled_prepare(x, dn, cv)
    for name, eager_value, compiled_value in zip(
        ("state", "q", "k", "v", "g", "beta"),
        inputs[:6],
        compiled_inputs[:6],
    ):
        compare(name, eager_value, compiled_value)

    eager_out, eager_state = kernel_call(*inputs)
    compiled = torch.compile(
        kernel_call, backend="neuron", fullgraph=True, dynamic=False
    )
    compiled_out, compiled_state = compiled(*inputs)

    eager_out = eager_out.float().cpu()
    compiled_out = compiled_out.float().cpu()
    eager_state = eager_state.float().cpu()
    compiled_state = compiled_state.float().cpu()
    out_cos = F.cosine_similarity(
        eager_out.reshape(-1), compiled_out.reshape(-1), dim=0
    ).item()
    out_diff = (eager_out - compiled_out).abs().max().item()
    state_cos = F.cosine_similarity(
        eager_state.reshape(-1), compiled_state.reshape(-1), dim=0
    ).item()
    state_diff = (eager_state - compiled_state).abs().max().item()

    if rank == 0:
        print(
            f"[dn-chunk-compile] seq={args.seq}: "
            f"out_cos={out_cos:.8f} out_maxdiff={out_diff:.4e} "
            f"state_cos={state_cos:.8f} state_maxdiff={state_diff:.4e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
