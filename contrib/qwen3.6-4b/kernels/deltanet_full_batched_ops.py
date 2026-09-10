"""PyTorch custom-op registration for the TP=4 Qwen3.5-4B DeltaNet kernel."""

import torch
from torch_neuronx import nki_op

from deltanet_full_batched import nki_deltanet_full_batched


@nki_op("deltanet4b::full_batched", mutates_args={})
def deltanet4b_full_batched(
    state: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    a_out: torch.Tensor,
    b_out: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return nki_deltanet_full_batched(
        state,
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        a_out,
        b_out,
        z,
        A_log,
        dt_bias,
        norm_weight,
    )


@nki_op(
    "deltanet4b::full_batched_direct",
    mutates_args={"state_out", "conv_state_out"},
)
def deltanet4b_full_batched_direct(
    state: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    a_out: torch.Tensor,
    b_out: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    state_out: torch.Tensor,
    conv_state_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return nki_deltanet_full_batched(
        state,
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        a_out,
        b_out,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state_out,
        conv_state_out,
    )
