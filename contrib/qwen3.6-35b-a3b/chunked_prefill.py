"""
Neuron-compatible chunked DeltaNet prefill.

Reimplements HF's torch_chunk_gated_delta_rule without 5D tensors.
All internal ops use at most 4D [B, H, C, D] by looping over chunks explicitly.
Mathematically identical to the HF version (produces same output given same inputs).

Usage:
    from chunked_prefill import neuron_chunk_gated_delta_rule
    # Drop-in replacement for torch_chunk_gated_delta_rule
"""
import torch
import torch.nn.functional as F


def neuron_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """
    Neuron-compatible reimplementation of torch_chunk_gated_delta_rule.

    Avoids 5D tensor broadcasts by processing chunks in explicit Python loops.
    All internal tensors are at most 4D [B, H, chunk_size, D].

    Args:
        query:  [B, S, H, K]
        key:    [B, S, H, K]
        value:  [B, S, H, V]
        g:      [B, S, H, 1] or [B, S, H]  (log decay per position per head)
        beta:   [B, S, H, 1] or [B, S, H]  (update gate)
        chunk_size: tokens per chunk (default 64, matching HF)
        initial_state: [B, H, K, V] or None
        output_final_state: whether to return final recurrent state
        use_qk_l2norm_in_kernel: apply L2 norm to q,k inside (default False)

    Returns:
        output: [B, S, H, V]
        final_state: [B, H, K, V] or None
    """
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1, eps=1e-6)
        key = F.normalize(key, p=2, dim=-1, eps=1e-6)

    # Transpose to [B, H, S, D] and cast to fp32
    query, key, value = [x.transpose(1, 2).contiguous().float() for x in (query, key, value)]
    beta = beta.transpose(1, 2).contiguous().float()
    g = g.transpose(1, 2).contiguous().float()

    # Squeeze trailing dim if present (g/beta may be [B,H,S,1] or [B,H,S])
    if g.dim() == 4 and g.shape[-1] == 1:
        g = g.squeeze(-1)
    if beta.dim() == 4 and beta.shape[-1] == 1:
        beta = beta.squeeze(-1)

    B, H, S, K = key.shape
    V = value.shape[-1]
    C = chunk_size

    # Pad sequence to multiple of chunk_size
    pad = (C - S % C) % C
    if pad > 0:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))

    total_S = S + pad
    num_chunks = total_S // C

    # Apply query scale (same as HF)
    scale = 1.0 / (K ** 0.5)
    query = query * scale

    # Precompute v_beta and k_beta (4D: [B, H, total_S, D])
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Phase 1 mask: zeros diagonal AND upper (strict lower triangle only)
    mask_phase1 = torch.triu(
        torch.ones(C, C, device=query.device, dtype=torch.bool), diagonal=0
    )
    # Phase 2 mask: zeros strict upper only (keeps diagonal)
    mask_phase2 = torch.triu(
        torch.ones(C, C, device=query.device, dtype=torch.bool), diagonal=1
    )
    eye_C = torch.eye(C, device=query.device, dtype=torch.float32)

    # Initialize state
    state = (
        initial_state.float().clone()
        if initial_state is not None
        else torch.zeros(B, H, K, V, device=query.device, dtype=torch.float32)
    )

    # Output buffer
    output = torch.zeros(B, H, total_S, V, device=query.device, dtype=torch.float32)

    for ci in range(num_chunks):
        cs = ci * C
        ce = cs + C

        # Extract chunk tensors — all 4D [B, H, C, D] or [B, H, C]
        q_c = query[:, :, cs:ce]        # [B, H, C, K]
        k_c = key[:, :, cs:ce]          # [B, H, C, K]
        v_c = value[:, :, cs:ce]        # [B, H, C, V]
        g_c = g[:, :, cs:ce]            # [B, H, C]
        k_beta_c = k_beta[:, :, cs:ce]  # [B, H, C, K]
        v_beta_c = v_beta[:, :, cs:ce]  # [B, H, C, V]

        # Cumulative decay within chunk: [B, H, C]
        g_cum = g_c.cumsum(dim=-1)

        # Decay mask [B, H, C, C] — the key 4D operation (no 5D needed!)
        # decay_mask[j,k] = exp(g_cum[j] - g_cum[k]) for j >= k, else 0
        dm = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp()  # [B, H, C, C]
        dm = dm.tril()

        # === Phase 1: Delta-rule value correction within chunk ===
        # A = -(k_beta @ key.T * decay_mask), strict lower triangle, then Woodbury correction
        A = -(k_beta_c @ k_c.transpose(-1, -2)) * dm  # [B, H, C, C]
        A = A.masked_fill(mask_phase1, 0)  # zero diagonal + upper (strict lower only)

        # Iterative Woodbury correction (resolvent)
        for i in range(1, C):
            row = A[..., i, :i].clone()       # [B, H, i]
            sub = A[..., :i, :i].clone()      # [B, H, i, i]
            # row + sum_j(row[j] * sub[j, :]) = row + row @ sub
            A[..., i, :i] = row + (row.unsqueeze(-2) @ sub).squeeze(-2)

        A = A + eye_C  # [B, H, C, C]

        # Corrected values and cumulative-decay-weighted keys
        v_corrected = A @ v_beta_c                              # [B, H, C, V]
        k_cumdecay = A @ (k_beta_c * g_cum.exp().unsqueeze(-1)) # [B, H, C, K]

        # === Phase 2: Intra-chunk attention + inter-chunk state interaction ===
        # Intra-chunk causal attention with decay (keeps diagonal)
        intra_attn = (q_c @ k_c.transpose(-1, -2)) * dm  # [B, H, C, C]
        intra_attn = intra_attn.masked_fill(mask_phase2, 0)

        # Inter-chunk: subtract state's contribution already captured in corrected values
        v_prime = k_cumdecay @ state          # [B, H, C, K] @ [B, H, K, V] = [B, H, C, V]
        v_new = v_corrected - v_prime         # [B, H, C, V]

        # Query interaction with state (scaled by cumulative decay)
        attn_inter = (q_c * g_cum.unsqueeze(-1).exp()) @ state  # [B, H, C, V]

        # Combine intra + inter
        output[:, :, cs:ce] = attn_inter + intra_attn @ v_new

        # === Phase 3: State update ===
        total_decay = g_cum[:, :, -1]  # [B, H] — total cumulative decay for this chunk
        state = state * total_decay[..., None, None].exp()

        # Keys weighted by positional decay relative to chunk end (using cumulative g)
        k_weighted = k_c * (total_decay[..., None] - g_cum).exp().unsqueeze(-1)  # [B, H, C, K]
        state = state + k_weighted.transpose(-1, -2) @ v_new  # [B, H, K, V]

    if not output_final_state:
        state = None

    # Remove padding and transpose back to [B, S, H, V]
    output = output[:, :, :S, :]
    output = output.transpose(1, 2).contiguous().to(initial_dtype)
    return output, state


# ─── Validation ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Validate against HF's torch_chunk_gated_delta_rule on CPU."""
    import os, sys
    torch.manual_seed(42)

    # Try to import HF reference
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    except ImportError:
        sys.path.insert(0, os.environ.get("TRANSFORMERS_SRC", ""))
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    B, S, H, K, V = 1, 37, 12, 128, 128  # odd seq_len to test padding
    chunk_size = 16

    query = torch.randn(B, S, H, K)
    key = torch.randn(B, S, H, K)
    value = torch.randn(B, S, H, V)
    g = torch.randn(B, S, H) * 0.1  # small decay values
    beta = torch.sigmoid(torch.randn(B, S, H))
    init_state = torch.randn(B, H, K, V) * 0.01

    # HF reference (uses 5D tensors, works on CPU)
    ref_out, ref_state = torch_chunk_gated_delta_rule(
        query.clone(), key.clone(), value.clone(),
        g=g.clone(), beta=beta.clone(),
        chunk_size=chunk_size,
        initial_state=init_state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    # Our implementation (no 5D tensors)
    our_out, our_state = neuron_chunk_gated_delta_rule(
        query.clone(), key.clone(), value.clone(),
        g=g.clone(), beta=beta.clone(),
        chunk_size=chunk_size,
        initial_state=init_state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    out_diff = (our_out.float() - ref_out.float()).abs().max().item()
    state_diff = (our_state.float() - ref_state.float()).abs().max().item()
    out_cos = F.cosine_similarity(our_out.float().reshape(-1), ref_out.float().reshape(-1), dim=0).item()

    print(f"Output max_diff:  {out_diff:.8f}")
    print(f"State  max_diff:  {state_diff:.8f}")
    print(f"Output cosine:    {out_cos:.8f}")
    print(f"Output norm ours: {our_out.float().norm():.4f} vs ref: {ref_out.float().norm():.4f}")

    passed = out_diff < 1e-4 and state_diff < 1e-4
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'} (chunk_size={chunk_size}, seq={S}, heads={H})")

    if not passed:
        # Debug: check per-position norms
        our_norms = our_out[0].float().norm(dim=-1)  # [S, H]
        ref_norms = ref_out[0].float().norm(dim=-1)
        ratio = our_norms / (ref_norms + 1e-10)
        print(f"Per-position norm ratio range: [{ratio.min():.4f}, {ratio.max():.4f}]")
        print(f"First 5 positions, head 0:")
        for t in range(min(5, S)):
            print(f"  t={t}: ours={our_norms[t,0]:.4f} ref={ref_norms[t,0]:.4f} ratio={ratio[t,0]:.4f}")

    # Also test with [B, S, H, 1] format for g/beta (as static_decode.py uses)
    print("\n--- Testing [B,S,H,1] input format ---")
    our_out2, our_state2 = neuron_chunk_gated_delta_rule(
        query.clone(), key.clone(), value.clone(),
        g=g.clone().unsqueeze(-1), beta=beta.clone().unsqueeze(-1),
        chunk_size=chunk_size,
        initial_state=init_state.clone(),
        output_final_state=True,
    )
    fmt_diff = (our_out2.float() - our_out.float()).abs().max().item()
    print(f"Format invariance: max_diff={fmt_diff:.8f} {'✓' if fmt_diff < 1e-7 else '✗'}")
