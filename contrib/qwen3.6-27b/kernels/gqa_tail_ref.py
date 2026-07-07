"""Host oracle for the GQA-tail mega-kernel (Phase 1 of the BS=8 throughput work).

Mirrors EXACTLY the attention-tail of static_decode._gqa_layer — from the
already-projected query/gate (and the already-updated KV cache) through to the
pre-o_proj gated attention output. The k-side norm+rope and the KV-cache write are
done in TORCH outside the kernel (lean-KV showed the write isn't the TPOT mover, and
keeping it out avoids a dynamic-offset DMA inside the kernel); the kernel receives
the updated cached_k/cached_v + a precomputed causal mask, so there is NO dynamic
`position` inside it.

What the kernel (and this ref) computes, per batch row b:
  q[h] = rope_partial( rms_norm(query[b,h], q_norm) )       for h in 0..5   (partial-64)
  scores[h,t] = (q[h] . cached_k[b,t]) / sqrt(HEAD_DIM)      t in 0..max_seq-1
  scores[h,t] += (1 - mask[t]) * -1e9                        (causal; mask precomputed)
  w[h,:] = softmax(scores[h,:])                              (fp32)
  o[h,:] = sum_t w[h,t] * cached_v[b,t,:]                    [256]
  attn_out[b, h*256:(h+1)*256] = o[h,:] * sigmoid(gate[b,h,:])
Output: attn_out [B, 1536] (= 6*256), pre-o_proj.

partial RoPE (ROPE_DIM=64 of 256): rotary applied to [...,:64], [64:] passes through.
"""
import torch
import torch.nn.functional as F

HEAD_DIM = 256
Q_HEADS = 6
ROPE_DIM = 64
RMS_EPS = 1e-6


def _rms_norm(x, w):  # x [...,256], w [256]  -> (1+w) * x*rsqrt(mean(x^2)+eps)
    xf = x.float()
    n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (1.0 + w.float()) * n


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope_partial(x, cos, sin):
    # x [...,256]; cos/sin [64]. rotary on first 64 dims, pass-through rest.
    rot, pas = x[..., :ROPE_DIM], x[..., ROPE_DIM:]
    rot_e = rot * cos + _rotate_half(rot) * sin
    return torch.cat((rot_e, pas), dim=-1)


def gqa_tail_ref(query, gate, q_norm, cos, sin, cached_k, cached_v, mask):
    """All fp32. Shapes:
       query [B,6,256], gate [B,6,256], q_norm [256], cos/sin [64],
       cached_k/cached_v [B,max_seq,256], mask [max_seq] (1.0 valid / 0.0 masked).
       Returns attn_out [B, 1536]."""
    B = query.shape[0]
    S = cached_k.shape[1]
    out = torch.zeros(B, Q_HEADS * HEAD_DIM, dtype=torch.float32)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    neg = (1.0 - mask.float()).reshape(1, S) * (-1e9)  # [1,S]
    for b in range(B):
        qn = _rms_norm(query[b], q_norm)                  # [6,256]
        qr = _apply_rope_partial(qn, cos, sin)            # [6,256]
        ck = cached_k[b].float()                          # [S,256]
        cv = cached_v[b].float()                          # [S,256]
        scores = (qr @ ck.t()) * scale                    # [6,S]
        scores = scores + neg                             # causal mask
        w = F.softmax(scores, dim=-1)                     # [6,S]
        o = w @ cv                                        # [6,256]
        g = torch.sigmoid(gate[b].float())                # [6,256]
        out[b] = (o * g).reshape(Q_HEADS * HEAD_DIM)
    return out


if __name__ == "__main__":
    # Cross-check the ref against the actual _gqa_layer torch math (inline copy).
    torch.manual_seed(0)
    B, S = 8, 128
    query = torch.randn(B, Q_HEADS, HEAD_DIM)
    gate = torch.randn(B, Q_HEADS, HEAD_DIM)
    q_norm = torch.randn(HEAD_DIM) * 0.1
    cos = torch.randn(ROPE_DIM); sin = torch.randn(ROPE_DIM)
    cached_k = torch.randn(B, S, HEAD_DIM)
    cached_v = torch.randn(B, S, HEAD_DIM)
    pos = 100
    mask = (torch.arange(S) <= pos).float()

    out = gqa_tail_ref(query, gate, q_norm, cos, sin, cached_k, cached_v, mask)

    # inline mirror of _gqa_layer tail (query already projected; cache already updated)
    qn = _rms_norm(query, q_norm)
    qn4 = qn.unsqueeze(2)  # [B,6,1,256]
    cos4 = cos.reshape(1, 1, 1, ROPE_DIM); sin4 = sin.reshape(1, 1, 1, ROPE_DIM)
    rot, pas = qn4[..., :ROPE_DIM], qn4[..., ROPE_DIM:]
    qr = torch.cat((rot * cos4 + _rotate_half(rot) * sin4, pas), dim=-1).squeeze(2)  # [B,6,256]
    scores = torch.matmul(qr, cached_k.transpose(1, 2)) / (HEAD_DIM ** 0.5)  # [B,6,S]
    scores = scores + (1.0 - mask).reshape(1, 1, -1) * (-1e9)
    aw = F.softmax(scores.float(), dim=-1)
    ao = torch.matmul(aw, cached_v)  # [B,6,256]
    ao = ao.reshape(B, Q_HEADS * HEAD_DIM) * torch.sigmoid(gate.reshape(B, Q_HEADS * HEAD_DIM))

    d = (out - ao).abs().max().item()
    print(f"ref vs _gqa_layer-mirror max_diff = {d:.3e}")
    print("PASS" if d < 1e-4 else "FAIL")
