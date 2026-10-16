"""Static-shape port of Qwen3-ASR (HF Qwen3ASRForConditionalGeneration) for torch.compile(backend="neuron").

Three fixed-shape graphs reuse the HF modules/weights:
  Encoder(C chunks):  feats [1,128,C*100] + n_valid -> projected audio embeds [C*13, 1024]
     * HF packs valid post-CNN frames with nonzero/index_select and runs windowed attention via cu_seqlens.
       For batch 1 the valid frames are a flat prefix of the [C*13] layout (only the last valid chunk is partial),
       so we keep the padded layout and use a block-diagonal mask: window id = p // 104, key valid iff k < n_valid.
  Prefill(P tokens):  ids [1,P] (right-padded) + audio embeds + last index -> next token, K/V caches [28,8,S,128]
  Decode (T=1):       token, pos, K/V caches -> next token, K/V caches  (slot write via torch.where(arange==pos))
"""
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

AUDIO_TOK = 151676
PROMPT_AUDIO_OFFSET = 9          # audio placeholders start at index 9 in the transcription template
TOK_PER_CHUNK = 13               # post-CNN frames per 100-frame mel chunk
NEG = -1e9


def post_cnn_len(l):
    for _ in range(3):
        l = (l - 1) // 2 + 1 if l > 0 else 0
    return l


def n_audio_tokens(feat_valid, chunk_len=100):
    full, rem = divmod(int(feat_valid), chunk_len)
    return full * TOK_PER_CHUNK + post_cnn_len(rem)


def greedy_argmax(lg):
    """argmax over the last dim, first index on ties (== torch.argmax). torch.argmax on the [1,151936] logits from
    h @ lm_w_t returned INT_MAX on neuron (compiler issue); max + masked-min lowers to two plain reductions."""
    if os.environ.get("QASR_ARGMAX", "manual") == "torch":
        return torch.argmax(lg, dim=-1)
    lg = lg.float()
    V = lg.shape[-1]
    m = lg.max(dim=-1, keepdim=True).values
    idx = torch.arange(V, device=lg.device, dtype=torch.int32).expand_as(lg)
    return torch.where(lg == m, idx, torch.full_like(idx, V)).min(dim=-1).values.to(torch.long)


class Encoder(nn.Module):
    def __init__(self, hf):
        super().__init__()
        self.at = hf.model.audio_tower
        self.proj = hf.model.multi_modal_projector
        cfg = self.at.config
        self.win = TOK_PER_CHUNK * (cfg.n_window_infer // (cfg.n_window * 2))  # 104
        self.nh = cfg.encoder_attention_heads

    def forward(self, feats, n_valid):
        at = self.at
        _, nmel, F_ = feats.shape
        C = F_ // 100
        x = feats.view(1, nmel, C, 100).permute(0, 2, 1, 3).reshape(C, 1, nmel, 100)
        x = F.gelu(at.conv2d1(x))
        x = F.gelu(at.conv2d2(x))
        x = F.gelu(at.conv2d3(x))
        c, ch, fb, ts = x.shape
        x = at.conv_out(x.permute(0, 3, 1, 2).reshape(c, ts, ch * fb))
        x = x + at.positional_embedding.positional_embedding[:ts].to(x.dtype)
        h = x.reshape(C * ts, -1)                                      # [T, 896] padded layout, valid prefix
        T = h.shape[0]
        p = torch.arange(T, device=h.device)
        blk = p // self.win
        allowed = (blk[:, None] == blk[None, :]) & (p[None, :] < n_valid)
        mask = torch.where(allowed, 0.0, NEG).to(torch.float32)[None]  # [1,T,T]
        nh = self.nh
        for layer in at.layers:
            a = layer.self_attn
            r = h
            hn = layer.self_attn_layer_norm(h)
            q = a.q_proj(hn).view(T, nh, -1).transpose(0, 1)
            k = a.k_proj(hn).view(T, nh, -1).transpose(0, 1)
            v = a.v_proj(hn).view(T, nh, -1).transpose(0, 1)
            s = torch.matmul(q, k.transpose(1, 2)).float() * a.scaling + mask
            pr = torch.softmax(s, dim=-1).to(v.dtype)
            o = torch.matmul(pr, v).transpose(0, 1).reshape(T, -1)
            h = r + a.out_proj(o)
            r = h
            hn = layer.final_layer_norm(h)
            h = r + layer.fc2(layer.activation_fn(layer.fc1(hn)))
        h = at.ln_post(h)
        return self.proj(h)                                            # [T, 1024]


class TextLM(nn.Module):
    """Qwen3 decoder with static KV cache, shared by prefill and decode."""

    def __init__(self, hf, max_seq):
        super().__init__()
        self.lm = hf.model.language_model
        self.lm_head = hf.lm_head
        # The tied lm_head as nn.Linear lowers to a per-call DMA transpose of the [V,1024] weight (~650 ms/step on
        # trn2). Keep a pre-transposed contiguous [1024, V] copy instead (QASR_LMHEAD=linear restores the HF path).
        self.lm_mode = os.environ.get("QASR_LMHEAD", "t")
        if self.lm_mode == "t":
            # r1: split the [1024,V] weight into QASR_LMSPLIT contiguous column blocks, one matmul each (+cat).
            # A single [1024,151936] matmul made the compiler re-read + transpose + spill the weight (~2.7 ms in a
            # microbench vs ~0.9 ms split 8-way). QASR_LMSPLIT=1 restores the single matmul.
            self.lm_split = int(os.environ.get("QASR_LMSPLIT", "8"))
            wt = hf.lm_head.weight.detach().t()
            for j, piece in enumerate(wt.tensor_split(self.lm_split, dim=1)):
                self.register_buffer(f"lm_w_t{j}", piece.contiguous(), persistent=False)
        # r1: QASR_WT=t keeps pre-transposed [in,out] copies of every decoder-layer weight (x @ Wt instead of
        # nn.Linear's x @ W.T, which the profile showed costs a Tensor-engine transpose per weight tile per call);
        # QASR_WT=tf additionally fuses q|k|v and gate|up; QASR_WT=linear restores the HF modules.
        self.wt = os.environ.get("QASR_WT", "tf")
        if self.wt in ("t", "tf"):
            for i, layer in enumerate(self.lm.layers):
                a, mlp = layer.self_attn, layer.mlp
                T_ = lambda lin: lin.weight.detach().t().contiguous()
                if self.wt == "tf":
                    self.register_buffer(f"wqkv{i}", torch.cat([T_(a.q_proj), T_(a.k_proj), T_(a.v_proj)], 1), persistent=False)
                    self.register_buffer(f"wgu{i}", torch.cat([T_(mlp.gate_proj), T_(mlp.up_proj)], 1), persistent=False)
                else:
                    for n, lin in (("wq", a.q_proj), ("wk", a.k_proj), ("wv", a.v_proj), ("wg", mlp.gate_proj), ("wu", mlp.up_proj)):
                        self.register_buffer(f"{n}{i}", T_(lin), persistent=False)
                self.register_buffer(f"wo{i}", T_(a.o_proj), persistent=False)
                self.register_buffer(f"wd{i}", T_(mlp.down_proj), persistent=False)
        # r2: QASR_Q=e5m2 + QASR_QW=<comma list of lm,qkv,o,gu,d> -> weight-only fp8 (e5m2, per-out-channel scale)
        # for the chosen groups; dequant (cast to activation dtype) in graph, scale applied on the matmul output.
        self.qmode = os.environ.get("QASR_Q", "none")
        self.qset = set(os.environ.get("QASR_QW", "lm,qkv,o,gu,d").split(",")) if self.qmode != "none" else set()
        if self.qset:
            pref = {"lm": "lm_w_t", "qkv": "wqkv", "o": "wo", "gu": "wgu", "d": "wd"}
            for name, w in list(self._buffers.items()):
                g = next((k for k, pf in pref.items() if name.startswith(pf) and name[len(pf):].isdigit()), None)
                if g in self.qset:
                    sc = (w.float().abs().amax(0, keepdim=True) / 57344.0).clamp_min(1e-30)
                    self._buffers[name] = (w.float() / sc).to(torch.float8_e5m2)
                    self.register_buffer("s_" + name, sc.to(w.dtype), persistent=False)
        self.gqa = os.environ.get("QASR_GQA", "group")
        # r3: QASR_KVLIST=1 -> K/V caches are passed/returned as per-layer lists of [nkv,S,hd] tensors instead of one
        # stacked [L,nkv,S,hd] tensor (hypothesis: the torch.stack forces every layer's rewritten cache to be spilled
        # and reloaded; profile_r2 spill = 89 MB/token = exactly one full K+V cache copy).
        self.kvlist = os.environ.get("QASR_KVLIST", "0") == "1"
        # r1: QASR_KV=inplace -> Decode mutates kc/vc in place (index_copy_ of slot pos) and returns only the token;
        # QASR_KV=where restores the functional full-cache torch.where rewrite + returned caches.
        # r2: QASR_KV=late -> attend over the OLD cache + the new k/v column per layer (like inplace), then ONE
        # torch.where over the stacked [28,8,S,128] cache at the end (no per-layer full-cache rewrite + stack copy).
        self.kv_mode = os.environ.get("QASR_KV", "where") if self.gqa == "group" else "where"
        self.kv_inplace = self.kv_mode == "inplace"
        self.kv_newcol = self.kv_mode in ("inplace", "late")      # r1: "expand" restores the materialized repeat_kv
        cfg = self.lm.config
        self.S = max_seq
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.nl = cfg.num_hidden_layers

    def W(self, x, name):
        """x @ weight buffer `name` (dequantized if it was quantized in __init__)."""
        w = getattr(self, name)
        if w.dtype == torch.float8_e5m2:
            return (x @ w.to(x.dtype)) * getattr(self, "s_" + name)
        return x @ w

    def _layers(self, x, pos_ids, mask, kc=None, vc=None, pos=None):
        """x [1,T,D]; mask [1,1,T,Tk] additive fp32. Prefill (kc None): attends to own K/V, returns caches padded
        to S. Decode: writes slot `pos` of the caches, attends over all S slots."""
        T = x.shape[1]
        cos, sin = self.lm.rotary_emb(x, pos_ids)
        nrep = self.nh // self.nkv
        ks, vs = [], []
        for i, layer in enumerate(self.lm.layers):
            a = layer.self_attn
            r = x
            h = layer.input_layernorm(x)
            if self.wt == "tf":
                qkv = self.W(h, f"wqkv{i}")
                nq, nk = self.nh * self.hd, self.nkv * self.hd
                qp, kp, vp = qkv[..., :nq], qkv[..., nq:nq + nk], qkv[..., nq + nk:]
            elif self.wt == "t":
                qp, kp, vp = h @ getattr(self, f"wq{i}"), h @ getattr(self, f"wk{i}"), h @ getattr(self, f"wv{i}")
            else:
                qp, kp, vp = a.q_proj(h), a.k_proj(h), a.v_proj(h)
            q = a.q_norm(qp.view(1, T, self.nh, self.hd)).transpose(1, 2)
            k = a.k_norm(kp.view(1, T, self.nkv, self.hd)).transpose(1, 2)
            v = vp.view(1, T, self.nkv, self.hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if kc is not None and self.kv_newcol:
                # r1 in-place cache: attend over the OLD cache (mask excludes slot pos) plus the new k/v as one
                # extra column; the caller writes only slot `pos` via index_copy_ (graph-input mutation).
                ks.append(k[0])
                vs.append(v[0])
                qg = q.reshape(1, self.nkv, nrep, self.hd)
                s_old = torch.matmul(qg, kc[i][None].transpose(2, 3)).float() * a.scaling + mask
                s_new = torch.matmul(qg, k.transpose(2, 3)).float() * a.scaling
                pr = torch.softmax(torch.cat([s_old, s_new], dim=-1), dim=-1).to(v.dtype)
                o = torch.matmul(pr[..., :self.S], vc[i][None]) + torch.matmul(pr[..., self.S:], v)
                o = o.reshape(1, 1, -1)
                x, h = self._mlp_block(layer, a, r, o)
                continue
            if kc is None:
                kk, vv = k, v
                ks.append(F.pad(k[0], (0, 0, 0, self.S - T)))
                vs.append(F.pad(v[0], (0, 0, 0, self.S - T)))
            else:
                sel = (torch.arange(self.S, device=x.device) == pos).view(1, self.S, 1)
                kn = torch.where(sel, k[0], kc[i])
                vn = torch.where(sel, v[0], vc[i])
                ks.append(kn)
                vs.append(vn)
                kk, vv = kn[None], vn[None]
            Tk = kk.shape[2]
            if self.gqa == "expand":
                kk = kk[:, :, None].expand(1, self.nkv, nrep, Tk, self.hd).reshape(1, self.nh, Tk, self.hd)
                vv = vv[:, :, None].expand(1, self.nkv, nrep, Tk, self.hd).reshape(1, self.nh, Tk, self.hd)
                s = torch.matmul(q, kk.transpose(2, 3)).float() * a.scaling + mask
                pr = torch.softmax(s, dim=-1).to(vv.dtype)
                o = torch.matmul(pr, vv).transpose(1, 2).reshape(1, T, -1)
            else:
                # grouped GQA: fold the nrep query heads of each kv head into the row dim (head = kv*nrep + r,
                # same order as HF repeat_kv) so K/V are never duplicated.
                qg = q.reshape(1, self.nkv, nrep * T, self.hd)
                mg = mask if mask.shape[2] == 1 else mask.repeat(1, 1, nrep, 1)
                s = torch.matmul(qg, kk.transpose(2, 3)).float() * a.scaling + mg
                pr = torch.softmax(s, dim=-1).to(vv.dtype)
                o = torch.matmul(pr, vv).reshape(1, self.nh, T, self.hd).transpose(1, 2).reshape(1, T, -1)
            x, h = self._mlp_block(layer, a, r, o)
        if self.kvlist:
            return self.lm.norm(x), tuple(ks), tuple(vs)
        return self.lm.norm(x), torch.stack(ks), torch.stack(vs)

    def _mlp_block(self, layer, a, r, o):
        i = layer.self_attn.layer_idx
        if True:
            if self.wt == "linear":
                x = r + a.o_proj(o)
                x = x + layer.mlp(layer.post_attention_layernorm(x))
            else:
                x = r + self.W(o, f"wo{i}")
                hn = layer.post_attention_layernorm(x)
                if self.wt == "tf":
                    gu = self.W(hn, f"wgu{i}")
                    g, u = gu[..., :gu.shape[-1] // 2], gu[..., gu.shape[-1] // 2:]
                else:
                    g, u = hn @ getattr(self, f"wg{i}"), hn @ getattr(self, f"wu{i}")
                x = x + self.W(F.silu(g) * u, f"wd{i}")
        return x, None

    def logits(self, h):
        if self.lm_mode == "t":
            ws = [f"lm_w_t{j}" for j in range(self.lm_split)]
            if len(ws) == 1:
                return self.W(h, ws[0])
            return torch.cat([self.W(h, w) for w in ws], dim=-1)
        return self.lm_head(h)


class Prefill(TextLM):
    def forward(self, ids, audio, last_idx):
        P = ids.shape[1]
        emb = self.lm.embed_tokens(ids)                                # [1,P,D]
        A = audio.shape[0]
        need = P - PROMPT_AUDIO_OFFSET
        aud = audio[:need] if A >= need else F.pad(audio, (0, 0, 0, need - A))
        aud = F.pad(aud, (0, 0, PROMPT_AUDIO_OFFSET, 0))[None].to(emb.dtype)
        x = torch.where((ids == AUDIO_TOK)[..., None], aud, emb)
        pos_ids = torch.arange(P, device=ids.device)[None]
        causal = pos_ids[0][None, :] <= pos_ids[0][:, None]
        mask = torch.where(causal, 0.0, NEG).to(torch.float32)[None, None]
        h, kc, vc = self._layers(x, pos_ids, mask)
        hl = torch.index_select(h[0], 0, last_idx)                     # [1,D]
        logits = self.logits(hl)
        return greedy_argmax(logits), kc, vc


class Decode(TextLM):
    def forward(self, tok, pos, kc, vc):
        return self.step(tok, pos, kc, vc)

    def step(self, tok, pos, kc, vc):
        x = self.lm.embed_tokens(tok.view(1, 1))
        pos_ids = pos.view(1, 1)
        ar = torch.arange(self.S, device=tok.device)
        valid = (ar < pos) if self.kv_newcol else (ar <= pos)
        mask = torch.where(valid, 0.0, NEG).to(torch.float32).view(1, 1, 1, self.S)
        h, kn, vn = self._layers(x, pos_ids, mask, kc, vc, pos)
        logits = self.logits(h[0])
        if self.kv_inplace:
            kc.index_copy_(2, pos, kn)
            vc.index_copy_(2, pos, vn)
            return greedy_argmax(logits)
        if self.kv_mode == "late":
            sel = (ar == pos).view(1, 1, self.S, 1)
            kn = torch.where(sel, kn, kc)
            vn = torch.where(sel, vn, vc)
        return greedy_argmax(logits), kn, vn


class DecodeK(Decode):
    """r2: K greedy decode steps unrolled in ONE graph (token fed back in-graph via embedding gather), so the fixed
    per-call host/runtime overhead (~1 ms of the 4.9 ms step) is paid once per K tokens. Returns (toks [K], last tok
    [1], kc, vc); the caller truncates at EOS (up to K-1 wasted steps, writes beyond S are no-ops)."""

    def __init__(self, hf, max_seq, K):
        super().__init__(hf, max_seq)
        self.K = K

    def forward(self, tok, pos, kc, vc):
        toks = []
        for j in range(self.K):
            tok, kc, vc = self.step(tok, pos + j, kc, vc)
            toks.append(tok)
        return torch.cat(toks), tok, kc, vc


class DecodeKLate(Decode):
    """r3 (QASR_KV=lateK): K unrolled greedy steps, but the full [L,nkv,S,hd] caches are NOT rewritten on every inner
    step. Inner step j attends over the OLD cache (mask: slots < pos0) plus the j+1 new k/v columns kept in a small
    per-layer list (all causal-valid), with a two-part softmax. After the K steps the K new columns are written into
    the caches ONCE (index_select gather + where). Motivation: profile_r2 spill = 89 MB/token = one full K+V cache
    copy per inner step."""

    def __init__(self, hf, max_seq, K):
        super().__init__(hf, max_seq)
        self.K = K
        self.late_norm = os.environ.get("QASR_LATENORM", "1") == "1"

    def _step_late(self, tok, pos, mask_old, kc, vc, newk, newv):
        x = self.lm.embed_tokens(tok.view(1, 1))
        pos_ids = pos.view(1, 1)
        cos, sin = self.lm.rotary_emb(x, pos_ids)
        nrep = self.nh // self.nkv
        for i, layer in enumerate(self.lm.layers):
            a = layer.self_attn
            r = x
            h = layer.input_layernorm(x)
            qkv = self.W(h, f"wqkv{i}")
            nq, nk = self.nh * self.hd, self.nkv * self.hd
            qp, kp, vp = qkv[..., :nq], qkv[..., nq:nq + nk], qkv[..., nq + nk:]
            q = a.q_norm(qp.view(1, 1, self.nh, self.hd)).transpose(1, 2)
            k = a.k_norm(kp.view(1, 1, self.nkv, self.hd)).transpose(1, 2)
            v = vp.view(1, 1, self.nkv, self.hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            newk[i].append(k[0])                                        # [nkv,1,hd]
            newv[i].append(v[0])
            Kn = torch.cat(newk[i], dim=1)[None] if len(newk[i]) > 1 else k   # [1,nkv,j+1,hd]
            Vn = torch.cat(newv[i], dim=1)[None] if len(newv[i]) > 1 else v
            qg = q.reshape(1, self.nkv, nrep, self.hd)
            s_old = torch.matmul(qg, kc[i][None].transpose(2, 3)).float() * a.scaling + mask_old
            s_new = torch.matmul(qg, Kn.transpose(2, 3)).float() * a.scaling
            m = torch.maximum(s_old.amax(-1, keepdim=True), s_new.amax(-1, keepdim=True))
            e_old = torch.exp(s_old - m)
            e_new = torch.exp(s_new - m)
            den = e_old.sum(-1, keepdim=True) + e_new.sum(-1, keepdim=True)
            if self.late_norm:      # normalize before the bf16 cast (closest to the softmax path)
                o = torch.matmul((e_old / den).to(v.dtype), vc[i][None]) + torch.matmul((e_new / den).to(v.dtype), Vn)
            else:
                o = (torch.matmul(e_old.to(v.dtype), vc[i][None]).float()
                     + torch.matmul(e_new.to(v.dtype), Vn).float()) / den
                o = o.to(v.dtype)
            o = o.reshape(1, 1, -1)
            x, _ = self._mlp_block(layer, a, r, o)
        return greedy_argmax(self.logits(self.lm.norm(x)[0]))

    def forward(self, tok, pos, kc, vc):
        L = len(self.lm.layers)
        ar = torch.arange(self.S, device=tok.device)
        mask_old = torch.where(ar < pos, 0.0, NEG).to(torch.float32).view(1, 1, 1, self.S)
        newk, newv = [[] for _ in range(L)], [[] for _ in range(L)]
        toks = []
        for j in range(self.K):
            tok = self._step_late(tok, pos + j, mask_old, kc, vc, newk, newv)
            toks.append(tok)
        off = ar - pos
        inwin = ((off >= 0) & (off < self.K)).view(1, self.S, 1)
        idx = off.clamp(0, self.K - 1)
        ks, vs = [], []
        for i in range(L):
            nk = torch.index_select(torch.cat(newk[i], dim=1), 1, idx)  # [nkv,S,hd]
            nv = torch.index_select(torch.cat(newv[i], dim=1), 1, idx)
            ks.append(torch.where(inwin, nk, kc[i]))
            vs.append(torch.where(inwin, nv, vc[i]))
        return torch.cat(toks), tok, torch.stack(ks), torch.stack(vs)
