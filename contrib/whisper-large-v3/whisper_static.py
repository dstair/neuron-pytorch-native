"""Whisper large-v3, PyTorch-native static-shape port for Neuron (torch.compile backend="neuron").

Why not HF generate(): its host bookkeeping runs eager index ops on the device (hits an LNC=1
bounds-check NKI kernel bug in this torch-neuronx build) and StaticCache mixes devices.

Graphs (all fixed shape):
  encoder(feats[B,128,3000])            -> enc[B,1500,1280]
  cross_kv(enc)                         -> per-layer ck [B,H,D,1500], cv [B,H,1500,D]  (once per utterance, PR #70 idea)
  step(tok[1], pos[1], ks, vs, cks, cvs, sup, sup_begin) -> next tok[1]; embedding gather,
       where-blend slot write, suppress biases and argmax all in-graph; per-layer ks [B,H,D,S] / vs [B,H,S,D] in place
Decoder linears use pre-transposed [in,out] weights (LinT / FusedQKV). See EXPERIMENTS.md for the why.
Host: enqueues steps without syncing, reads tokens back every 8 steps for the EOS check.
(In-graph gather/argmax are fine; only *eager* aten::index hits the LNC=1 bounds-check bug. Cache slot write
 stays a where-blend: index_copy_ compiled but cost +3 ms/step -- see EXPERIMENTS.md round 2.)
"""
import argparse
import json
import os
import time

import jiwer
import torch
import torch.nn.functional as F
from torch import nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from cpu_ref import load_samples

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("ASR_WORK", HERE)  # ref/ outputs land here
MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3")  # HF id or local snapshot dir
REF = os.path.join(WORK, "ref", "whisper_float32.json")  # written by cpu_ref.py
PROMPT = [50258, 50259, 50360, 50364]  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
EOS = 50257
# ablation switches for decoder-step experiments, e.g. ASR_ABLATE=nofuse,where,noargmax (see EXPERIMENTS.md)
ABLATE = set(filter(None, os.environ.get("ASR_ABLATE", "").split(",")))
norm = BasicTextNormalizer()


class Encoder(nn.Module):
    def __init__(self, hf):
        super().__init__()
        self.enc = hf.model.encoder

    def forward(self, feats):
        e = self.enc
        x = F.gelu(e.conv1(feats))
        x = F.gelu(e.conv2(x)).transpose(1, 2)
        x = x + e.embed_positions.weight
        for layer in e.layers:
            x = _enc_layer(layer, x)
        return e.layer_norm(x)


def _heads(t, h):
    b, s, d = t.shape
    return t.view(b, s, h, d // h).transpose(1, 2)


def _merge(t):
    b, h, s, d = t.shape
    return t.transpose(1, 2).reshape(b, s, h * d)


def _enc_layer(layer, x):
    a = layer.self_attn
    h = layer.self_attn_layer_norm(x)
    q, k, v = (_heads(p(h), a.num_heads) for p in (a.q_proj, a.k_proj, a.v_proj))
    o = F.scaled_dot_product_attention(q, k, v)
    x = x + a.out_proj(_merge(o))
    h = layer.final_layer_norm(x)
    return x + layer.fc2(F.gelu(layer.fc1(h)))


class LinT(nn.Module):
    """nn.Linear with the weight stored pre-transposed [in, out].

    Fix 3 (2026-10-15): with HF's [out, in] layout the compiler transposed every decoder weight on
    TensorE each step (243 GF of TRANSPOSE vs 1.9 GF of real matmul).
    """

    def __init__(self, lin):
        super().__init__()
        self.wT = nn.Parameter(lin.weight.detach().t().contiguous(), requires_grad=False)
        self.b = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        y = x @ self.wT
        return y if self.b is None else y + self.b


def _kT(t, h):
    """[B,S,H*D] -> [B,H,D,S]: K stored pre-transposed so q @ K needs no per-step K^T (fix 2)."""
    b, s, d = t.shape
    return t.view(b, s, h, d // h).permute(0, 2, 3, 1)


def _attend(q, kT, v, mask=None):
    s = (q @ kT) * (q.shape[-1] ** -0.5)
    if mask is not None:
        s = s + mask
    return torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype) @ v


def _attend_with_new(q, kT, v, mask, kT_new, v_new):
    """softmax over [cached slots (masked to < pos), the current token] without writing the cache first."""
    scale = q.shape[-1] ** -0.5
    s = torch.cat([(q @ kT) * scale + mask, (q @ kT_new) * scale], dim=-1)
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    return p[..., :-1] @ v + p[..., -1:] * v_new


class CrossKV(nn.Module):
    """Per-utterance cross-attention K/V: per-layer lists, K as [B,H,D,T], V as [B,H,T,D]."""

    def __init__(self, hf):
        super().__init__()
        layers = hf.model.decoder.layers
        self.H = layers[0].encoder_attn.num_heads
        self.k = nn.ModuleList(LinT(l.encoder_attn.k_proj) for l in layers)
        self.v = nn.ModuleList(LinT(l.encoder_attn.v_proj) for l in layers)

    def forward(self, enc):
        ks = [_kT(k(enc), self.H).contiguous() for k in self.k]
        vs = [_heads(v(enc), self.H).contiguous() for v in self.v]
        return ks, vs


class DecLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        a, c = layer.self_attn, layer.encoder_attn
        self.H = a.num_heads
        self.ln1, self.ln2, self.ln3 = layer.self_attn_layer_norm, layer.encoder_attn_layer_norm, layer.final_layer_norm
        self.qkv = SplitQKV(a.q_proj, a.k_proj, a.v_proj) if "nofuse" in ABLATE else FusedQKV(a.q_proj, a.k_proj, a.v_proj)
        self.o = LinT(a.out_proj)
        self.cq, self.co = LinT(c.q_proj), LinT(c.out_proj)
        self.fc1, self.fc2 = LinT(layer.fc1), LinT(layer.fc2)


class FusedQKV(nn.Module):
    """Round 2 C: self-attn q/k/v as one pre-transposed [in, 3*out] matmul (k_proj has no bias -> zeros)."""

    def __init__(self, q, k, v):
        super().__init__()
        self.n = q.out_features
        self.wT = nn.Parameter(torch.cat([p.weight.detach() for p in (q, k, v)]).t().contiguous(), requires_grad=False)
        bk = torch.zeros_like(q.bias) if k.bias is None else k.bias
        self.b = nn.Parameter(torch.cat([q.bias, bk, v.bias]).detach().clone(), requires_grad=False)

    def forward(self, x):
        return (x @ self.wT + self.b).split(self.n, dim=-1)


class SplitQKV(nn.Module):
    """Ablation (ASR_ABLATE=nofuse): separate q/k/v LinT, as in round 1."""

    def __init__(self, q, k, v):
        super().__init__()
        self.q, self.k, self.v = LinT(q), LinT(k), LinT(v)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)


class DecoderStep(nn.Module):
    """One greedy decode step, fully device-resident (round 2 A).

    In: tok [1] int64 (previous step's output, or a forced prompt token), pos [1] int64, per-layer caches,
    suppress / begin-suppress additive biases [V]. Out: next token [1] int64 (suppress rules + argmax in-graph).
    """

    def __init__(self, hf, first_gen_pos):
        super().__init__()
        dec = hf.model.decoder
        self.layers = nn.ModuleList(DecLayer(l) for l in dec.layers)
        self.ln = dec.layer_norm
        self.emb = nn.Parameter(dec.embed_tokens.weight.detach().clone(), requires_grad=False)
        self.pemb = nn.Parameter(dec.embed_positions.weight.detach().clone(), requires_grad=False)
        self.lm = nn.Parameter(dec.embed_tokens.weight.detach().t().contiguous(), requires_grad=False)
        self.max_len = hf.config.max_target_positions
        self.first_gen_pos = first_gen_pos

    def forward(self, tok, pos, ks, vs, cks, cvs, sup, sup_begin):
        # Fix 1: one cache tensor per layer (ks[i] [B,H,D,S], vs[i] [B,H,S,D]).
        x = (F.embedding(tok, self.emb) + F.embedding(pos, self.pemb)).unsqueeze(1)  # [B,1,d]
        # `mask` (slots < pos) is only used by the rejected index_copy_ path (ASR_ABLATE=index)
        slots = torch.arange(self.max_len, device=x.device)
        mask = torch.where(slots < pos, 0.0, float("-inf")).to(x.dtype).view(1, 1, 1, -1)
        for i, L in enumerate(self.layers):
            h = L.ln1(x)
            q, k, v = L.qkv(h)
            q, kT_new, v_new = _heads(q, L.H), _kT(k, L.H), _heads(v, L.H)
            if "index" in ABLATE:  # round 2 B/2b (rejected): slot write via scatter, +3 ms/step on device
                x = x + L.o(_merge(_attend_with_new(q, ks[i], vs[i], mask, kT_new, v_new)))
                ks[i].index_copy_(3, pos, kT_new)
                vs[i].index_copy_(2, pos, v_new)
            else:  # where-blend of the per-layer cache, then attend (faster than the scatter despite more bytes)
                hit = slots == pos
                ks[i].copy_(torch.where(hit.view(1, 1, 1, -1), kT_new, ks[i]))
                vs[i].copy_(torch.where(hit.view(1, 1, -1, 1), v_new, vs[i]))
                incl = torch.where(slots <= pos, 0.0, float("-inf")).to(x.dtype).view(1, 1, 1, -1)
                x = x + L.o(_merge(_attend(q, ks[i], vs[i], incl)))

            h = L.ln2(x)
            x = x + L.co(_merge(_attend(_heads(L.cq(h), L.H), cks[i], cvs[i])))

            h = L.ln3(x)
            x = x + L.fc2(F.gelu(L.fc1(h)))
        logits = (self.ln(x)[:, -1] @ self.lm).float() + sup
        if "noargmax" in ABLATE:  # timing ablation only: return logits, skip the in-graph argmax
            return logits
        logits = logits + torch.where(pos == self.first_gen_pos, sup_begin, torch.zeros_like(sup_begin))
        return torch.argmax(logits, dim=-1)


class StaticWhisper:
    def __init__(self, path, dtype, device, compile_):
        self.hf = WhisperForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation="eager").eval()
        gc = self.hf.generation_config
        cfg = self.hf.config
        self.L, self.H = cfg.decoder_layers, cfg.decoder_attention_heads
        self.D, self.S = cfg.d_model // self.H, cfg.max_target_positions
        self.dtype, self.device = dtype, torch.device(device)
        # build the pre-transposed decoder modules on host, then move (no eager transposes on device)
        self.encoder, self.cross_kv = Encoder(self.hf), CrossKV(self.hf)
        self.step = DecoderStep(self.hf, first_gen_pos=len(PROMPT) - 1)
        for m in (self.encoder, self.cross_kv, self.step):
            m.to(self.device)
        V = cfg.vocab_size
        sup, sup_begin = torch.zeros(V), torch.zeros(V)
        sup[gc.suppress_tokens] = float("-inf")
        sup_begin[gc.begin_suppress_tokens] = float("-inf")
        self.sup, self.sup_begin = sup.to(self.device), sup_begin.to(self.device)
        # device-resident position / prompt-token scalars, created once (no per-step H2D)
        self.pos_t = [torch.tensor([p], device=self.device) for p in range(self.S)]
        self.prompt_t = [torch.tensor([t], device=self.device) for t in PROMPT]
        self._prompt_cache = {1: self.prompt_t}
        if compile_:
            self.encoder = torch.compile(self.encoder, backend="neuron", fullgraph=True, dynamic=False)
            self.cross_kv = torch.compile(self.cross_kv, backend="neuron", fullgraph=True, dynamic=False)
            self.step = torch.compile(self.step, backend="neuron", fullgraph=True, dynamic=False)

    def new_cache(self, B=1):
        """Per-layer self-attn caches: K [B,H,D,S] (pre-transposed), V [B,H,S,D]."""
        kw = dict(dtype=self.dtype, device=self.device)
        ks = [torch.zeros(B, self.H, self.D, self.S, **kw) for _ in range(self.L)]
        vs = [torch.zeros(B, self.H, self.S, self.D, **kw) for _ in range(self.L)]
        return ks, vs

    @torch.no_grad()
    def transcribe(self, feats, max_new_tokens=128, flush_every=8):
        """Batch-1 convenience wrapper: returns (token ids, timings)."""
        outs, t = self.transcribe_batch(feats, max_new_tokens, flush_every)
        return outs[0], t

    def _prompt(self, B):
        if B not in self._prompt_cache:
            self._prompt_cache[B] = [torch.full((B,), t, device=self.device) for t in PROMPT]
        return self._prompt_cache[B]

    @torch.no_grad()
    def transcribe_batch(self, feats, max_new_tokens=128, flush_every=8):
        """Greedy decode of B utterances in lockstep (same prompt -> same position for all), token kept on device.
        The host reads tokens back every `flush_every` steps for the EOS check (round 2 A); the batch stops once
        every sequence has emitted EOS, so short utterances ride along until the longest finishes."""
        dev, B = self.device, feats.shape[0]
        t = {}
        t0 = time.perf_counter()
        enc = self.encoder(feats.to(dev, self.dtype))
        ck, cv = self.cross_kv(enc)
        cv[-1].cpu()  # sync for the stage timer
        t["encoder"] = time.perf_counter() - t0
        k_cache, v_cache = self.new_cache(B)
        prompt = self._prompt(B)

        n_prompt = len(PROMPT)
        last = min(n_prompt + max_new_tokens - 1, self.S)
        outs, done, pending = [[] for _ in range(B)], [False] * B, []
        t0 = time.perf_counter()
        nxt = None
        for pos in range(last):
            tok = prompt[pos] if pos < n_prompt else nxt
            nxt = self.step(tok, self.pos_t[pos], k_cache, v_cache, ck, cv, self.sup, self.sup_begin)
            if pos >= n_prompt - 1:
                pending.append(nxt)
            if len(pending) == flush_every or pos == last - 1:
                block = torch.stack(pending, dim=1).tolist()  # [B, flushed steps]
                for b in range(B):
                    for x in block[b]:
                        if done[b]:
                            break
                        outs[b].append(x)
                        done[b] = x == EOS
                pending = []
                if all(done):
                    break
        t["decode"] = time.perf_counter() - t0
        t["steps"] = pos + 1
        return outs, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    proc = WhisperProcessor.from_pretrained(MODEL)
    model = StaticWhisper(MODEL, dtype, args.device, not args.no_compile)
    ref = {s["id"]: s for s in json.load(open(REF))["samples"]}
    res = []
    for i, s in enumerate(load_samples(args.n)):
        feats = proc(s["audio"], sampling_rate=s["sr"], return_tensors="pt").input_features
        t0 = time.perf_counter()
        ids, t = model.transcribe(feats, args.max_new_tokens)
        dt = time.perf_counter() - t0
        text = proc.decode(ids, skip_special_tokens=True).strip()
        r = ref[s["id"]]
        match = ids == [x for x in r["ids"] if x != EOS] + [EOS] or ids == r["ids"]
        res.append(dict(id=s["id"], ref=s["ref"], hyp=text, ids=ids, secs=dt, token_match=match, **t))
        print(f"[{i}] {dt:7.2f}s enc {t['encoder']*1e3:7.1f}ms dec {t['decode']*1e3:8.1f}ms/{t['steps']} steps "
              f"({t['decode']/t['steps']*1e3:.1f} ms/step) match={match}  {text}", flush=True)

    wer = jiwer.wer([norm(r["ref"]) for r in res], [norm(r["hyp"]) for r in res])
    warm = res[1:] or res
    print(f"== {args.device} {args.dtype} compile={not args.no_compile}: WER {wer*100:.2f}%  "
          f"token-exact {sum(r['token_match'] for r in res)}/{len(res)}  "
          f"warm enc {sum(r['encoder'] for r in warm)/len(warm)*1e3:.1f}ms  "
          f"warm ms/step {sum(r['decode'] for r in warm)/sum(r['steps'] for r in warm)*1e3:.2f}", flush=True)
    tag = f"{args.device}_{args.dtype}{'_eager' if args.no_compile else ''}"
    json.dump(dict(wer=wer, samples=res), open(os.path.join(WORK, "ref", f"whisper_static_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
