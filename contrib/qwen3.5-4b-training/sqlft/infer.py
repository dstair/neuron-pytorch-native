"""Recurrent-cache decode: prefill once, then O(1)-per-token steps.

Why this exists: `eval_sql.greedy_generate` re-forwards the whole prefix for every
generated token (O(n^2) in sequence length) because the TRAINABLE GDN kernel is a chunked
form with no single-token step. The kernel repo's INFERENCE kernels are the recurrent form
of the same math:

  - prefill: nkilib `gdn_cte` (Schur inverse, state OUT) via the v15
    `NeuronGatedDeltaNet.forward`, bound onto the stock HF modules -- the weight layout is
    identical, so nothing else changes.
  - decode: `deltanet_tkg_batched_bh` -- one token, state in/out, all B*H items in one
    launch.
  - the 8 full-attention layers get a fixed-size KV cache: preallocated (1, H_kv, max_len,
    head_dim) buffers returned in full every step, with the additive attention mask
    supplying correctness. Static shapes are the whole point -- every decode step is the
    SAME graph, so it compiles once and is then a few milliseconds.

Correctness contract: `labs/lab09_cached_decode_parity.py` compares per-step logits
against the full re-forward path under teacher forcing (top-1 agreement + cos > 0.999),
the same standard lab08 set for the bucketed rewrite. Decode is greedy over bf16 logits,
so decoded STRINGS can still legitimately diverge from the old path on near-ties.

    ./env/container.sh exec python3 labs/lab09_cached_decode_parity.py
"""

from __future__ import annotations

import importlib
import pathlib
import types

import torch

from sqlft.device import get_device, sync
from sqlft.gdn import KERNEL_ROOTS, load_kernel_package

RUNS = pathlib.Path("/work/runs")


def load_inference_layer_module(root: pathlib.Path | None = None):
    """The kernel repo's `layers` module (v15 inference path), as a package submodule."""
    pkg = load_kernel_package(root or KERNEL_ROOTS["nki-v15"])
    return importlib.import_module(f"{pkg.__name__}.layers")


# ---------------------------------------------------------------------- cache


class _GDNLayerState:
    """What the v15 forward reads at `cache_params.layers[layer_idx]`."""

    __slots__ = ("conv_states", "recurrent_states")

    def __init__(self):
        self.conv_states = None
        self.recurrent_states = None


class DecodeCache:
    """One cache object serving both layer families.

    GDN layers (v15 NeuronGatedDeltaNet.forward) use:
        .layers[i].conv_states / .recurrent_states
        .has_previous_state(i)            -- False during prefill, True during decode
        .update_conv_state(t, i)
        .update_recurrent_state(t, i)

    Full-attention layers (stock HF Qwen3_5Attention.forward) use:
        .update(k, v, layer_idx) -> (k_full, v_full)
    The returned tensors are always the FULL preallocated buffers; the additive attention
    mask decides which positions are visible. That keeps every step's shapes static.
    """

    def __init__(self, model, max_len: int, device, dtype: torch.dtype):
        text = model.model
        cfg = text.config
        n_kv = cfg.num_key_value_heads
        self.max_len = max_len
        self.cursor = 0  # absolute write position of the next decode token
        self._prefilled = False
        self._dev = device
        self.layers: dict[int, _GDNLayerState] = {}
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in text.layers:
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                buf = lambda: torch.zeros(  # noqa: E731
                    1, n_kv, max_len, attn.head_dim, dtype=dtype, device=device)
                self.kv[attn.layer_idx] = (buf(), buf())
            else:
                self.layers[layer.linear_attn.layer_idx] = _GDNLayerState()

    # --- GDN interface (v15 layer) ---
    def has_previous_state(self, layer_idx: int) -> bool:
        return self._prefilled

    def update_conv_state(self, conv_state, layer_idx: int) -> None:
        self.layers[layer_idx].conv_states = conv_state

    def update_recurrent_state(self, state, layer_idx: int) -> None:
        self.layers[layer_idx].recurrent_states = state

    # --- full-attention interface (HF layer) ---
    def update(self, key_states, value_states, layer_idx: int):
        kbuf, vbuf = self.kv[layer_idx]
        if key_states.shape[2] == 1:
            # Decode: one position, written at `cursor`. The cursor is a device tensor so
            # it is graph DATA, not a shape -- same NEFF every step.
            cur = torch.tensor([self.cursor], device=self._dev)
            kbuf.index_copy_(2, cur, key_states)
            vbuf.index_copy_(2, cur, value_states)
        else:
            kbuf[:, :, : key_states.shape[2]] = key_states
            vbuf[:, :, : value_states.shape[2]] = value_states
        return kbuf, vbuf


# ---------------------------------------------------------------------- install


def install_inference_gdn(model, root: pathlib.Path | None = None, verbose: bool = True) -> int:
    """Bind the v15 inference forward onto every GatedDeltaNet module. Returns count.

    Done per-instance (not on the class) so the stock HF forward -- and with it the
    training-kernel swap -- remains reachable for the reference path BEFORE this is called.
    Requires LoRA to be merged+unwrapped already: the fused in-projection reads `.weight`.
    """
    layers_mod = load_inference_layer_module(root)
    fwd = layers_mod.NeuronGatedDeltaNet.forward
    n = 0
    for layer in model.model.layers:
        gdn = getattr(layer, "linear_attn", None)
        if gdn is not None:
            gdn.forward = types.MethodType(fwd, gdn)
            n += 1
    if verbose:
        prefill = getattr(layers_mod, "PREFILL_CORE", "?")
        print(f"inference GDN installed on {n} layers (prefill core: {prefill}, "
              f"decode: deltanet_tkg_batched_bh)")
    return n


# ---------------------------------------------------------------------- forward pieces


def _prefill_mask(L: int, W: int, max_len: int, device, dtype) -> torch.Tensor:
    """Additive (1, 1, W, max_len) mask for a left-padded prompt: L real tokens at the
    right edge, W-L pads at the left.

    Real query i attends real keys j <= i. Pad queries attend pad keys causally (their
    outputs are zeros anyway -- every block maps an all-zero hidden row to zero -- but the
    rows must stay FINITE, so they need at least one visible key).
    """
    pad = W - L
    i = torch.arange(W)[:, None]
    j = torch.arange(max_len)[None, :]
    allowed = (j <= i) & (j < W) & ((j >= pad) | (i < pad))
    return torch.where(allowed, 0.0, float("-inf")).to(dtype)[None, None].to(device)


def _decode_mask(pad: int, cursor: int, max_len: int, device, dtype) -> torch.Tensor:
    """Additive (1, 1, 1, max_len): the new token sees every real key up to itself."""
    j = torch.arange(max_len)[None, :]
    allowed = (j >= pad) & (j <= cursor)
    return torch.where(allowed, 0.0, float("-inf")).to(dtype)[None, None].to(device)


def _forward_last(model, ids, attn_mask, pos_ids, cache, input_mask=None) -> torch.Tensor:
    """One forward through the stack; returns logits for the FINAL position only."""
    text = model.model
    h = text.embed_tokens(ids)
    if input_mask is not None:
        # What HF's apply_mask_to_padding_states does inside the stock GDN layer, done once
        # at the input instead: pad rows start at zero and every block maps zero -> zero.
        h = h * input_mask
    pos_emb = text.rotary_emb(h, pos_ids)
    for layer in text.layers:
        h = layer(
            h,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=cache,
        )
    h = text.norm(h[:, -1:])
    return model.lm_head(h)


# ---------------------------------------------------------------------- generation


@torch.no_grad()
def cached_generate(model, tokenizer, prompts: list[str], max_new_tokens: int = 48,
                    seq_len: int = 1024, chunk: int = 128, device: str = "neuron",
                    progress: bool = True,
                    stop_ids: "set[int] | None" = None) -> tuple[list[str], list[bool]]:
    """Greedy decode with a recurrent/KV cache. Returns (texts, finished_flags).

    finished[i] is True when the row emitted EOS inside the cap; False means the output
    was cut off and a parse failure on it is a truncation, not a syntax error.

    Layout matches the current eval exactly: content is LEFT-padded into the smallest
    128-multiple bucket that holds the prompt (the GDN prefill pads internally to its own
    chunk regardless), and position ids are mask-aware (first real token = position 0).
    bs=1: batching was already slower on the old path, and this kernel batches heads, not
    requests.
    """
    import time

    dev = get_device(str(device))
    dt = next(model.parameters()).dtype
    eos = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    # Chat-format generations end with <|im_end|> (== eos for Qwen3.5) but may also emit
    # <|endoftext|>; stop on any of them. Default: eos only (the plain format's behaviour).
    stops = {eos} if stop_ids is None else set(stop_ids) | {eos}

    texts: list[str] = []
    finished: list[bool] = []
    t_start = time.perf_counter()

    for n, prompt in enumerate(prompts):
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        L = len(ids)
        # A prompt too long to leave decode room in the seq_len-wide buffer is
        # truncated to its LAST tokens (the schema tail, the question, and the
        # '-- Answer:' marker are what matter; schema middle is what is lost).
        max_L = seq_len - chunk
        if L > max_L:
            ids = ids[-max_L:]
            L = max_L
        W = -(-L // chunk) * chunk
        # Decode tokens must fit the seq_len-wide KV buffer: cursor W+t < seq_len. A
        # prompt whose bucket fills the buffer gets zero decode steps -- the old
        # re-forward path also produced nothing for it (it breaks at content_len >=
        # seq_len), so both score it the same way.
        n_decode = max(0, min(max_new_tokens - 1, seq_len - W))
        pad = W - L
        cache = DecodeCache(model, seq_len, dev, dt)

        # ---- prefill ----
        row = [pad_id] * pad + ids
        ids_t = torch.tensor([row], device=dev)
        mask2d = torch.tensor([[0] * pad + [1] * L], device=dev)
        pos = (mask2d.long().cumsum(-1) - 1).clamp(min=0)
        attn_mask = _prefill_mask(L, W, seq_len, dev, dt)
        logits = _forward_last(model, ids_t, attn_mask, pos, cache,
                               input_mask=mask2d.unsqueeze(-1).to(dt))
        cache._prefilled = True
        cache.cursor = W

        gen: list[int] = [int(logits[:, -1, :].argmax(-1).cpu())]

        # ---- decode ----
        for t in range(n_decode):
            if gen[-1] in stops:
                break
            ids_t = torch.tensor([[gen[-1]]], device=dev)
            pos = torch.tensor([[L + t]], device=dev)
            attn_mask = _decode_mask(pad, cache.cursor, seq_len, dev, dt)
            logits = _forward_last(model, ids_t, attn_mask, pos, cache)
            gen.append(int(logits[:, -1, :].argmax(-1).cpu()))
            cache.cursor += 1
        sync(dev)

        stop_at = next((k for k, t in enumerate(gen) if t in stops), None)
        finished.append(stop_at is not None)
        if stop_at is not None:
            gen = gen[:stop_at]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True))
        if progress:
            el = time.perf_counter() - t_start
            print(f"  cached gen {n + 1}/{len(prompts)}  prompt={L} tok  new={len(gen)}  "
                  f"elapsed {el / 60:.1f} min", flush=True)
    return texts, finished


def evaluate_model_cached(model, tokenizer, rows, *, limit: int = 128,
                          max_new_tokens: int = 48, seq_len: int = 1024,
                          device: str = "neuron", tag: str | None = None) -> dict:
    """`eval_sql.evaluate_model` with cached decode. Identical grading."""
    from sqlft.data import render_prompt
    from sqlft.eval_sql import evaluate, gold_over_cap

    rows = list(rows)[:limit]
    prompts = [render_prompt(r) for r in rows]
    gens, finished = cached_generate(model, tokenizer, prompts,
                                     max_new_tokens=max_new_tokens, seq_len=seq_len,
                                     device=device)
    result = evaluate([(g, r.schema, f) for g, r, f in zip(gens, rows, finished)],
                      verbose=True)
    result["n_gold_over_cap"] = gold_over_cap(rows, tokenizer, max_new_tokens)
    if result["n_gold_over_cap"]:
        print(f"note: {result['n_gold_over_cap']} gold queries exceed the {max_new_tokens}-token "
              f"decode cap (unwinnable rows)")
    result["samples"] = [
        {"question": r.prompt, "gold": r.query, "generated": g}
        for r, g in list(zip(rows, gens))[:5]
    ]
    if tag:
        import json

        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{tag}.json").write_text(json.dumps(result, indent=2))
        print(f"wrote {RUNS / f'{tag}.json'}")
    return result
