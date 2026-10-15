"""Whisper decode step timing: how much does the host sync cost per token?

A: sync every step (token read back each step, like the original host-argmax loop)
B: flush every 8 steps (what transcribe() does)
C: fully async, one sync at the end (device-throughput bound)
"""
import time

import torch

from whisper_static import MODEL, StaticWhisper

N = 64
torch.set_grad_enabled(False)
m = StaticWhisper(MODEL, torch.bfloat16, "neuron", True)
dev = m.device
enc = m.encoder(torch.zeros(1, 128, 3000, dtype=torch.bfloat16, device=dev))
ck, cv = m.cross_kv(enc)
k, v = m.new_cache(1)


def run(flush):
    tok, pend = m.prompt_t[0], []
    t0 = time.perf_counter()
    for p in range(N):
        y = m.step(tok, m.pos_t[p], k, v, ck, cv, m.sup, m.sup_begin)
        is_tok = y.dtype == torch.int64  # ASR_ABLATE=noargmax returns logits; keep feeding a fixed token
        tok = y if is_tok else m.prompt_t[0]
        pend.append(y if is_tok else y[:, :1])
        if len(pend) == flush:
            torch.cat(pend).tolist()
            pend = []
    if pend:
        torch.cat(pend).tolist()
    return (time.perf_counter() - t0) / N


run(N)  # warm
a, b, c = run(1), run(8), run(N)
print(f"A sync every step {a*1e3:.2f} ms/step | B flush every 8 {b*1e3:.2f} ms/step | C fully async {c*1e3:.2f} ms/step")
