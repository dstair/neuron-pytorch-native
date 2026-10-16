"""Probe: do in-graph gather (embedding), index_copy_ (cache slot write) and argmax compile + run on Neuron?

Eager aten::index hit an LNC=1 bounds-check NKI kernel (NCC_EVRF066); check the torch.compile path before relying on it.
"""
import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401

torch.set_grad_enabled(False)


def f(tok, pos, E, P, cache_k, cache_v, w, bias):
    x = F.embedding(tok, E) + F.embedding(pos, P)  # [1,D]
    k = (x @ w).view(1, 4, 16, 1)  # [B,H,D,1]
    v = (x @ w).view(1, 4, 1, 16)  # [B,H,1,D]
    cache_k.index_copy_(3, pos, k)
    cache_v.index_copy_(2, pos, v)
    logits = (x @ E.T).float() + bias
    return torch.argmax(logits, dim=-1)


V, D, S = 1000, 64, 32
g = torch.Generator().manual_seed(0)
E, P = torch.randn(V, D, generator=g), torch.randn(S, D, generator=g)
w = torch.randn(D, 64, generator=g)
bias = torch.zeros(V)
bias[torch.randperm(V, generator=g)[:50]] = float("-inf")
tok, pos = torch.tensor([7]), torch.tensor([5])

ck, cv = torch.zeros(1, 4, 16, S), torch.zeros(1, 4, S, 16)
ref = f(tok, pos, E, P, ck, cv, w, bias)

dev = torch.device("neuron")
cf = torch.compile(f, backend="neuron", fullgraph=True, dynamic=False)
nk, nv = torch.zeros(1, 4, 16, S, device=dev), torch.zeros(1, 4, S, 16, device=dev)
args = [t.to(dev) for t in (tok, pos, E, P)]
out = cf(*args, nk, nv, w.to(dev), bias.to(dev))
print("argmax  cpu", ref.tolist(), "neuron", out.cpu().tolist())
print("k slot  maxdiff", (nk.cpu() - ck).abs().max().item(), " nonzero slots", (nk.cpu().abs().sum((0, 1, 2)) > 0).nonzero().flatten().tolist())
print("v slot  maxdiff", (nv.cpu() - cv).abs().max().item())
# second call, different pos, reusing the device output token as input (device-resident loop)
pos2 = torch.tensor([9], device=dev)
out2 = cf(out, pos2, *args[2:], nk, nv, w.to(dev), bias.to(dev))
ref2 = f(ref, torch.tensor([9]), E, P, ck, cv, w, bias)
print("step2 argmax cpu", ref2.tolist(), "neuron", out2.cpu().tolist(),
      " slots", (nk.cpu().abs().sum((0, 1, 2)) > 0).nonzero().flatten().tolist())
print("PROBE OK")
