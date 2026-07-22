"""Populate device-generation-specific Torch NeuronX startup artifacts."""

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401
import static_decode_35b  # noqa: F401


dist.init_process_group(backend="neuron")
device = torch.neuron.current_device()
dtype = torch.bfloat16

# Match the eager state initialization performed before the prefill graph runs.
dn_states = torch.zeros(30, 2, 512, 128, dtype=dtype, device=device)
conv_states = torch.zeros(30, 2, 1024, 3, dtype=dtype, device=device)
kv_k = torch.zeros(10, 2, 1, 20480, 256, dtype=dtype, device=device)
kv_v = torch.zeros_like(kv_k)
one = torch.tensor(1, dtype=torch.long, device=device)
pid = torch.zeros(2, 20000, dtype=torch.long, device=device)

torch.neuron.synchronize()
dist.barrier()
del dn_states, conv_states, kv_k, kv_v, one, pid
dist.destroy_process_group()
