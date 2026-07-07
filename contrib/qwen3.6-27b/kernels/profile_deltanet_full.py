"""Generate a clean NEFF for nki_deltanet_full so neuron-explorer can profile it.

Runs the kernel with realistic-magnitude inputs. CPU reference is intentionally
omitted — we only want this kernel's NEFF in the output dir (no XLA NEFFs from
side ops). Two invocations so --profile-nth-exec=2 captures the steady-state.
"""
import os

os.environ.setdefault('NEURON_RT_INSPECT_ENABLE', '1')
os.environ.setdefault('NEURON_RT_INSPECT_DEVICE_PROFILE', '1')
os.environ.setdefault('NEURON_RT_INSPECT_OUTPUT_DIR', './output')
os.environ.setdefault('NEURON_CC_FLAGS', '--target trn2 --lnc 1')
os.environ.setdefault('NEURON_RT_VISIBLE_CORES', '0')

import torch
from deltanet_full import nki_deltanet_full

K_DIM = V_DIM = 128
K_HEADS, V_HEADS = 4, 12
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM

torch.manual_seed(0)
device = torch.device("neuron")

inputs = dict(
    state=torch.randn(V_HEADS * K_DIM, V_DIM, dtype=torch.float32) * 0.1,
    mixed_qkv=torch.randn(QKV_DIM, dtype=torch.bfloat16),
    conv_state=torch.randn(QKV_DIM, 3, dtype=torch.bfloat16),
    conv_weight=torch.randn(QKV_DIM, 4, dtype=torch.float32) * 0.5,
    conv_bias=torch.randn(QKV_DIM, dtype=torch.float32) * 0.1,
    a_out=torch.randn(V_HEADS, dtype=torch.float32),
    b_out=torch.randn(V_HEADS, dtype=torch.float32),
    z=torch.randn(V_HEADS, V_DIM, dtype=torch.bfloat16),
    A_log=torch.randn(V_HEADS, dtype=torch.float32) * 0.5,
    dt_bias=torch.randn(V_HEADS, dtype=torch.float32),
    norm_weight=torch.randn(V_DIM, dtype=torch.float32) * 0.1 + 1.0,
)
dev_inputs = {k: v.to(device) for k, v in inputs.items()}

# Two invocations — the second one is what --profile-nth-exec=2 will capture.
for i in range(3):
    out = nki_deltanet_full(**dev_inputs)
    # Force completion so each run produces an independent execution.
    out[0].cpu()
print("Profiling runs done.")
