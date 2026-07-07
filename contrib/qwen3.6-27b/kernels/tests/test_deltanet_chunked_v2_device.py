"""ON-DEVICE smoke test for nki_deltanet_chunked_prefill_v2.

The simulator (nki.simulate, numpy) validates this kernel bit-exact, but serving
shows decode gibberish that A/B-isolates to THIS kernel building a corrupt
recurrent state on-device — a simulator-vs-device gap (cf. the gqa_tail
device-only bugs). This harness drives the kernel through the SAME lowering the
plugin uses (vllm_neuron.nki.nki_hop.wrap_nki -> XLA AwsNeuronNkiKernel custom
call) on torch.device('xla'), so the on-device numerics are what we compare —
not the simulator's.

It diffs the kernel's on-device (output, new_state) against ref_chunk_single_head
(the proven CPU mirror) per head, and reports per-head state error so we can see
whether corruption is uniform (a precision/algorithm issue) or localized.

Run INSIDE the 28ce3c3 serve image with the neuron device, e.g.:

  docker run --rm --privileged --device=/dev/neuron0 \
    -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 -e PJRT_DEVICE=NEURON \
    -v /home/ubuntu/qwen3_6:/work -w /work/kernels/tests \
    --entrypoint python3 <serve-image> \
    test_deltanet_chunked_v2_device.py --C 64 --S 256
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # kernels/
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deltanet_chunked_v2_ref import build_constants, ref_chunk_single_head

K_DIM = 128
V_DIM = 128
V_HEADS = 12


def _build_inputs(C, S, seed, realistic_gates=True):
    torch.manual_seed(seed)
    H = V_HEADS
    q = torch.randn(H * S, K_DIM)
    k = torch.randn(H * S, K_DIM)
    v = torch.randn(H * S, V_DIM)
    if realistic_gates:
        A_log = torch.randn(H, 1)
        dt = torch.randn(H, 1) * 0.1
        a = torch.randn(H * S, 1)
        g = ((-torch.exp(A_log)).repeat_interleave(S, dim=0)
             * F.softplus(a + dt.repeat_interleave(S, dim=0)))
    else:
        g = torch.randn(H * S, 1) * 0.1
    beta = torch.sigmoid(torch.randn(H * S, 1))
    state = torch.randn(H * K_DIM, V_DIM) * 0.01
    m_incl, m_strict, eye = build_constants(C)
    return q, k, v, g, beta, state, m_incl, m_strict, eye


def _reference(q, k, v, g, beta, state, C, m_incl, m_strict, eye, S):
    H = V_HEADS
    ref_out = torch.zeros(H * S, V_DIM)
    ref_state = torch.zeros(H * K_DIM, V_DIM)
    for h in range(H):
        qh = F.normalize(q[h * S:(h + 1) * S], p=2, dim=-1)
        kh = F.normalize(k[h * S:(h + 1) * S], p=2, dim=-1)
        oh, nsh = ref_chunk_single_head(
            state[h * K_DIM:(h + 1) * K_DIM], qh, kh, v[h * S:(h + 1) * S],
            g[h * S:(h + 1) * S], beta[h * S:(h + 1) * S], C, m_incl, m_strict, eye)
        ref_out[h * S:(h + 1) * S] = oh
        ref_state[h * K_DIM:(h + 1) * K_DIM] = nsh
    return ref_out, ref_state


def run(C=64, S=256, seed=0, lnc=2):
    # Importing vllm_neuron registers the "vllm_neuron" torch.compile backend
    # (vllm_neuron/__init__.py). The kernel ONLY lowers correctly through that
    # backend — an eager kernel[grid](...) + mark_step hits "Unknown custom-call
    # API version enum 0" because the raw XLA path never sets the custom-call
    # API version. torch.compile(backend="vllm_neuron") is the production path.
    # libtorch_neuronx_lite renames the privateuse1 backend to "neuron" (its
    # __init__ calls torch.utils.rename_privateuse1_backend("neuron")), so an XLA
    # tensor reports device.type == "neuron" — which the vllm_neuron compile
    # backend's _validate_inputs_on_device REQUIRES.
    import libtorch_neuronx_lite  # noqa: F401  (renames privateuse1 -> "neuron")
    import vllm_neuron  # noqa: F401  (registers the "vllm_neuron" compile backend)
    import torch_xla.core.xla_model as xm  # noqa: F401
    from deltanet_chunked_v2 import nki_deltanet_chunked_prefill_v2
    from vllm_neuron.nki.nki_hop import wrap_nki

    q, k, v, g, beta, state, m_incl, m_strict, eye = _build_inputs(C, S, seed)
    ref_out, ref_state = _reference(q, k, v, g, beta, state, C, m_incl, m_strict, eye, S)

    dev = torch.device("neuron:0")
    args_cpu = [state, q, k, v, g, beta, m_incl, m_strict, eye]
    args_dev = [t.to(torch.float32).to(dev) for t in args_cpu]

    kernel = wrap_nki(nki_deltanet_chunked_prefill_v2)

    def _call(state, q, k, v, g, beta, m_incl, m_strict, eye):
        # grid = LNC (matches the plugin: lnc=2 on trn2).
        return kernel[lnc](state, q, k, v, g, beta, m_incl, m_strict, eye)

    compiled = torch.compile(_call, backend="vllm_neuron", fullgraph=True)
    out_dev, ns_dev = compiled(*args_dev)
    xm.mark_step()
    out = out_dev.cpu().float()
    ns = ns_dev.cpu().float()

    od = (out - ref_out).abs().max().item()
    sd = (ns - ref_state).abs().max().item()
    ocos = F.cosine_similarity(out.reshape(-1), ref_out.reshape(-1), dim=0).item()
    print(f"[DEVICE] C={C} S={S} lnc={lnc}: out_max_diff={od:.3e} "
          f"state_max_diff={sd:.3e} out_cos={ocos:.6f}")

    # per-head state error — uniform vs localized corruption
    print("per-head state_max_diff:")
    for h in range(V_HEADS):
        hsd = (ns[h * K_DIM:(h + 1) * K_DIM] - ref_state[h * K_DIM:(h + 1) * K_DIM]).abs().max().item()
        hod = (out[h * S:(h + 1) * S] - ref_out[h * S:(h + 1) * S]).abs().max().item()
        print(f"  h={h:2d}  state={hsd:.3e}  out={hod:.3e}")

    # within head 0, per-chunk output error — localize to a chunk boundary
    print("head0 per-chunk out_max_diff:")
    for ci in range(S // C):
        cs, ce = ci * C, ci * C + C
        cd = (out[cs:ce] - ref_out[cs:ce]).abs().max().item()
        print(f"  chunk {ci} [{cs}:{ce}]  out={cd:.3e}")

    ok = od < 1e-2 and sd < 1e-2
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=int, default=64)
    p.add_argument("--S", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lnc", type=int, default=2)
    a = p.parse_args()
    run(C=a.C, S=a.S, seed=a.seed, lnc=a.lnc)
