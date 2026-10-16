"""Dump Whisper static-port NEFFs (encoder / cross_kv / decoder step) for neuron-explorer profiling.

Env per neuron-nki-profiling + profile-querying skills: debug info for source attribution,
runtime inspect for NEFF dump, single pinned core. Run via launch.sh; then capture with neuron-explorer.
"""
import os

os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_SYSTEM_PROFILE"] = "0"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.environ.get("ASR_WORK", os.path.dirname(os.path.abspath(__file__))), "profiles", "whisper", "neffs")

import time  # noqa: E402

import torch  # noqa: E402
from transformers import WhisperProcessor  # noqa: E402

from cpu_ref import load_samples  # noqa: E402
from whisper_static import MODEL, StaticWhisper  # noqa: E402

proc = WhisperProcessor.from_pretrained(MODEL)
model = StaticWhisper(MODEL, torch.bfloat16, "neuron", True)
B = int(os.environ.get("ASR_BS", "1"))  # batch size of the dumped graphs
ss = load_samples(B)
feats = torch.cat([proc(s["audio"], sampling_rate=s["sr"], return_tensors="pt").input_features for s in ss])
for i in range(2):
    t0 = time.perf_counter()
    outs, t = model.transcribe_batch(feats, max_new_tokens=32)
    print(f"run {i}: BS={B} {time.perf_counter()-t0:.2f}s enc {t['encoder']*1e3:.1f}ms "
          f"{t['decode']/t['steps']*1e3:.2f} ms/step  {proc.decode(outs[0], skip_special_tokens=True)}", flush=True)
