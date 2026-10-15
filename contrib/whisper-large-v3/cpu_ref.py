"""CPU fp32 reference transcripts + WER for Whisper large-v3 (HF `generate()`).

Writes $ASR_WORK/ref/whisper_float32.json: per-sample reference text, hypothesis, token ids, and timing.
This is the ground truth the Neuron port is compared against (token-exact + WER).
"""
import argparse
import io
import json
import os
import time

import jiwer
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("ASR_WORK", HERE)
MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3")
OUT = os.path.join(WORK, "ref")
norm = BasicTextNormalizer()


def load_samples(n):
    # decode=False + soundfile avoids the torchcodec dependency (needs a CUDA/ffmpeg build)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for row in ds.select(range(n)):
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        out.append(dict(id=row["id"], audio=audio, sr=sr, ref=row["text"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(os.cpu_count())
    os.makedirs(OUT, exist_ok=True)

    proc = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=dtype).eval()
    res = []
    for s in load_samples(args.n):
        feats = proc(s["audio"], sampling_rate=s["sr"], return_tensors="pt").input_features.to(dtype)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(feats, language="en", task="transcribe", max_new_tokens=200)
        dt = time.perf_counter() - t0
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        res.append(dict(id=s["id"], ref=s["ref"], hyp=text, ids=ids[0].tolist(), secs=dt))
        print(f"[whisper] {dt:6.2f}s  {text}", flush=True)
    wer = jiwer.wer([norm(r["ref"]) for r in res], [norm(r["hyp"]) for r in res])
    with open(f"{OUT}/whisper_{args.dtype}.json", "w") as f:
        json.dump(dict(model="whisper", dtype=args.dtype, wer=wer, samples=res), f, indent=1)
    print(f"== whisper {args.dtype}: WER {wer*100:.2f}% over {len(res)} samples", flush=True)


if __name__ == "__main__":
    main()
