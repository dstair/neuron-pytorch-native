"""CPU fp32 reference for Qwen3-ASR-0.6B-hf: greedy generate(), language forced to English.
Saves ref/ref.json (ids, prompt ids, generated ids, text, WER, cpu time)."""
import json, time, os, io, re, sys
import numpy as np, torch, soundfile as sf
from datasets import load_dataset, Audio
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

RUN = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10


def load_samples(n):
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for i in range(n):
        a = ds[i]["audio"]
        wav, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")
        assert sr == 16000
        out.append(dict(id=ds[i]["id"], audio=wav, text=ds[i]["text"]))
    return out


def norm_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return " ".join(s.split())


def wer(refs, hyps):
    import jiwer
    return jiwer.wer([norm_text(r) for r in refs], [norm_text(h) for h in hyps])


if __name__ == "__main__":
    torch.manual_seed(0)
    MODEL = os.environ.get("QASR_MODEL", f"{RUN}/model")
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    samples = load_samples(N)
    recs = []
    for s in samples:
        inp = proc.apply_transcription_request(audio=s["audio"], language="English")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=False)
        dt = time.perf_counter() - t0
        gen = out[0, inp["input_ids"].shape[1]:].tolist()
        text = proc.decode([gen], return_format="transcription_only")[0]
        rec = dict(id=s["id"], ref_text=s["text"], prompt_ids=inp["input_ids"][0].tolist(),
                   feat_shape=list(inp["input_features"].shape),
                   feat_valid=int(inp["input_features_mask"].sum()),
                   gen_ids=gen, text=text, cpu_s=dt)
        print(s["id"], rec["feat_shape"], rec["feat_valid"], len(rec["prompt_ids"]), len(gen), f"{dt:.2f}s", text, flush=True)
        recs.append(rec)
    w = wer([r["ref_text"] for r in recs], [r["text"] for r in recs])
    os.makedirs(f"{RUN}/ref", exist_ok=True)
    json.dump(dict(samples=recs, wer=w, keys=list(inp.keys())), open(f"{RUN}/ref/ref.json", "w"), indent=1)
    print(f"CPU fp32 WER={w:.4f} median_cpu_s={np.median([r['cpu_s'] for r in recs]):.2f}")
