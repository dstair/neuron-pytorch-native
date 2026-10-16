"""Whisper large-v3 batch-size sweep point: throughput / latency / accuracy at one batch size.

usage: python whisper_batch.py --bs 8     (one BS per process so each gets its own compiled graphs)
Uses the librispeech_asr_dummy split (73 utterances); times >= 32 utterances after one warm-up batch.
"""
import argparse
import json
import os
import time

import jiwer
import torch
from transformers import WhisperProcessor

from cpu_ref import load_samples
from whisper_static import EOS, MODEL, REF, WORK, StaticWhisper, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--min-utts", type=int, default=32)
    args = ap.parse_args()
    B = args.bs
    nb = max(1, -(-args.min_utts // B))  # timed batches
    samples = load_samples(min(73, B * (nb + 1)))
    while len(samples) < B * (nb + 1):  # wrap around for large B
        samples += samples[: B * (nb + 1) - len(samples)]

    proc = WhisperProcessor.from_pretrained(MODEL)
    feats = [proc(s["audio"], sampling_rate=s["sr"], return_tensors="pt").input_features for s in samples]
    ref = {s["id"]: s["ids"] for s in json.load(open(REF))["samples"]}
    model = StaticWhisper(MODEL, torch.bfloat16, "neuron", True)

    t0 = time.perf_counter()
    model.transcribe_batch(torch.cat(feats[:B]))  # warm-up / compile
    compile_s = time.perf_counter() - t0

    res, enc_s, dec_s, steps = [], 0.0, 0.0, 0
    t_all = time.perf_counter()
    for i in range(1, nb + 1):
        idx = range(i * B, (i + 1) * B)
        outs, t = model.transcribe_batch(torch.cat([feats[j] for j in idx]))
        enc_s, dec_s, steps = enc_s + t["encoder"], dec_s + t["decode"], steps + t["steps"]
        for j, ids in zip(idx, outs):
            s = samples[j]
            r = ref.get(s["id"])
            match = None if r is None else (ids == r or ids == [x for x in r if x != EOS] + [EOS])
            res.append(dict(id=s["id"], ref=s["ref"], hyp=proc.decode(ids, skip_special_tokens=True).strip(),
                            ntok=len(ids), secs_audio=len(s["audio"]) / s["sr"], token_match=match))
    wall = time.perf_counter() - t_all

    wer = jiwer.wer([norm(r["ref"]) for r in res], [norm(r["hyp"]) for r in res])
    checked = [r["token_match"] for r in res if r["token_match"] is not None]
    ntok = sum(r["ntok"] for r in res)
    audio = sum(r["secs_audio"] for r in res)
    summ = dict(bs=B, utts=len(res), compile_s=round(compile_s, 1),
                enc_ms_per_batch=round(enc_s / nb * 1e3, 1), ms_per_step=round(dec_s / steps * 1e3, 2),
                steps_per_batch=round(steps / nb, 1), tok_per_s=round(ntok / dec_s, 1),
                utt_per_s=round(len(res) / wall, 2), rtfx=round(audio / wall, 1), wall_s=round(wall, 2),
                wer=round(wer * 100, 2), token_exact=f"{sum(checked)}/{len(checked)}")
    print("SWEEP " + json.dumps(summ), flush=True)
    json.dump(dict(summary=summ, samples=res), open(os.path.join(WORK, "ref", f"whisper_batch_bs{B}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
