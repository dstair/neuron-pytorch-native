"""Entry script: Qwen3-ASR-0.6B static-shape port on Neuron (or CPU for validation).
  python run.py -n 10                       # Neuron bf16, compiled
  python run.py -n 10 --device cpu --no-compile --dtype fp32   # port validation vs CPU reference
Prints per-sample token-exact match vs ref/ref.json and a final SUMMARY line."""
import argparse, json, os, sys, time
import numpy as np, torch
RUN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUN)
from cpu_ref import load_samples, wer
from qasr_port import Encoder, Prefill, Decode, DecodeK, DecodeKLate, n_audio_tokens, AUDIO_TOK, PROMPT_AUDIO_OFFSET

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=10)
ap.add_argument("--device", default="neuron")
ap.add_argument("--no-compile", action="store_true")
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--enc-buckets", default="8,16,32", help="mel chunks (x100 frames)")
ap.add_argument("--prefill-buckets", default="128,256,512")
ap.add_argument("--max-seq", type=int, default=768)
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--repeats", type=int, default=3, help="timed warm repeats per sample")
args = ap.parse_args()

if args.device == "neuron":
    import torch_neuronx  # noqa
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

dev = torch.device(args.device)
dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
EOS = {151643, 151645}
PAD = 151645
ENC_B = [int(x) for x in args.enc_buckets.split(",")]
PRE_B = [int(x) for x in args.prefill_buckets.split(",")]


def sync(x=None):
    if args.device == "neuron":
        if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
            torch.neuron.synchronize()
        elif x is not None:
            x.cpu()


def pick(buckets, n):
    for b in buckets:
        if b >= n:
            return b
    raise ValueError(f"length {n} exceeds largest bucket {buckets[-1]}")


MODEL = os.environ.get("QASR_MODEL", f"{RUN}/model")
proc = AutoProcessor.from_pretrained(MODEL)
hf = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, dtype=dt).eval()
enc_m = Encoder(hf).eval().to(dev)
pre_m = Prefill(hf, args.max_seq).eval().to(dev)
# r2 (kept): QASR_MULTI=K (default 4) -> K decode steps per graph call (DecodeK); QASR_MULTI=1 = one token per call (r1 path).
MULTI = int(os.environ.get("QASR_MULTI", "4"))
# r3: QASR_KV=lateK -> DecodeKLate (new k/v columns kept aside within a call, caches written once per call).
if MULTI > 1 and os.environ.get("QASR_KV") == "lateK":
    dec_m = DecodeKLate(hf, args.max_seq, MULTI).eval().to(dev)
else:
    dec_m = (DecodeK(hf, args.max_seq, MULTI) if MULTI > 1 else Decode(hf, args.max_seq)).eval().to(dev)
if args.no_compile:
    enc_f, pre_f, dec_f = enc_m, pre_m, dec_m
else:
    cc = lambda m: torch.compile(m, backend="neuron", fullgraph=True, dynamic=False)
    enc_f, pre_f, dec_f = cc(enc_m), cc(pre_m), cc(dec_m)

ref = json.load(open(f"{RUN}/ref/ref.json"))["samples"]
samples = load_samples(args.n)
compile_s = {}
# r2: QASR_LAG=k -> lagged EOS check in the decode loop (REJECTED: no gain, 5.00 vs 4.93 ms/token); default 0 = synced loop.
LAG = int(os.environ.get("QASR_LAG", "0"))
# r2: QASR_POSDEV=1 -> device position tensors created once (no gain: 4.93 vs 4.95 ms); default 0.
POSDEV = os.environ.get("QASR_POSDEV", "0") == "1"
POSD = [torch.tensor([p], dtype=torch.long).to(dev) for p in range(args.max_seq)] if (POSDEV or LAG > 0) else None


@torch.no_grad()
def transcribe(s, timed):
    inp = proc.apply_transcription_request(audio=s["audio"], language="English")
    ids = inp["input_ids"][0]
    fvalid = int(inp["input_features_mask"].sum())
    nA = n_audio_tokens(fvalid)
    assert int((ids == AUDIO_TOK).sum()) == nA and bool((ids[PROMPT_AUDIO_OFFSET:PROMPT_AUDIO_OFFSET + nA] == AUDIO_TOK).all())
    feats = inp["input_features"]                       # [1,128,F], F multiple of 100
    C = pick(ENC_B, feats.shape[2] // 100)
    feats = torch.nn.functional.pad(feats, (0, C * 100 - feats.shape[2]))
    L = ids.shape[0]
    P = pick(PRE_B, L)
    assert P + args.max_new <= args.max_seq
    ids_p = torch.full((1, P), PAD, dtype=torch.long)
    ids_p[0, :L] = ids
    t = {}
    t0 = time.perf_counter()
    f_d = feats.to(dt).to(dev)
    nv = torch.tensor(nA, dtype=torch.long).to(dev)
    audio = enc_f(f_d, nv)
    sync(audio)
    t1 = time.perf_counter()
    tok, kc, vc = pre_f(ids_p.to(dev), audio, torch.tensor([L - 1], dtype=torch.long).to(dev))
    first = int(tok.cpu()[0])
    t2 = time.perf_counter()
    gen = [first]
    step_t = []
    pos = L
    if MULTI > 1:
        posd = torch.tensor([pos], dtype=torch.long)
        while gen[-1] not in EOS and len(gen) < args.max_new:
            toks, tok, kc, vc = dec_f(tok, posd.to(dev), kc, vc)
            for t_ in toks.cpu().tolist():
                gen.append(int(t_))
                if t_ in EOS or len(gen) >= args.max_new:
                    break
            posd = posd + MULTI
        step_t = [float("nan")]
    elif LAG > 0 and not dec_m.kv_inplace:
        # r2: lagged EOS check -- dispatch step n+1 (input = device token of step n) before the blocking D2H read of
        # token n, so host dispatch/D2H overlap device execution. Overshoots EOS by <= LAG wasted steps (discarded).
        q = []
        cur = tok
        while gen[-1] not in EOS:
            if len(gen) + len(q) < args.max_new:
                cur, kc, vc = dec_f(cur, POSD[pos], kc, vc)
                pos += 1
                q.append(cur)
            if not q:
                break
            if len(q) > LAG or len(gen) + len(q) >= args.max_new:
                gen.append(int(q.pop(0).cpu()[0]))
        step_t = [float("nan")]
    else:
        while gen[-1] not in EOS and len(gen) < args.max_new:
            ts = time.perf_counter()
            pd = POSD[pos] if POSDEV else torch.tensor([pos], dtype=torch.long).to(dev)
            if dec_m.kv_inplace:     # r1: decode writes slot pos of kc/vc in place, returns only the token
                tok = dec_f(tok, pd, kc, vc)
            else:
                tok, kc, vc = dec_f(tok, pd, kc, vc)
            gen.append(int(tok.cpu()[0]))
            step_t.append(time.perf_counter() - ts)
            pos += 1
    t3 = time.perf_counter()
    t.update(enc=t1 - t0, prefill=t2 - t1, decode_total=t3 - t2, e2e=t3 - t0, C=C, P=P,
             step=float(np.median(step_t)) if step_t else float("nan"), nsteps=len(gen) - 1,
             step_amort=(t3 - t2) / max(len(gen) - 1, 1))
    return gen, t


results = []
for i, s in enumerate(samples):
    tc = time.perf_counter()
    gen, t_cold = transcribe(s, False)          # first call: includes compile for new buckets
    cold = time.perf_counter() - tc
    reps = [transcribe(s, True) for _ in range(args.repeats)]
    assert all(g == gen for g, _ in reps), "non-deterministic output across repeats"
    tt = {k: float(np.median([r[1][k] for r in reps])) for k in ("enc", "prefill", "step", "step_amort", "e2e")}
    text = proc.decode([gen], return_format="transcription_only")[0]
    r = ref[i]
    match = gen == r["gen_ids"]
    # first divergence
    div = next((j for j, (a, b) in enumerate(zip(gen, r["gen_ids"])) if a != b), None)
    if div is None and not match:
        div = min(len(gen), len(r["gen_ids"]))
    results.append(dict(id=s["id"], match=match, div=div, text=text, ref_text=s["text"], cold=cold, C=t_cold["C"],
                        P=t_cold["P"], ntok=len(gen), cpu_s=r["cpu_s"], **tt))
    print(f"[{i}] {s['id']} C={t_cold['C']} P={t_cold['P']} ntok={len(gen)} match={match} div={div} "
          f"cold={cold:.1f}s enc={tt['enc']*1e3:.2f}ms prefill={tt['prefill']*1e3:.2f}ms "
          f"step={tt['step']*1e3:.2f}ms amort={tt['step_amort']*1e3:.2f}ms e2e={tt['e2e']*1e3:.1f}ms cpu={r['cpu_s']*1e3:.0f}ms | {text}", flush=True)

w = wer([r["ref_text"] for r in results], [r["text"] for r in results])
nm = sum(r["match"] for r in results)
med = lambda k: float(np.median([r[k] for r in results])) * 1e3
os.makedirs(f"{RUN}/artifacts", exist_ok=True)
tag = os.environ.get("QASR_RUNTAG") or f"{args.device}_{args.dtype}_{'eager' if args.no_compile else 'compiled'}"
json.dump(dict(args=vars(args), results=results, wer=w, matches=nm), open(f"{RUN}/artifacts/run_{tag}.json", "w"), indent=1)
print(f"SUMMARY device={args.device} dtype={args.dtype} compiled={not args.no_compile} token_exact={nm}/{len(results)} "
      f"WER={w:.4f} (cpu_ref WER={json.load(open(f'{RUN}/ref/ref.json'))['wer']:.4f}) "
      f"enc_ms={med('enc'):.2f} prefill_ms={med('prefill'):.2f} decode_step_ms={med('step'):.2f} step_amort_ms={med('step_amort'):.2f} "
      f"e2e_ms={med('e2e'):.1f} cpu_fp32_e2e_ms={med('cpu_s'):.0f}", flush=True)
