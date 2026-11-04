#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Functional tests for the FL2VA-patched pipeline methods.

Extracts the FL2VA methods from the patched pipeline_minimax_h3.py and binds them
to a mock pipeline, verifying:
1. Layout: FL2VA anchors match the reference math; T2V layout unchanged.
2. Renoise: condition rows frozen bit-exactly; generated rows updated; the
   generator consumption matches the no-anchor path's per-row draw.
3. T2V equivalence: with no anchors the renoise is bit-identical to the old code.

Run: python contrib/pika-h3-flex/test_fl2va_patched.py /tmp/pipeline_test.py
"""

import sys
import types

import numpy as np
import torch

# Constants mirrored from the patched pipeline.
KEYFRAME_NOISE_AUG = 0.999
MINIMAX_H3_AUDIO_TAG = 2
MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_CHANNELS = 2


def load_pipeline_class(path: str):
    """Import the patched pipeline module with heavy deps stubbed out."""
    stubs = [
        "diffusers",
        "diffusers.models",
        "diffusers.models.transformers",
        "diffusers.models.transformers.transformer_minimax_h3",
        "diffusers.modular_pipelines",
        "diffusers.modular_pipelines.minimax_h3",
        "diffusers.modular_pipelines.minimax_h3.before_denoise",
        "diffusers.modular_pipelines.minimax_h3.modular_pipeline",
        "diffusers.schedulers",
        "diffusers.utils",
        "diffusers.utils.torch_utils",
        "transformers",
        "vllm",
        "vllm.distributed",
        "vllm.distributed.parallel_state",
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.data",
        "vllm_omni.diffusion.distributed",
        "vllm_omni.diffusion.distributed.utils",
        "vllm_omni_neuron",
        "vllm_omni_neuron.diffusion",
        "vllm_omni_neuron.diffusion.distributed",
        "vllm_omni_neuron.diffusion.distributed.autoencoders",
        "vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_minimax_h3",
        "vllm_omni_neuron.diffusion.models",
        "vllm_omni_neuron.diffusion.models.minimax_h3",
        "vllm_omni_neuron.diffusion.models.minimax_h3.encoders",
        "vllm_omni_neuron.diffusion.models.minimax_h3.minimax_h3_transformer",
    ]
    saved = {}
    for name in stubs:
        saved[name] = sys.modules.get(name)
        module = types.ModuleType(name)
        sys.modules[name] = module

    # Populate the specific names the pipeline imports.
    bd = sys.modules["diffusers.modular_pipelines.minimax_h3.before_denoise"]

    def patchify_video_latents(latents, patch_size):
        pt, ph, pw = patch_size
        b, c, f, h, w = latents.shape
        latents = latents.reshape(b, c, f // pt, pt, h // ph, ph, w // pw, pw)
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return latents.reshape(-1, c * pt * ph * pw).contiguous()

    bd.patchify_video_latents = patchify_video_latents
    bd.MiniMaxH3PrepareLayoutStep = object

    mp = sys.modules["diffusers.modular_pipelines.minimax_h3.modular_pipeline"]
    mp.MINIMAX_H3_AUDIO_CHANNELS = 2
    mp.MINIMAX_H3_AUDIO_TAG = 2
    mp.MINIMAX_H3_FPS = 24
    mp.MINIMAX_H3_TEXT_TAG = 1
    mp.MINIMAX_H3_VIDEO_TAG = 0
    mp.align_num_frames = lambda n, fpc, lpc: n
    mp.audio_latent_num_frames = lambda n: n
    mp.video_latent_num_frames = lambda n, fpc, lpc: n

    sch = sys.modules["diffusers.schedulers"]
    sch.MiniMaxH3Scheduler = object

    tu = sys.modules["diffusers.utils.torch_utils"]

    def randn_tensor(shape, generator=None, device=None, dtype=None):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    tu.randn_tensor = randn_tensor

    sys.modules["diffusers.models.transformers.transformer_minimax_h3"].MiniMaxH3Transformer3DModel = object
    sys.modules["transformers"].AutoTokenizer = object
    sys.modules["vllm.distributed.parallel_state"].get_tp_group = lambda: None
    sys.modules["vllm_omni.diffusion.data"].DiffusionOutput = object
    sys.modules["vllm_omni.diffusion.distributed.utils"].get_local_device = lambda: torch.device("cpu")
    enc = sys.modules["vllm_omni_neuron.diffusion.models.minimax_h3.encoders"]
    enc.MINIMAX_H3_TEXT_SEQ_BUCKET = 64
    enc.NeuronQwen3VLTextEncoder = object
    sys.modules[
        "vllm_omni_neuron.diffusion.models.minimax_h3.minimax_h3_transformer"
    ].MiniMaxH3Transformer3DModel = object
    ae = sys.modules[
        "vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_minimax_h3"
    ]
    ae.DistributedAutoencoderKLMiniMaxH3 = object
    ae.NeuronAutoencoderKLMiniMaxH3 = object

    import importlib.util
    import re

    # Local Python is 3.9: rewrite `X | None` annotations (3.10+) before exec.
    with open(path) as f:
        src = f.read()
    src = re.sub(r"torch\.Tensor \| None", "object", src)
    src = re.sub(r"tuple\[(.*?)\]", "tuple", src)
    src = re.sub(r"list\[(.*?)\]", "list", src)
    src = re.sub(r"dict \| None", "object", src)

    spec = importlib.util.spec_from_loader("patched_pipeline", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(src, path, "exec"), module.__dict__)
    return module


def make_mock_pipeline(module, num_condition_rows=0):
    pipe = module.NeuronMiniMaxH3Pipeline.__new__(module.NeuronMiniMaxH3Pipeline)
    pipe._num_condition_video_rows = num_condition_rows
    pipe.device = torch.device("cpu")
    pipe.dtype = torch.bfloat16
    pipe.is_output_rank = False
    return pipe


def make_layout(pipe, num_text=11, num_frames=6, lh=4, lw=6, num_audio=7, anchors=()):
    tags = torch.full((num_text,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    return pipe._build_fl2va_layout(tags, num_frames, lh, lw, num_audio, (1, 2, 2), anchors)


def test_layout(module):
    pipe = make_mock_pipeline(module)
    # T2V: no anchors.
    pos, tags, vid, aud, txt, n_cond, n_cond_a = make_layout(pipe, anchors=())
    assert n_cond == 0 and n_cond_a == 0
    assert pos.shape[0] == 11 + 6 * (2 * 3) + 7 * 2
    # FL2VA: first + last.
    pos, tags, vid, aud, txt, n_cond, _ = make_layout(pipe, anchors=("first", "last"))
    rows_per_frame = 2 * 3
    assert n_cond == 2 * rows_per_frame
    # First anchor at t = num_text_tokens.
    assert pos[11, 0].item() == 11.0
    # Last anchor: numpy pairwise sum of the frame spans, minus one rescale.
    spans = np.ones(6, dtype=np.float64) * (5.0 / 3.0)
    for off, mult in enumerate((1, 4, 4, 4, 4)):
        spans[off::5] *= mult
    expected = 11.0 + float(spans.sum()) - 5.0 / 3.0
    assert abs(pos[11 + rows_per_frame, 0].item() - expected) < 1e-12
    # Condition rows are video-tagged; indices lead the video block.
    assert torch.all(tags[vid[:n_cond]] == MINIMAX_H3_VIDEO_TAG)
    assert vid[0].item() == 11
    print("layout OK")


class FakeScheduler:
    class config:
        shift = 12.0

    def scale_noise(self, sample, t, noise):
        return t * sample + (1.0 - t) * noise


class FakeAudioScheduler:
    class config:
        shift = 3.0


class FakeTransformer:
    """Zero-velocity DiT: noise_pred = 0 everywhere, so x0 = latents."""

    class config:
        adaln_cache = False
        first_block_cache_threshold = 0.0

    def __call__(self, hidden_states=None, audio_hidden_states=None, **kwargs):
        # Real DiT returns (1, rows, dim); _denoise_pika indexes [0] after .cpu().
        return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)


def run_pika_loop(pipe, video_rows, audio_rows, layout, generator):
    """Run the patched _denoise_pika with the fake DiT, on CPU."""
    pipe.scheduler = FakeScheduler()
    pipe.audio_scheduler = FakeAudioScheduler()
    pipe.transformer = FakeTransformer()
    pipe.block_modulations = lambda ts: ()
    return pipe._denoise_pika(
        torch.zeros(1, 4, 8), layout, video_rows, audio_rows, generator
    )


def test_frozen_anchors(module):
    pipe = make_mock_pipeline(module, num_condition_rows=4)
    layout = make_layout(pipe, anchors=("first",))
    n_cond = 4
    gen_rows = 6 * (2 * 3)
    video_rows = torch.randn(n_cond + gen_rows, 8)
    anchors_before = video_rows[:n_cond].clone()
    audio_rows = torch.randn(7 * 2, 3)

    g1 = torch.Generator().manual_seed(7)
    out, _ = run_pika_loop(pipe, video_rows.clone(), audio_rows.clone(), layout, g1)

    # Anchors bit-identical after all 6 steps.
    assert torch.equal(out[:n_cond], anchors_before), "anchors drifted!"

    # With zero velocity, x0 = latents; the renoise is (1-s)*x0 + s*noise, so generated
    # rows must differ from their start (except by astronomical chance).
    assert not torch.equal(out[n_cond:], video_rows[n_cond:])

    # Determinism: same seed -> same result.
    g2 = torch.Generator().manual_seed(7)
    out2, _ = run_pika_loop(pipe, video_rows.clone(), audio_rows.clone(), layout, g2)
    assert torch.equal(out, out2)
    print("frozen anchors OK")


def test_t2v_bit_equivalence(module):
    """With no anchors, the new gen-slice renoise must match the old rebind math."""
    pipe = make_mock_pipeline(module, num_condition_rows=0)
    layout = make_layout(pipe, anchors=())
    video_rows = torch.randn(36, 8)
    audio_rows = torch.randn(14, 3)

    # New path.
    g = torch.Generator().manual_seed(3)
    out_new, _ = run_pika_loop(pipe, video_rows.clone(), audio_rows.clone(), layout, g)

    # Reference math (old code): full-shape draws, rebinding.
    sig_v = module.NeuronMiniMaxH3Pipeline._pika_shifted_sigmas(12.0)
    sig_a = module.NeuronMiniMaxH3Pipeline._pika_shifted_sigmas(3.0)
    g = torch.Generator().manual_seed(3)
    v, a = video_rows.clone(), audio_rows.clone()
    for i in range(sig_v.numel() - 1):
        x0_v = v  # zero velocity
        x0_a = a
        noise_v = torch.randn(v.shape, generator=g, dtype=torch.float32)
        noise_a = torch.randn(a.shape, generator=g, dtype=torch.float32)
        v = (1.0 - sig_v[i + 1]) * x0_v + sig_v[i + 1] * noise_v
        a = (1.0 - sig_a[i + 1]) * x0_a + sig_a[i + 1] * noise_a
    assert torch.equal(out_new, v), "T2V video rows changed"
    print("T2V bit-equivalence OK")


def test_generator_consumption(module):
    """FL2VA draws generated-rows-shaped noise, not full-shape — generator state parity."""
    pipe = make_mock_pipeline(module, num_condition_rows=4)
    layout = make_layout(pipe, anchors=("first",))
    n_cond, gen_rows = 4, 36
    video_rows = torch.randn(n_cond + gen_rows, 8)
    audio_rows = torch.randn(14, 3)

    g = torch.Generator().manual_seed(11)
    run_pika_loop(pipe, video_rows.clone(), audio_rows.clone(), layout, g)
    state_after = torch.randn(1, generator=g)

    # Expected: per step, generated-rows draw then audio draw.
    g = torch.Generator().manual_seed(11)
    for _ in range(6):
        torch.randn(gen_rows, 8, generator=g)
        torch.randn(14, 3, generator=g)
    expected_after = torch.randn(1, generator=g)
    assert torch.equal(state_after, expected_after), "generator consumption mismatch"
    print("generator consumption OK")


def test_keyframe_noise_level(module):
    """Anchors are noised to t=0.999 via scale_noise, one draw per keyframe."""
    pipe = make_mock_pipeline(module)
    pipe.scheduler = FakeScheduler()
    cond = torch.ones(1, 24, 1, 4, 6)
    g = torch.Generator().manual_seed(5)
    rows = pipe._noise_keyframe_latents([cond], g, (1, 2, 2))
    noise = torch.randn(cond.shape, generator=torch.Generator().manual_seed(5))
    expected = 0.999 * cond + 0.001 * noise
    # patchify of a single latent frame: reshape is lossless; compare on values.
    b, c, f, h, w = expected.shape
    exp = expected.reshape(b, c, f, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6)
    exp = exp.reshape(-1, c * 4)
    assert torch.allclose(rows, exp, atol=1e-7), (rows - exp).abs().max()
    print("keyframe noise level OK")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pipeline_test.py"
    module = load_pipeline_class(path)
    test_layout(module)
    test_keyframe_noise_level(module)
    test_frozen_anchors(module)
    test_t2v_bit_equivalence(module)
    test_generator_consumption(module)
    print("\nAll patched-pipeline tests passed!")
