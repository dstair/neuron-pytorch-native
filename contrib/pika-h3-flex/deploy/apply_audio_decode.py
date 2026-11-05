#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add audio decoding to pipeline_minimax_h3.py (video+audio output for Pika tasks).

The pipeline already denoises the audio rows in the packed sequence; this decodes them
through the MiniMax-H3 audio VAE (mono BigVGAN-style decoder, stereo as two batch items)
on the host on rank 0 and writes a 32 kHz stereo WAV per request, following diffusers
main's ``PikaH3FlexAudioDecodeStep``:

    rows -> (2, C, T) -> denormalize -> decode -> truncate to num_frames/fps -> (1, 2, N)

The WAV lands in ``$MINIMAX_H3_OUT/request_NNN.wav`` (or ``WORKLOAD_OUTPUT_RW``) so the
runner can mux it into the MP4. Decoding is on by default; ``decode_audio: false`` in the
stage config's model_config disables it.

Usage:
    python apply_audio_decode.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

AUDIO_INIT = '''        # Audio output: decode the denoised audio rows through the audio VAE on the host
        # (rank 0 only). Off with decode_audio: false.
        self._decode_audio = bool(model_config.pop("decode_audio", True))
        self.audio_vae = None
        self._audio_request_index = 0
'''

AUDIO_LOAD = '''        if self.vae is not None:
            self.vae.load_weights(os.path.join(model_path, "vae"))
            if self._vae_fused_vit:
                from vllm_omni_neuron.diffusion.distributed.autoencoders.minimax_h3_vit_block import (
                    fuse_decoder,
                )

                with torch.no_grad():
                    fuse_decoder(self.vae.decoder)
        # The audio VAE stays on the host in FP32: only rank 0 decodes, once per request.
        if self._decode_audio and self.is_output_rank:
            from diffusers.models import AutoencoderKLMiniMaxH3Audio

            self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
                model_path, subfolder="audio_vae", local_files_only=os.path.isdir(model_path),
                torch_dtype=torch.float32,
            )
            self.audio_vae.eval()
'''

AUDIO_METHOD = '''
    def _decode_audio_rows(self, audio_rows: torch.Tensor, num_audio_latents: int, num_frames: int):
        """Decode denoised audio rows to a 32 kHz stereo WAV path (rank 0, host).

        Follows diffusers main ``PikaH3FlexAudioDecodeStep``: channel-major rows become
        ``(2, C, T)`` (the mono audio VAE takes the stereo channels as two batch items),
        latents are denormalized, decoded, and truncated to the video's duration.
        """
        if self.audio_vae is None:
            return None
        import wave

        rows = audio_rows[self._num_condition_audio_rows :] if getattr(self, "_num_condition_audio_rows", 0) else audio_rows
        channels = rows.shape[-1]
        audio_latents = rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, channels).permute(0, 2, 1).contiguous()
        mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(self.audio_vae.config.latents_std).view(1, -1, 1)
        z = (audio_latents.float() * std + mean).to(self.audio_vae.dtype)
        audio = self.audio_vae.decode(z, return_dict=False)[0]
        sampling_rate = int(self.audio_vae.config.sampling_rate)
        audio = audio[..., : round(num_frames / MINIMAX_H3_FPS * sampling_rate)]
        audio = audio.float().permute(1, 0, 2)  # (1, 2, samples)

        directory = os.environ.get("MINIMAX_H3_OUT") or os.environ.get("WORKLOAD_OUTPUT_RW")
        if not directory:
            return None
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"request_{self._audio_request_index:03d}.wav")
        self._audio_request_index += 1
        pcm = (audio[0].T.clamp(-1, 1).numpy() * 32767.0).round().astype("<i2")
        with wave.open(path, "wb") as handle:
            handle.setnchannels(pcm.shape[1])
            handle.setsampwidth(2)
            handle.setframerate(sampling_rate)
            handle.writeframes(pcm.tobytes())
        logger.info("Audio: wrote %s (%.2fs, %d Hz stereo)", path, pcm.shape[0] / sampling_rate, sampling_rate)
        return path

'''

OLD_AUDIO_ROWS = '''        t0 = time.perf_counter()
        video_rows = patchify_video_latents(latents, patch_size)
        if condition_rows is not None:
            video_rows = torch.cat([condition_rows, video_rows])
        video_rows, step_seconds = self.denoise(
            prompt_embeds,
            layout,
            video_rows,
            audio_rows,
            num_steps,
            generator=generator,
        )
'''

NEW_AUDIO_ROWS = '''        t0 = time.perf_counter()
        video_rows = patchify_video_latents(latents, patch_size)
        if condition_rows is not None:
            video_rows = torch.cat([condition_rows, video_rows])
        video_rows, audio_rows, step_seconds = self.denoise(
            prompt_embeds,
            layout,
            video_rows,
            audio_rows,
            num_steps,
            generator=generator,
        )
        audio_path = self._decode_audio_rows(audio_rows, num_audio_latents, num_frames)
'''


def apply(src: str) -> str:
    if "_decode_audio_rows" in src:
        raise SystemExit("audio decode already applied; refusing to re-apply")

    # 1. __init__ state after _num_condition_video_rows.
    anchor = "        # FL2VA: number of leading video rows that are frozen keyframe conditioning.\n        self._num_condition_video_rows = 0\n"
    assert anchor in src, "init anchor not found"
    src = src.replace(anchor, anchor + AUDIO_INIT + "        self._num_condition_audio_rows = 0\n", 1)

    # 2. Audio VAE load in load_weights (after the video VAE block).
    old_load = '''        if self.vae is not None:
            self.vae.load_weights(os.path.join(model_path, "vae"))
            if self._vae_fused_vit:
                from vllm_omni_neuron.diffusion.distributed.autoencoders.minimax_h3_vit_block import (
                    fuse_decoder,
                )

                with torch.no_grad():
                    fuse_decoder(self.vae.decoder)
'''
    assert old_load in src, "load_weights anchor not found"
    src = src.replace(old_load, AUDIO_LOAD, 1)

    # 3. _decode_audio_rows method before block_modulations.
    anchor = "    def block_modulations(self, timestep: torch.Tensor)"
    assert anchor in src, "block_modulations anchor not found"
    src = src.replace(anchor, AUDIO_METHOD + anchor, 1)

    # 4. denoise() returns audio rows too; forward() decodes them.
    old_sig = "    def denoise(self, prompt_embeds, layout, video_rows, audio_rows, num_steps, generator=None):\n"
    assert old_sig in src
    # The wrappers just pass through; update _denoise and _denoise_pika returns.
    src = src.replace(
        "            self._latent_dump_request = request_index + 1\n        return video_rows, step_seconds\n",
        "            self._latent_dump_request = request_index + 1\n        return video_rows, audio_rows, step_seconds\n",
        2,
    )
    assert OLD_AUDIO_ROWS in src, "denoise call anchor not found"
    src = src.replace(OLD_AUDIO_ROWS, NEW_AUDIO_ROWS, 1)

    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    backup = path + ".pre_audio"
    with open(backup, "w") as f:
        f.write(src)
    with open(path, "w") as f:
        f.write(apply(src))
    print(f"Applied audio decode to {path} (backup: {backup})")


if __name__ == "__main__":
    main()
