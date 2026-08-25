"""MiniMax-H3 joint video/audio-to-video/audio inference.

This module leaves the stock MiniMaxH3Pipeline unchanged.  It initializes both
modalities from VAE-encoded GT latents plus FlowMatch noise instead of starting
from pure noise.
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from ..core import ModelConfig
from ..diffusion.base_pipeline import PipelineUnit
from ..utils.data.audio import convert_to_stereo, resample_waveform
from .minimax_h3_audio_video import MiniMaxH3Pipeline


class MiniMaxH3Unit_AVToAVInitializer(PipelineUnit):
    """Encode aligned GT video/audio and add noise at each initial sigma."""

    def __init__(self):
        super().__init__(
            input_params=(
                "seed", "num_frames", "height", "width", "rand_device",
                "tiled", "tile_size", "tile_overlap",
                "input_video", "input_audio",
            ),
            output_params=("video_latents", "audio_latents"),
            onload_model_names=("video_vae", "audio_vae"),
        )

    def process(
        self,
        pipe: "MiniMaxH3AV2AVPipeline",
        seed,
        num_frames,
        height,
        width,
        rand_device,
        tiled,
        tile_size,
        tile_overlap,
        input_video,
        input_audio,
    ):
        if input_video is None or input_audio is None:
            raise ValueError("AV-to-AV inference requires both input_video and input_audio")
        if len(input_video) != num_frames:
            raise ValueError(f"input_video has {len(input_video)} frames; expected {num_frames}")

        video_latent_t = ((num_frames - 5) // 17) * 5 + 2
        latent_h, latent_w = height // 16, width // 16
        video_noise = pipe.generate_noise(
            (1, 24, video_latent_t, latent_h, latent_w),
            seed=seed,
            rand_device=rand_device,
            rand_torch_dtype=pipe.torch_dtype,
        )
        audio_latent_t = round(num_frames / 24.0 * 40.0)
        audio_noise = pipe.generate_noise(
            (2, 32, audio_latent_t),
            seed=seed,
            rand_device=rand_device,
            rand_torch_dtype=pipe.torch_dtype,
        )

        pipe.load_models_to_device(["video_vae"])
        frames_tensor = pipe.preprocess_video(
            input_video,
            torch_dtype=torch.float32,
            min_value=0,
            device=pipe.device,
        )
        video_input_latents = pipe.video_vae.encode_video(
            frames_tensor,
            dtype=pipe.torch_dtype,
            tiled=tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        ).to(device=pipe.device, dtype=pipe.torch_dtype)
        if video_input_latents.shape != video_noise.shape:
            raise ValueError(
                "GT video latent shape does not match generated noise: "
                f"{tuple(video_input_latents.shape)} != {tuple(video_noise.shape)}"
            )
        del frames_tensor

        waveform, sample_rate = input_audio
        if waveform is None or sample_rate is None:
            raise ValueError("GT media has no readable audio stream")
        waveform = waveform.squeeze(0) if waveform.dim() == 3 else waveform
        if waveform.dim() != 2:
            raise ValueError(f"input_audio must be [C, T], got {tuple(waveform.shape)}")
        pipe.load_models_to_device(["audio_vae"])
        waveform = resample_waveform(
            convert_to_stereo(waveform).float(),
            sample_rate,
            pipe.audio_vae.sample_rate,
        )
        audio_input_latents = pipe.audio_vae.encode_audio(
            waveform[:2].to(pipe.device),
            dtype=pipe.torch_dtype,
        ).to(device=pipe.device, dtype=pipe.torch_dtype)
        if audio_input_latents.shape[-1] > audio_latent_t:
            audio_input_latents = audio_input_latents[..., :audio_latent_t]
        elif audio_input_latents.shape[-1] < audio_latent_t:
            missing = audio_latent_t - audio_input_latents.shape[-1]
            audio_input_latents = torch.cat(
                [audio_input_latents, audio_input_latents[..., -1:].repeat(1, 1, missing)],
                dim=-1,
            )
        if audio_input_latents.shape != audio_noise.shape:
            raise ValueError(
                "GT audio latent shape does not match generated noise: "
                f"{tuple(audio_input_latents.shape)} != {tuple(audio_noise.shape)}"
            )

        video_latents = pipe.scheduler.add_noise(
            video_input_latents,
            video_noise,
            timestep=pipe.scheduler.timesteps[0],
        )
        audio_latents = pipe.scheduler_audio.add_noise(
            audio_input_latents,
            audio_noise,
            timestep=pipe.scheduler_audio.timesteps[0],
        )
        return {"video_latents": video_latents, "audio_latents": audio_latents}


class MiniMaxH3AV2AVPipeline(MiniMaxH3Pipeline):
    """MiniMax-H3 pipeline whose video and audio both start from noised GT."""

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        # Stock order: shape, noise, training-video, training-audio, retake, ...
        # Replace the first four initialization units while retaining all
        # conditioning, packing, denoising, and decoding behavior.
        self.units = [self.units[0], MiniMaxH3Unit_AVToAVInitializer(), *self.units[4:]]

    @staticmethod
    def from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = ModelConfig(
            model_id="MiniMax/MiniMax-H3",
            origin_file_pattern="FL2VA/processor/",
        ),
        vram_limit: float | None = None,
    ):
        pipe = MiniMaxH3AV2AVPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("minimax_h3_text_encoder")
        pipe.dit = model_pool.fetch_model("minimax_h3_dit")
        pipe.video_vae = model_pool.fetch_model("minimax_h3_video_vae")
        pipe.audio_vae = model_pool.fetch_model("minimax_h3_audio_vae")
        if processor_config is not None:
            processor_config.download_if_necessary()
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
            pipe.tokenizer = pipe.processor.tokenizer
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        input_video,
        input_audio: torch.Tensor,
        input_audio_sample_rate: int,
        negative_prompt: str = " ",
        height: int = 768,
        width: int = 1344,
        num_frames: int = 124,
        num_inference_steps: int = 50,
        seed: int = 42,
        rand_device: str = "cpu",
        cfg_scale: float = 1.0,
        flow_shift: float = 12.0,
        audio_flow_shift: float = 3.0,
        denoising_strength: float = 0.1,
        audio_denoising_strength: float | None = None,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        keyframes=None,
        keyframe_indices=None,
        references=None,
        ref_image_short_edge: int = 2048,
        ref_video_short_edge: int = 768,
        ref_video_max_pixels: int = 768 * 1344,
        progress_bar_cmd=tqdm,
    ):
        if not 0.0 < denoising_strength <= 1.0:
            raise ValueError("denoising_strength must be in (0, 1]")
        if audio_denoising_strength is None:
            audio_denoising_strength = denoising_strength
        if not 0.0 < audio_denoising_strength <= 1.0:
            raise ValueError("audio_denoising_strength must be in (0, 1]")

        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=flow_shift,
        )
        self.scheduler_audio.set_timesteps(
            num_inference_steps,
            denoising_strength=audio_denoising_strength,
            shift=audio_flow_shift,
        )

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "seed": seed,
            "rand_device": rand_device,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
            "input_video": input_video,
            "input_audio": (input_audio, input_audio_sample_rate),
            "keyframes": keyframes,
            "keyframe_indices": keyframe_indices,
            "references": references,
            "ref_image_short_edge": ref_image_short_edge,
            "ref_video_short_edge": ref_video_short_edge,
            "ref_video_max_pixels": ref_video_max_pixels,
            "retake_video": None,
            "frame_regions_to_retake": None,
            "retake_audio": None,
            "seconds_regions_to_retake": None,
            "imgvid_cond_noise_aug": self.imgvid_cond_noise_aug,
            "audio_cond_noise_aug": self.audio_cond_noise_aug,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep_video in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep_video = timestep_video.unsqueeze(0).to(
                dtype=torch.float32, device=self.device
            )
            timestep_audio = self.scheduler_audio.timesteps[progress_id].unsqueeze(0).to(
                dtype=torch.float32, device=self.device
            )
            noise_pred_video, noise_pred_audio = self.cfg_guided_model_fn(
                self.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **models,
                timestep_video=timestep_video,
                timestep_audio=timestep_audio,
            )
            inputs_shared["video_latents"] = self.step(
                self.scheduler,
                inputs_shared["video_latents"],
                progress_id,
                noise_pred=noise_pred_video,
            )
            inputs_shared["audio_latents"] = self.step(
                self.scheduler_audio,
                inputs_shared["audio_latents"],
                progress_id,
                noise_pred=noise_pred_audio,
            )

        self.load_models_to_device(["video_vae"])
        frames = self.video_vae.decode_video(
            inputs_shared["video_latents"],
            dtype=self.torch_dtype,
            tiled=tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        video = self.vae_output_to_video(frames, min_value=0, max_value=1)

        self.load_models_to_device(["audio_vae"])
        waveform = self.audio_vae.decode_audio(
            inputs_shared["audio_latents"], dtype=self.torch_dtype
        )
        audio = self.output_audio_format_check(waveform)
        return video, audio
