#!/usr/bin/env python3
"""Repair a short silent video with MiniMax-H3 keyframe-guided video-to-video.

The first frame is used as the clean FL2AV keyframe.  All input frames are
encoded as the video-to-video source, padded by repeating the last frame to the
next MiniMax-H3-supported 17n+5 length, noised at a low FlowMatch strength, and
denoised jointly with a synthetic silent audio latent.  Source frames are
normalized to the requested generation size with a direct resize before VAE
encoding/noising.  The decoded result is cropped back to the original input
frame count, aspect-fit and padded to the output size, and is silent by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageFilter


DEFAULT_DIFFSYNTH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "MiniMax/MiniMax-H3"
H3_FPS = 24
DEFAULT_PROMPT = """<SUBJECT>: The Brandenburg Gate in Berlin, a monumental neoclassical sandstone landmark with six large columns, shown with accurate architectural proportions and detailed stone textures.

<SCENE>: A photorealistic daytime view of the Brandenburg Gate in Berlin. The architecture remains sharp, rigid, and geometrically consistent across all frames. Natural daylight, stable exposure, realistic sandstone colors, clean edges, fine architectural details, and temporally consistent textures. No structural deformation, no flickering, no duplicated columns, and no objects appearing or disappearing.

<EVENT>: The camera performs a smooth horizontal lateral tracking movement across the scene while keeping the Brandenburg Gate stable and in focus. The camera motion is continuous and steady, with consistent perspective changes and no sudden jumps, shaking, warping, or zooming. The scene remains coherent throughout the entire shot. The video is silent, with no speech, no music, and no environmental sound."""


def align_up(value: int, factor: int) -> int:
    if value <= 0:
        raise ValueError(f"Expected a positive size, got {value}")
    return math.ceil(value / factor) * factor


def supported_frame_count_at_least(frame_count: int) -> int:
    if frame_count < 5:
        raise ValueError("MiniMax-H3 repair requires at least 5 input frames")
    return math.ceil((frame_count - 5) / 17) * 17 + 5


def direct_resize_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    """Normalize a possibly stretched/scaled source without discarding pixels."""
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def fit_pad_resize_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    """Fit the full image inside the target and pad, never crop or stretch."""
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = min(width / src_w, height / src_h)
    resized_w = min(width, max(1, round(src_w * scale)))
    resized_h = min(height, max(1, round(src_h * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    left = (width - resized_w) // 2
    top = (height - resized_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def sharpen_frame(image: Image.Image, amount: float) -> Image.Image:
    """Apply deterministic light unsharp masking at the internal resolution."""
    image = image.convert("RGB")
    if amount <= 0:
        return image
    return image.filter(
        ImageFilter.UnsharpMask(radius=1.25, percent=round(amount * 100), threshold=2)
    )


def read_video_frames(path: Path, max_frames: int | None = None) -> tuple[list[Image.Image], float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[Image.Image] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    if not frames or width <= 0 or height <= 0:
        raise RuntimeError(f"No readable frames in input video: {path}")
    return frames, fps, width, height


def build_pipeline(args: argparse.Namespace):
    if str(args.diffsynth_root) not in sys.path:
        sys.path.insert(0, str(args.diffsynth_root))
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.minimax_h3_audio_video_av2av import MiniMaxH3AV2AVPipeline

    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"

    free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
    free_gb, total_gb = free_bytes / 1024**3, total_bytes / 1024**3
    vram_limit = args.vram_limit_gb
    if vram_limit is None:
        vram_limit = max(1.0, total_gb - args.vram_reserve_gb)
    print(
        f"[vram] free={free_gb:.2f}GB total={total_gb:.2f}GB "
        f"limit={vram_limit:.2f}GB reserve={args.vram_reserve_gb:.2f}GB"
    )
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": args.device,
        "computation_dtype": torch.bfloat16,
        "computation_device": args.device,
    }
    return MiniMaxH3AV2AVPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
            ModelConfig(model_id=args.model_id, origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
            ModelConfig(model_id=args.model_id, origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
            ModelConfig(model_id=args.model_id, origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        ],
        processor_config=ModelConfig(model_id=args.model_id, origin_file_pattern="FL2VA/processor/"),
        vram_limit=vram_limit,
    )


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return args.prompt.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair a silent short video using a clean first-frame keyframe and low-strength H3 V2V."
    )
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--width", type=int, default=1024, help="Internal H3 repair width.")
    parser.add_argument("--height", type=int, default=576, help="Internal H3 repair height.")
    parser.add_argument("--output-width", type=int, default=736)
    parser.add_argument("--output-height", type=int, default=416)
    parser.add_argument("--fps", type=int, default=H3_FPS)
    parser.add_argument("--max-input-frames", type=int)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--denoising-strength", type=float, default=0.01)
    parser.add_argument("--audio-denoising-strength", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram-limit-gb", type=float, default=60.0)
    parser.add_argument("--vram-reserve-gb", type=float, default=2.0)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--keep-generated-audio", action="store_true")
    parser.add_argument("--no-preserve-first-frame", action="store_true")
    parser.add_argument(
        "--save-internal-frames",
        action="store_true",
        help="Save raw decoded internal-resolution frames as lossless PNGs before sharpening/downsampling.",
    )
    parser.add_argument(
        "--sharpen-amount",
        type=float,
        default=0.0,
        help="Internal-resolution unsharp-mask amount; 0 disables it, 0.3-0.6 is recommended.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input_video.is_file():
        raise FileNotFoundError(args.input_video)
    if args.prompt_file is not None and not args.prompt_file.is_file():
        raise FileNotFoundError(args.prompt_file)
    if args.fps != H3_FPS:
        raise ValueError(f"MiniMax-H3 timing is fixed at {H3_FPS} fps")
    if not 0.0 < args.denoising_strength <= 1.0:
        raise ValueError("--denoising-strength must be in (0, 1]")
    if not 0.0 < args.audio_denoising_strength <= 1.0:
        raise ValueError("--audio-denoising-strength must be in (0, 1]")
    if not 0.0 <= args.sharpen_amount <= 3.0:
        raise ValueError("--sharpen-amount must be in [0, 3]")
    if args.output_width <= 0 or args.output_height <= 0 or args.output_width % 2 or args.output_height % 2:
        raise ValueError("Output dimensions must be positive even numbers")
    if args.output_video.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {args.output_video}")
    internal_frames_dir = args.output_video.with_suffix("").with_name(
        f"{args.output_video.stem}_internal_frames"
    )
    if (
        args.save_internal_frames
        and internal_frames_dir.exists()
        and any(internal_frames_dir.glob("*.png"))
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Internal-frame directory already contains PNGs; pass --overwrite or choose another output: "
            f"{internal_frames_dir}"
        )

    source_frames, source_fps, source_width, source_height = read_video_frames(
        args.input_video, args.max_input_frames
    )
    input_frame_count = len(source_frames)
    padded_frame_count = supported_frame_count_at_least(input_frame_count)
    generation_width, generation_height = align_up(args.width, 32), align_up(args.height, 32)
    prompt = load_prompt(args)
    if not prompt:
        raise ValueError("Prompt is empty")

    print(f"[input] {args.input_video}")
    print(
        f"[source] frames={input_frame_count} reported_fps={source_fps:.6f} "
        f"size={source_width}x{source_height}"
    )
    print(
        f"[repair] input_frames={input_frame_count} padded_frames={padded_frame_count} "
        f"generation={generation_width}x{generation_height}@{args.fps} "
        f"output={args.output_width}x{args.output_height}@{args.fps} steps={args.num_inference_steps}"
    )
    print("[spatial] input=direct-resize-before-noise output=aspect-fit-plus-pad")
    print(
        f"[detail] save_internal_frames={args.save_internal_frames} "
        f"sharpen_amount={args.sharpen_amount:.3f}"
    )
    video_sigma = args.flow_shift * args.denoising_strength / (
        1 + (args.flow_shift - 1) * args.denoising_strength
    )
    audio_sigma = args.audio_flow_shift * args.audio_denoising_strength / (
        1 + (args.audio_flow_shift - 1) * args.audio_denoising_strength
    )
    print(
        f"[noise] video_strength={args.denoising_strength} sigma={video_sigma:.6f} "
        f"audio_strength={args.audio_denoising_strength} sigma={audio_sigma:.6f}"
    )
    if args.dry_run:
        print(f"[prompt] {prompt}")
        return

    input_frames = [
        direct_resize_frame(frame, generation_width, generation_height)
        for frame in source_frames
    ]
    clean_first_frame = input_frames[0].copy()
    input_frames.extend([input_frames[-1].copy() for _ in range(padded_frame_count - input_frame_count)])

    pipe = build_pipeline(args)
    audio_samples = round(padded_frame_count / args.fps * pipe.audio_vae.sample_rate)
    silent_audio = torch.zeros(2, audio_samples, dtype=torch.float32)
    video, audio = pipe(
        prompt=prompt,
        input_video=input_frames,
        input_audio=silent_audio,
        input_audio_sample_rate=pipe.audio_vae.sample_rate,
        negative_prompt=args.negative_prompt,
        height=generation_height,
        width=generation_width,
        num_frames=padded_frame_count,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        audio_flow_shift=args.audio_flow_shift,
        denoising_strength=args.denoising_strength,
        audio_denoising_strength=args.audio_denoising_strength,
        tiled=not args.no_tiled,
        keyframes=[clean_first_frame],
        keyframe_indices=[0],
    )
    if len(video) < input_frame_count:
        raise RuntimeError(f"Pipeline returned {len(video)} frames, need at least {input_frame_count}")
    internal_video = [frame.convert("RGB") for frame in video[:input_frame_count]]
    if args.save_internal_frames:
        internal_frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in enumerate(internal_video):
            frame.save(internal_frames_dir / f"{frame_index:06d}.png", compress_level=1)
        print(f"[saved] raw_internal_frames={internal_frames_dir}")
    processed_video = [sharpen_frame(frame, args.sharpen_amount) for frame in internal_video]
    video = [
        fit_pad_resize_frame(frame, args.output_width, args.output_height)
        for frame in processed_video
    ]
    if not args.no_preserve_first_frame:
        video[0] = fit_pad_resize_frame(clean_first_frame, args.output_width, args.output_height)

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    from diffsynth.utils.data.audio_video import write_video_audio
    write_video_audio(
        video=video,
        audio=audio if args.keep_generated_audio else None,
        output_path=str(args.output_video),
        fps=args.fps,
        audio_sample_rate=pipe.audio_vae.sample_rate,
        video_quality=8,
    )
    metadata = {
        "input_video": str(args.input_video),
        "output_video": str(args.output_video),
        "source_reported_fps": source_fps,
        "output_fps": args.fps,
        "input_frame_count": input_frame_count,
        "padded_h3_frame_count": padded_frame_count,
        "output_frame_count": len(video),
        "generation_width": generation_width,
        "generation_height": generation_height,
        "output_width": args.output_width,
        "output_height": args.output_height,
        "input_resize_mode": "direct_resize_before_vae_and_noise",
        "output_resize_mode": "aspect_fit_then_pad_no_crop_no_stretch",
        "raw_internal_frames": str(internal_frames_dir) if args.save_internal_frames else None,
        "sharpen_amount": args.sharpen_amount,
        "sharpen_radius": 1.25 if args.sharpen_amount > 0 else None,
        "sharpen_threshold": 2 if args.sharpen_amount > 0 else None,
        "num_inference_steps": args.num_inference_steps,
        "denoising_strength": args.denoising_strength,
        "audio_denoising_strength": args.audio_denoising_strength,
        "initial_video_sigma": video_sigma,
        "initial_audio_sigma": audio_sigma,
        "first_frame_preserved": not args.no_preserve_first_frame,
        "has_audio": bool(args.keep_generated_audio),
        "prompt": prompt,
    }
    metadata_path = args.output_video.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] video={args.output_video}")
    print(f"[saved] metadata={metadata_path}")


if __name__ == "__main__":
    main()
