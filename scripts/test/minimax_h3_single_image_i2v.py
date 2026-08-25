#!/usr/bin/env python3
"""Generate one MiniMax-H3 FL2AV video from a single image and prompt.

This is a single-input counterpart of minimax_h3_stage1_pair_render.py.  It
uses the original MiniMax-H3 checkpoint (no LoRA), pure-noise FL2AV sampling,
24 fps, and the same 20-step inference defaults.  The source image is directly
resized to the H3-aligned generation size before it is used as frame-0 keyframe.
The decoded video is aspect-fit and padded to the exact requested output size,
so postprocessing never crops or stretches generated content.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image


DEFAULT_DIFFSYNTH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "MiniMax/MiniMax-H3"
H3_FPS = 24
NEGATIVE_PROMPT = " "


def align_up(value: int, factor: int) -> int:
    if value <= 0:
        raise ValueError(f"Expected a positive size, got {value}")
    return math.ceil(value / factor) * factor


def validate_frame_count(frame_count: int) -> None:
    if frame_count < 5 or (frame_count - 5) % 17 != 0:
        raise ValueError("--num-frames must satisfy 17n+5, e.g. 22, 39, 124, or 175")


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\[time_range:[^\]]*\]", "", prompt).strip()


def direct_resize_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def fit_pad_resize_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    """Fit the entire frame inside the target; never crop or stretch it."""
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = min(width / src_w, height / src_h)
    resized_w = min(width, max(1, round(src_w * scale)))
    resized_h = min(height, max(1, round(src_h * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - resized_w) // 2, (height - resized_h) // 2))
    return canvas


def load_prompt(args: argparse.Namespace) -> str:
    value = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    prompt = normalize_prompt(value)
    if not prompt:
        raise ValueError("Prompt is empty")
    return prompt


def build_pipeline(args: argparse.Namespace):
    if str(args.diffsynth_root) not in sys.path:
        sys.path.insert(0, str(args.diffsynth_root))
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

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
    return MiniMaxH3Pipeline.from_pretrained(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a high-resolution MiniMax-H3 video from one image and prompt."
    )
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--width", type=int, default=1280, help="Exact output width.")
    parser.add_argument("--height", type=int, default=720, help="Exact output height.")
    parser.add_argument("--num-frames", type=int, default=39)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=H3_FPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram-limit-gb", type=float, default=60.0)
    parser.add_argument("--vram-reserve-gb", type=float, default=2.0)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument(
        "--keep-aligned-output",
        action="store_true",
        help="Write the aligned 1280x736 generation directly instead of fit-padding to 1280x720.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input_image.is_file():
        raise FileNotFoundError(args.input_image)
    if args.prompt_file is not None and not args.prompt_file.is_file():
        raise FileNotFoundError(args.prompt_file)
    if not args.diffsynth_root.is_dir():
        raise FileNotFoundError(args.diffsynth_root)
    if args.fps != H3_FPS:
        raise ValueError(f"MiniMax-H3 timing is fixed at {H3_FPS} fps")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("Output dimensions must be positive even numbers")
    validate_frame_count(args.num_frames)
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    if args.output_video.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output_video}")

    prompt = load_prompt(args)
    with Image.open(args.input_image) as image:
        source_width, source_height = image.size
        source_image = image.convert("RGB")
    generation_width = align_up(args.width, 32)
    generation_height = align_up(args.height, 32)
    keyframe = direct_resize_frame(source_image, generation_width, generation_height)
    output_width = generation_width if args.keep_aligned_output else args.width
    output_height = generation_height if args.keep_aligned_output else args.height

    print(f"[model] {args.model_id} (base checkpoint, no LoRA)")
    print(f"[input] {args.input_image} source={source_width}x{source_height}")
    print(
        f"[render] input_resize=direct generation={generation_width}x{generation_height}@{args.fps} "
        f"output={output_width}x{output_height} frames={args.num_frames} "
        f"steps={args.num_inference_steps} seed={args.seed}"
    )
    if not args.keep_aligned_output and (generation_width, generation_height) != (args.width, args.height):
        print("[spatial] output=aspect-fit-plus-pad (no crop, no stretch)")
    print(f"[prompt] {prompt}")
    if args.dry_run:
        return

    pipe = build_pipeline(args)
    video, audio = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=generation_height,
        width=generation_width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        audio_flow_shift=args.audio_flow_shift,
        tiled=not args.no_tiled,
        keyframes=[keyframe],
        keyframe_indices=[0],
    )
    if len(video) != args.num_frames:
        raise RuntimeError(f"Pipeline returned {len(video)} frames, expected {args.num_frames}")
    if not args.keep_aligned_output:
        video = [fit_pad_resize_frame(frame, args.width, args.height) for frame in video]

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    from diffsynth.utils.data.audio_video import write_video_audio

    write_video_audio(
        video=video,
        audio=None if args.no_audio else audio,
        output_path=str(args.output_video),
        fps=args.fps,
        audio_sample_rate=pipe.audio_vae.sample_rate,
        video_quality=8,
    )
    metadata = {
        "input_image": str(args.input_image),
        "output_video": str(args.output_video),
        "source_width": source_width,
        "source_height": source_height,
        "input_resize_mode": "direct_resize_to_h3_aligned_generation_size",
        "generation_width": generation_width,
        "generation_height": generation_height,
        "output_width": output_width,
        "output_height": output_height,
        "output_resize_mode": "aligned_direct" if args.keep_aligned_output else "aspect_fit_then_pad_no_crop_no_stretch",
        "num_frames": len(video),
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "flow_shift": args.flow_shift,
        "audio_flow_shift": args.audio_flow_shift,
        "has_audio": not args.no_audio,
        "model_id": args.model_id,
        "lora_path": None,
        "prompt": prompt,
    }
    metadata_path = args.output_video.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] video={args.output_video}")
    print(f"[saved] metadata={metadata_path}")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
