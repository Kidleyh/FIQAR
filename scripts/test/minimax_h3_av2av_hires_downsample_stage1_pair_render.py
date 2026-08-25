#!/usr/bin/env python3
"""Render original MiniMax-H3 AV-to-AV pairs at high resolution, then downsample.

For each sample in a source JSON list, this script creates one output folder
named after the source video stem and writes:

- render.mp4: output initialized from noised GT video and audio latents
- gt.mp4: source video resampled/cropped to exactly match the render
- sample.json: the original record plus render metadata

MiniMax-H3 runs at 24 fps, requires spatial sizes divisible by 32, and aligns
frame counts to 17n+5. Requested sizes are aligned before both GT preparation
and generation so the pair has identical dimensions and frame counts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_DIFFSYNTH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_JSON = Path(
    "/gemini/platform/public/aigc/human_guozz2/data/basemodel_train/"
    "classification/classification_0722/stage1/wuda_data.json"
)
DEFAULT_MODEL_ID = "MiniMax/MiniMax-H3"
DEFAULT_OUTPUT_DIR = Path("results/minimax_h3_av2av_hires_downsample_stage1_pairs")
H3_FPS = 24

# Keep the generation defaults aligned with MiniMax-H3-TI2VA.py. The pipeline's
# own default negative prompt is a single blank space.
NEGATIVE_PROMPT = " "

PROMPT_KEYS = (
    "prompt",
    "audio_video_description",
    "text",
    "caption",
    "audiovisual_caption",
    "audio_content",
)
VIDEO_KEYS = ("file_path", "video_path", "path", "video")
CAPTION_KEYS = ("video_caption_path", "caption_path", "prompt_path")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_text(data: Any, keys: tuple[str, ...] = PROMPT_KEYS) -> str | None:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            text = _first_text(value, keys)
            if text:
                return text
    elif isinstance(data, list):
        for value in data:
            text = _first_text(value, keys)
            if text:
                return text
    return None


def load_prompt_from_record(record: dict[str, Any]) -> str:
    text = _first_text(record)
    if text:
        return text
    for key in CAPTION_KEYS:
        caption_path = record.get(key)
        if isinstance(caption_path, str) and caption_path:
            path = Path(caption_path)
            if path.exists():
                text = _first_text(_load_json(path))
                if text:
                    return text
    raise KeyError(f"No prompt-like key found in record or caption path; keys={list(record.keys())}")


def normalize_prompt(prompt: str) -> str:
    """Match MiniMax-H3-TI2VA.py prompt preprocessing."""
    return re.sub(r"\[time_range:[^\]]*\]", "", prompt).strip()


def load_video_path_from_record(record: dict[str, Any]) -> Path:
    for key in VIDEO_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            return Path(value)
    raise KeyError(f"No video path key found; keys={list(record.keys())}")


def safe_folder_name(video_path: Path, used: set[str]) -> str:
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", video_path.stem).strip("._") or "sample"
    if name not in used:
        used.add(name)
        return name
    base = name
    suffix = 2
    while f"{base}_{suffix}" in used:
        suffix += 1
    name = f"{base}_{suffix}"
    used.add(name)
    return name


def get_video_info(video_path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Unable to read video metadata: {video_path}")
    return frame_count, fps, width, height


def align_up(value: int, factor: int) -> int:
    if value <= 0:
        raise ValueError(f"Expected a positive size, got {value}")
    return math.ceil(value / factor) * factor


def h3_supported_frame_count(
    raw_frame_count: int,
    raw_fps: float,
    requested: str,
    target_fps: int = H3_FPS,
) -> int:
    available = max(1, int(math.floor(raw_frame_count / raw_fps * target_fps)))
    candidate = available if requested == "auto" else min(available, int(requested))
    frame_count = ((candidate - 5) // 17) * 17 + 5
    if frame_count < 5:
        raise ValueError(
            "Video is too short for MiniMax-H3 frame rule 17n+5: "
            f"raw_frame_count={raw_frame_count}, raw_fps={raw_fps:.6f}, target_fps={target_fps}"
        )
    return frame_count


def get_ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError("ffmpeg not found in PATH and imageio_ffmpeg is unavailable") from exc


def trim_gt_video(src: Path, dst: Path, frame_count: int, fps: int, width: int, height: int) -> None:
    if dst.exists():
        try:
            existing_frames, existing_fps, existing_w, existing_h = get_video_info(dst)
            if (
                existing_frames == frame_count
                and abs(existing_fps - fps) < 0.01
                and existing_w == width
                and existing_h == height
            ):
                return
        except Exception:
            pass

    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = get_ffmpeg_exe()
    video_filter = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"trim=end_frame={frame_count},setpts=PTS-STARTPTS"
    )
    duration = frame_count / fps
    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        video_filter,
        "-af",
        f"atrim=end={duration:.9f},apad=whole_dur={duration:.9f},asetpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-frames:v",
        str(frame_count),
        "-c:a",
        "aac",
        "-t",
        f"{duration:.9f}",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        fallback = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            video_filter,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-frames:v",
            str(frame_count),
            str(dst),
        ]
        subprocess.run(fallback, check=True)

    actual_frames, actual_fps, actual_w, actual_h = get_video_info(dst)
    if (
        actual_frames != frame_count
        or abs(actual_fps - fps) >= 0.01
        or actual_w != width
        or actual_h != height
    ):
        raise RuntimeError(
            f"Aligned GT verification failed for {dst}: "
            f"got {actual_frames} frames at {actual_w}x{actual_h}@{actual_fps:.6f}, "
            f"expected {frame_count} frames at {width}x{height}@{fps}"
        )


def read_first_frame(video_path: Path, width: int, height: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read first frame: {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    src_h, src_w = frame.shape[:2]
    target_aspect = width / height
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        crop_w = int(src_h * target_aspect)
        left = (src_w - crop_w) // 2
        frame = frame[:, left : left + crop_w]
    else:
        crop_h = int(src_w / target_aspect)
        top = (src_h - crop_h) // 2
        frame = frame[top : top + crop_h, :]
    return Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS).convert("RGB")


def read_aligned_gt_video_audio(
    path: Path,
    frame_count: int,
    fps: int,
    width: int,
    height: int,
    audio_sample_rate: int,
) -> tuple[list[Image.Image], torch.Tensor, int]:
    """Read aligned GT frames/audio without the environment's broken TorchCodec."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open aligned GT: {path}")
    frames: list[Image.Image] = []
    try:
        while len(frames) < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                raise RuntimeError(
                    f"Aligned GT frame is {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}: {path}"
                )
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    if len(frames) != frame_count:
        raise RuntimeError(f"Decoded GT has {len(frames)} frames, expected {frame_count}: {path}")

    duration = frame_count / fps
    cmd = [
        get_ffmpeg_exe(),
        "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-t", f"{duration:.9f}",
        "-ac", "2", "-ar", str(audio_sample_rate),
        "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Unable to decode GT audio: {path}: {message}")
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if samples.size == 0 or samples.size % 2:
        raise RuntimeError(f"GT has no readable stereo audio samples: {path}")
    waveform = torch.from_numpy(samples.reshape(-1, 2).T.copy())
    return frames, waveform, audio_sample_rate


def center_crop_resize_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop a generated frame to the output aspect ratio and downsample."""
    image = image.convert("RGB")
    src_w, src_h = image.size
    target_aspect = width / height
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        crop_w = max(1, round(src_h * target_aspect))
        left = (src_w - crop_w) // 2
        image = image.crop((left, 0, left + crop_w, src_h))
    elif src_aspect < target_aspect:
        crop_h = max(1, round(src_w / target_aspect))
        top = (src_h - crop_h) // 2
        image = image.crop((0, top, src_w, top + crop_h))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def iter_records(data_json: Path, start: int, limit: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    data = _load_json(data_json)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise TypeError(f"Expected list-like dataset JSON, got {type(data).__name__}: {data_json}")
    end = len(data) if limit is None else min(len(data), start + limit)
    for idx in range(start, end):
        item = data[idx]
        if not isinstance(item, dict):
            print(f"[skip] idx={idx}: expected dict item, got {type(item).__name__}")
            continue
        yield idx, item


def build_pipeline(args: argparse.Namespace):
    if str(args.diffsynth_root) not in sys.path:
        sys.path.insert(0, str(args.diffsynth_root))
    from diffsynth.core import load_state_dict
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.minimax_h3_audio_video_av2av import MiniMaxH3AV2AVPipeline

    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"

    free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
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
    pipe = MiniMaxH3AV2AVPipeline.from_pretrained(
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
    if args.lora_path is not None:
        if not args.lora_path.is_file():
            raise FileNotFoundError(args.lora_path)
        if not getattr(pipe.dit, "vram_management_enabled", False):
            raise RuntimeError(
                "Turbo LoRA requires DiffSynth VRAM management so it can be applied "
                "at runtime instead of being fused into bf16 base weights."
            )
        print(
            f"[lora] loading={args.lora_path} strength={args.lora_strength} "
            "storage_device=cpu"
        )
        # Keep hot-loaded LoRA tensors on CPU. LoRAHotLoadMixin transfers each
        # layer's tensors to the active device on demand; loading the whole
        # state dict directly on CUDA needlessly consumes another ~744 MiB and
        # can fail before inference when the base model already fills the GPU.
        lora_state_dict = load_state_dict(
            str(args.lora_path),
            torch_dtype=pipe.torch_dtype,
            device="cpu",
        )
        pipe.load_lora(
            pipe.dit,
            alpha=args.lora_strength,
            hotload=True,
            state_dict=lora_state_dict,
        )
        del lora_state_dict
        patched = sum(
            bool(getattr(module, "lora_A_weights", []))
            for module in pipe.dit.modules()
        )
        if patched == 0:
            raise RuntimeError(f"LoRA loaded but patched zero DiT modules: {args.lora_path}")
        print(f"[lora] patched_modules={patched}")
    return pipe


def validate_requested_frames(value: str) -> None:
    if value == "auto":
        return
    try:
        requested = int(value)
    except ValueError as exc:
        raise ValueError("--num-frames must be auto or an integer") from exc
    if requested < 5 or (requested - 5) % 17 != 0:
        raise ValueError("--num-frames must satisfy MiniMax-H3 frame rule 17n+5, e.g. 22, 39, 124, 175")


def main(
    *,
    default_steps: int = 20,
    default_lora_path: Path | None = None,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    turbo_mode: bool = False,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render GT/render stage-1 classification pairs with MiniMax-H3 Turbo AV-to-AV"
            if turbo_mode
            else "Render GT/render stage-1 classification pairs with original MiniMax-H3 AV-to-AV"
        )
    )
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--height", type=int, default=576, help="Internal H3 generation height.")
    parser.add_argument("--width", type=int, default=1024, help="Internal H3 generation width.")
    parser.add_argument("--output-height", type=int, default=416)
    parser.add_argument("--output-width", type=int, default=736)
    parser.add_argument(
        "--num-frames",
        default="auto",
        help="auto or an integer satisfying 17n+5; auto uses the largest supported count within the GT duration",
    )
    parser.add_argument("--num-inference-steps", type=int, default=default_steps)
    parser.add_argument("--fps", type=int, default=H3_FPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument(
        "--denoising-strength",
        type=float,
        default=0.1,
        help="GT video noise strength before FlowMatch shift; lower values preserve more GT structure.",
    )
    parser.add_argument(
        "--audio-denoising-strength",
        type=float,
        help="GT audio noise strength; defaults to --denoising-strength.",
    )
    parser.add_argument("--lora-path", type=Path, default=default_lora_path)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram-limit-gb", type=float)
    parser.add_argument("--vram-reserve-gb", type=float, default=2.0)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model downloads. By default downloads are disabled because models may be a shared symlink.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate records and report aligned shapes without loading the model or writing outputs.",
    )
    args = parser.parse_args()

    validate_requested_frames(args.num_frames)
    if args.fps != H3_FPS:
        raise ValueError(f"MiniMax-H3 audio/video timing is fixed at {H3_FPS} fps; got --fps={args.fps}")
    if args.start < 0 or args.limit is not None and args.limit < 0:
        raise ValueError("--start and --limit must be non-negative")
    if args.output_width <= 0 or args.output_height <= 0:
        raise ValueError("--output-width and --output-height must be positive")
    if args.output_width % 2 or args.output_height % 2:
        raise ValueError("H.264 output dimensions must be even")
    if args.lora_strength <= 0:
        raise ValueError("--lora-strength must be positive")
    if not 0.0 < args.denoising_strength <= 1.0:
        raise ValueError("--denoising-strength must be in (0, 1]")
    if args.audio_denoising_strength is not None and not 0.0 < args.audio_denoising_strength <= 1.0:
        raise ValueError("--audio-denoising-strength must be in (0, 1]")
    if turbo_mode:
        if not 4 <= args.num_inference_steps <= 8:
            raise ValueError("MiniMax-H3 Turbo requires --num-inference-steps in the range 4..8")
        if args.num_inference_steps < 6:
            print("[warning] Turbo 4–5 steps are faster, but 6–8 steps are recommended for quality.")
    if not args.data_json.is_file():
        raise FileNotFoundError(args.data_json)
    if not args.diffsynth_root.is_dir():
        raise FileNotFoundError(args.diffsynth_root)

    generation_width = align_up(args.width, 32)
    generation_height = align_up(args.height, 32)
    output_width = args.output_width
    output_height = args.output_height
    records = list(iter_records(args.data_json, args.start, args.limit))
    print(f"[model] {args.model_id}")
    print(f"[data] {args.data_json}")
    print(f"[output] {args.output_dir}")
    if args.lora_path is not None:
        print(f"[lora] {args.lora_path} strength={args.lora_strength}")
    print(
        f"[render] fps={args.fps}, requested_generation_size={args.width}x{args.height}, "
        f"aligned_generation_size={generation_width}x{generation_height}, "
        f"output_size={output_width}x{output_height}, steps={args.num_inference_steps}, "
        f"cfg_scale={args.cfg_scale}, num_frames={args.num_frames}"
    )
    audio_strength = (
        args.denoising_strength
        if args.audio_denoising_strength is None
        else args.audio_denoising_strength
    )
    video_sigma = float(args.flow_shift * args.denoising_strength / (1 + (args.flow_shift - 1) * args.denoising_strength))
    audio_sigma = float(args.audio_flow_shift * audio_strength / (1 + (args.audio_flow_shift - 1) * audio_strength))
    print(
        f"[av2av] video_strength={args.denoising_strength} initial_sigma={video_sigma:.6f}, "
        f"audio_strength={audio_strength} initial_sigma={audio_sigma:.6f}"
    )

    if args.dry_run:
        valid = 0
        for idx, record in records:
            try:
                video_path = load_video_path_from_record(record)
                prompt = normalize_prompt(load_prompt_from_record(record))
                raw_frames, raw_fps, raw_w, raw_h = get_video_info(video_path)
                frames = h3_supported_frame_count(raw_frames, raw_fps, args.num_frames, args.fps)
                print(
                    f"[dry-run] idx={idx} video={video_path} source={raw_w}x{raw_h}@{raw_fps:.3f} "
                    f"source_frames={raw_frames} generation={generation_width}x{generation_height} "
                    f"output={output_width}x{output_height}@{args.fps} "
                    f"render_frames={frames} prompt_chars={len(prompt)}"
                )
                valid += 1
            except Exception as exc:
                print(f"[skip] idx={idx}: {type(exc).__name__}: {exc}")
        print(f"[done] dry_run_valid={valid} total={len(records)}")
        if records and valid == 0:
            raise SystemExit(1)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(args)
    from diffsynth.utils.data.audio_video import write_video_audio

    used_names: set[str] = set()
    rendered = 0
    failed = 0
    skipped_existing = 0
    for idx, record in records:
        try:
            video_path = load_video_path_from_record(record)
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            prompt = normalize_prompt(load_prompt_from_record(record))
            raw_frames, raw_fps, raw_w, raw_h = get_video_info(video_path)
            frame_count = h3_supported_frame_count(raw_frames, raw_fps, args.num_frames, args.fps)
            folder = args.output_dir / safe_folder_name(video_path, used_names)
            render_path = folder / "render.mp4"
            gt_path = folder / "gt.mp4"
            sample_path = folder / "sample.json"
            if args.skip_existing and render_path.exists() and gt_path.exists() and sample_path.exists():
                print(f"[skip-existing] idx={idx} {folder.name}")
                skipped_existing += 1
                continue

            folder.mkdir(parents=True, exist_ok=True)
            trim_gt_video(video_path, gt_path, frame_count, args.fps, output_width, output_height)
            with tempfile.TemporaryDirectory(prefix=f"minimax_h3_av2av_{idx}_") as temp_dir:
                conditioning_path = Path(temp_dir) / "conditioning_gt.mp4"
                trim_gt_video(
                    video_path,
                    conditioning_path,
                    frame_count,
                    args.fps,
                    generation_width,
                    generation_height,
                )
                gt_video, gt_audio, gt_audio_sample_rate = read_aligned_gt_video_audio(
                    conditioning_path,
                    frame_count=frame_count,
                    fps=args.fps,
                    width=generation_width,
                    height=generation_height,
                    audio_sample_rate=pipe.audio_vae.sample_rate,
                )
            first_frame = gt_video[0]

            print(f"[case] idx={idx} name={folder.name}")
            print(f"  gt={video_path}")
            print(
                f"  source={raw_w}x{raw_h}@{raw_fps:.3f} source_frames={raw_frames} "
                f"generation={generation_width}x{generation_height} "
                f"output={output_width}x{output_height}@{args.fps} render_frames={frame_count}"
            )
            print(f"  prompt={prompt}")
            video, audio = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=generation_height,
                width=generation_width,
                num_frames=frame_count,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                flow_shift=args.flow_shift,
                audio_flow_shift=args.audio_flow_shift,
                denoising_strength=args.denoising_strength,
                audio_denoising_strength=args.audio_denoising_strength,
                tiled=not args.no_tiled,
                input_video=gt_video,
                input_audio=gt_audio,
                input_audio_sample_rate=gt_audio_sample_rate,
                keyframes=[first_frame],
                keyframe_indices=[0],
            )
            if len(video) != frame_count:
                raise RuntimeError(f"Pipeline returned {len(video)} frames, expected {frame_count}")
            video = [
                center_crop_resize_frame(frame, output_width, output_height)
                for frame in video
            ]
            write_video_audio(
                video=video,
                audio=audio,
                output_path=str(render_path),
                fps=args.fps,
                audio_sample_rate=pipe.audio_vae.sample_rate,
                video_quality=8,
            )

            sample_record = dict(record)
            sample_record.update(
                {
                    "render_frame_count": len(video),
                    "render_fps": args.fps,
                    "render_width": output_width,
                    "render_height": output_height,
                    "generation_width": generation_width,
                    "generation_height": generation_height,
                    "render_downsample_method": "center_crop_lanczos",
                    "source_frame_count": raw_frames,
                    "source_fps": raw_fps,
                    "source_width": raw_w,
                    "source_height": raw_h,
                    "render_model": args.model_id,
                    "render_num_inference_steps": args.num_inference_steps,
                    "render_seed": args.seed,
                    "render_cfg_scale": args.cfg_scale,
                    "render_flow_shift": args.flow_shift,
                    "render_audio_flow_shift": args.audio_flow_shift,
                    "render_init_mode": "gt_audio_video_latents_plus_noise",
                    "render_denoising_strength": args.denoising_strength,
                    "render_audio_denoising_strength": audio_strength,
                    "render_initial_video_sigma": video_sigma,
                    "render_initial_audio_sigma": audio_sigma,
                    "render_lora_path": None if args.lora_path is None else str(args.lora_path),
                    "render_lora_strength": None if args.lora_path is None else args.lora_strength,
                }
            )
            with sample_path.open("w", encoding="utf-8") as f:
                json.dump(sample_record, f, ensure_ascii=False, indent=2)

            rendered += 1
            print(f"  saved={render_path}")
        except Exception as exc:
            failed += 1
            print(f"[skip] idx={idx}: {type(exc).__name__}: {exc}")
            del exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[done] rendered={rendered} skipped_existing={skipped_existing} "
        f"failed={failed} total={len(records)}"
    )
    if records and rendered == 0 and skipped_existing == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
