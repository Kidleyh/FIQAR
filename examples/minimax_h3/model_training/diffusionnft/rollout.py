#!/usr/bin/env python3
"""Generate MiniMax-H3 multi-seed rollouts and score them with MagFace."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[4]
INFERENCE_HELPER_DIR = REPO_ROOT / "scripts" / "test"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(INFERENCE_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_HELPER_DIR))

from minimax_h3_stage1_pair_render import (  # noqa: E402
    DEFAULT_MODEL_ID,
    H3_FPS,
    NEGATIVE_PROMPT,
    align_up,
    build_pipeline,
    get_video_info,
    h3_supported_frame_count,
    iter_records,
    load_prompt_from_record,
    load_video_path_from_record,
    normalize_prompt,
    read_first_frame,
    safe_folder_name,
    validate_requested_frames,
)
from reward_face_quality import DEFAULT_EVALUATOR, evaluate_face_quality  # noqa: E402


DEFAULT_DATA_JSON = Path(
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/"
    "guangdian_20251114_small_clear_faces/guangdian_20251114_small_clear_faces.json"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_latents(path: Path, latents: dict[str, torch.Tensor]) -> None:
    required = {"video_latents", "audio_latents"}
    if set(latents) != required:
        raise ValueError(f"Unexpected rollout latent keys: {sorted(latents)}")
    state = {
        key: value.detach().to("cpu").contiguous()
        for key, value in latents.items()
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(state, str(temporary))
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_seeds(num_samples: int | None, seeds: list[int] | None) -> list[int]:
    if seeds is not None:
        if not seeds:
            raise ValueError("--seeds requires at least one value")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Seeds must be unique: {seeds}")
        if num_samples is not None and num_samples != len(seeds):
            raise ValueError(
                f"--num-samples={num_samples} does not match {len(seeds)} explicit --seeds"
            )
        resolved = seeds
    else:
        count = 2 if num_samples is None else num_samples
        if count <= 0:
            raise ValueError("--num-samples must be positive")
        resolved = list(range(count))
    if any(seed < 0 for seed in resolved):
        raise ValueError(f"Seeds must be non-negative: {resolved}")
    return resolved


def _resolve_records(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    if args.prompt is not None or args.input_video is not None:
        if args.prompt is None or args.input_video is None:
            raise ValueError("--prompt and --input-video must be provided together")
        return [(0, {"prompt": args.prompt, "file_path": str(args.input_video)})]
    if not args.data_json.is_file():
        raise FileNotFoundError(args.data_json)
    return list(iter_records(args.data_json, args.start, args.limit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-H3 current-policy rollout plus MagFace reward.")
    parser.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", default="124")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram-limit-gb", type=float, default=None)
    parser.add_argument("--vram-reserve-gb", type=float, default=2.0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--reward-evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--reward-python", default=sys.executable)
    parser.add_argument("--conda-executable", default="/root/miniconda3/bin/conda")
    parser.add_argument("--reward-frame-stride", type=int, default=25)
    parser.add_argument("--reward-max-frames", type=int, default=0)
    parser.add_argument("--missing-face-reward", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 0 or args.limit <= 0:
        raise ValueError("--start must be non-negative and --limit must be positive")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    if args.lora_strength <= 0:
        raise ValueError("--lora-strength must be positive")
    validate_requested_frames(args.num_frames)
    seeds = _resolve_seeds(args.num_samples, args.seeds)
    records = _resolve_records(args)
    if not records:
        raise RuntimeError("No rollout records were selected")

    args.diffsynth_root = REPO_ROOT
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_width = align_up(args.width, 32)
    render_height = align_up(args.height, 32)
    print(
        f"[rollout] prompts={len(records)} samples_per_prompt={len(seeds)} "
        f"seeds={seeds} output={args.output_dir}",
        flush=True,
    )
    print(
        f"[render] size={render_width}x{render_height} frames={args.num_frames} "
        f"steps={args.num_inference_steps}",
        flush=True,
    )

    pipe = build_pipeline(args)
    from diffsynth.utils.data.audio_video import write_video_audio

    generated: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for record_index, record in records:
        source_video = load_video_path_from_record(record).expanduser().resolve()
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        prompt = normalize_prompt(load_prompt_from_record(record))
        raw_frames, raw_fps, _, _ = get_video_info(source_video)
        frame_count = h3_supported_frame_count(
            raw_frames, raw_fps, args.num_frames, H3_FPS
        )
        first_frame = read_first_frame(source_video, render_width, render_height)
        folder_name = safe_folder_name(source_video, used_names)
        prompt_dir = args.output_dir / f"prompt_{record_index:06d}_{folder_name}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        condition_path = (prompt_dir / "condition_image.png").resolve()
        first_frame.save(condition_path, format="PNG")
        with Image.open(condition_path) as saved_condition:
            condition_image = saved_condition.convert("RGB").copy()
        condition_sha256 = _sha256(condition_path)
        if condition_image.size != (render_width, render_height):
            raise RuntimeError(
                f"Saved condition size {condition_image.size} does not match "
                f"rollout size {(render_width, render_height)}"
            )

        for seed in seeds:
            video_path = (prompt_dir / f"seed_{seed}.mp4").resolve()
            latent_path = (prompt_dir / f"seed_{seed}_latents.safetensors").resolve()
            if (
                args.skip_existing
                and video_path.is_file()
                and video_path.stat().st_size > 0
                and latent_path.is_file()
                and latent_path.stat().st_size > 0
            ):
                print(f"[rollout] reuse seed={seed} video={video_path}", flush=True)
            else:
                print(f"[rollout] generate record={record_index} seed={seed}", flush=True)
                video, audio, clean_latents = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=render_height,
                    width=render_width,
                    num_frames=frame_count,
                    num_inference_steps=args.num_inference_steps,
                    seed=seed,
                    cfg_scale=args.cfg_scale,
                    flow_shift=args.flow_shift,
                    audio_flow_shift=args.audio_flow_shift,
                    tiled=not args.no_tiled,
                    keyframes=[condition_image],
                    keyframe_indices=[0],
                    return_latents=True,
                )
                if len(video) != frame_count:
                    raise RuntimeError(
                        f"Pipeline returned {len(video)} frames, expected {frame_count}"
                    )
                write_video_audio(
                    video=video,
                    audio=audio,
                    output_path=str(video_path),
                    fps=H3_FPS,
                    audio_sample_rate=pipe.audio_vae.sample_rate,
                    video_quality=8,
                )
                _write_latents(latent_path, clean_latents)
            generated.append(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "video_path": str(video_path),
                    "latent_path": str(latent_path),
                    "condition_image_path": str(condition_path),
                    "condition_image_sha256": condition_sha256,
                    "height": render_height,
                    "width": render_width,
                    "num_frames": frame_count,
                    "fps": H3_FPS,
                    "num_inference_steps": args.num_inference_steps,
                    "flow_shift": args.flow_shift,
                    "audio_flow_shift": args.audio_flow_shift,
                }
            )

    # The external evaluator needs only the rendered files. Release H3 model
    # allocations before SCRFD and MagFace enter their own CUDA environments.
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reward_dir = args.output_dir / "reward_eval"
    rewards = evaluate_face_quality(
        [item["video_path"] for item in generated],
        reward_dir,
        evaluator_path=args.reward_evaluator,
        python_executable=args.reward_python,
        conda_executable=args.conda_executable,
        frame_stride=args.reward_frame_stride,
        max_frames_per_video=args.reward_max_frames,
        missing_face_reward=args.missing_face_reward,
    )

    rollouts = []
    for item in generated:
        reward = rewards[str(Path(item["video_path"]).resolve())]
        result = {
            **item,
            "reward": reward["reward"],
            "mean_quality": reward["mean_quality"],
            "face_visible_ratio": reward["face_visible_ratio"],
            "num_faces": reward["num_faces"],
        }
        rollouts.append(result)
        print(
            f"[result] seed={result['seed']} reward={result['reward']:.6f} "
            f"visible_ratio={result['face_visible_ratio']:.3f} "
            f"num_faces={result['num_faces']} video={result['video_path']}",
            flush=True,
        )

    rollout_path = args.output_dir / "rollout.json"
    _write_json(rollout_path, rollouts)
    print(f"[done] rollout_json={rollout_path}", flush=True)


if __name__ == "__main__":
    main()
