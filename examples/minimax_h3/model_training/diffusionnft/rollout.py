#!/usr/bin/env python3
"""Generate MiniMax-H3 multi-seed rollouts and score them with MagFace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
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
from reward_face_quality import (  # noqa: E402
    DEFAULT_EVALUATOR,
    DEFAULT_MAGFACE_CHECKPOINT,
    DEFAULT_SCRFD_MODEL,
    FaceQualityReward,
)


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


def _valid_complete_state(
    state_path: Path, expected: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            return None, "state is not an object"
        for key, value in expected.items():
            if state.get(key) != value:
                return None, f"{key} mismatch: {state.get(key)!r} != {value!r}"
        if state.get("format_version") != 1 or state.get("complete") is not True:
            return None, "state is not a committed format-version 1 sample"
        latent_path = Path(state["latent_path"]).resolve()
        condition_path = Path(state["condition_image_path"]).resolve()
        for path, label in ((latent_path, "latent"), (condition_path, "condition")):
            if not path.is_file() or path.stat().st_size <= 0:
                return None, f"missing or empty {label}: {path}"
        if _sha256(latent_path) != state.get("latent_sha256"):
            return None, "latent sha256 mismatch"
        if _sha256(condition_path) != state.get("condition_image_sha256"):
            return None, "condition sha256 mismatch"
        reward = state.get("reward")
        ratio = state.get("face_visible_ratio")
        faces = state.get("num_faces")
        quality = state.get("mean_quality")
        if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            return None, "invalid reward"
        if quality is not None and (
            not isinstance(quality, (int, float)) or not math.isfinite(float(quality))
        ):
            return None, "invalid mean_quality"
        if not isinstance(ratio, (int, float)) or not 0 <= float(ratio) <= 1:
            return None, "invalid face_visible_ratio"
        if not isinstance(faces, int) or isinstance(faces, bool) or faces < 0:
            return None, "invalid num_faces"
        return state, "valid"
    except (OSError, TypeError, ValueError, KeyError) as error:
        return None, str(error)


def _resolve_policy_provenance(args: argparse.Namespace) -> dict[str, Any]:
    lora_path = (
        args.lora_path.expanduser().resolve() if args.lora_path is not None else None
    )
    source_checkpoint = (
        args.source_checkpoint.expanduser().resolve()
        if args.source_checkpoint is not None
        else None
    )
    if lora_path is not None and not lora_path.is_file():
        raise FileNotFoundError(lora_path)
    actual_sha256 = _sha256(lora_path) if lora_path is not None else None
    if args.policy_lora_sha256 is not None and args.policy_lora_sha256 != actual_sha256:
        raise ValueError(
            "--policy-lora-sha256 does not match --lora-path: "
            f"expected={args.policy_lora_sha256} actual={actual_sha256}"
        )
    if args.policy_role == "base":
        if lora_path is not None or source_checkpoint is not None:
            raise ValueError("Base-policy rollout cannot use LoRA or a source checkpoint")
        if args.global_step_before != 0:
            raise ValueError("Base-policy rollout requires --global-step-before 0")
    elif args.policy_role == "old":
        if lora_path is None or source_checkpoint is None:
            raise ValueError("Old-policy rollout requires --lora-path and --source-checkpoint")
        if args.global_step_before is None or args.global_step_before <= 0:
            raise ValueError("Old-policy rollout requires a positive --global-step-before")
        expected_old = (source_checkpoint / "old_lora.safetensors").resolve()
        if lora_path != expected_old:
            raise ValueError(
                f"Old-policy LoRA must be the source checkpoint old adapter: {expected_old}"
            )
    elif any(
        value is not None
        for value in (
            args.policy_lora_sha256,
            args.source_checkpoint,
            args.global_step_before,
        )
    ):
        raise ValueError("Policy provenance fields require --policy-role base or old")
    return {
        "policy_role": args.policy_role,
        "policy_lora_path": str(lora_path) if lora_path is not None else None,
        "policy_lora_sha256": actual_sha256,
        "source_checkpoint": (
            str(source_checkpoint) if source_checkpoint is not None else None
        ),
        "global_step_before": args.global_step_before,
    }


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
    parser.add_argument(
        "--policy-role", choices=("unspecified", "base", "old"), default="unspecified"
    )
    parser.add_argument("--policy-lora-sha256", default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--global-step-before", type=int, default=None)
    parser.add_argument("--reward-evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--scrfd-model", type=Path, default=DEFAULT_SCRFD_MODEL)
    parser.add_argument(
        "--magface-checkpoint", type=Path, default=DEFAULT_MAGFACE_CHECKPOINT
    )
    parser.add_argument("--reward-frame-stride", type=int, default=25)
    parser.add_argument("--reward-max-frames", type=int, default=0)
    parser.add_argument(
        "--reward-frame-face-aggregation",
        choices=("mean", "min", "max"),
        default="mean",
    )
    parser.add_argument("--missing-face-reward", type=float, default=0.0)
    parser.add_argument(
        "--save-rollout-video",
        action="store_true",
        help="Write debug mp4 artifacts; formal training defaults to latent/state only.",
    )
    parser.add_argument(
        "--engineering-stop-after-state-seed", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--engineering-stop-after-latent-seed", type=int, default=None, help=argparse.SUPPRESS
    )
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
    policy_provenance = _resolve_policy_provenance(args)
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
    reward_init_started = time.monotonic()
    reward_model = FaceQualityReward(
        evaluator_path=args.reward_evaluator,
        scrfd_model=args.scrfd_model,
        magface_checkpoint=args.magface_checkpoint,
        device=args.device,
    )
    reward_init_seconds = time.monotonic() - reward_init_started
    print(
        f"[reward-inmemory] model_init_seconds={reward_init_seconds:.6f}", flush=True
    )
    if args.save_rollout_video:
        from diffsynth.utils.data.audio_video import write_video_audio

    rollouts: list[dict[str, Any]] = []
    reward_wall_time = reward_init_seconds
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
        condition_temporary = condition_path.with_suffix(
            condition_path.suffix + f".tmp-{os.getpid()}"
        )
        first_frame.save(condition_temporary, format="PNG")
        condition_temporary.replace(condition_path)
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
            state_path = (prompt_dir / f"seed_{seed}_state.json").resolve()
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            expected_state = {
                "format_version": 1,
                "complete": True,
                "seed": seed,
                "latent_path": str(latent_path),
                "condition_image_path": str(condition_path),
                "condition_image_sha256": condition_sha256,
                "prompt": prompt,
                "prompt_sha256": prompt_sha256,
                **policy_provenance,
                "height": render_height,
                "width": render_width,
                "num_frames": frame_count,
                "num_inference_steps": args.num_inference_steps,
                "flow_shift": args.flow_shift,
                "audio_flow_shift": args.audio_flow_shift,
                "save_rollout_video": args.save_rollout_video,
                "video_path": str(video_path) if args.save_rollout_video else None,
            }
            state = None
            if args.skip_existing and state_path.is_file():
                state, reason = _valid_complete_state(state_path, expected_state)
                if state is not None and args.save_rollout_video and (
                    not video_path.is_file() or video_path.stat().st_size <= 0
                ):
                    print(
                        f"[rollout-resume] seed={seed} formal_state_valid=true "
                        "debug_video_missing=true regenerate_for_requested_video=true",
                        flush=True,
                    )
                    state = None
                elif state is not None:
                    print(
                        f"[rollout-resume] reuse seed={seed} state={state_path}",
                        flush=True,
                    )
                else:
                    print(
                        f"[rollout-resume] reject seed={seed} reason={reason}", flush=True
                    )
            if state is None:
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                reward_started = time.monotonic()
                reward = reward_model.score_frames(
                    video,
                    frame_stride=args.reward_frame_stride,
                    max_frames_per_video=args.reward_max_frames,
                    frame_face_aggregation=args.reward_frame_face_aggregation,
                    missing_face_reward=args.missing_face_reward,
                )
                elapsed = time.monotonic() - reward_started
                reward_wall_time += elapsed
                print(
                    f"[reward-inmemory] seed={seed} reward={reward['reward']:.8f} "
                    f"elapsed_seconds={elapsed:.6f}",
                    flush=True,
                )
                _write_latents(latent_path, clean_latents)
                if args.engineering_stop_after_latent_seed == seed:
                    raise RuntimeError(
                        f"Engineering stop after latent before state for seed={seed}"
                    )
                if args.save_rollout_video:
                    write_video_audio(
                        video=video,
                        audio=audio,
                        output_path=str(video_path),
                        fps=H3_FPS,
                        audio_sample_rate=pipe.audio_vae.sample_rate,
                        video_quality=8,
                    )
                state = {
                    **expected_state,
                    "reward": float(reward["reward"]),
                    "mean_quality": reward["mean_quality"],
                    "face_visible_ratio": float(reward["face_visible_ratio"]),
                    "num_faces": int(reward["num_faces"]),
                    "latent_sha256": _sha256(latent_path),
                    "fps": H3_FPS,
                    "reward_backend": "in_memory",
                }
                _write_json(state_path, state)
                if args.engineering_stop_after_state_seed == seed:
                    raise RuntimeError(f"Engineering stop after state for seed={seed}")
                del video, audio, clean_latents
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            result = {**state, "sample_state_path": str(state_path)}
            rollouts.append(result)
            print(
                f"[result] seed={result['seed']} reward={result['reward']:.6f} "
                f"visible_ratio={result['face_visible_ratio']:.3f} "
                f"num_faces={result['num_faces']} video={result['video_path']} "
                f"state={state_path}",
                flush=True,
            )

    rollout_path = args.output_dir / "rollout.json"
    _write_json(rollout_path, rollouts)
    print(
        f"[reward-inmemory-summary] reward_wall_time_seconds={reward_wall_time:.6f}",
        flush=True,
    )
    del reward_model, pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[done] rollout_json={rollout_path}", flush=True)


if __name__ == "__main__":
    main()
