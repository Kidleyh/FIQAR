#!/usr/bin/env python3
"""Numerical regression: legacy mp4 evaluator versus in-memory reward API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from PIL import Image

from reward_face_quality import (
    FaceQualityReward,
    evaluate_face_quality_from_video_files,
)


PHASE7_PROMPT_DIR = Path(
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/"
    "DiffSynth-Studio-minimaxh3/outputs/minimax_h3_diffusionnft_online/"
    "phase7_engineering_20iter_20260827/iteration_000000/rollout/"
    "prompt_000000_20251119_guangdian_b6_folder_01_BV11DVozQEar_00000"
)


def decode_rgb_frames(path: Path) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    frames: list[Image.Image] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path, nargs="*")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/minimax_h3_diffusionnft_reward_compat_phase8"),
    )
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--frame-face-aggregation", choices=("mean", "min", "max"), default="mean")
    parser.add_argument(
        "--atol",
        type=float,
        default=0.05,
        help=(
            "Absolute quality tolerance for CUDA-provider floating-point landmark "
            "drift; face counts and visibility must remain exact."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = args.videos or [PHASE7_PROMPT_DIR / f"seed_{seed}.mp4" for seed in range(3)]
    videos = [path.expanduser().resolve() for path in videos]
    if len(videos) < 3:
        raise ValueError("Compatibility regression requires at least three videos")
    legacy_started = time.monotonic()
    legacy = evaluate_face_quality_from_video_files(
        videos,
        args.output_dir / "legacy",
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        frame_face_aggregation=args.frame_face_aggregation,
        keep_aligned_faces=True,
    )
    legacy_wall_time = time.monotonic() - legacy_started
    in_memory_started = time.monotonic()
    scorer = FaceQualityReward()
    rows = []
    for path in videos:
        current = scorer.score_frames(
            decode_rgb_frames(path),
            frame_stride=args.frame_stride,
            max_frames_per_video=args.max_frames_per_video,
            frame_face_aggregation=args.frame_face_aggregation,
        )
        reference = legacy[str(path)]
        quality_delta = (
            None
            if current["mean_quality"] is None or reference["mean_quality"] is None
            else abs(float(current["mean_quality"]) - float(reference["mean_quality"]))
        )
        reward_delta = abs(float(current["reward"]) - float(reference["reward"]))
        visible_delta = abs(
            float(current["face_visible_ratio"])
            - float(reference["face_visible_ratio"])
        )
        matches = (
            current["num_faces"] == reference["num_faces"]
            and visible_delta <= 1e-12
            and reward_delta <= args.atol
            and (quality_delta is None or quality_delta <= args.atol)
        )
        row = {
            "video_path": str(path),
            "legacy": reference,
            "in_memory": current,
            "mean_quality_abs_delta": quality_delta,
            "reward_abs_delta": reward_delta,
            "reward_relative_delta": (
                reward_delta / abs(float(reference["reward"]))
                if float(reference["reward"]) != 0
                else 0.0
            ),
            "face_visible_ratio_abs_delta": visible_delta,
            "num_faces_equal": current["num_faces"] == reference["num_faces"],
            "matches": matches,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    output = args.output_dir / "compatibility_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    max_delta = max(float(row["reward_abs_delta"]) for row in rows)
    in_memory_wall_time = time.monotonic() - in_memory_started
    summary = {
        "video_count": len(rows),
        "legacy_wall_time_seconds": legacy_wall_time,
        "in_memory_wall_time_seconds": in_memory_wall_time,
        "wall_time_change_seconds": in_memory_wall_time - legacy_wall_time,
        "max_reward_abs_delta": max_delta,
        "max_reward_relative_delta": max(
            float(row["reward_relative_delta"]) for row in rows
        ),
        "face_counts_exact": all(row["num_faces_equal"] for row in rows),
        "face_visible_ratios_exact": all(
            row["face_visible_ratio_abs_delta"] == 0 for row in rows
        ),
    }
    summary_path = args.output_dir / "compatibility_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not all(row["matches"] for row in rows):
        raise AssertionError(
            f"Reward compatibility failed: max_reward_abs_delta={max_delta} atol={args.atol}"
        )
    if not any(int(row["legacy"]["num_faces"]) > 1 for row in rows):
        raise AssertionError("Compatibility set did not exercise a multi-face sample")
    if len({float(row["legacy"]["reward"]) for row in rows}) < 2:
        raise AssertionError("Compatibility set did not contain different rewards")
    print(
        f"reward_compatibility=true videos={len(rows)} max_reward_abs_delta={max_delta:.10g} "
        f"legacy_wall_time_seconds={legacy_wall_time:.6f} "
        f"in_memory_wall_time_seconds={in_memory_wall_time:.6f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
