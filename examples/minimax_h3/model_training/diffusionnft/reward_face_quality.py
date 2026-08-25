#!/usr/bin/env python3
"""Thin adapter around the existing SCRFD + MagFace video evaluator.

This module does not import or modify either reward model. It writes the input
manifest expected by ``eval_video_face_quality.py``, launches that script as a
subprocess, and normalizes its ``video_quality.jsonl`` output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


DEFAULT_EVALUATOR = Path(
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/MagFace/"
    "inference/eval_video_face_quality.py"
)


def _absolute_video_paths(video_paths: Sequence[str | Path]) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in video_paths]
    if not paths:
        raise ValueError("At least one generated mp4 is required")
    if len(set(paths)) != len(paths):
        raise ValueError("Generated mp4 paths must be unique")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Generated mp4 files do not exist: {missing}")
    return paths


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_face_quality(
    video_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    evaluator_path: str | Path = DEFAULT_EVALUATOR,
    python_executable: str | Path = sys.executable,
    conda_executable: str | Path = "/root/miniconda3/bin/conda",
    frame_stride: int = 25,
    max_frames_per_video: int = 0,
    frame_face_aggregation: str = "mean",
    missing_face_reward: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Evaluate generated videos and return results keyed by absolute path."""

    paths = _absolute_video_paths(video_paths)
    evaluator = Path(evaluator_path).expanduser().resolve()
    if not evaluator.is_file():
        raise FileNotFoundError(f"MagFace evaluator does not exist: {evaluator}")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if max_frames_per_video < 0:
        raise ValueError("max_frames_per_video must be non-negative")

    reward_dir = Path(output_dir).expanduser().resolve()
    reward_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reward_dir / "rollout_reward_input.json"
    manifest = [
        {"name": f"rollout_{index:06d}", "render_video_path": str(path)}
        for index, path in enumerate(paths)
    ]
    _write_json(manifest_path, manifest)

    command = [
        str(python_executable),
        str(evaluator),
        "--input-json",
        str(manifest_path),
        "--render-field",
        "render_video_path",
        "--output-dir",
        str(reward_dir),
        "--allow-unpaired",
        "--frame-stride",
        str(frame_stride),
        "--max-frames-per-video",
        str(max_frames_per_video),
        "--frame-face-aggregation",
        frame_face_aggregation,
        "--pair-score-stat",
        "mean_quality",
        "--conda-exe",
        str(conda_executable),
    ]
    print(f"[reward] evaluating {len(paths)} videos", flush=True)
    print(f"[reward] command={' '.join(command)}", flush=True)
    subprocess.run(command, check=True)

    video_quality_path = reward_dir / "video_quality.jsonl"
    if not video_quality_path.is_file():
        raise RuntimeError(f"MagFace output is missing: {video_quality_path}")

    raw_by_path: dict[str, dict[str, Any]] = {}
    with video_quality_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            video_path = row.get("video_path")
            if isinstance(video_path, str):
                raw_by_path[str(Path(video_path).expanduser().resolve())] = row

    rewards: dict[str, dict[str, Any]] = {}
    for path in paths:
        key = str(path)
        row = raw_by_path.get(key)
        if row is None:
            raise RuntimeError(f"MagFace output has no row for generated video: {key}")
        mean_quality = row.get("mean_quality")
        valid_quality = isinstance(mean_quality, (int, float))
        rewards[key] = {
            "video_path": key,
            "mean_quality": float(mean_quality) if valid_quality else None,
            "face_visible_ratio": float(row.get("face_visible_ratio", 0.0)),
            "num_faces": int(row.get("detected_face_count", 0)),
            "reward": float(mean_quality) if valid_quality else float(missing_face_reward),
        }

    _write_json(reward_dir / "reward.json", list(rewards.values()))
    return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated mp4 files with the existing MagFace evaluator.")
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator-path", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--conda-executable", default="/root/miniconda3/bin/conda")
    parser.add_argument("--frame-stride", type=int, default=25)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--frame-face-aggregation", choices=("mean", "min", "max"), default="mean")
    parser.add_argument("--missing-face-reward", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rewards = evaluate_face_quality(
        args.videos,
        args.output_dir,
        evaluator_path=args.evaluator_path,
        python_executable=args.python_executable,
        conda_executable=args.conda_executable,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        frame_face_aggregation=args.frame_face_aggregation,
        missing_face_reward=args.missing_face_reward,
    )
    print(json.dumps(list(rewards.values()), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
