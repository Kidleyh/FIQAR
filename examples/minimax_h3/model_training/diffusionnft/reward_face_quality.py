#!/usr/bin/env python3
"""In-process and legacy SCRFD + MagFace reward interfaces."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_EVALUATOR = Path(
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/MagFace/"
    "inference/eval_video_face_quality.py"
)
DEFAULT_SCRFD_MODEL = Path(
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/insightface/"
    "models/scrfd_10g_bnkps.onnx"
)
DEFAULT_MAGFACE_CHECKPOINT = Path(
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/MagFace/"
    "checkpoints/magface_iresnet100_quality.pth"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_existing_evaluator(path: Path) -> ModuleType:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MagFace evaluator does not exist: {path}")
    spec = importlib.util.spec_from_file_location("fiqar_magface_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import MagFace evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FaceQualityReward:
    """Reusable scorer matching the legacy evaluator exactly.

    Input frames are RGB (the H3 pipeline returns PIL RGB). Detection and
    alignment run in BGR. The JPEG-95 roundtrip is intentional: the legacy
    evaluator writes and rereads every aligned crop before MagFace inference.
    """

    def __init__(
        self,
        *,
        evaluator_path: str | Path = DEFAULT_EVALUATOR,
        scrfd_model: str | Path = DEFAULT_SCRFD_MODEL,
        magface_checkpoint: str | Path = DEFAULT_MAGFACE_CHECKPOINT,
        device: str = "cuda:0",
        det_thresh: float = 0.5,
        det_size: int = 640,
        magface_batch_size: int = 256,
    ) -> None:
        if device not in ("cuda", "cuda:0"):
            raise ValueError("The verified reward path currently supports cuda:0 only")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SCRFD + MagFace reward inference")
        if det_size <= 0 or magface_batch_size <= 0:
            raise ValueError("det_size and magface_batch_size must be positive")
        self.device = torch.device("cuda:0")
        self.magface_batch_size = magface_batch_size

        scrfd_path = Path(scrfd_model).expanduser().resolve()
        checkpoint_path = Path(magface_checkpoint).expanduser().resolve()
        for path, label in (
            (scrfd_path, "SCRFD model"),
            (checkpoint_path, "MagFace checkpoint"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")

        from insightface.model_zoo import get_model
        from insightface.utils.face_align import norm_crop

        evaluator = _load_existing_evaluator(Path(evaluator_path))
        self.detector = get_model(
            str(scrfd_path), providers=["CUDAExecutionProvider"]
        )
        self.detector.prepare(
            ctx_id=0, input_size=(det_size, det_size), det_thresh=det_thresh
        )
        providers = self.detector.session.get_providers()
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"SCRFD CUDA provider was not selected: {providers}")
        self.magface, model_torch = evaluator.load_magface_model(str(checkpoint_path))
        if model_torch is not torch:
            raise RuntimeError("MagFace evaluator imported a different torch module")
        if next(self.magface.parameters()).device != self.device:
            raise RuntimeError("MagFace model did not initialize on cuda:0")
        self._norm_crop = norm_crop
        print(
            f"[reward-inmemory] initialized scrfd={scrfd_path} "
            f"magface={checkpoint_path} providers={providers}",
            flush=True,
        )

    @staticmethod
    def _frame_to_bgr(frame: Image.Image | np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(frame, Image.Image):
            rgb = np.asarray(frame.convert("RGB"))
        elif isinstance(frame, torch.Tensor):
            rgb = frame.detach().cpu().numpy()
            if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4):
                rgb = np.transpose(rgb, (1, 2, 0))
            if np.issubdtype(rgb.dtype, np.floating):
                if rgb.size and float(rgb.max()) <= 1.0:
                    rgb = rgb * 255.0
                rgb = np.clip(rgb, 0, 255).round().astype(np.uint8)
        elif isinstance(frame, np.ndarray):
            rgb = frame
        else:
            raise TypeError(f"Unsupported reward frame type: {type(frame).__name__}")
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(f"Expected RGB/RGBA HWC frame, got shape={rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _legacy_jpeg_roundtrip(aligned_bgr: np.ndarray) -> np.ndarray:
        ok, encoded = cv2.imencode(
            ".jpg", aligned_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not ok:
            raise RuntimeError("Failed to encode aligned face as JPEG-95")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != (112, 112, 3):
            raise RuntimeError("Failed to decode aligned JPEG-95 face")
        return decoded

    def score_frames(
        self,
        frames: Sequence[Image.Image | np.ndarray | torch.Tensor],
        frame_stride: int,
        max_frames_per_video: int = 0,
        frame_face_aggregation: str = "mean",
        missing_face_reward: float = 0.0,
    ) -> dict[str, float | int | None]:
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if max_frames_per_video < 0:
            raise ValueError("max_frames_per_video must be non-negative")
        if frame_face_aggregation not in ("mean", "min", "max"):
            raise ValueError("frame_face_aggregation must be mean, min, or max")
        if not isinstance(frames, Sequence) or not frames:
            raise ValueError("At least one in-memory frame is required")

        sampled = list(frames[::frame_stride])
        if max_frames_per_video:
            sampled = sampled[:max_frames_per_video]
        aligned_faces: list[np.ndarray] = []
        face_sizes: list[float] = []
        face_size_weights: list[float] = []
        frame_face_ranges: list[tuple[int, int]] = []
        detected_face_count = 0
        for frame in sampled:
            bgr = self._frame_to_bgr(frame)
            bboxes, landmarks = self.detector.detect(bgr, max_num=0)
            start = len(aligned_faces)
            if landmarks is not None:
                if bboxes is None or len(bboxes) != len(landmarks):
                    raise RuntimeError(
                        "SCRFD returned inconsistent bbox and landmark counts"
                    )
                detected_face_count += len(landmarks)
                for bbox, landmark in zip(bboxes, landmarks):
                    width = max(0.0, float(bbox[2]) - float(bbox[0]))
                    height = max(0.0, float(bbox[3]) - float(bbox[1]))
                    face_size = math.sqrt(width * height)
                    face_size_weight = (
                        3.0
                        if face_size == 0.0
                        else float(np.clip(96.0 / face_size, 1.0, 3.0))
                    )
                    aligned = self._norm_crop(
                        bgr, landmark=np.asarray(landmark), image_size=112
                    )
                    aligned_faces.append(self._legacy_jpeg_roundtrip(aligned))
                    face_sizes.append(face_size)
                    face_size_weights.append(face_size_weight)
            frame_face_ranges.append((start, len(aligned_faces)))

        face_scores: list[float] = []
        for start in range(0, len(aligned_faces), self.magface_batch_size):
            arrays = [
                np.ascontiguousarray(image.transpose(2, 0, 1))
                for image in aligned_faces[start : start + self.magface_batch_size]
            ]
            tensor = torch.from_numpy(np.stack(arrays)).float().div_(255.0).cuda()
            with torch.inference_mode():
                embeddings = self.magface(tensor)
                scores = torch.linalg.vector_norm(
                    embeddings, ord=2, dim=1
                ).cpu().tolist()
            face_scores.extend(float(score) for score in scores)

        frame_qualities: list[float] = []
        for start, end in frame_face_ranges:
            scores = face_scores[start:end]
            if scores:
                weights = face_size_weights[start:end]
                frame_qualities.append(
                    float(
                        {
                            "mean": sum(
                                weight * score
                                for weight, score in zip(weights, scores)
                            )
                            / sum(weights),
                            "min": min(scores),
                            "max": max(scores),
                        }[frame_face_aggregation]
                    )
                )
        mean_quality = (
            float(statistics.fmean(frame_qualities)) if frame_qualities else None
        )
        return {
            "reward": (
                mean_quality if mean_quality is not None else float(missing_face_reward)
            ),
            "mean_quality": mean_quality,
            "face_visible_ratio": (
                len(frame_qualities) / len(sampled) if sampled else 0.0
            ),
            "num_faces": int(detected_face_count),
            "mean_face_size_px": (
                float(statistics.fmean(face_sizes)) if face_sizes else None
            ),
            "mean_face_size_weight": (
                float(statistics.fmean(face_size_weights))
                if face_size_weights
                else None
            ),
        }


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


def evaluate_face_quality_from_video_files(
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
    keep_aligned_faces: bool = False,
) -> dict[str, dict[str, Any]]:
    """Legacy file/subprocess evaluator retained for tests and debugging."""

    paths = _absolute_video_paths(video_paths)
    evaluator = Path(evaluator_path).expanduser().resolve()
    if not evaluator.is_file():
        raise FileNotFoundError(f"MagFace evaluator does not exist: {evaluator}")
    if frame_stride <= 0 or max_frames_per_video < 0:
        raise ValueError("Invalid frame sampling configuration")
    reward_dir = Path(output_dir).expanduser().resolve()
    reward_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reward_dir / "rollout_reward_input.json"
    _write_json(
        manifest_path,
        [
            {"name": f"rollout_{index:06d}", "render_video_path": str(path)}
            for index, path in enumerate(paths)
        ],
    )
    command = [
        str(python_executable), str(evaluator),
        "--input-json", str(manifest_path),
        "--render-field", "render_video_path",
        "--output-dir", str(reward_dir),
        "--allow-unpaired",
        "--frame-stride", str(frame_stride),
        "--max-frames-per-video", str(max_frames_per_video),
        "--frame-face-aggregation", frame_face_aggregation,
        "--pair-score-stat", "mean_quality",
        "--conda-exe", str(conda_executable),
    ]
    if keep_aligned_faces:
        command.append("--keep-aligned-faces")
    print(f"[reward-legacy] evaluating {len(paths)} videos", flush=True)
    subprocess.run(command, check=True)
    output = reward_dir / "video_quality.jsonl"
    if not output.is_file():
        raise RuntimeError(f"MagFace output is missing: {output}")
    raw: dict[str, dict[str, Any]] = {}
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("video_path"), str):
                    raw[str(Path(row["video_path"]).resolve())] = row
    rewards: dict[str, dict[str, Any]] = {}
    for path in paths:
        key = str(path)
        row = raw.get(key)
        if row is None:
            raise RuntimeError(f"MagFace output has no row for video: {key}")
        quality = row.get("mean_quality")
        valid = isinstance(quality, (int, float))
        rewards[key] = {
            "video_path": key,
            "mean_quality": float(quality) if valid else None,
            "face_visible_ratio": float(row.get("face_visible_ratio", 0.0)),
            "num_faces": int(row.get("detected_face_count", 0)),
            "reward": float(quality) if valid else float(missing_face_reward),
        }
    _write_json(reward_dir / "reward.json", list(rewards.values()))
    return rewards


evaluate_face_quality = evaluate_face_quality_from_video_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy file-based reward adapter")
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
    rewards = evaluate_face_quality_from_video_files(
        args.videos, args.output_dir,
        evaluator_path=args.evaluator_path,
        python_executable=args.python_executable,
        conda_executable=args.conda_executable,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        frame_face_aggregation=args.frame_face_aggregation,
        missing_face_reward=args.missing_face_reward,
    )
    print(json.dumps(list(rewards.values()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
