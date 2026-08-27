#!/usr/bin/env python3
"""Bounded, resumable online DiffusionNFT orchestration for MiniMax-H3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
ROLLOUT_SCRIPT = Path(__file__).with_name("rollout.py")
TRAIN_SCRIPT = Path(__file__).with_name("train.py")
DEFAULT_DATA_JSON = Path(
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/"
    "guangdian_20251114_small_clear_faces/guangdian_20251114_small_clear_faces.json"
)
CHECKPOINT_FILES = (
    "current_lora.safetensors",
    "old_lora.safetensors",
    "optimizer.pt",
    "training_state.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_state(path: Path, expected_step: int | None = None) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    missing = [name for name in CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete checkpoint {path}: missing={missing}")
    state = read_json(path / "training_state.json")
    step = state.get("global_step")
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"Invalid checkpoint global_step in {path}: {step}")
    if expected_step is not None and step != expected_step:
        raise ValueError(
            f"Checkpoint step mismatch for {path}: {step} != {expected_step}"
        )
    return state


def expected_policy(
    latest_checkpoint: Path | None, global_step: int
) -> dict[str, Any]:
    if global_step == 0:
        if latest_checkpoint is not None:
            raise ValueError("global_step=0 cannot have a latest checkpoint")
        return {
            "policy_role": "base",
            "policy_lora_path": None,
            "policy_lora_sha256": None,
            "source_checkpoint": None,
            "global_step_before": 0,
        }
    if latest_checkpoint is None:
        raise ValueError("A positive global_step requires a latest checkpoint")
    latest_checkpoint = latest_checkpoint.resolve()
    checkpoint_state(latest_checkpoint, global_step)
    old_lora = (latest_checkpoint / "old_lora.safetensors").resolve()
    return {
        "policy_role": "old",
        "policy_lora_path": str(old_lora),
        "policy_lora_sha256": sha256_file(old_lora),
        "source_checkpoint": str(latest_checkpoint),
        "global_step_before": global_step,
    }


def validate_rollout(
    rollout_json: Path,
    expected: dict[str, Any],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(rollout_json)
    records = payload.get("rollouts") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or len(records) != len(seeds):
        raise ValueError(
            f"Expected {len(seeds)} rollout records in {rollout_json}, "
            f"found {len(records) if isinstance(records, list) else type(records)}"
        )
    actual_seeds = [record.get("seed") for record in records]
    if actual_seeds != seeds:
        raise ValueError(f"Rollout seeds mismatch: {actual_seeds} != {seeds}")
    prompts = {record.get("prompt") for record in records}
    if len(prompts) != 1 or not next(iter(prompts), None):
        raise ValueError("Online iteration must contain exactly one non-empty prompt group")
    rewards: list[float] = []
    mean_qualities: list[float | None] = []
    visible_ratios: list[float] = []
    face_counts: list[int] = []
    missing_face_samples = 0
    for index, record in enumerate(records):
        for key in ("video_path", "latent_path", "condition_image_path"):
            value = record.get(key)
            if not isinstance(value, str) or not Path(value).is_file():
                raise FileNotFoundError(f"Rollout record {index} missing {key}: {value}")
        actual = {key: record.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"Rollout policy mismatch at record {index}: "
                f"actual={actual} expected={expected}"
            )
        reward = record.get("reward")
        if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ValueError(f"Invalid reward at rollout record {index}: {reward}")
        rewards.append(float(reward))
        mean_quality = record.get("mean_quality")
        if mean_quality is not None and (
            not isinstance(mean_quality, (int, float))
            or not math.isfinite(float(mean_quality))
        ):
            raise ValueError(
                f"Invalid mean_quality at rollout record {index}: {mean_quality}"
            )
        visible_ratio = record.get("face_visible_ratio")
        face_count = record.get("num_faces")
        if (
            not isinstance(visible_ratio, (int, float))
            or not math.isfinite(float(visible_ratio))
            or not 0 <= float(visible_ratio) <= 1
        ):
            raise ValueError(
                f"Invalid face_visible_ratio at rollout record {index}: {visible_ratio}"
            )
        if not isinstance(face_count, int) or isinstance(face_count, bool) or face_count < 0:
            raise ValueError(f"Invalid num_faces at rollout record {index}: {face_count}")
        mean_qualities.append(
            float(mean_quality) if mean_quality is not None else None
        )
        visible_ratios.append(float(visible_ratio))
        face_counts.append(face_count)
        if mean_quality is None or face_count == 0:
            missing_face_samples += 1
    valid_qualities = [value for value in mean_qualities if value is not None]
    stats = {
        "rewards": rewards,
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "mean_quality": mean_qualities,
        "mean_quality_mean": (
            statistics.fmean(valid_qualities) if valid_qualities else None
        ),
        "face_visible_ratio": visible_ratios,
        "face_visible_ratio_mean": statistics.fmean(visible_ratios),
        "num_faces": face_counts,
        "num_faces_total": sum(face_counts),
        "missing_face_samples": missing_face_samples,
        "missing_face_ratio": missing_face_samples / len(records),
    }
    return records, stats


def run_streaming(command: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[command] {' '.join(command)}", flush=True)
    captured: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            captured.append(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return "".join(captured)


def parse_train_metrics(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    patterns = {
        "positive_loss": r"\[group\].*?positive_loss=([0-9eE+.-]+)",
        "negative_loss": r"\[group\].*?negative_loss=([0-9eE+.-]+)",
        "policy_loss": r"\[group\].*?policy_loss=([0-9eE+.-]+)",
        "kl_loss": r"\[group\].*?kl_loss=([0-9eE+.-]+)",
        "total_loss": r"\[group\].*?total_loss=([0-9eE+.-]+)",
        "gradient_norm": r"\[grad\].*?gradient_norm=([0-9eE+.-]+)",
        "current_reference_prediction_distance": (
            r"\[group\].*?current_reference_prediction_distance=([0-9eE+.-]+)"
        ),
        "current_old_parameter_distance": (
            r"current_old_parameter_distance_after=([0-9eE+.-]+)"
        ),
        "old_decay": r"\[policies\].*?old_decay=([0-9eE+.-]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            metrics[key] = float(matches[-1])
    integer_patterns = {
        "global_step_before": r"\[step\] global_step_before=(\d+)",
        "global_step_after": r"\[step\] global_step_after=(\d+)",
    }
    for key, pattern in integer_patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            metrics[key] = int(matches[-1])
    advantage_matches = re.findall(r"\[nft\] advantages=(\[[^\n]+\])", output)
    if advantage_matches:
        advantages = json.loads(advantage_matches[-1])
        if not isinstance(advantages, list) or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in advantages
        ):
            raise ValueError(f"Invalid advantages in training output: {advantages}")
        metrics["advantages"] = [float(value) for value in advantages]
    required = {
        "positive_loss",
        "negative_loss",
        "policy_loss",
        "kl_loss",
        "total_loss",
        "gradient_norm",
        "current_reference_prediction_distance",
        "current_old_parameter_distance",
        "old_decay",
        "global_step_before",
        "global_step_after",
        "advantages",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"Training output is missing required metrics: {missing}")
    return metrics


def iteration_metrics(
    iteration: int,
    group_size: int,
    policy: dict[str, Any],
    rollout_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "group_size": group_size,
        "group_size_note": (
            "K=2 engineering smoke only; normalized advantages are nearly +/-1"
            if group_size == 2
            else "K>=4 stability/experiment group"
        ),
        **policy,
        **rollout_metrics,
        **train_metrics,
        "checkpoint": str(checkpoint),
        "iteration_success": True,
    }


def write_observability_outputs(run_dir: Path, state: dict[str, Any]) -> None:
    rows = [
        item["metrics"]
        for item in sorted(state.get("iterations", []), key=lambda value: value["iteration"])
        if item.get("status") == "complete" and isinstance(item.get("metrics"), dict)
    ]
    metrics_path = run_dir / "metrics.jsonl"
    temporary = metrics_path.with_suffix(metrics_path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )
    temporary.replace(metrics_path)

    reward_means = [float(row["reward_mean"]) for row in rows]
    reward_stds = [float(row["reward_std"]) for row in rows]
    kl_values = [float(row["kl_loss"]) for row in rows]
    gradient_norms = [float(row["gradient_norm"]) for row in rows]
    reference_distances = [
        float(row["current_reference_prediction_distance"]) for row in rows
    ]
    visible_means = [float(row["face_visible_ratio_mean"]) for row in rows]
    missing_counts = [int(row["missing_face_samples"]) for row in rows]
    sample_count = sum(len(row["rewards"]) for row in rows)
    checkpoints = [row["checkpoint"] for row in rows]
    summary = {
        "iteration_count": len(rows),
        "group_size": state.get("config", {}).get("num_samples_per_prompt"),
        "group_size_note": (
            "K=2 is an engineering-only smoke configuration"
            if state.get("config", {}).get("num_samples_per_prompt") == 2
            else "K>=4 provides non-degenerate group-relative reward information"
        ),
        "reward_mean_by_iteration": reward_means,
        "reward_std_by_iteration": reward_stds,
        "reward_initial": reward_means[0] if reward_means else None,
        "reward_final": reward_means[-1] if reward_means else None,
        "reward_change": (
            reward_means[-1] - reward_means[0] if reward_means else None
        ),
        "kl_max": max(kl_values) if kl_values else None,
        "kl_final": kl_values[-1] if kl_values else None,
        "gradient_norm_max": max(gradient_norms) if gradient_norms else None,
        "current_reference_distance_by_iteration": reference_distances,
        "current_reference_distance_max": (
            max(reference_distances) if reference_distances else None
        ),
        "current_reference_distance_final": (
            reference_distances[-1] if reference_distances else None
        ),
        "face_visible_ratio_mean_by_iteration": visible_means,
        "face_visible_ratio_initial": visible_means[0] if visible_means else None,
        "face_visible_ratio_final": visible_means[-1] if visible_means else None,
        "missing_face_samples_by_iteration": missing_counts,
        "missing_face_samples": sum(missing_counts),
        "missing_face_ratio": sum(missing_counts) / sample_count if sample_count else None,
        "checkpoints": checkpoints,
    }
    write_json_atomic(run_dir / "training_summary.json", summary)


def stable_config(args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    return {
        "data_json": str(args.data_json.expanduser().resolve()),
        "num_samples_per_prompt": args.num_samples_per_prompt,
        "start": args.start,
        "limit": args.limit,
        "seeds": seeds,
        "model_id": args.model_id,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "flow_shift": args.flow_shift,
        "audio_flow_shift": args.audio_flow_shift,
        "reward_frame_stride": args.reward_frame_stride,
        "lora_rank": args.lora_rank,
        "lora_target_modules": args.lora_target_modules,
        "timestep_fraction": args.timestep_fraction,
        "adv_clip_max": args.adv_clip_max,
        "policy_beta": args.policy_beta,
        "kl_beta": args.kl_beta,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "old_decay_type": args.old_decay_type,
    }


def load_or_create_state(
    run_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    state_path = run_dir / "online_state.json"
    if args.resume_from is not None:
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        state = read_json(state_path)
        if state.get("format_version") != 1:
            raise ValueError(f"Unsupported online state format: {state.get('format_version')}")
        if state.get("config") != config:
            raise ValueError("Resume online configuration does not match online_state.json")
        checkpoint_value = state.get("latest_checkpoint")
        if checkpoint_value is not None:
            checkpoint_state(Path(checkpoint_value), int(state["global_step"]))
        print(
            f"[online-resume] run_dir={run_dir} completed_iteration="
            f"{state['completed_iteration']} global_step={state['global_step']} "
            "resume_success=true",
            flush=True,
        )
        return state
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "completed_iteration": -1,
        "latest_checkpoint": None,
        "global_step": 0,
        "dataset_position": args.start,
        "seeds": config["seeds"],
        "config": config,
        "iterations": [],
    }
    write_json_atomic(state_path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded, resumable MiniMax-H3 online DiffusionNFT loop."
    )
    parser.add_argument("--num-iterations", type=int, required=True)
    parser.add_argument("--num-samples-per-prompt", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-id", default="MiniMax/MiniMax-H3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", default="124")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--vram-limit-gb", type=float, default=None)
    parser.add_argument("--reward-frame-stride", type=int, default=25)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-target-modules", default="qkv_proj,out_proj")
    parser.add_argument("--timestep-fraction", type=float, default=0.99)
    parser.add_argument("--adv-clip-max", type=float, default=5.0)
    parser.add_argument("--policy-beta", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--old-decay-type", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--use-gradient-checkpointing-offload", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, list[int]]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Online training supports one process/GPU only")
    if args.num_iterations <= 0 or args.num_samples_per_prompt < 2:
        raise ValueError("Iterations must be positive and samples per prompt at least 2")
    if args.start < 0 or args.limit <= 0:
        raise ValueError("--start must be non-negative and --limit positive")
    if args.output_dir is None and args.resume_from is None:
        raise ValueError("Provide --output-dir for a new run or --resume-from")
    if args.resume_from is not None:
        run_dir = args.resume_from.expanduser().resolve()
        if args.output_dir is not None and args.output_dir.expanduser().resolve() != run_dir:
            raise ValueError("--output-dir and --resume-from must identify the same run")
    else:
        run_dir = args.output_dir.expanduser().resolve()
    seeds = list(range(args.num_samples_per_prompt)) if args.seeds is None else args.seeds
    if len(seeds) != args.num_samples_per_prompt or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain exactly K unique values")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative")
    if not args.data_json.expanduser().is_file():
        raise FileNotFoundError(args.data_json)
    return run_dir, seeds


def main() -> None:
    args = parse_args()
    run_dir, seeds = validate_args(args)
    config = stable_config(args, seeds)
    state_path = run_dir / "online_state.json"
    state = load_or_create_state(run_dir, args, config)
    write_observability_outputs(run_dir, state)
    print(
        f"[group-size] K={args.num_samples_per_prompt} "
        f"engineering_smoke_only={str(args.num_samples_per_prompt == 2).lower()}",
        flush=True,
    )
    first_iteration = int(state["completed_iteration"]) + 1
    if first_iteration >= args.num_iterations:
        print(
            f"[online] target already complete: completed_iteration="
            f"{state['completed_iteration']} global_step={state['global_step']}",
            flush=True,
        )
        return

    for iteration in range(first_iteration, args.num_iterations):
        global_step_before = int(state["global_step"])
        latest_value = state.get("latest_checkpoint")
        latest_checkpoint = Path(latest_value) if latest_value else None
        policy = expected_policy(latest_checkpoint, global_step_before)
        dataset_position = args.start + (iteration % args.limit)
        iteration_dir = run_dir / f"iteration_{iteration:06d}"
        rollout_dir = iteration_dir / "rollout"
        rollout_json = rollout_dir / "rollout.json"
        checkpoint_dir = run_dir / "checkpoints" / f"checkpoint-{global_step_before + 1}"

        existing = next(
            (item for item in state["iterations"] if item["iteration"] == iteration),
            None,
        )
        if existing is None:
            existing = {
                "iteration": iteration,
                "status": "planned",
                "dataset_position": dataset_position,
                "seeds": seeds,
                "rollout_path": str(rollout_json),
                **policy,
            }
            state["iterations"].append(existing)
            state["dataset_position"] = dataset_position
            write_json_atomic(state_path, state)
        else:
            stored_policy = {key: existing.get(key) for key in policy}
            if stored_policy != policy or existing.get("dataset_position") != dataset_position:
                raise ValueError("Incomplete iteration state does not match expected policy/data")

        print(
            f"[online] iteration={iteration} rollout_policy_role={policy['policy_role']} "
            f"rollout_policy_sha256={policy['policy_lora_sha256']} "
            f"global_step_before={global_step_before}",
            flush=True,
        )

        rollout_valid = False
        if rollout_json.is_file():
            try:
                records, rollout_metrics = validate_rollout(
                    rollout_json, policy, seeds
                )
                rollout_valid = True
                print(f"[online-resume] reuse_valid_rollout={rollout_json}", flush=True)
            except (OSError, TypeError, ValueError) as error:
                print(f"[online-resume] rollout_not_complete={error}", flush=True)
        if not rollout_valid:
            rollout_command = [
                args.python,
                str(ROLLOUT_SCRIPT),
                "--data-json", str(args.data_json.expanduser().resolve()),
                "--start", str(dataset_position),
                "--limit", "1",
                "--num-samples", str(args.num_samples_per_prompt),
                "--seeds", *map(str, seeds),
                "--output-dir", str(rollout_dir),
                "--model-id", args.model_id,
                "--height", str(args.height),
                "--width", str(args.width),
                "--num-frames", str(args.num_frames),
                "--num-inference-steps", str(args.num_inference_steps),
                "--flow-shift", str(args.flow_shift),
                "--audio-flow-shift", str(args.audio_flow_shift),
                "--reward-frame-stride", str(args.reward_frame_stride),
                "--policy-role", policy["policy_role"],
                "--global-step-before", str(global_step_before),
                "--skip-existing",
            ]
            if args.vram_limit_gb is not None:
                rollout_command.extend(["--vram-limit-gb", str(args.vram_limit_gb)])
            if args.allow_download:
                rollout_command.append("--allow-download")
            if policy["policy_role"] == "old":
                rollout_command.extend(
                    [
                        "--lora-path", policy["policy_lora_path"],
                        "--policy-lora-sha256", policy["policy_lora_sha256"],
                        "--source-checkpoint", policy["source_checkpoint"],
                    ]
                )
            run_streaming(rollout_command, iteration_dir / "rollout.log")
            records, rollout_metrics = validate_rollout(
                rollout_json, policy, seeds
            )

        if policy["policy_role"] == "old":
            rollout_log = iteration_dir / "rollout.log"
            if not rollout_log.is_file():
                raise FileNotFoundError(
                    f"Old-policy rollout requires its patch-verification log: {rollout_log}"
                )
            patched = re.findall(
                r"\[lora\] patched_modules=(\d+)",
                rollout_log.read_text(encoding="utf-8"),
            )
            if not patched or int(patched[-1]) != 104:
                raise RuntimeError(
                    f"Old LoRA rollout did not patch 104 DiT modules: {patched}"
                )
            existing["rollout_policy_patched_modules"] = 104

        rewards = [float(record["reward"]) for record in records]
        reward_stats = {
            "mean": rollout_metrics["reward_mean"],
            "std": rollout_metrics["reward_std"],
            "min": rollout_metrics["reward_min"],
            "max": rollout_metrics["reward_max"],
        }
        existing.update(
            {
                "status": "rollout_complete",
                "rewards": rewards,
                "reward_stats": reward_stats,
                "rollout_metrics": rollout_metrics,
            }
        )
        write_json_atomic(state_path, state)
        print(
            f"[online] iteration={iteration} rewards={rewards} "
            f"reward_mean={rollout_metrics['reward_mean']:.8f} "
            f"reward_std={rollout_metrics['reward_std']:.8f} "
            f"face_visible_ratio_mean="
            f"{rollout_metrics['face_visible_ratio_mean']:.8f} "
            f"missing_face_samples={rollout_metrics['missing_face_samples']}",
            flush=True,
        )

        if checkpoint_dir.exists():
            checkpoint_state(checkpoint_dir, global_step_before + 1)
            print(
                f"[online-resume] reuse_valid_checkpoint={checkpoint_dir}", flush=True
            )
            train_metrics = existing.get("train_metrics")
            if not isinstance(train_metrics, dict):
                train_log = iteration_dir / "train.log"
                if not train_log.is_file():
                    raise FileNotFoundError(
                        f"Recovered checkpoint has no training metrics log: {train_log}"
                    )
                train_metrics = parse_train_metrics(
                    train_log.read_text(encoding="utf-8")
                )
        else:
            train_command = [
                args.python,
                str(TRAIN_SCRIPT),
                "--mode", "nft-step",
                "--rollout-json", str(rollout_json),
                "--rollout-index", "0",
                "--group-size", str(args.num_samples_per_prompt),
                "--model-id", args.model_id,
                "--device", args.device,
                "--lora-rank", str(args.lora_rank),
                "--lora-target-modules", args.lora_target_modules,
                "--timestep-fraction", str(args.timestep_fraction),
                "--adv-clip-max", str(args.adv_clip_max),
                "--policy-beta", str(args.policy_beta),
                "--kl-beta", str(args.kl_beta),
                "--learning-rate", str(args.learning_rate),
                "--max-grad-norm", str(args.max_grad_norm),
                "--old-decay-type", str(args.old_decay_type),
                "--checkpoint-output", str(checkpoint_dir),
                "--require-policy-provenance",
            ]
            if latest_checkpoint is not None:
                train_command.extend(["--resume-from", str(latest_checkpoint)])
            if args.use_gradient_checkpointing_offload:
                train_command.append("--use-gradient-checkpointing-offload")
            if args.allow_download:
                train_command.append("--allow-download")
            train_output = run_streaming(train_command, iteration_dir / "train.log")
            train_metrics = parse_train_metrics(train_output)
            checkpoint_state(checkpoint_dir, global_step_before + 1)

        global_step_after = global_step_before + 1
        if train_metrics["global_step_before"] != global_step_before or train_metrics[
            "global_step_after"
        ] != global_step_after:
            raise ValueError(
                "Parsed training global steps do not match online state: "
                f"{train_metrics['global_step_before']} -> "
                f"{train_metrics['global_step_after']} vs "
                f"{global_step_before} -> {global_step_after}"
            )
        if len(train_metrics["advantages"]) != args.num_samples_per_prompt:
            raise ValueError(
                f"Advantage count {len(train_metrics['advantages'])} does not match "
                f"group size {args.num_samples_per_prompt}"
            )
        metrics = iteration_metrics(
            iteration,
            args.num_samples_per_prompt,
            policy,
            rollout_metrics,
            train_metrics,
            checkpoint_dir,
        )
        existing.update(
            {
                "status": "complete",
                "checkpoint": str(checkpoint_dir),
                "global_step_after": global_step_after,
                "train_metrics": train_metrics,
                "metrics": metrics,
                "iteration_success": True,
            }
        )
        state.update(
            {
                "completed_iteration": iteration,
                "latest_checkpoint": str(checkpoint_dir),
                "global_step": global_step_after,
                "dataset_position": args.start + ((iteration + 1) % args.limit),
            }
        )
        write_json_atomic(state_path, state)
        write_observability_outputs(run_dir, state)
        print(
            f"[online] iteration={iteration} global_step_before={global_step_before} "
            f"global_step_after={global_step_after} "
            f"current_old_distance="
            f"{train_metrics.get('current_old_parameter_distance')} "
            f"kl_loss={train_metrics.get('kl_loss')} checkpoint={checkpoint_dir} "
            "iteration_success=true",
            flush=True,
        )


if __name__ == "__main__":
    main()
