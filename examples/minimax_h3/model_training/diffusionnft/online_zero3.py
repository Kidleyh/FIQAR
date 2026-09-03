#!/usr/bin/env python3
"""Bounded online DiffusionNFT orchestration for ZeRO-3 rollout and training.

This file contains no rollout, reward, NFT-loss, scheduler, EMA, or checkpoint
math. Each large-video seed is generated serially by ``rollout_zero3.sh``;
the validated records are merged and passed unchanged to ``train_zero3.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_JSON = Path(
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/"
    "guangdian_20251114_small_clear_faces/"
    "guangdian_20251114_small_clear_faces.json"
)
FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZeRO-3 rollout -> reward -> ZeRO-3 DiffusionNFT loop."
    )
    parser.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--rollout-world-size", type=int, default=2)
    parser.add_argument("--train-world-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--width", type=int, default=1088)
    parser.add_argument("--height", type=int, default=736)
    parser.add_argument("--num-frames", type=int, default=175)
    parser.add_argument("--num-inference-steps", type=int, default=3)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--policy-beta", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=1e-4)
    parser.add_argument("--adv-clip-max", type=float, default=5.0)
    parser.add_argument("--timestep-fraction", type=float, default=0.99)
    parser.add_argument("--old-decay-type", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--timestep-seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=5678)
    parser.add_argument("--reward-frame-stride", type=int, default=25)
    parser.add_argument("--reward-max-frames", type=int, default=0)
    parser.add_argument(
        "--reward-frame-face-aggregation",
        choices=("mean", "min", "max"),
        default="mean",
    )
    parser.add_argument("--missing-face-reward", type=float, default=0.0)
    parser.add_argument("--save-rollout-video", action="store_true")
    parser.add_argument(
        "--rollout-launcher",
        type=Path,
        default=SCRIPT_DIR / "rollout_zero3.sh",
    )
    parser.add_argument(
        "--train-launcher",
        type=Path,
        default=SCRIPT_DIR / "train_zero3.sh",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def validate_args(args: argparse.Namespace) -> list[str]:
    positive_ints = {
        "limit": args.limit,
        "num-iterations": args.num_iterations,
        "rollout-world-size": args.rollout_world_size,
        "train-world-size": args.train_world_size,
        "group-size": args.group_size,
        "width": args.width,
        "height": args.height,
        "num-frames": args.num_frames,
        "num-inference-steps": args.num_inference_steps,
        "lora-rank": args.lora_rank,
        "reward-frame-stride": args.reward_frame_stride,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.start < 0 or args.reward_max_frames < 0:
        raise ValueError("--start and --reward-max-frames must be non-negative")
    if args.group_size != len(args.seeds):
        raise ValueError(
            f"GROUP_SIZE={args.group_size} must equal seed count={len(args.seeds)}"
        )
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError(f"Seeds must be unique and non-negative: {args.seeds}")
    if args.group_size % args.train_world_size:
        raise ValueError(
            f"GROUP_SIZE={args.group_size} must be divisible by "
            f"TRAIN_WORLD_SIZE={args.train_world_size}"
        )
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU IDs must be unique: {gpu_ids}")
    if len(gpu_ids) < max(args.rollout_world_size, args.train_world_size):
        raise ValueError(
            f"GPU_IDS exposes {len(gpu_ids)} devices but rollout/train require "
            f"{args.rollout_world_size}/{args.train_world_size}"
        )
    if args.old_decay_type not in (0, 1, 2):
        raise ValueError("--old-decay-type must be 0, 1, or 2")
    if not 0 < args.timestep_fraction <= 1:
        raise ValueError("--timestep-fraction must be in (0, 1]")
    if args.learning_rate <= 0 or args.policy_beta <= 0 or args.adv_clip_max <= 0:
        raise ValueError("Learning rate, policy beta, and advantage clip must be positive")
    if args.kl_beta < 0 or args.max_grad_norm <= 0:
        raise ValueError("KL beta must be non-negative and max grad norm positive")
    if not args.data_json.expanduser().is_file():
        raise FileNotFoundError(args.data_json)
    for launcher in (args.rollout_launcher, args.train_launcher):
        if not launcher.expanduser().is_file():
            raise FileNotFoundError(launcher)
    return gpu_ids


def stable_config(args: argparse.Namespace, gpu_ids: list[str]) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "data_json": str(args.data_json.expanduser().resolve()),
        "start": args.start,
        "limit": args.limit,
        "gpu_ids": gpu_ids,
        "rollout_world_size": args.rollout_world_size,
        "train_world_size": args.train_world_size,
        "group_size": args.group_size,
        "seeds": args.seeds,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "flow_shift": args.flow_shift,
        "audio_flow_shift": args.audio_flow_shift,
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "policy_beta": args.policy_beta,
        "kl_beta": args.kl_beta,
        "adv_clip_max": args.adv_clip_max,
        "timestep_fraction": args.timestep_fraction,
        "old_decay_type": args.old_decay_type,
        "max_grad_norm": args.max_grad_norm,
        "timestep_seed": args.timestep_seed,
        "noise_seed": args.noise_seed,
        "reward_frame_stride": args.reward_frame_stride,
        "reward_max_frames": args.reward_max_frames,
        "reward_frame_face_aggregation": args.reward_frame_face_aggregation,
        "missing_face_reward": args.missing_face_reward,
        "save_rollout_video": args.save_rollout_video,
        "rollout_launch": str(args.rollout_launcher.expanduser().resolve()),
        "train_launch": str(args.train_launcher.expanduser().resolve()),
    }


def initial_state(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "run_dir": str(run_dir),
        "stable_config": config,
        "completed_iteration": -1,
        "global_step": 0,
        "latest_checkpoint": None,
        "current_iteration": None,
        "iterations": [],
    }


def load_or_create_state(
    args: argparse.Namespace, run_dir: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    state_path = run_dir / "online_zero3_state.json"
    if args.resume_from is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"New run directory is not empty; use --resume-from: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        state = initial_state(run_dir, config)
        atomic_json(state_path, state)
        return state, state_path
    if not nonempty_file(state_path):
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported online_zero3 state format")
    if state.get("stable_config") != config:
        raise ValueError("Resume stable configuration mismatch")
    if state.get("run_dir") != str(run_dir):
        raise ValueError("Resume run directory mismatch")
    return state, state_path


def checkpoint_state(checkpoint: Path, expected_step: int) -> dict[str, Any] | None:
    required = (
        checkpoint / "current_lora.safetensors",
        checkpoint / "old_lora.safetensors",
        checkpoint / "training_state.json",
    )
    if not checkpoint.is_dir() or not all(nonempty_file(path) for path in required):
        return None
    deepspeed_dir = checkpoint / "deepspeed"
    if not deepspeed_dir.is_dir() or not any(deepspeed_dir.rglob("*optim_states.pt")):
        return None
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != expected_step:
        return None
    return state


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_seed_rollout(seed_dir: Path, expected_seed: int) -> dict[str, Any] | None:
    rollout_path = seed_dir / "rollout.json"
    if not nonempty_file(rollout_path):
        return None
    try:
        records = json.loads(rollout_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or len(records) != 1:
            return None
        record = records[0]
        if record.get("seed") != expected_seed or record.get("complete") is not True:
            return None
        if not finite_number(record.get("reward")):
            return None
        for key in ("latent_path", "condition_image_path", "sample_state_path"):
            path = Path(record[key]).expanduser().resolve()
            if not nonempty_file(path):
                return None
        latent_path = Path(record["latent_path"]).expanduser().resolve()
        condition_path = Path(record["condition_image_path"]).expanduser().resolve()
        if sha256_file(latent_path) != record.get("latent_sha256"):
            return None
        if sha256_file(condition_path) != record.get("condition_image_sha256"):
            return None
        sample_state = json.loads(
            Path(record["sample_state_path"]).read_text(encoding="utf-8")
        )
        for key, value in sample_state.items():
            if record.get(key) != value:
                return None
        return record
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def common_value(records: list[dict[str, Any]], key: str) -> Any:
    values = [record.get(key) for record in records]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Merged rollout mismatch for {key}: {values}")
    return values[0]


def validate_merged_records(
    records: list[dict[str, Any]], args: argparse.Namespace, expected_policy: dict[str, Any]
) -> None:
    if len(records) != args.group_size:
        raise ValueError(f"Expected K={args.group_size}, got {len(records)}")
    if [record.get("seed") for record in records] != args.seeds:
        raise ValueError("Merged seed order/content mismatch")
    if len({record["seed"] for record in records}) != len(records):
        raise ValueError("Merged rollout contains duplicate seeds")
    common_keys = (
        "prompt",
        "prompt_sha256",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "flow_shift",
        "audio_flow_shift",
        "policy_role",
        "policy_lora_path",
        "policy_lora_sha256",
        "source_checkpoint",
        "global_step_before",
        "reward_frame_stride",
        "reward_max_frames",
        "reward_frame_face_aggregation",
        "missing_face_reward",
        "scrfd_model_sha256",
        "magface_checkpoint_sha256",
        "reward_backend",
    )
    for key in common_keys:
        common_value(records, key)
    for key, expected in expected_policy.items():
        if common_value(records, key) != expected:
            raise ValueError(f"Rollout policy mismatch for {key}")
    geometry = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "flow_shift": args.flow_shift,
        "audio_flow_shift": args.audio_flow_shift,
    }
    for key, expected in geometry.items():
        if common_value(records, key) != expected:
            raise ValueError(f"Rollout geometry/scheduler mismatch for {key}")


def valid_merged_rollout(
    path: Path, args: argparse.Namespace, expected_policy: dict[str, Any]
) -> list[dict[str, Any]] | None:
    if not nonempty_file(path):
        return None
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            return None
        validate_merged_records(records, args, expected_policy)
        if any(load_seed_rollout(path.parent / f"seed_{r['seed']}", r["seed"]) is None for r in records):
            return None
        return records
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[online-zero3-command] {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    elapsed = time.monotonic() - started
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}; log={log_path}"
        )
    return elapsed


def archive_incomplete_seed_dir(seed_dir: Path) -> None:
    if not seed_dir.exists():
        return
    archived = seed_dir.with_name(
        f"{seed_dir.name}_incomplete_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    )
    seed_dir.replace(archived)
    print(f"[online-zero3] archived_incomplete={archived}", flush=True)


def expected_policy(
    global_step: int, source_checkpoint: Path | None
) -> dict[str, Any]:
    if global_step == 0:
        if source_checkpoint is not None:
            raise ValueError("Base policy cannot have a source checkpoint")
        return {
            "policy_role": "base",
            "policy_lora_path": None,
            "policy_lora_sha256": None,
            "source_checkpoint": None,
            "global_step_before": 0,
        }
    if source_checkpoint is None:
        raise ValueError("Old policy rollout requires latest checkpoint")
    old_lora = (source_checkpoint / "old_lora.safetensors").resolve()
    if not nonempty_file(old_lora):
        raise FileNotFoundError(old_lora)
    return {
        "policy_role": "old",
        "policy_lora_path": str(old_lora),
        "policy_lora_sha256": sha256_file(old_lora),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "global_step_before": global_step,
    }


def rollout_environment(
    args: argparse.Namespace,
    gpu_ids: list[str],
    seed: int,
    seed_dir: Path,
    dataset_position: int,
    policy: dict[str, Any],
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "NUM_PROCESSES": str(args.rollout_world_size),
            "DATA_JSON": str(args.data_json.expanduser().resolve()),
            "START": str(dataset_position),
            "OUTPUT_DIR": str(seed_dir.resolve()),
            "WIDTH": str(args.width),
            "HEIGHT": str(args.height),
            "NUM_FRAMES": str(args.num_frames),
            "NUM_INFERENCE_STEPS": str(args.num_inference_steps),
            "SEED": str(seed),
            "FLOW_SHIFT": str(args.flow_shift),
            "AUDIO_FLOW_SHIFT": str(args.audio_flow_shift),
            "POLICY_ROLE": policy["policy_role"],
            "LORA_PATH": policy["policy_lora_path"] or "",
            "LORA_RANK": str(args.lora_rank),
            "SOURCE_CHECKPOINT": policy["source_checkpoint"] or "",
            "GLOBAL_STEP_BEFORE": str(policy["global_step_before"]),
            "SAVE_ROLLOUT_VIDEO": "1" if args.save_rollout_video else "0",
            "REWARD_FRAME_STRIDE": str(args.reward_frame_stride),
            "REWARD_MAX_FRAMES": str(args.reward_max_frames),
            "REWARD_FRAME_FACE_AGGREGATION": args.reward_frame_face_aggregation,
            "MISSING_FACE_REWARD": str(args.missing_face_reward),
        }
    )
    return env


def training_environment(
    args: argparse.Namespace,
    gpu_ids: list[str],
    merged_rollout: Path,
    checkpoint_output: Path,
    resume_checkpoint: Path | None,
    condition_cache: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "NUM_PROCESSES": str(args.train_world_size),
            "GROUP_SIZE": str(args.group_size),
            "ROLLOUT_JSON": str(merged_rollout.resolve()),
            "CHECKPOINT_OUTPUT": str(checkpoint_output.resolve()),
            "CONDITION_CACHE_DIR": str(condition_cache.resolve()),
            "LORA_RANK": str(args.lora_rank),
            "LEARNING_RATE": str(args.learning_rate),
            "POLICY_BETA": str(args.policy_beta),
            "KL_BETA": str(args.kl_beta),
            "ADV_CLIP_MAX": str(args.adv_clip_max),
            "TIMESTEP_FRACTION": str(args.timestep_fraction),
            "OLD_DECAY_TYPE": str(args.old_decay_type),
            "MAX_GRAD_NORM": str(args.max_grad_norm),
            "TIMESTEP_SEED": str(args.timestep_seed),
            "NOISE_SEED": str(args.noise_seed),
            "RESUME_FROM": str(resume_checkpoint.resolve()) if resume_checkpoint else "",
            "INITIAL_LORA_PATH": "",
        }
    )
    return env


def parse_training_peaks(log_path: Path) -> dict[str, float]:
    pattern = re.compile(
        r"\[rank (\d+)\].*peak_gpu_allocated_mb=([0-9.]+).*"
        r"peak_gpu_reserved_mb=([0-9.]+)"
    )
    peaks: dict[str, float] = {}
    if not log_path.is_file():
        return peaks
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rank = match.group(1)
            peaks[f"rank_{rank}_allocated_mb"] = float(match.group(2))
            peaks[f"rank_{rank}_reserved_mb"] = float(match.group(3))
    return peaks


def commit_completed_iteration(
    state: dict[str, Any],
    state_path: Path,
    iteration: int,
    iteration_record: dict[str, Any],
    checkpoint: Path,
    global_step_after: int,
) -> None:
    if state["completed_iteration"] >= iteration:
        return
    iteration_record["status"] = "complete"
    iteration_record["global_step_after"] = global_step_after
    iteration_record["checkpoint"] = str(checkpoint.resolve())
    state["iterations"].append(iteration_record)
    state["completed_iteration"] = iteration
    state["global_step"] = global_step_after
    state["latest_checkpoint"] = str(checkpoint.resolve())
    state["current_iteration"] = None
    atomic_json(state_path, state)


def run_iteration(
    args: argparse.Namespace,
    gpu_ids: list[str],
    state: dict[str, Any],
    state_path: Path,
    run_dir: Path,
    iteration: int,
) -> None:
    started = time.monotonic()
    global_step_before = int(state["global_step"])
    source_checkpoint = (
        Path(state["latest_checkpoint"]).resolve()
        if state.get("latest_checkpoint")
        else None
    )
    policy = expected_policy(global_step_before, source_checkpoint)
    dataset_position = args.start + iteration % args.limit
    iteration_dir = run_dir / f"iteration_{iteration:06d}"
    rollout_dir = iteration_dir / "rollout"
    checkpoint = run_dir / "checkpoints" / f"checkpoint-{global_step_before + 1}"
    merged_rollout = rollout_dir / "rollout.json"
    condition_cache = iteration_dir / "training_condition_cache"
    iteration_record: dict[str, Any] = {
        "iteration": iteration,
        "dataset_position": dataset_position,
        "global_step_before": global_step_before,
        "policy": policy,
        "seed_rollouts": [],
        "merged_rollout": str(merged_rollout.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "status": "running",
    }
    state["current_iteration"] = iteration_record
    atomic_json(state_path, state)

    # Recovery point C: checkpoint commit is the optimizer-step transaction.
    committed = checkpoint_state(checkpoint, global_step_before + 1)
    if committed is not None:
        iteration_record["checkpoint_reused"] = True
        iteration_record["iteration_wall_time_seconds"] = 0.0
        commit_completed_iteration(
            state, state_path, iteration, iteration_record, checkpoint,
            global_step_before + 1,
        )
        print(
            f"[online-zero3] iteration={iteration} reuse_valid_checkpoint=true "
            f"global_step={global_step_before}->{global_step_before + 1}",
            flush=True,
        )
        return
    if checkpoint.exists():
        raise RuntimeError(f"Existing checkpoint is incomplete/invalid: {checkpoint}")

    records = valid_merged_rollout(merged_rollout, args, policy)
    rollout_seconds = 0.0
    if records is None:
        records = []
        for seed in args.seeds:
            seed_dir = rollout_dir / f"seed_{seed}"
            record = load_seed_rollout(seed_dir, seed)
            reused = record is not None
            elapsed = 0.0
            if record is None:
                archive_incomplete_seed_dir(seed_dir)
                env = rollout_environment(
                    args, gpu_ids, seed, seed_dir, dataset_position, policy
                )
                print(
                    f"[online-zero3] iteration={iteration} seed={seed} "
                    f"rollout_reused=false policy={policy['policy_role']}",
                    flush=True,
                )
                elapsed = run_logged(
                    ["bash", str(args.rollout_launcher.expanduser().resolve())],
                    env,
                    rollout_dir / f"seed_{seed}.log",
                )
                record = load_seed_rollout(seed_dir, seed)
                if record is None:
                    raise RuntimeError(f"Rollout launcher did not commit seed {seed}")
            else:
                print(
                    f"[online-zero3] iteration={iteration} seed={seed} "
                    "rollout_reused=true",
                    flush=True,
                )
            records.append(record)
            rollout_seconds += elapsed
            iteration_record["seed_rollouts"].append(
                {
                    "seed": seed,
                    "rollout_json": str((seed_dir / "rollout.json").resolve()),
                    "sample_state": record["sample_state_path"],
                    "reward": record["reward"],
                    "reused": reused,
                    "wall_time_seconds": elapsed,
                    "rank_peak_allocated_mb": record.get("rank_peak_allocated_mb"),
                    "rank_peak_reserved_mb": record.get("rank_peak_reserved_mb"),
                }
            )
            state["current_iteration"] = iteration_record
            atomic_json(state_path, state)
        validate_merged_records(records, args, policy)
        rollout_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(merged_rollout, records)
        print(
            f"[online-zero3] iteration={iteration} merged_rollout={merged_rollout}",
            flush=True,
        )
    else:
        print(
            f"[online-zero3] iteration={iteration} reuse_merged_rollout=true",
            flush=True,
        )
        iteration_record["seed_rollouts"] = [
            {
                "seed": record["seed"],
                "rollout_json": str(
                    (rollout_dir / f"seed_{record['seed']}" / "rollout.json").resolve()
                ),
                "sample_state": record["sample_state_path"],
                "reward": record["reward"],
                "reused": True,
                "wall_time_seconds": 0.0,
                "rank_peak_allocated_mb": record.get("rank_peak_allocated_mb"),
                "rank_peak_reserved_mb": record.get("rank_peak_reserved_mb"),
            }
            for record in records
        ]
    iteration_record["rollout_wall_time_seconds"] = rollout_seconds
    iteration_record["rewards"] = [float(record["reward"]) for record in records]
    state["current_iteration"] = iteration_record
    atomic_json(state_path, state)

    training_log = iteration_dir / "training.log"
    train_env = training_environment(
        args, gpu_ids, merged_rollout, checkpoint, source_checkpoint, condition_cache
    )
    print(
        f"[online-zero3] iteration={iteration} training_resume={source_checkpoint} "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    training_seconds = run_logged(
        ["bash", str(args.train_launcher.expanduser().resolve())],
        train_env,
        training_log,
    )
    committed = checkpoint_state(checkpoint, global_step_before + 1)
    if committed is None:
        raise RuntimeError(f"Training did not commit a valid checkpoint: {checkpoint}")
    iteration_record["training_wall_time_seconds"] = training_seconds
    iteration_record["training_peak_vram"] = parse_training_peaks(training_log)
    iteration_record["iteration_wall_time_seconds"] = time.monotonic() - started
    iteration_record["checkpoint_reused"] = False
    commit_completed_iteration(
        state, state_path, iteration, iteration_record, checkpoint,
        global_step_before + 1,
    )
    print(
        f"[online-zero3] iteration={iteration} iteration_success=true "
        f"global_step={global_step_before}->{global_step_before + 1} "
        f"wall_time_seconds={iteration_record['iteration_wall_time_seconds']:.3f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    gpu_ids = validate_args(args)
    run_dir = (
        args.resume_from.expanduser().resolve()
        if args.resume_from is not None
        else args.output_dir.expanduser().resolve()
    )
    if args.resume_from is not None and args.output_dir.expanduser().resolve() != run_dir:
        raise ValueError("--output-dir must equal --resume-from for a resumed run")
    config = stable_config(args, gpu_ids)
    state, state_path = load_or_create_state(args, run_dir, config)
    next_iteration = int(state["completed_iteration"]) + 1
    if args.num_iterations < next_iteration:
        raise ValueError(
            f"Requested target {args.num_iterations} is behind completed "
            f"iteration count {next_iteration}"
        )
    print(
        f"[online-zero3] run_dir={run_dir} target_iterations={args.num_iterations} "
        f"next_iteration={next_iteration} gpu_ids={gpu_ids} "
        f"rollout_world_size={args.rollout_world_size} "
        f"train_world_size={args.train_world_size} group_size={args.group_size}",
        flush=True,
    )
    for iteration in range(next_iteration, args.num_iterations):
        run_iteration(args, gpu_ids, state, state_path, run_dir, iteration)
    print(
        f"[online-zero3] complete=true global_step={state['global_step']} "
        f"completed_iteration={state['completed_iteration']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
