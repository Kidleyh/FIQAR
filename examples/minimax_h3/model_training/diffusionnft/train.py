#!/usr/bin/env python3
"""MiniMax-H3 LoRA diffusion backward and repeatable offline DiffusionNFT updates.

Modes:

* ``backward-smoke`` keeps the Phase-2 weighted FlowMatch MSE smoke.
* ``nft-step`` reads a same-prompt rollout group, reconstructs its inference
  timestep schedule, computes the official DiffusionNFT objective at multiple
  timesteps, and performs one checkpointable optimizer step.

Repeated updates are performed by invoking this entrypoint again with
``--resume-from``. This file intentionally contains no online rollout loop,
multi-GPU support, or audio reward/loss.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer, set_adapter
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[4]
SFT_TRAIN_PATH = REPO_ROOT / "examples" / "minimax_h3" / "model_training" / "train.py"
DEFAULT_ROLLOUT_JSON = REPO_ROOT / "outputs" / "minimax_h3_diffusionnft_rollout_smoke" / "rollout.json"
DEFAULT_MODEL_ID = "MiniMax/MiniMax-H3"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import OffloadTrainingManager  # noqa: E402


def _load_sft_training_module_class():
    spec = importlib.util.spec_from_file_location("minimax_h3_sft_train", SFT_TRAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load existing H3 trainer: {SFT_TRAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniMaxH3TrainingModule


def load_rollout_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("rollouts") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise TypeError(f"Expected non-empty rollout list or object with 'rollouts': {path}")
    validated = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Rollout record {index} must be an object")
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"Rollout record {index} has no prompt")
        video_path = record.get("video_path")
        if video_path is not None and not isinstance(video_path, str):
            raise TypeError(f"Rollout record {index} has invalid video_path: {video_path}")
        validated.append(record)
    return validated


def select_rollout_record(records: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if index < 0 or index >= len(records):
        raise IndexError(f"--rollout-index={index} is outside 0..{len(records) - 1}")
    return records[index]


def select_prompt_group(
    records: list[dict[str, Any]], index: int, group_size: int | None
) -> list[dict[str, Any]]:
    anchor = select_rollout_record(records, index)
    group = [record for record in records if record["prompt"] == anchor["prompt"]]
    if group_size is not None:
        if group_size < 2:
            raise ValueError("--group-size must be at least 2")
        group = group[:group_size]
    if len(group) < 2:
        raise ValueError("DiffusionNFT requires at least two same-prompt rollout samples")
    for group_index, record in enumerate(group):
        reward = record.get("reward")
        if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ValueError(f"Group record {group_index} has invalid reward: {reward}")
    return group


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_nft_rollout_artifacts(group: list[dict[str, Any]]) -> None:
    required_metadata = ("height", "width", "num_frames", "seed")
    expected_geometry = None
    for index, record in enumerate(group):
        for key in ("latent_path", "condition_image_path"):
            value = record.get(key)
            if not isinstance(value, str) or not Path(value).is_file():
                raise FileNotFoundError(
                    f"Formal nft-step requires an existing {key} for rollout "
                    f"record {index}; legacy mp4-only rollout is not accepted"
                )
        for key in required_metadata:
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Rollout record {index} has invalid {key}: {value}")
        geometry = (record["num_frames"], record["height"], record["width"])
        if expected_geometry is None:
            expected_geometry = geometry
        elif geometry != expected_geometry:
            raise ValueError(
                f"Same-prompt rollout geometry mismatch: {geometry} != {expected_geometry}"
            )
        if record["height"] % 32 or record["width"] % 32:
            raise ValueError(f"Invalid H3 rollout geometry: {geometry}")
        if record["num_frames"] < 5 or (record["num_frames"] - 5) % 17:
            raise ValueError(f"Invalid H3 rollout frame count: {record['num_frames']}")


def validate_rollout_policy_provenance(
    group: list[dict[str, Any]],
    resume_dir: Path | None,
    global_step: int,
) -> None:
    expected_role = "base" if resume_dir is None else "old"
    expected_lora = None
    expected_sha256 = None
    expected_checkpoint = None
    if resume_dir is not None:
        expected_checkpoint = resume_dir.resolve()
        expected_lora = (expected_checkpoint / "old_lora.safetensors").resolve()
        if not expected_lora.is_file():
            raise FileNotFoundError(expected_lora)
        expected_sha256 = _sha256(expected_lora)
    for index, record in enumerate(group):
        role = record.get("policy_role")
        lora_value = record.get("policy_lora_path")
        sha256 = record.get("policy_lora_sha256")
        checkpoint_value = record.get("source_checkpoint")
        step = record.get("global_step_before")
        actual_lora = Path(lora_value).expanduser().resolve() if lora_value else None
        actual_checkpoint = (
            Path(checkpoint_value).expanduser().resolve() if checkpoint_value else None
        )
        actual = (role, actual_lora, sha256, actual_checkpoint, step)
        expected = (
            expected_role,
            expected_lora,
            expected_sha256,
            expected_checkpoint,
            global_step,
        )
        if actual != expected:
            raise ValueError(
                f"Rollout policy provenance mismatch at record {index}: "
                f"actual={actual} expected={expected}"
            )
    print(
        f"[provenance] rollout_policy_role={expected_role} "
        f"rollout_policy_sha256={expected_sha256} "
        f"source_checkpoint={expected_checkpoint} "
        f"global_step_before={global_step} policy_match=true",
        flush=True,
    )


def load_rollout_clean_state(
    record: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, Image.Image]:
    latent_path = Path(record["latent_path"]).expanduser().resolve()
    state = load_file(str(latent_path), device="cpu")
    required = {"video_latents", "audio_latents"}
    if set(state) != required:
        raise ValueError(
            f"Rollout latent file must contain exactly {sorted(required)}, got {sorted(state)}"
        )
    video_x0 = state["video_latents"]
    audio_x0 = state["audio_latents"]
    expected_video_shape = (
        1,
        24,
        ((record["num_frames"] - 5) // 17) * 5 + 2,
        record["height"] // 16,
        record["width"] // 16,
    )
    expected_audio_shape = (
        2,
        32,
        round(record["num_frames"] / 24.0 * 40.0),
    )
    if tuple(video_x0.shape) != expected_video_shape:
        raise ValueError(
            f"Saved video latent shape {tuple(video_x0.shape)} != {expected_video_shape}"
        )
    if tuple(audio_x0.shape) != expected_audio_shape:
        raise ValueError(
            f"Saved audio latent shape {tuple(audio_x0.shape)} != {expected_audio_shape}"
        )
    condition_path = Path(record["condition_image_path"]).expanduser().resolve()
    saved_hash = record.get("condition_image_sha256")
    if saved_hash is not None and _sha256(condition_path) != saved_hash:
        raise ValueError(f"Condition image hash mismatch: {condition_path}")
    with Image.open(condition_path) as image:
        condition_image = image.convert("RGB").copy()
    if condition_image.size != (record["width"], record["height"]):
        raise ValueError(
            f"Condition image size {condition_image.size} does not match rollout geometry"
        )
    return (
        video_x0.to(device=device, dtype=dtype),
        audio_x0.to(device=device, dtype=dtype),
        condition_image,
    )


def rollout_scheduler_config(group: list[dict[str, Any]]) -> dict[str, int | float]:
    aliases = {
        "num_inference_steps": ("num_inference_steps", "render_num_inference_steps"),
        "flow_shift": ("flow_shift", "render_flow_shift"),
        "audio_flow_shift": ("audio_flow_shift", "render_audio_flow_shift"),
    }
    resolved: dict[str, int | float] = {}
    for output_key, candidate_keys in aliases.items():
        values = []
        for record_index, record in enumerate(group):
            value = next((record[key] for key in candidate_keys if key in record), None)
            if value is None:
                raise ValueError(
                    f"Rollout record {record_index} is missing {output_key}; regenerate it "
                    "with the Phase-4 rollout.py so training can reconstruct the exact scheduler"
                )
            values.append(value)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Same-prompt rollout group has inconsistent {output_key}: {values}")
        resolved[output_key] = values[0]
    num_steps = resolved["num_inference_steps"]
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError(f"Invalid rollout num_inference_steps: {num_steps}")
    for key in ("flow_shift", "audio_flow_shift"):
        value = resolved[key]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"Invalid rollout {key}: {value}")
        resolved[key] = float(value)
    return resolved


def selected_timestep_indices(
    num_inference_steps: int,
    timestep_fraction: float,
    *,
    shuffle: bool,
    seed: int,
) -> list[int]:
    # Match official DiffusionNFT: train the leading fraction of the rollout's
    # actual inference schedule. Unlike the former Phase-3 path, there is no
    # sampling from a separate 1000-step training schedule.
    count = int(num_inference_steps * timestep_fraction)
    if count < 1:
        raise ValueError(
            f"int(num_inference_steps={num_inference_steps} * "
            f"timestep_fraction={timestep_fraction}) is zero"
        )
    indices = list(range(count))
    if shuffle:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(count, generator=generator).tolist()
        indices = [indices[index] for index in permutation]
    return indices


def decode_video_frames(path: str | Path) -> tuple[list[Image.Image], float]:
    video_path = Path(path).expanduser().resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open rollout video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[Image.Image] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame).convert("RGB"))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from rollout video: {video_path}")
    if fps <= 0:
        raise RuntimeError(f"Invalid fps={fps} in rollout video: {video_path}")
    width, height = frames[0].size
    if width % 32 or height % 32:
        raise ValueError(f"MiniMax-H3 video size must be divisible by 32, got {width}x{height}")
    if len(frames) < 5 or (len(frames) - 5) % 17:
        raise ValueError(f"MiniMax-H3 frame count must satisfy 17n+5, got {len(frames)}")
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError("Rollout video contains inconsistent frame sizes")
    if abs(fps - 24.0) >= 0.01:
        raise ValueError(f"MiniMax-H3 training video must be 24fps, got {fps}")
    return frames, fps


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return len(parameters), sum(parameter.numel() for parameter in parameters)


def gradient_norm(parameters: list[torch.nn.Parameter]) -> tuple[float, int]:
    squared_norm = 0.0
    tensors_with_grad = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm(2).item()
        squared_norm += norm * norm
        tensors_with_grad += 1
    return math.sqrt(squared_norm), tensors_with_grad


def parameter_delta(
    before: list[torch.Tensor], parameters: list[torch.nn.Parameter]
) -> float:
    squared_delta = 0.0
    for previous, parameter in zip(before, parameters, strict=True):
        delta = parameter.detach().float() - previous.to(parameter.device, torch.float32)
        squared_delta += delta.square().sum().item()
    return math.sqrt(squared_delta)


def named_adapter_parameters(
    model: torch.nn.Module, adapter_name: str
) -> list[tuple[str, torch.nn.Parameter]]:
    marker = f".{adapter_name}."
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if marker in name and ("lora_A." in name or "lora_B." in name)
    ]


def add_old_adapter(
    dit: torch.nn.Module,
    lora_rank: int,
    target_modules: str,
    *,
    initialize_from_current: bool,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=target_modules.split(","),
    )
    inject_adapter_in_model(config, dit, adapter_name="old")
    current_named = named_adapter_parameters(dit, "default")
    old_named = named_adapter_parameters(dit, "old")
    if not current_named or len(current_named) != len(old_named):
        raise RuntimeError(
            f"Adapter parameter mismatch: current={len(current_named)} old={len(old_named)}"
        )
    old_by_name = {name: parameter for name, parameter in old_named}
    paired_old = []
    for current_name, current_parameter in current_named:
        old_name = current_name.replace(".default.", ".old.")
        if old_name not in old_by_name:
            raise RuntimeError(f"Missing old adapter parameter for {current_name}")
        old_parameter = old_by_name[old_name]
        if initialize_from_current:
            with torch.no_grad():
                old_parameter.copy_(current_parameter)
        # Keep the frozen EMA adapter in fp32. With bf16, the official early
        # decay values (for example 0.001 at global step 1) round the old state
        # back to current exactly and destroy the intended policy lag.
        old_parameter.data = old_parameter.data.float()
        paired_old.append(old_parameter)
    current = [parameter for _, parameter in current_named]
    # Both adapter copies must be classified as always-on-GPU by the existing
    # offload manager. Old is frozen immediately after manager construction.
    for parameter in current + paired_old:
        parameter.requires_grad_(True)
    return current, paired_old


def set_policy_adapter(dit: torch.nn.Module, name: str, *, trainable: bool) -> None:
    set_adapter(dit, name, inference_mode=not trainable)


def disable_all_adapters(dit: torch.nn.Module) -> None:
    for module in dit.modules():
        if isinstance(module, BaseTunerLayer):
            module.enable_adapters(False)


def enable_current_adapter(dit: torch.nn.Module) -> None:
    for module in dit.modules():
        if isinstance(module, BaseTunerLayer):
            module.enable_adapters(True)
    set_policy_adapter(dit, "default", trainable=True)


def no_adapter_has_grad(model: torch.nn.Module, adapter_name: str) -> bool:
    return all(parameter.grad is None for _, parameter in named_adapter_parameters(model, adapter_name))


def frozen_base_has_grad(model: torch.nn.Module) -> bool:
    for name, parameter in model.named_parameters():
        if "lora_A." in name or "lora_B." in name:
            continue
        if parameter.grad is not None:
            return True
    return False


def official_return_decay(step: int, decay_type: int) -> float:
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError("--old-decay-type must be 0, 1, or 2")
    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)


def update_old_policy(
    current_parameters: list[torch.nn.Parameter],
    old_parameters: list[torch.nn.Parameter],
    decay: float,
) -> None:
    with torch.no_grad():
        for current, old in zip(current_parameters, old_parameters, strict=True):
            old.copy_(
                old.detach().float() * decay
                + current.detach().float() * (1.0 - decay)
            )


def paired_parameter_distance(
    left: list[torch.nn.Parameter], right: list[torch.nn.Parameter]
) -> float:
    squared_distance = 0.0
    with torch.no_grad():
        for left_parameter, right_parameter in zip(left, right, strict=True):
            difference = left_parameter.detach().float() - right_parameter.detach().float()
            squared_distance += difference.square().sum().item()
    return math.sqrt(squared_distance)


def _rollout_lora_key(parameter_name: str, adapter_name: str) -> str:
    key = parameter_name.replace(f".lora_A.{adapter_name}.weight", ".lora_A.weight")
    key = key.replace(f".lora_B.{adapter_name}.weight", ".lora_B.weight")
    if key == parameter_name:
        raise ValueError(f"Unable to export adapter parameter name: {parameter_name}")
    return key


def adapter_export_state(dit: torch.nn.Module, adapter_name: str) -> dict[str, torch.Tensor]:
    return {
        _rollout_lora_key(name, adapter_name): parameter.detach().cpu().contiguous()
        for name, parameter in named_adapter_parameters(dit, adapter_name)
    }


def load_adapter_state(dit: torch.nn.Module, adapter_name: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = load_file(str(path), device="cpu")
    named = named_adapter_parameters(dit, adapter_name)
    expected = {_rollout_lora_key(name, adapter_name) for name, _ in named}
    if set(state) != expected:
        missing = sorted(expected - set(state))[:5]
        unexpected = sorted(set(state) - expected)[:5]
        raise RuntimeError(
            f"Adapter checkpoint keys do not match {adapter_name}: "
            f"missing={missing} unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in named:
            value = state[_rollout_lora_key(name, adapter_name)]
            if value.shape != parameter.shape:
                raise RuntimeError(
                    f"Adapter tensor shape mismatch for {name}: {value.shape} != {parameter.shape}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def checkpoint_config(
    args: argparse.Namespace, scheduler_config: dict[str, int | float]
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_id": args.model_id,
        "lora_rank": args.lora_rank,
        "lora_target_modules": args.lora_target_modules,
        "adv_eps": args.adv_eps,
        "adv_clip_max": args.adv_clip_max,
        "policy_beta": args.policy_beta,
        "kl_beta": args.kl_beta,
        "timestep_fraction": args.timestep_fraction,
        "shuffle_timesteps": args.shuffle_timesteps,
        "old_decay_type": args.old_decay_type,
        "learning_rate": args.learning_rate,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        **scheduler_config,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_checkpoint_state(checkpoint_dir: Path) -> dict[str, Any]:
    state_path = checkpoint_dir / "training_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state.get("global_step"), int) or state["global_step"] < 0:
        raise ValueError(f"Invalid checkpoint global_step in {state_path}")
    if not isinstance(state.get("nft_config"), dict):
        raise ValueError(f"Missing nft_config in {state_path}")
    return state


def validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    mismatches = {
        key: (saved.get(key), value)
        for key, value in current.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume NFT configuration mismatch: {mismatches}")


def save_checkpoint(
    checkpoint_dir: Path,
    dit: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    nft_config: dict[str, Any],
) -> None:
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_dir}")
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_dir.with_name(
        f".{checkpoint_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    current_state = adapter_export_state(dit, "default")
    old_state = adapter_export_state(dit, "old")
    if not current_state or set(current_state) != set(old_state):
        raise RuntimeError("Current/old LoRA export keys are empty or inconsistent")
    temporary.mkdir()
    try:
        # current_lora.safetensors intentionally uses the exact key convention
        # consumed by rollout.py --lora-path / BasePipeline.load_lora.
        save_file(current_state, str(temporary / "current_lora.safetensors"))
        save_file(old_state, str(temporary / "old_lora.safetensors"))
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        _write_json_atomic(
            temporary / "training_state.json",
            {"global_step": global_step, "nft_config": nft_config},
        )
        temporary.replace(checkpoint_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def scheduler_sigma(scheduler, timestep: torch.Tensor, device: torch.device) -> torch.Tensor:
    timestep_cpu = timestep.detach().to(scheduler.timesteps.device)
    timestep_id = torch.argmin((scheduler.timesteps - timestep_cpu).abs())
    return scheduler.sigmas[timestep_id].to(device=device, dtype=torch.float32)


def prepare_pipeline_inputs(model, frames: list[Image.Image], prompt: str):
    inputs = model.get_pipeline_inputs({"video": frames, "prompt": prompt})
    inputs = model.transfer_data_to_device(inputs, model.pipe.device, model.pipe.torch_dtype)
    for unit in model.pipe.units:
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    return inputs


def prepare_nft_pipeline_inputs(
    model,
    record: dict[str, Any],
    condition_image: Image.Image,
):
    # Reuse the existing training-module input contract for dimensions, then
    # explicitly remove input_video so no generated mp4 is decoded or VAE
    # encoded. The saved PNG is used for both FL2AV prompt presentation and the
    # keyframe VAE anchor, exactly as it was during rollout.
    placeholder_frames = [condition_image] * record["num_frames"]
    inputs_shared, inputs_posi, inputs_nega = model.get_pipeline_inputs(
        {"video": placeholder_frames, "prompt": record["prompt"]}
    )
    inputs_shared["input_video"] = None
    inputs_shared["keyframes"] = [condition_image]
    inputs_shared["keyframe_indices"] = [0]
    inputs_shared["seed"] = record["seed"]
    inputs = model.transfer_data_to_device(
        (inputs_shared, inputs_posi, inputs_nega),
        model.pipe.device,
        model.pipe.torch_dtype,
    )
    for unit in model.pipe.units:
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    inputs_shared, inputs_posi, inputs_nega = inputs
    if "input_latents" in inputs_shared:
        raise RuntimeError("Formal nft-step unexpectedly VAE-encoded an input video")
    if inputs_shared.get("keyframe_cond_anchor") is None:
        raise RuntimeError("Formal nft-step did not construct the FL2AV keyframe condition")
    return inputs_shared, inputs_posi, inputs_nega


def make_forward_diffusion(
    model,
    inputs_shared: dict[str, Any],
    timestep_id: int,
    noise_seed: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    input_latents = inputs_shared["input_latents"]
    timestep_video = model.pipe.scheduler.timesteps[timestep_id].to(
        dtype=torch.float32, device=device
    )
    timestep_audio = model.pipe.scheduler_audio.timesteps[timestep_id].to(
        dtype=torch.float32, device=device
    )
    noise_generator = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn(
        input_latents.shape,
        generator=noise_generator,
        device=device,
        dtype=input_latents.dtype,
    )
    x_t = model.pipe.scheduler.add_noise(input_latents, noise, timestep_video)
    velocity_target = model.pipe.scheduler.training_target(input_latents, noise, timestep_video)
    sigma = scheduler_sigma(model.pipe.scheduler, timestep_video, device)
    return input_latents, x_t, velocity_target, timestep_video, timestep_audio, sigma


def make_joint_forward_diffusion(
    model,
    video_x0: torch.Tensor,
    audio_x0: torch.Tensor,
    timestep_id: int,
    noise_seed: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    timestep_video = model.pipe.scheduler.timesteps[timestep_id].to(
        dtype=torch.float32, device=device
    )
    timestep_audio = model.pipe.scheduler_audio.timesteps[timestep_id].to(
        dtype=torch.float32, device=device
    )
    video_generator = torch.Generator(device=device).manual_seed(noise_seed)
    audio_generator = torch.Generator(device=device).manual_seed(noise_seed + 10_000_019)
    video_noise = torch.randn(
        video_x0.shape,
        generator=video_generator,
        device=device,
        dtype=video_x0.dtype,
    )
    audio_noise = torch.randn(
        audio_x0.shape,
        generator=audio_generator,
        device=device,
        dtype=audio_x0.dtype,
    )
    video_xt = model.pipe.scheduler.add_noise(video_x0, video_noise, timestep_video)
    audio_xt = model.pipe.scheduler_audio.add_noise(audio_x0, audio_noise, timestep_audio)
    sigma_video = scheduler_sigma(model.pipe.scheduler, timestep_video, device)
    sigma_audio = scheduler_sigma(model.pipe.scheduler_audio, timestep_audio, device)
    return video_xt, audio_xt, timestep_video, timestep_audio, sigma_video, sigma_audio


def model_velocity(
    model,
    inputs_shared: dict[str, Any],
    inputs_posi: dict[str, Any],
    x_t: torch.Tensor,
    timestep_video: torch.Tensor,
    timestep_audio: torch.Tensor,
    *,
    audio_x_t: torch.Tensor | None = None,
    checkpoint: bool,
) -> torch.Tensor:
    forward_inputs = dict(inputs_shared)
    forward_inputs["video_latents"] = x_t
    if audio_x_t is not None:
        forward_inputs["audio_latents"] = audio_x_t
    forward_inputs["use_gradient_checkpointing"] = checkpoint
    forward_inputs["use_gradient_checkpointing_offload"] = checkpoint and bool(
        inputs_shared.get("use_gradient_checkpointing_offload", False)
    )
    video_prediction, _ = model.pipe.model_fn(
        dit=model.pipe.dit,
        **inputs_posi,
        **forward_inputs,
        timestep_video=timestep_video,
        timestep_audio=timestep_audio,
    )
    return video_prediction


def sample_timestep_id(args: argparse.Namespace, sample_index: int) -> int:
    if args.timestep_id is not None:
        return args.timestep_id
    generator = torch.Generator(device="cpu").manual_seed(args.timestep_seed + sample_index)
    return int(torch.randint(0, 1000, (1,), generator=generator).item())


def validate_timestep(model, timestep_id: int) -> None:
    if timestep_id < 0 or timestep_id >= len(model.pipe.scheduler.timesteps):
        raise ValueError(f"--timestep-id must be in 0..{len(model.pipe.scheduler.timesteps) - 1}")


def run_backward_smoke(
    model,
    offload_manager: OffloadTrainingManager,
    record: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    current_parameters: list[torch.nn.Parameter],
) -> None:
    video_path = record.get("video_path")
    if not isinstance(video_path, str) or not Path(video_path).is_file():
        raise FileNotFoundError(
            "backward-smoke is the legacy mp4 diagnostic and requires video_path: "
            f"{video_path}"
        )
    frames, fps = decode_video_frames(video_path)
    print(
        f"[input] video={record['video_path']} frames={len(frames)} fps={fps:.3f} "
        f"size={frames[0].size[0]}x{frames[0].size[1]}",
        flush=True,
    )
    inputs_shared, inputs_posi, _ = prepare_pipeline_inputs(model, frames, record["prompt"])
    offload_manager.after_backward()
    input_latents = inputs_shared["input_latents"]
    print(f"[latent] shape={list(input_latents.shape)} dtype={input_latents.dtype}", flush=True)
    timestep_id = sample_timestep_id(args, 0)
    validate_timestep(model, timestep_id)
    x0, x_t, target, timestep_video, timestep_audio, _ = make_forward_diffusion(
        model, inputs_shared, timestep_id, args.noise_seed, device
    )
    prediction = model_velocity(
        model,
        inputs_shared,
        inputs_posi,
        x_t,
        timestep_video,
        timestep_audio,
        checkpoint=True,
    )
    mse = F.mse_loss(prediction.float(), target.float())
    weight = model.pipe.scheduler.training_weight(timestep_video)
    loss = mse * weight
    print(
        f"[forward] timestep_id={timestep_id} timestep={float(timestep_video):.6f} "
        f"mse={mse.item():.8f} weight={float(weight):.8f} loss={loss.item():.8f}",
        flush=True,
    )
    loss.backward()
    grad_norm, tensors_with_grad = gradient_norm(current_parameters)
    offload_manager.after_backward()
    print(
        f"[backward] gradient_tensors={tensors_with_grad} gradient_norm={grad_norm:.8f}",
        flush=True,
    )
    if tensors_with_grad == 0 or not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("Backward produced invalid LoRA gradients")
    print("[done] backward_success=true optimizer_step=false", flush=True)


def run_nft_step(
    model,
    offload_manager: OffloadTrainingManager,
    group: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    current_parameters: list[torch.nn.Parameter],
    old_parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    global_step: int,
    nft_config: dict[str, Any],
) -> int:
    rewards = torch.tensor([float(record["reward"]) for record in group], dtype=torch.float32)
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    advantages = (rewards - reward_mean) / (reward_std + args.adv_eps)
    clipped_advantages = advantages.clamp(-args.adv_clip_max, args.adv_clip_max)
    ratios = ((clipped_advantages / args.adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)
    print(f"[nft] rewards={rewards.tolist()}", flush=True)
    print(f"[nft] advantages={advantages.tolist()}", flush=True)
    print(f"[nft] clipped_advantages={clipped_advantages.tolist()}", flush=True)
    print(f"[nft] r={ratios.tolist()}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    before_step = [parameter.detach().cpu().float().clone() for parameter in current_parameters]
    metrics = {
        key: []
        for key in (
            "positive",
            "negative",
            "policy",
            "kl",
            "total",
            "current_reference_distance",
        )
    }
    num_inference_steps = int(nft_config["num_inference_steps"])
    num_train_timesteps = int(num_inference_steps * args.timestep_fraction)
    total_backward_passes = len(group) * num_train_timesteps
    current_old_distance_before = paired_parameter_distance(
        current_parameters, old_parameters
    )
    print(
        f"[step] global_step_before={global_step} samples={len(group)} "
        f"inference_timesteps={num_inference_steps} "
        f"training_timesteps_per_sample={num_train_timesteps} "
        f"total_backward_passes={total_backward_passes}",
        flush=True,
    )
    print(
        f"[policies] current_old_parameter_distance_before="
        f"{current_old_distance_before:.12f}",
        flush=True,
    )

    for sample_index, (record, ratio) in enumerate(zip(group, ratios, strict=True)):
        video_x0, audio_x0, condition_image = load_rollout_clean_state(
            record, device, model.pipe.torch_dtype
        )
        inputs_shared, inputs_posi, _ = prepare_nft_pipeline_inputs(
            model, record, condition_image
        )
        print(
            f"[rollout-state {sample_index}] video_latent_shape={list(video_x0.shape)} "
            f"audio_latent_shape={list(audio_x0.shape)} "
            f"condition_image_path={Path(record['condition_image_path']).resolve()} "
            "uses_rollout_clean_latent=true uses_exact_rollout_condition=true",
            flush=True,
        )
        # Frozen condition/text forwards do not participate in backward. Reset
        # the existing hook manager before the three joint policy forwards.
        offload_manager.after_backward()
        timestep_indices = selected_timestep_indices(
            num_inference_steps,
            args.timestep_fraction,
            shuffle=args.shuffle_timesteps,
            seed=args.timestep_seed + global_step * 100_003 + sample_index,
        )
        print(
            f"[sample {sample_index}] seed={record.get('seed')} "
            f"timestep_indices={timestep_indices}",
            flush=True,
        )
        for timestep_position, timestep_id in enumerate(timestep_indices):
            validate_timestep(model, timestep_id)
            noise_seed = (
                args.noise_seed
                + global_step * 1_000_003
                + sample_index * num_inference_steps
                + timestep_id
            )
            (
                video_xt,
                audio_xt,
                timestep_video,
                timestep_audio,
                sigma_video,
                sigma_audio,
            ) = make_joint_forward_diffusion(
                model,
                video_x0,
                audio_x0,
                timestep_id,
                noise_seed,
                device,
            )

            set_policy_adapter(model.pipe.dit, "old", trainable=False)
            with torch.no_grad():
                old_prediction = model_velocity(
                    model,
                    inputs_shared,
                    inputs_posi,
                    video_xt,
                    timestep_video,
                    timestep_audio,
                    audio_x_t=audio_xt,
                    checkpoint=False,
                ).detach()
            offload_manager.after_backward()

            disable_all_adapters(model.pipe.dit)
            with torch.no_grad():
                reference_prediction = model_velocity(
                    model,
                    inputs_shared,
                    inputs_posi,
                    video_xt,
                    timestep_video,
                    timestep_audio,
                    audio_x_t=audio_xt,
                    checkpoint=False,
                ).detach()
            offload_manager.after_backward()

            enable_current_adapter(model.pipe.dit)
            current_prediction = model_velocity(
                model,
                inputs_shared,
                inputs_posi,
                video_xt,
                timestep_video,
                timestep_audio,
                audio_x_t=audio_xt,
                checkpoint=True,
            )

            positive_prediction = (
                args.policy_beta * current_prediction
                + (1.0 - args.policy_beta) * old_prediction
            )
            implicit_negative_prediction = (
                (1.0 + args.policy_beta) * old_prediction
                - args.policy_beta * current_prediction
            )
            sigma_expanded = sigma_video.view(*([1] * video_x0.ndim))
            positive_x0 = video_xt - sigma_expanded * positive_prediction
            negative_x0 = video_xt - sigma_expanded * implicit_negative_prediction
            reduce_dims = tuple(range(1, video_x0.ndim))
            with torch.no_grad():
                positive_weight = (
                    (positive_x0.double() - video_x0.double())
                    .abs()
                    .mean(dim=reduce_dims, keepdim=True)
                    .clamp(min=1e-5)
                )
                negative_weight = (
                    (negative_x0.double() - video_x0.double())
                    .abs()
                    .mean(dim=reduce_dims, keepdim=True)
                    .clamp(min=1e-5)
                )
            positive_loss = (
                (positive_x0 - video_x0).square() / positive_weight
            ).mean(dim=reduce_dims)
            negative_loss = (
                (negative_x0 - video_x0).square() / negative_weight
            ).mean(dim=reduce_dims)
            original_policy_loss = (
                ratio.to(device) * positive_loss / args.policy_beta
                + (1.0 - ratio.to(device)) * negative_loss / args.policy_beta
            )
            policy_loss = (original_policy_loss * args.adv_clip_max).mean()
            kl_loss = F.mse_loss(
                current_prediction.float(), reference_prediction.float()
            )
            current_reference_distance = math.sqrt(max(kl_loss.item(), 0.0))
            total_loss = policy_loss + args.kl_beta * kl_loss
            (total_loss / total_backward_passes).backward()
            metrics["positive"].append(positive_loss.detach().mean().float().cpu())
            metrics["negative"].append(negative_loss.detach().mean().float().cpu())
            metrics["policy"].append(policy_loss.detach().float().cpu())
            metrics["kl"].append(kl_loss.detach().float().cpu())
            metrics["total"].append(total_loss.detach().float().cpu())
            metrics["current_reference_distance"].append(
                torch.tensor(current_reference_distance)
            )
            print(
                f"[sample {sample_index} timestep {timestep_position}] "
                f"seed={record.get('seed')} reward={float(record['reward']):.8f} "
                f"advantage={float(advantages[sample_index]):.8f} r={float(ratio):.8f} "
                f"timestep_id={timestep_id} timestep={float(timestep_video):.8f} "
                f"video_sigma={float(sigma_video):.8f} "
                f"audio_sigma={float(sigma_audio):.8f} "
                f"positive_loss={positive_loss.mean().item():.8f} "
                f"negative_loss={negative_loss.mean().item():.8f} "
                f"policy_loss={policy_loss.item():.8f} kl_loss={kl_loss.item():.8f} "
                f"current_reference_prediction_distance={current_reference_distance:.8f} "
                f"total_loss={total_loss.item():.8f}",
                flush=True,
            )
            offload_manager.after_backward()

    old_policy_no_grad = no_adapter_has_grad(model.pipe.dit, "old")
    reference_policy_no_grad = not frozen_base_has_grad(model.pipe.dit)
    preclip_grad_norm, tensors_with_grad = gradient_norm(current_parameters)
    clipped_norm = torch.nn.utils.clip_grad_norm_(current_parameters, args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    delta = parameter_delta(before_step, current_parameters)
    global_step += 1
    decay = official_return_decay(global_step, args.old_decay_type)
    update_old_policy(current_parameters, old_parameters, decay)
    current_old_distance_after = paired_parameter_distance(
        current_parameters, old_parameters
    )

    means = {key: torch.stack(values).mean().item() for key, values in metrics.items()}
    print(
        f"[group] positive_loss={means['positive']:.8f} negative_loss={means['negative']:.8f} "
        f"policy_loss={means['policy']:.8f} kl_loss={means['kl']:.8f} "
        f"current_reference_prediction_distance="
        f"{means['current_reference_distance']:.8f} total_loss={means['total']:.8f}",
        flush=True,
    )
    print(
        f"[grad] tensors={tensors_with_grad} gradient_norm={preclip_grad_norm:.8f} "
        f"clip_return_norm={float(clipped_norm):.8f} max_grad_norm={args.max_grad_norm}",
        flush=True,
    )
    print(
        f"[policies] old_policy_no_grad={str(old_policy_no_grad).lower()} "
        f"reference_policy_no_grad={str(reference_policy_no_grad).lower()} "
        f"old_decay_type={args.old_decay_type} decay_global_step={global_step} "
        f"old_decay={decay:.8f} current_old_parameter_distance_after="
        f"{current_old_distance_after:.12f}",
        flush=True,
    )
    print(f"[optimizer] current_parameter_delta={delta:.12f}", flush=True)
    if tensors_with_grad != len(current_parameters):
        raise RuntimeError(
            f"Not all current LoRA parameters received gradients: {tensors_with_grad}/{len(current_parameters)}"
        )
    if not math.isfinite(preclip_grad_norm) or preclip_grad_norm <= 0:
        raise RuntimeError(f"Invalid current-policy gradient norm: {preclip_grad_norm}")
    if delta <= 0 or not math.isfinite(delta):
        raise RuntimeError(f"Optimizer produced invalid parameter delta: {delta}")
    if not old_policy_no_grad:
        raise RuntimeError("Old policy unexpectedly received gradients")
    if not reference_policy_no_grad:
        raise RuntimeError("Reference/base policy unexpectedly received gradients")
    checkpoint_path = None
    if args.checkpoint_output is not None:
        checkpoint_path = args.checkpoint_output.expanduser().resolve()
        save_checkpoint(
            checkpoint_path,
            model.pipe.dit,
            optimizer,
            global_step,
            nft_config,
        )
        print(f"[checkpoint] path={checkpoint_path}", flush=True)
    print(
        f"[step] global_step_after={global_step} optimizer_step=true "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )
    return global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiniMax-H3 LoRA backward or checkpointable DiffusionNFT update."
    )
    parser.add_argument("--mode", choices=("backward-smoke", "nft-step"), default="backward-smoke")
    parser.add_argument("--rollout-json", type=Path, default=DEFAULT_ROLLOUT_JSON)
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-target-modules", default="qkv_proj,out_proj")
    parser.add_argument("--timestep-id", type=int, default=None)
    parser.add_argument("--timestep-seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=5678)
    parser.add_argument("--timestep-fraction", type=float, default=0.99)
    parser.add_argument(
        "--shuffle-timesteps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cpu-offload-split-threshold-mb", type=int, default=None)
    parser.add_argument("--use-gradient-checkpointing-offload", action="store_true")
    parser.add_argument("--adv-eps", type=float, default=1e-4)
    parser.add_argument("--adv-clip-max", type=float, default=5.0)
    parser.add_argument("--policy-beta", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--old-decay-type", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--require-policy-provenance", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This entrypoint supports single-process/single-GPU only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MiniMax-H3 training forward")
    if torch.device(args.device).type != "cuda":
        raise ValueError("--device must select one CUDA GPU")
    positive_values = {
        "lora-rank": args.lora_rank,
        "adv-eps": args.adv_eps,
        "adv-clip-max": args.adv_clip_max,
        "policy-beta": args.policy_beta,
        "learning-rate": args.learning_rate,
        "max-grad-norm": args.max_grad_norm,
        "timestep-fraction": args.timestep_fraction,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.timestep_fraction > 1:
        raise ValueError("--timestep-fraction must be in (0, 1]")
    if args.kl_beta < 0 or args.weight_decay < 0:
        raise ValueError("KL beta and weight decay must be non-negative")
    if args.mode != "nft-step" and (
        args.resume_from is not None or args.checkpoint_output is not None
    ):
        raise ValueError("Checkpoint save/resume is supported only in --mode nft-step")
    if (
        args.cpu_offload_split_threshold_mb is not None
        and args.cpu_offload_split_threshold_mb <= 0
    ):
        raise ValueError("--cpu-offload-split-threshold-mb must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    rollout_path = args.rollout_json.expanduser().resolve()
    if not rollout_path.is_file():
        raise FileNotFoundError(rollout_path)
    records = load_rollout_records(rollout_path)
    selected = (
        select_prompt_group(records, args.rollout_index, args.group_size)
        if args.mode == "nft-step"
        else [select_rollout_record(records, args.rollout_index)]
    )
    scheduler_config: dict[str, int | float] = {}
    nft_config: dict[str, Any] = {}
    resume_dir = None
    resume_state = None
    global_step = 0
    if args.mode == "nft-step":
        validate_nft_rollout_artifacts(selected)
        scheduler_config = rollout_scheduler_config(selected)
        # Validate the official leading-fraction selection before loading H3.
        selected_timestep_indices(
            int(scheduler_config["num_inference_steps"]),
            args.timestep_fraction,
            shuffle=False,
            seed=args.timestep_seed,
        )
        nft_config = checkpoint_config(args, scheduler_config)
        if args.resume_from is not None:
            resume_dir = args.resume_from.expanduser().resolve()
            if not resume_dir.is_dir():
                raise FileNotFoundError(resume_dir)
            resume_state = read_checkpoint_state(resume_dir)
            validate_resume_config(resume_state["nft_config"], nft_config)
            global_step = int(resume_state["global_step"])
        if args.require_policy_provenance:
            validate_rollout_policy_provenance(selected, resume_dir, global_step)
    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
    torch.cuda.set_device(device)
    torch.manual_seed(args.noise_seed)
    torch.cuda.manual_seed_all(args.noise_seed)

    MiniMaxH3TrainingModule = _load_sft_training_module_class()
    model_paths = (
        f"{args.model_id}:FL2VA/text_encoder/model*.safetensors,"
        f"{args.model_id}:FL2VA/transformer/model*.safetensors,"
        f"{args.model_id}:FL2VA/video_vae/source/model.safetensors,"
        f"{args.model_id}:FL2VA/audio_vae/model.safetensors"
    )
    print(
        f"[input] rollout={rollout_path} mode={args.mode} samples={len(selected)}",
        flush=True,
    )
    print("[model] loading existing MiniMaxH3TrainingModule on CPU", flush=True)
    model = MiniMaxH3TrainingModule(
        model_id_with_origin_paths=model_paths,
        processor_path=f"{args.model_id}:FL2VA/processor/",
        lora_base_model="dit",
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=None,
        device="cpu",
        task="sft",
    )
    if args.mode == "nft-step":
        model.pipe.scheduler.set_timesteps(
            int(scheduler_config["num_inference_steps"]),
            shift=float(scheduler_config["flow_shift"]),
            training=True,
        )
        model.pipe.scheduler_audio.set_timesteps(
            int(scheduler_config["num_inference_steps"]),
            shift=float(scheduler_config["audio_flow_shift"]),
            training=True,
        )
        print(
            f"[scheduler] num_inference_steps={scheduler_config['num_inference_steps']} "
            f"flow_shift={scheduler_config['flow_shift']} "
            f"audio_flow_shift={scheduler_config['audio_flow_shift']} "
            f"video_timesteps={model.pipe.scheduler.timesteps.tolist()}",
            flush=True,
        )
    else:
        model.pipe.scheduler.set_timesteps(1000, training=True)
        model.pipe.scheduler_audio.set_timesteps(1000, training=True)
    current_parameters = [
        parameter for _, parameter in named_adapter_parameters(model.pipe.dit, "default")
    ]
    old_parameters: list[torch.nn.Parameter] = []
    if args.mode == "nft-step":
        current_parameters, old_parameters = add_old_adapter(
            model.pipe.dit,
            args.lora_rank,
            args.lora_target_modules,
            initialize_from_current=resume_dir is None,
        )
        if resume_dir is not None:
            load_adapter_state(
                model.pipe.dit, "default", resume_dir / "current_lora.safetensors"
            )
            load_adapter_state(
                model.pipe.dit, "old", resume_dir / "old_lora.safetensors"
            )
    if not current_parameters:
        raise RuntimeError("LoRA injection produced zero current-policy parameters")
    model.pipe.device = str(device)
    offload_manager = OffloadTrainingManager(
        model,
        device,
        enable_optimizer_cpu_offload=False,
        cpu_offload_split_threshold=args.cpu_offload_split_threshold_mb,
    )
    set_policy_adapter(model.pipe.dit, "default", trainable=True)
    for parameter in old_parameters:
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    trainable_tensors, trainable_params = count_trainable_parameters(model)
    print(
        f"[lora] trainable_parameter_tensors={trainable_tensors} "
        f"trainable_parameters={trainable_params} old_parameter_tensors={len(old_parameters)}",
        flush=True,
    )
    if trainable_tensors != len(current_parameters):
        raise RuntimeError(
            f"Only current LoRA may be trainable: model={trainable_tensors}, current={len(current_parameters)}"
        )

    if args.mode == "nft-step":
        optimizer = torch.optim.AdamW(
            current_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        if resume_dir is not None:
            optimizer_path = resume_dir / "optimizer.pt"
            if not optimizer_path.is_file():
                raise FileNotFoundError(optimizer_path)
            optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=device, weights_only=True)
            )
            print(
                f"[resume] path={resume_dir} resume_success=true "
                f"global_step={global_step} current_old_parameter_distance="
                f"{paired_parameter_distance(current_parameters, old_parameters):.12f}",
                flush=True,
            )
        else:
            print(
                f"[resume] path=None resume_success=false global_step={global_step}",
                flush=True,
            )
        run_nft_step(
            model,
            offload_manager,
            selected,
            args,
            device,
            current_parameters,
            old_parameters,
            optimizer,
            global_step,
            nft_config,
        )
    else:
        run_backward_smoke(
            model,
            offload_manager,
            selected[0],
            args,
            device,
            current_parameters,
        )


if __name__ == "__main__":
    main()
