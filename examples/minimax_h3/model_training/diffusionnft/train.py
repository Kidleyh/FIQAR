#!/usr/bin/env python3
"""One-sample MiniMax-H3 LoRA diffusion forward/backward smoke.

This is intentionally not an RL training loop. It reads one generated video
from ``rollout.json``, reuses the existing MiniMax-H3 training module and
pipeline units, computes one video FlowMatch loss, and calls ``backward()``.
There is no reward weighting, advantage, optimizer, or model update.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F
from PIL import Image


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


def load_rollout_record(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("rollouts") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"Expected rollout list or object with 'rollouts': {path}")
    if index < 0 or index >= len(records):
        raise IndexError(f"--rollout-index={index} is outside 0..{len(records) - 1}")
    record = records[index]
    if not isinstance(record, dict):
        raise TypeError(f"Rollout record {index} must be an object")
    if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
        raise ValueError(f"Rollout record {index} has no prompt")
    video_path = record.get("video_path")
    if not isinstance(video_path, str) or not Path(video_path).is_file():
        raise FileNotFoundError(f"Rollout record {index} video does not exist: {video_path}")
    return record


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
    return frames, fps


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return len(parameters), sum(parameter.numel() for parameter in parameters)


def gradient_norm(model: torch.nn.Module) -> tuple[float, int]:
    squared_norm = 0.0
    tensors_with_grad = 0
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().float().norm(2).item()
        squared_norm += grad_norm * grad_norm
        tensors_with_grad += 1
    return math.sqrt(squared_norm), tensors_with_grad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MiniMax-H3 LoRA diffusion backward pass.")
    parser.add_argument("--rollout-json", type=Path, default=DEFAULT_ROLLOUT_JSON)
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-target-modules", default="qkv_proj,out_proj")
    parser.add_argument("--timestep-id", type=int, default=None)
    parser.add_argument("--timestep-seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=5678)
    parser.add_argument("--cpu-offload-split-threshold-mb", type=int, default=None)
    parser.add_argument("--use-gradient-checkpointing-offload", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This minimal training forward supports single-process/single-GPU only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MiniMax-H3 training forward")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select one CUDA GPU")
    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive")
    if (
        args.cpu_offload_split_threshold_mb is not None
        and args.cpu_offload_split_threshold_mb <= 0
    ):
        raise ValueError("--cpu-offload-split-threshold-mb must be positive")
    rollout_path = args.rollout_json.expanduser().resolve()
    if not rollout_path.is_file():
        raise FileNotFoundError(rollout_path)
    record = load_rollout_record(rollout_path, args.rollout_index)
    frames, fps = decode_video_frames(record["video_path"])
    if abs(fps - 24.0) >= 0.01:
        raise ValueError(f"MiniMax-H3 training video must be 24fps, got {fps}")

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
    print(f"[input] rollout={rollout_path} index={args.rollout_index}", flush=True)
    print(
        f"[input] video={record['video_path']} frames={len(frames)} "
        f"fps={fps:.3f} size={frames[0].size[0]}x{frames[0].size[1]}",
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
    model.pipe.device = str(device)
    offload_manager = OffloadTrainingManager(
        model,
        device,
        enable_optimizer_cpu_offload=False,
        cpu_offload_split_threshold=args.cpu_offload_split_threshold_mb,
    )
    trainable_tensors, trainable_params = count_trainable_parameters(model)
    if trainable_params == 0:
        raise RuntimeError("LoRA injection produced zero trainable parameters")
    model.zero_grad(set_to_none=True)
    print(
        f"[lora] trainable_parameter_tensors={trainable_tensors} "
        f"trainable_parameters={trainable_params}",
        flush=True,
    )

    data = {"video": frames, "prompt": record["prompt"]}
    inputs = model.get_pipeline_inputs(data)
    inputs = model.transfer_data_to_device(inputs, model.pipe.device, model.pipe.torch_dtype)
    for unit in model.pipe.units:
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    inputs_shared, inputs_posi, _ = inputs
    input_latents = inputs_shared["input_latents"]
    print(f"[latent] shape={list(input_latents.shape)} dtype={input_latents.dtype}", flush=True)

    model.pipe.scheduler.set_timesteps(1000, training=True)
    model.pipe.scheduler_audio.set_timesteps(1000, training=True)
    if args.timestep_id is None:
        generator = torch.Generator(device="cpu").manual_seed(args.timestep_seed)
        timestep_id = int(torch.randint(0, 1000, (1,), generator=generator).item())
    else:
        timestep_id = args.timestep_id
    if timestep_id < 0 or timestep_id >= len(model.pipe.scheduler.timesteps):
        raise ValueError(f"--timestep-id must be in 0..{len(model.pipe.scheduler.timesteps) - 1}")
    timestep_video = model.pipe.scheduler.timesteps[timestep_id].to(
        dtype=torch.float32, device=device
    )
    timestep_audio = model.pipe.scheduler_audio.timesteps[timestep_id].to(
        dtype=torch.float32, device=device
    )
    noise_generator = torch.Generator(device=device).manual_seed(args.noise_seed)
    noise = torch.randn(
        input_latents.shape,
        generator=noise_generator,
        device=device,
        dtype=input_latents.dtype,
    )
    inputs_shared["video_latents"] = model.pipe.scheduler.add_noise(
        input_latents, noise, timestep_video
    )
    target = model.pipe.scheduler.training_target(input_latents, noise, timestep_video)

    noise_pred_video, _ = model.pipe.model_fn(
        dit=model.pipe.dit,
        **inputs_posi,
        **inputs_shared,
        timestep_video=timestep_video,
        timestep_audio=timestep_audio,
    )
    mse = F.mse_loss(noise_pred_video.float(), target.float())
    timestep_weight = model.pipe.scheduler.training_weight(timestep_video)
    loss = mse * timestep_weight
    print(
        f"[forward] timestep_id={timestep_id} timestep={float(timestep_video):.6f} "
        f"mse={mse.item():.8f} weight={float(timestep_weight):.8f} "
        f"loss={loss.item():.8f}",
        flush=True,
    )
    loss.backward()
    grad_norm, tensors_with_grad = gradient_norm(model)
    offload_manager.after_backward()
    print(
        f"[backward] gradient_tensors={tensors_with_grad} gradient_norm={grad_norm:.8f}",
        flush=True,
    )
    if tensors_with_grad == 0 or not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(
            f"Backward produced invalid LoRA gradients: tensors={tensors_with_grad}, norm={grad_norm}"
        )
    print("[done] backward_success=true optimizer_step=false", flush=True)


if __name__ == "__main__":
    main()
