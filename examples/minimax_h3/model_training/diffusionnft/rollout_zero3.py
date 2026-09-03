#!/usr/bin/env python3
"""Two-rank MiniMax-H3 rollout with a ZeRO-3-sharded DiT.

The launcher runs three bounded stages:

1. ``prepare-condition`` on GPU 0 caches the exact frozen Qwen/keyframe path.
2. ``denoise`` makes every rank execute every DiT timestep through one
   Accelerate/DeepSpeed ZeRO-3 engine. Rank 0 commits temporary clean latents.
3. ``finalize`` on GPU 0 decodes, scores in memory, and writes the existing
   Phase-8.1 rollout artifact contract.

Only the DiT is registered with DeepSpeed. The verified single-GPU rollout,
training, scheduler, reward, and online entrypoints are not modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import accelerate
import torch
from accelerate import Accelerator
from PIL import Image
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_HELPER_DIR = REPO_ROOT / "scripts" / "test"
DEFAULT_MODEL_ID = "MiniMax/MiniMax-H3"
FORMAT_VERSION = 1

for entry in (REPO_ROOT, SCRIPT_DIR, INFERENCE_HELPER_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import rollout as single_rollout  # noqa: E402
import train_zero3 as zero3  # noqa: E402
from diffsynth.core import OffloadTrainingManager  # noqa: E402
from minimax_h3_stage1_pair_render import (  # noqa: E402
    H3_FPS,
    NEGATIVE_PROMPT,
    align_up,
    get_video_info,
    h3_supported_frame_count,
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


class Zero3RolloutModule(torch.nn.Module):
    """Register only the H3 DiT (plus one optimizer token) with ZeRO-3."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        object.__setattr__(self, "core", core)
        self.dit = core.pipe.dit
        # Accelerate's DeepSpeed path expects an optimizer. This scalar is
        # never read by the model and no backward/step occurs during rollout.
        self.engine_token = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        *,
        inputs_shared: dict[str, Any],
        inputs_posi: dict[str, Any],
        timestep_video: torch.Tensor,
        timestep_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.pipe.model_fn(
            dit=self.core.pipe.dit,
            **inputs_posi,
            **inputs_shared,
            timestep_video=timestep_video,
            timestep_audio=timestep_audio,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiniMax-H3 single-seed rollout using a two-rank ZeRO-3 DiT."
    )
    parser.add_argument(
        "--mode",
        choices=("prepare-condition", "denoise", "finalize"),
        required=True,
    )
    parser.add_argument("--data-json", type=Path, default=single_rollout.DEFAULT_DATA_JSON)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--height", type=int, default=736)
    parser.add_argument("--width", type=int, default=1088)
    parser.add_argument("--num-frames", default="175")
    parser.add_argument("--num-inference-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-target-modules", default="qkv_proj,out_proj")
    parser.add_argument(
        "--policy-role", choices=("base", "old"), default="base"
    )
    parser.add_argument("--policy-lora-sha256", default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--global-step-before", type=int, default=0)
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
    parser.add_argument("--save-rollout-video", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_latents(path: Path, payload: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in payload.items()},
        str(temporary),
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported cached value: {type(value)!r}")


def tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to_device(item, device) for item in value)
    return value


def validate_args(args: argparse.Namespace) -> None:
    if args.start < 0 or args.seed < 0:
        raise ValueError("--start and --seed must be non-negative")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    if args.cfg_scale != 1.0:
        raise ValueError("Phase 10 first version supports the verified cfg_scale=1 only")
    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive")
    validate_requested_frames(args.num_frames)
    if int(args.num_frames) != 175:
        raise ValueError("Phase 10 first version supports exactly 175 frames")
    if args.width != 1088 or args.height != 736:
        raise ValueError("Phase 10 first version supports exactly 1088x736")
    if args.policy_role == "base":
        if args.lora_path is not None or args.source_checkpoint is not None:
            raise ValueError("Base policy cannot use LoRA/source checkpoint")
        if args.global_step_before != 0:
            raise ValueError("Base policy requires --global-step-before 0")
    else:
        if args.lora_path is None or args.source_checkpoint is None:
            raise ValueError("Old policy requires --lora-path and --source-checkpoint")
        expected = (
            args.source_checkpoint.expanduser().resolve() / "old_lora.safetensors"
        )
        if args.lora_path.expanduser().resolve() != expected:
            raise ValueError(f"Old policy LoRA must be {expected}")
        if args.global_step_before <= 0:
            raise ValueError("Old policy requires positive --global-step-before")


def resolve_record(args: argparse.Namespace) -> dict[str, Any]:
    if (args.prompt is None) != (args.input_video is None):
        raise ValueError("--prompt and --input-video must be provided together")
    if args.prompt is not None:
        record_index = 0
        record = {"prompt": args.prompt, "file_path": str(args.input_video)}
    else:
        records = list(single_rollout.iter_records(args.data_json, args.start, 1))
        if len(records) != 1:
            raise RuntimeError("Expected exactly one input record")
        record_index, record = records[0]
    source_video = load_video_path_from_record(record).expanduser().resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    prompt = normalize_prompt(load_prompt_from_record(record))
    raw_frames, raw_fps, _, _ = get_video_info(source_video)
    frame_count = h3_supported_frame_count(
        raw_frames, raw_fps, args.num_frames, H3_FPS
    )
    if frame_count != 175:
        raise ValueError(
            f"Source video only supports {frame_count} frames; Phase 10 requires 175"
        )
    width, height = align_up(args.width, 32), align_up(args.height, 32)
    folder = safe_folder_name(source_video, set())
    prompt_dir = (
        args.output_dir.expanduser().resolve()
        / f"prompt_{record_index:06d}_{folder}"
    )
    lora_path = (
        args.lora_path.expanduser().resolve() if args.lora_path is not None else None
    )
    if lora_path is not None and not lora_path.is_file():
        raise FileNotFoundError(lora_path)
    lora_sha = sha256_file(lora_path) if lora_path is not None else None
    if args.policy_lora_sha256 is not None and args.policy_lora_sha256 != lora_sha:
        raise ValueError("--policy-lora-sha256 does not match --lora-path")
    source_checkpoint = (
        args.source_checkpoint.expanduser().resolve()
        if args.source_checkpoint is not None
        else None
    )
    return {
        "record_index": record_index,
        "source_video": str(source_video),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_dir": str(prompt_dir),
        "condition_image_path": str((prompt_dir / "condition_image.png").resolve()),
        "height": height,
        "width": width,
        "num_frames": frame_count,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "flow_shift": args.flow_shift,
        "audio_flow_shift": args.audio_flow_shift,
        "cfg_scale": args.cfg_scale,
        "negative_prompt": args.negative_prompt,
        "model_id": args.model_id,
        "policy_role": args.policy_role,
        "policy_lora_path": str(lora_path) if lora_path is not None else None,
        "policy_lora_sha256": lora_sha,
        "source_checkpoint": (
            str(source_checkpoint) if source_checkpoint is not None else None
        ),
        "global_step_before": args.global_step_before,
    }


def contract(args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        **{
            key: record[key]
            for key in (
                "record_index",
                "source_video",
                "prompt",
                "prompt_sha256",
                "condition_image_path",
                "height",
                "width",
                "num_frames",
                "seed",
                "num_inference_steps",
                "flow_shift",
                "audio_flow_shift",
                "cfg_scale",
                "negative_prompt",
                "model_id",
                "policy_role",
                "policy_lora_path",
                "policy_lora_sha256",
                "source_checkpoint",
                "global_step_before",
            )
        },
        "tiled": not args.no_tiled,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
    }


def cache_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = args.output_dir.expanduser().resolve() / "zero3_rollout_cache"
    return root, root / "manifest.json", root / "condition_inputs.pt"


def load_cache(
    args: argparse.Namespace, record: dict[str, Any], device: torch.device
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _, manifest_path, payload_path = cache_paths(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = contract(args, record)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Condition cache mismatch for {key}")
    condition_path = Path(record["condition_image_path"])
    if sha256_file(condition_path) != manifest.get("condition_image_sha256"):
        raise ValueError("Condition image SHA256 mismatch")
    if sha256_file(payload_path) != manifest.get("payload_sha256"):
        raise ValueError("Condition payload SHA256 mismatch")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if set(payload) != {"inputs_shared", "inputs_posi"}:
        raise ValueError(f"Unexpected condition payload keys: {sorted(payload)}")
    return (
        tree_to_device(payload["inputs_shared"], device),
        tree_to_device(payload["inputs_posi"], device),
        manifest,
    )


def prepare_condition(args: argparse.Namespace, record: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
    output_dir = args.output_dir.expanduser().resolve()
    if (output_dir / "rollout.json").exists():
        raise FileExistsError(f"Refusing to overwrite complete rollout: {output_dir}")
    cache_dir, manifest_path, payload_path = cache_paths(args)
    if manifest_path.exists() or payload_path.exists():
        raise FileExistsError(f"Phase 10 has no resume support: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = Path(record["prompt_dir"])
    prompt_dir.mkdir(parents=True, exist_ok=True)
    condition_path = Path(record["condition_image_path"])
    first_frame = read_first_frame(
        Path(record["source_video"]), record["width"], record["height"]
    )
    temporary_condition = condition_path.with_name(
        f".{condition_path.name}.{uuid.uuid4().hex}.tmp"
    )
    first_frame.save(temporary_condition, format="PNG")
    temporary_condition.replace(condition_path)
    with Image.open(condition_path) as image:
        condition_image = image.convert("RGB").copy()
    if condition_image.size != (record["width"], record["height"]):
        raise RuntimeError(f"Unexpected condition size: {condition_image.size}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    TrainingModule = zero3.single._load_sft_training_module_class()
    model_paths = (
        f"{args.model_id}:FL2VA/text_encoder/model*.safetensors,"
        f"{args.model_id}:FL2VA/video_vae/source/model.safetensors"
    )
    print("[phase10] stage=condition_prepare loading frozen Qwen/video-VAE", flush=True)
    model = TrainingModule(
        model_id_with_origin_paths=model_paths,
        processor_path=f"{args.model_id}:FL2VA/processor/",
        lora_base_model=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        device="cpu",
        task="sft",
    )
    model.pipe.device = str(device)
    model.pipe.scheduler.set_timesteps(
        args.num_inference_steps, shift=args.flow_shift
    )
    model.pipe.scheduler_audio.set_timesteps(
        args.num_inference_steps, shift=args.audio_flow_shift
    )
    offload_manager = OffloadTrainingManager(
        model, device, enable_optimizer_cpu_offload=False
    )
    inputs_posi = {"prompt": record["prompt"]}
    inputs_nega = {"negative_prompt": args.negative_prompt}
    inputs_shared = {
        "cfg_scale": args.cfg_scale,
        "height": record["height"],
        "width": record["width"],
        "num_frames": record["num_frames"],
        "seed": args.seed,
        "rand_device": "cpu",
        "tiled": not args.no_tiled,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "keyframes": [condition_image],
        "keyframe_indices": [0],
        "references": None,
        "ref_image_short_edge": 2048,
        "ref_video_short_edge": 768,
        "ref_video_max_pixels": 768 * 1344,
        "retake_video": None,
        "frame_regions_to_retake": None,
        "retake_audio": None,
        "seconds_regions_to_retake": None,
        "imgvid_cond_noise_aug": model.pipe.imgvid_cond_noise_aug,
        "audio_cond_noise_aug": model.pipe.audio_cond_noise_aug,
    }
    with torch.no_grad():
        for unit in model.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = model.pipe.unit_runner(
                unit, model.pipe, inputs_shared, inputs_posi, inputs_nega
            )
    offload_manager.after_backward()
    shared_keys = (
        "video_latents",
        "audio_latents",
        "keyframe_cond_anchor",
        "ref_visual_anchor",
        "ref_audio_anchor",
        "input_latents_video",
        "denoise_mask_video",
        "input_latents_audio",
        "denoise_mask_audio",
        "imgvid_cond_noise_aug",
        "audio_cond_noise_aug",
        "use_gradient_checkpointing",
        "use_gradient_checkpointing_offload",
    )
    cached_shared = {
        key: inputs_shared[key] for key in shared_keys if key in inputs_shared
    }
    cached_posi = {
        key: inputs_posi[key] for key in ("prompt_embeds", "packed")
    }
    if set(cached_posi) != {"prompt_embeds", "packed"}:
        raise RuntimeError("Frozen condition preparation missed prompt embeddings/packed")
    if cached_shared.get("keyframe_cond_anchor") is None:
        raise RuntimeError("Frozen condition preparation missed I2V keyframe anchor")
    atomic_torch_save(
        payload_path,
        {
            "inputs_shared": tree_to_cpu(cached_shared),
            "inputs_posi": tree_to_cpu(cached_posi),
        },
    )
    manifest = {
        **contract(args, record),
        "complete": True,
        "condition_image_sha256": sha256_file(condition_path),
        "payload_file": payload_path.name,
        "payload_sha256": sha256_file(payload_path),
    }
    atomic_json(manifest_path, manifest)
    print(
        f"[phase10] condition_cache_complete=true path={cache_dir} "
        f"condition={condition_path}",
        flush=True,
    )


def all_rank_contract(accelerator: Accelerator, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    local = torch.tensor(list(digest), dtype=torch.uint8, device=accelerator.device)
    gathered = accelerator.gather(local).view(accelerator.num_processes, -1)
    if not torch.equal(gathered, gathered[0].expand_as(gathered)):
        raise RuntimeError("Generation contract differs between ranks")
    return digest.hex()


def rank_consistency(
    accelerator: Accelerator,
    video: torch.Tensor,
    audio: torch.Tensor,
    timestep_index: int,
) -> None:
    summary = torch.stack(
        (
            video.float().mean(),
            video.float().square().mean(),
            audio.float().mean(),
            audio.float().square().mean(),
        )
    )
    gathered = accelerator.gather(summary).view(accelerator.num_processes, -1)
    if not torch.allclose(gathered, gathered[0].expand_as(gathered), rtol=0, atol=0):
        delta = float((gathered - gathered[0]).abs().max().item())
        raise RuntimeError(
            f"Rank latent divergence after timestep {timestep_index}: max_abs={delta}"
        )


def denoise(args: argparse.Namespace, record: dict[str, Any]) -> None:
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    if accelerator.distributed_type != accelerate.DistributedType.DEEPSPEED:
        raise RuntimeError(f"DeepSpeed required, got {accelerator.distributed_type}")
    plugin = accelerator.state.deepspeed_plugin
    plugin.deepspeed_config["gradient_clipping"] = 1.0
    zero_stage = int(plugin.deepspeed_config["zero_optimization"]["stage"])
    if accelerator.num_processes != 2 or zero_stage != 3:
        raise RuntimeError(
            f"Phase 10 requires world_size=2 and ZeRO-3, got "
            f"{accelerator.num_processes}/{zero_stage}"
        )
    zero3.rank_log(
        accelerator,
        f"world_size={accelerator.num_processes} rank={accelerator.process_index} "
        f"local_rank={accelerator.local_process_index} device={accelerator.device} "
        f"deepspeed_enabled=true zero_stage={zero_stage}",
    )
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    inputs_shared, inputs_posi, manifest = load_cache(
        args, record, accelerator.device
    )
    cache_digest = all_rank_contract(
        accelerator,
        {
            **contract(args, record),
            "condition_image_sha256": manifest["condition_image_sha256"],
            "payload_sha256": manifest["payload_sha256"],
        },
    )
    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
    TrainingModule = zero3.single._load_sft_training_module_class()
    model_paths = f"{args.model_id}:FL2VA/transformer/model*.safetensors"
    zero3.rank_log(accelerator, "stage=model_load loading H3 DiT on CPU")
    core = TrainingModule(
        model_id_with_origin_paths=model_paths,
        processor_path=None,
        lora_base_model="dit" if args.lora_path is not None else None,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        device="cpu",
        task="sft",
    )
    core.pipe.scheduler.set_timesteps(
        args.num_inference_steps, shift=args.flow_shift
    )
    core.pipe.scheduler_audio.set_timesteps(
        args.num_inference_steps, shift=args.audio_flow_shift
    )
    if args.lora_path is not None:
        zero3.single.load_adapter_state(
            core.pipe.dit, "default", args.lora_path.expanduser().resolve()
        )
        zero3.single.set_policy_adapter(core.pipe.dit, "default", trainable=False)
        patched = len(zero3.single.named_adapter_parameters(core.pipe.dit, "default")) // 2
        zero3.rank_log(accelerator, f"policy_role=old patched_modules={patched}")
        if patched != 104:
            raise RuntimeError(f"Expected 104 patched DiT modules, got {patched}")
    else:
        zero3.rank_log(accelerator, "policy_role=base patched_modules=0")
    for parameter in core.pipe.dit.parameters():
        parameter.requires_grad_(False)
    core.pipe.device = str(accelerator.device)
    wrapper = Zero3RolloutModule(core)
    optimizer = torch.optim.AdamW([wrapper.engine_token], lr=0.0)
    engine, optimizer = accelerator.prepare(wrapper, optimizer)
    del optimizer
    unwrapped = accelerator.unwrap_model(engine)
    zero_count, ordinary_count, moved_bytes = zero3.move_zero3_partition_storage_to_device(
        unwrapped, accelerator.device
    )
    moved_buffers = zero3.move_registered_buffers_to_device(
        unwrapped, accelerator.device
    )
    zero3.rank_log(
        accelerator,
        f"zero3_partition_tensors_moved={zero_count} "
        f"ordinary_parameter_tensors_moved={ordinary_count} "
        f"partition_storage_moved_mb={moved_bytes / 2**20:.2f} "
        f"registered_buffers_moved={moved_buffers}",
    )
    core = unwrapped.core
    video_latents = inputs_shared["video_latents"]
    audio_latents = inputs_shared["audio_latents"]
    expected_video_shape = (1, 24, 52, 46, 68)
    if tuple(video_latents.shape) != expected_video_shape:
        raise RuntimeError(
            f"Unexpected initial video latent shape {tuple(video_latents.shape)}; "
            f"expected {expected_video_shape}"
        )
    zero3.rank_log(
        accelerator,
        f"condition_contract_sha256={cache_digest} "
        f"video_latent_shape={list(video_latents.shape)} "
        f"audio_latent_shape={list(audio_latents.shape)}",
    )
    accelerator.wait_for_everyone()
    with torch.no_grad():
        for progress_id, timestep_video_cpu in enumerate(core.pipe.scheduler.timesteps):
            zero3.rank_log(
                accelerator,
                f"dit_forward=true timestep_index={progress_id} "
                f"timestep_video={float(timestep_video_cpu):.6f}",
            )
            timestep_video = timestep_video_cpu.unsqueeze(0).to(
                device=accelerator.device, dtype=torch.float32
            )
            timestep_audio = core.pipe.scheduler_audio.timesteps[
                progress_id
            ].unsqueeze(0).to(device=accelerator.device, dtype=torch.float32)
            forward_shared = dict(inputs_shared)
            forward_shared["video_latents"] = video_latents
            forward_shared["audio_latents"] = audio_latents
            forward_shared["use_gradient_checkpointing"] = False
            forward_shared["use_gradient_checkpointing_offload"] = False
            try:
                noise_video, noise_audio = engine(
                    inputs_shared=forward_shared,
                    inputs_posi=inputs_posi,
                    timestep_video=timestep_video,
                    timestep_audio=timestep_audio,
                )
            except torch.OutOfMemoryError:
                zero3.rank_log(
                    accelerator,
                    f"oom=true stage=dit_forward timestep_index={progress_id}",
                )
                raise
            video_latents = core.pipe.step(
                core.pipe.scheduler,
                video_latents,
                progress_id,
                noise_pred=noise_video,
                inpaint_mask=inputs_shared.get("denoise_mask_video"),
                input_latents=inputs_shared.get("input_latents_video"),
            )
            audio_latents = core.pipe.step(
                core.pipe.scheduler_audio,
                audio_latents,
                progress_id,
                noise_pred=noise_audio,
                inpaint_mask=inputs_shared.get("denoise_mask_audio"),
                input_latents=inputs_shared.get("input_latents_audio"),
            )
            del noise_video, noise_audio, forward_shared
            rank_consistency(
                accelerator, video_latents, audio_latents, progress_id
            )
            zero3.rank_log(
                accelerator,
                f"dit_forward_complete=true timestep_index={progress_id}",
            )
    peak = torch.tensor(
        [
            torch.cuda.max_memory_allocated(accelerator.device) / 2**20,
            torch.cuda.max_memory_reserved(accelerator.device) / 2**20,
        ],
        device=accelerator.device,
        dtype=torch.float64,
    )
    peaks = accelerator.gather(peak).view(accelerator.num_processes, 2).cpu()
    zero3.rank_log(
        accelerator,
        f"peak_gpu_allocated_mb={peak[0].item():.2f} "
        f"peak_gpu_reserved_mb={peak[1].item():.2f}",
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cache_dir, _, _ = cache_paths(args)
        denoise_latent_path = cache_dir / "denoised_clean_latents.safetensors"
        atomic_latents(
            denoise_latent_path,
            {
                "video_latents": video_latents,
                "audio_latents": audio_latents,
            },
        )
        atomic_json(
            cache_dir / "denoise_state.json",
            {
                **contract(args, record),
                "complete": True,
                "world_size": accelerator.num_processes,
                "zero_stage": zero_stage,
                "all_ranks_executed_timesteps": args.num_inference_steps,
                "clean_latent_path": str(denoise_latent_path.resolve()),
                "clean_latent_sha256": sha256_file(denoise_latent_path),
                "video_latent_shape": list(video_latents.shape),
                "audio_latent_shape": list(audio_latents.shape),
                "rank_peak_allocated_mb": [float(row[0]) for row in peaks],
                "rank_peak_reserved_mb": [float(row[1]) for row in peaks],
            },
        )
        print(
            f"[phase10] denoise_complete=true clean_latent={denoise_latent_path}",
            flush=True,
        )
    accelerator.wait_for_everyone()


def validate_denoise_state(
    args: argparse.Namespace, record: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    cache_dir, _, _ = cache_paths(args)
    state_path = cache_dir / "denoise_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for key, value in contract(args, record).items():
        if state.get(key) != value:
            raise ValueError(f"Denoise state mismatch for {key}")
    if state.get("complete") is not True:
        raise ValueError("Denoise state is not complete")
    if state.get("world_size") != 2 or state.get("zero_stage") != 3:
        raise ValueError("Denoise state is not a two-rank ZeRO-3 result")
    if state.get("all_ranks_executed_timesteps") != args.num_inference_steps:
        raise ValueError("Not all ranks completed every timestep")
    latent_path = Path(state["clean_latent_path"])
    if not latent_path.is_file() or sha256_file(latent_path) != state.get(
        "clean_latent_sha256"
    ):
        raise ValueError("Denoise clean latent is missing or corrupted")
    return state, latent_path


def finalize(args: argparse.Namespace, record: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
    denoise_state, denoise_latent_path = validate_denoise_state(args, record)
    output_dir = args.output_dir.expanduser().resolve()
    rollout_path = output_dir / "rollout.json"
    if rollout_path.exists():
        raise FileExistsError(f"Phase 10 has no resume/overwrite support: {rollout_path}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    latents_cpu = load_file(str(denoise_latent_path), device="cpu")
    if set(latents_cpu) != {"video_latents", "audio_latents"}:
        raise ValueError(f"Unexpected denoise latent keys: {sorted(latents_cpu)}")
    print("[phase10] stage=vae_decode rank0_only=true", flush=True)
    # Use the verified inference loader for decode. The training wrapper turns
    # the VAE's WarpedTensor register tokens into a training-time representation
    # that is not valid for inference decode; the inference VRAM manager keeps
    # their callable shape contract intact.
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Pipeline,
        ModelConfig,
    )
    _, total_bytes = torch.cuda.mem_get_info(device)
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": str(device),
        "computation_dtype": torch.bfloat16,
        "computation_device": str(device),
    }
    vae_configs = [
        ModelConfig(
            model_id=args.model_id,
            origin_file_pattern="FL2VA/video_vae/source/model.safetensors",
            **vram_config,
        )
    ]
    if args.save_rollout_video:
        vae_configs.append(
            ModelConfig(
                model_id=args.model_id,
                origin_file_pattern="FL2VA/audio_vae/model.safetensors",
                **vram_config,
            )
        )
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=str(device),
        model_configs=vae_configs,
        processor_config=None,
        vram_limit=max(1.0, total_bytes / 2**30 - 2.0),
    )
    try:
        pipe.load_models_to_device(["video_vae"])
        frames_tensor = pipe.video_vae.decode_video(
            latents_cpu["video_latents"].to(device=device),
            dtype=pipe.torch_dtype,
            tiled=not args.no_tiled,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )
        video = pipe.vae_output_to_video(frames_tensor, min_value=0, max_value=1)
    except torch.OutOfMemoryError:
        print("[phase10] oom=true stage=vae_decode", flush=True)
        raise
    if len(video) != record["num_frames"]:
        raise RuntimeError(
            f"Decoded {len(video)} frames, expected {record['num_frames']}"
        )
    audio = None
    audio_sample_rate = None
    if args.save_rollout_video:
        pipe.load_models_to_device(["audio_vae"])
        waveform = pipe.audio_vae.decode_audio(
            latents_cpu["audio_latents"].to(device=device),
            dtype=pipe.torch_dtype,
        )
        audio = pipe.output_audio_format_check(waveform)
        audio_sample_rate = pipe.audio_vae.sample_rate
    finalize_decode_peak = torch.cuda.max_memory_allocated(device) / 2**20
    del frames_tensor, pipe
    gc.collect()
    torch.cuda.empty_cache()

    scrfd = args.scrfd_model.expanduser().resolve()
    magface = args.magface_checkpoint.expanduser().resolve()
    print("[phase10] stage=reward rank0_only=true backend=in_memory", flush=True)
    try:
        reward_model = FaceQualityReward(
            evaluator_path=args.reward_evaluator,
            scrfd_model=scrfd,
            magface_checkpoint=magface,
            device="cuda:0",
        )
        reward_started = time.monotonic()
        reward = reward_model.score_frames(
            video,
            frame_stride=args.reward_frame_stride,
            max_frames_per_video=args.reward_max_frames,
            frame_face_aggregation=args.reward_frame_face_aggregation,
            missing_face_reward=args.missing_face_reward,
        )
        reward_seconds = time.monotonic() - reward_started
    except torch.OutOfMemoryError:
        print("[phase10] oom=true stage=reward", flush=True)
        raise
    prompt_dir = Path(record["prompt_dir"])
    latent_path = (prompt_dir / f"seed_{args.seed}_latents.safetensors").resolve()
    atomic_latents(latent_path, latents_cpu)
    video_path = (prompt_dir / f"seed_{args.seed}.mp4").resolve()
    if args.save_rollout_video:
        from diffsynth.utils.data.audio_video import write_video_audio

        write_video_audio(
            video=video,
            audio=audio,
            output_path=str(video_path),
            fps=H3_FPS,
            audio_sample_rate=audio_sample_rate,
            video_quality=8,
        )
    condition_path = Path(record["condition_image_path"])
    state_path = (prompt_dir / f"seed_{args.seed}_state.json").resolve()
    state = {
        "format_version": 1,
        "complete": True,
        "seed": args.seed,
        "reward": float(reward["reward"]),
        "mean_quality": reward["mean_quality"],
        "face_visible_ratio": float(reward["face_visible_ratio"]),
        "num_faces": int(reward["num_faces"]),
        "latent_path": str(latent_path),
        "latent_sha256": sha256_file(latent_path),
        "condition_image_path": str(condition_path),
        "condition_image_sha256": sha256_file(condition_path),
        "prompt": record["prompt"],
        "prompt_sha256": record["prompt_sha256"],
        "policy_role": record["policy_role"],
        "policy_lora_path": record["policy_lora_path"],
        "policy_lora_sha256": record["policy_lora_sha256"],
        "source_checkpoint": record["source_checkpoint"],
        "global_step_before": record["global_step_before"],
        "height": record["height"],
        "width": record["width"],
        "num_frames": record["num_frames"],
        "num_inference_steps": record["num_inference_steps"],
        "flow_shift": record["flow_shift"],
        "audio_flow_shift": record["audio_flow_shift"],
        "reward_frame_stride": args.reward_frame_stride,
        "reward_max_frames": args.reward_max_frames,
        "reward_frame_face_aggregation": args.reward_frame_face_aggregation,
        "missing_face_reward": args.missing_face_reward,
        "scrfd_model_sha256": sha256_file(scrfd),
        "magface_checkpoint_sha256": sha256_file(magface),
        "save_rollout_video": args.save_rollout_video,
        "video_path": str(video_path) if args.save_rollout_video else None,
        "fps": H3_FPS,
        "reward_backend": "in_memory",
        "rollout_backend": "accelerate-deepspeed-zero3",
        "rollout_world_size": 2,
        "rollout_zero_stage": 3,
        "rank_peak_allocated_mb": denoise_state["rank_peak_allocated_mb"],
        "rank_peak_reserved_mb": denoise_state["rank_peak_reserved_mb"],
        "rank0_decode_peak_allocated_mb": finalize_decode_peak,
        "reward_wall_time_seconds": reward_seconds,
    }
    atomic_json(state_path, state)
    result = {**state, "sample_state_path": str(state_path)}
    atomic_json(rollout_path, [result])
    print(
        f"[phase10] reward={state['reward']:.8f} "
        f"mean_quality={state['mean_quality']} "
        f"visible_ratio={state['face_visible_ratio']:.6f} "
        f"num_faces={state['num_faces']}",
        flush=True,
    )
    print(
        f"[phase10] clean_video_latent_shape={denoise_state['video_latent_shape']} "
        f"clean_audio_latent_shape={denoise_state['audio_latent_shape']}",
        flush=True,
    )
    print(
        f"[phase10] rollout_success=true mp4_saved={str(args.save_rollout_video).lower()} "
        f"rollout_json={rollout_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    record = resolve_record(args)
    if args.mode == "prepare-condition":
        prepare_condition(args, record)
    elif args.mode == "denoise":
        denoise(args, record)
    else:
        finalize(args, record)


if __name__ == "__main__":
    main()
