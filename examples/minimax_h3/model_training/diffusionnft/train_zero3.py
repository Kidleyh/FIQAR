#!/usr/bin/env python3
"""DeepSpeed ZeRO-3 MiniMax-H3 DiffusionNFT optimizer step.

This is deliberately separate from the verified single-GPU ``train.py``.
It imports that file's rollout, scheduler, noise, adapter and loss helpers,
while making every joint-DiT policy forward enter through the DeepSpeed
engine returned by ``Accelerator.prepare``.
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
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import accelerate
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import save_file

from prepare_zero3_conditions import load_condition_cache


REPO_ROOT = Path(__file__).resolve().parents[4]
SINGLE_TRAIN_PATH = Path(__file__).with_name("train.py")
DEFAULT_MODEL_ID = "MiniMax/MiniMax-H3"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.diffusion.runner import (  # noqa: E402
    initialize_deepspeed_gradient_checkpointing,
)


def _load_single_helpers():
    spec = importlib.util.spec_from_file_location(
        "minimax_h3_diffusionnft_single", SINGLE_TRAIN_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load single-GPU helpers: {SINGLE_TRAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


single = _load_single_helpers()


class Zero3NFTModule(torch.nn.Module):
    """Route condition preparation and all policy forwards through ZeRO-3."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        # Keep the pipeline as a non-registered helper. Frozen conditioning is
        # prepared in a separate verified single-GPU process and loaded from a
        # strict cache; only the NFT DiT belongs to the ZeRO-3 training engine.
        object.__setattr__(self, "core", core)
        self.dit = core.pipe.dit

    def forward(self, action: str, **kwargs):
        if action != "velocity":
            raise ValueError(f"Unknown ZeRO-3 forward action: {action}")
        policy = kwargs.pop("policy")
        if policy == "old":
            single.set_policy_adapter(self.core.pipe.dit, "old", trainable=False)
        elif policy == "reference":
            single.disable_all_adapters(self.core.pipe.dit)
        elif policy == "current":
            single.enable_current_adapter(self.core.pipe.dit)
        else:
            raise ValueError(f"Unknown policy role: {policy}")
        return single.model_velocity(self.core, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiniMax-H3 DiffusionNFT nft-step with Accelerate/DeepSpeed ZeRO-3."
    )
    parser.add_argument("--mode", choices=("nft-step",), default="nft-step")
    parser.add_argument("--rollout-json", type=Path, required=True)
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-target-modules", default="qkv_proj,out_proj")
    parser.add_argument("--timestep-seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=5678)
    parser.add_argument("--timestep-fraction", type=float, default=0.99)
    parser.add_argument(
        "--shuffle-timesteps", action=argparse.BooleanOptionalAction, default=True
    )
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
    parser.add_argument("--initial-lora-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--condition-cache-dir", type=Path, required=True)
    parser.add_argument("--require-policy-provenance", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, accelerator: Accelerator) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MiniMax-H3 ZeRO-3 training")
    if accelerator.distributed_type != accelerate.DistributedType.DEEPSPEED:
        raise RuntimeError(
            f"This entrypoint requires Accelerate DeepSpeed, got "
            f"{accelerator.distributed_type}"
        )
    plugin = accelerator.state.deepspeed_plugin
    zero_stage = int(plugin.deepspeed_config["zero_optimization"]["stage"])
    if zero_stage != 3:
        raise RuntimeError(f"DeepSpeed ZeRO stage 3 is required, got {zero_stage}")
    if accelerator.num_processes < 1:
        raise RuntimeError("Invalid distributed world size")
    positive = {
        "lora-rank": args.lora_rank,
        "adv-eps": args.adv_eps,
        "adv-clip-max": args.adv_clip_max,
        "policy-beta": args.policy_beta,
        "learning-rate": args.learning_rate,
        "max-grad-norm": args.max_grad_norm,
        "timestep-fraction": args.timestep_fraction,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.timestep_fraction > 1:
        raise ValueError("--timestep-fraction must be in (0, 1]")
    if args.kl_beta < 0 or args.weight_decay < 0:
        raise ValueError("KL beta and weight decay must be non-negative")
    if args.initial_lora_path is not None and args.resume_from is not None:
        raise ValueError("--initial-lora-path and --resume-from are mutually exclusive")


def sha256_tensor_contract(
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    ratios: torch.Tensor,
) -> str:
    payload = json.dumps(
        {
            "rewards": rewards.tolist(),
            "advantages": advantages.tolist(),
            "ratios": ratios.tolist(),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_global_reward_contract(
    accelerator: Accelerator,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    ratios: torch.Tensor,
) -> str:
    digest = bytes.fromhex(sha256_tensor_contract(rewards, advantages, ratios))
    local = torch.tensor(list(digest), device=accelerator.device, dtype=torch.uint8)
    gathered = accelerator.gather(local).view(accelerator.num_processes, -1)
    if not torch.equal(gathered, gathered[0].expand_as(gathered)):
        raise RuntimeError("Reward/advantage/r hash differs between ranks")
    return digest.hex()


def distributed_mean(
    values: list[torch.Tensor], accelerator: Accelerator
) -> float:
    if values:
        local_sum = torch.stack([value.detach().float() for value in values]).sum()
        local_count = torch.tensor(float(len(values)), device=accelerator.device)
    else:
        local_sum = torch.zeros((), device=accelerator.device)
        local_count = torch.zeros((), device=accelerator.device)
    packed = torch.stack((local_sum.to(accelerator.device), local_count))
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    if packed[1].item() <= 0:
        raise RuntimeError("Cannot average empty distributed metrics")
    return float((packed[0] / packed[1]).item())


def gathered_parameter_context(parameters, modifier_rank=None):
    import deepspeed

    parameters = list(parameters)
    partitioned = [hasattr(parameter, "ds_id") for parameter in parameters]
    if not any(partitioned):
        # Frozen old LoRA is intentionally injected after prepare and remains
        # replicated FP32 on each GPU. It is not a ZeRO-sharded parameter.
        return nullcontext()
    if not all(partitioned):
        raise RuntimeError("Cannot gather a mixed ZeRO/ordinary parameter list")
    return deepspeed.zero.GatheredParameters(
        parameters, modifier_rank=modifier_rank, enabled=True
    )


def gathered_adapter_snapshot(
    accelerator: Accelerator,
    current_parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor] | None:
    with gathered_parameter_context(current_parameters):
        if accelerator.is_main_process:
            return [p.detach().cpu().float().clone() for p in current_parameters]
    return None


def gathered_delta(
    accelerator: Accelerator,
    before: list[torch.Tensor] | None,
    current_parameters: list[torch.nn.Parameter],
) -> float | None:
    value = None
    with gathered_parameter_context(current_parameters):
        if accelerator.is_main_process:
            if before is None:
                raise RuntimeError("Main rank has no pre-step snapshot")
            value = single.parameter_delta(before, current_parameters)
    return value


def gathered_policy_distance(
    accelerator: Accelerator,
    current_parameters: list[torch.nn.Parameter],
    old_parameters: list[torch.nn.Parameter],
) -> float | None:
    value = None
    # ZeRO coalesced all-gather requires one dtype per gather. Current LoRA is
    # bf16 while the EMA old adapter intentionally remains fp32, so gather the
    # two logical policies in nested, dtype-homogeneous collectives.
    with gathered_parameter_context(current_parameters):
        with gathered_parameter_context(old_parameters):
            if accelerator.is_main_process:
                value = single.paired_parameter_distance(
                    current_parameters, old_parameters
                )
    return value


def update_old_policy_zero3(
    accelerator: Accelerator,
    current_parameters: list[torch.nn.Parameter],
    old_parameters: list[torch.nn.Parameter],
    decay: float,
) -> None:
    # Current is gathered by every rank. Old is a replicated frozen FP32
    # adapter injected after prepare, so every rank applies the same EMA and
    # retains an identical logical old policy without BF16 rounding collapse.
    with gathered_parameter_context(current_parameters):
        with gathered_parameter_context(old_parameters):
            single.update_old_policy(current_parameters, old_parameters, decay)
    accelerator.wait_for_everyone()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def raise_main_process_error(
    accelerator: Accelerator, error: str | None, operation: str
) -> None:
    payload = [error if accelerator.is_main_process else None]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"Main-process {operation} failed: {payload[0]}")


def save_zero3_checkpoint(
    accelerator: Accelerator,
    engine,
    core,
    current_parameters: list[torch.nn.Parameter],
    old_parameters: list[torch.nn.Parameter],
    checkpoint_dir: Path,
    global_step: int,
    nft_config: dict[str, Any],
) -> None:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    # The run directory is on shared storage. Make every rank fail before any
    # collective if the final path already exists; a rank-0-only exception here
    # would otherwise strand peers in the following broadcast.
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_dir}")
    temporary_value = None
    if accelerator.is_main_process:
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_value = str(
            checkpoint_dir.with_name(
                f".{checkpoint_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            )
        )
    objects = [temporary_value]
    dist.broadcast_object_list(objects, src=0)
    temporary = Path(objects[0])
    if accelerator.is_main_process:
        temporary.mkdir()
    accelerator.wait_for_everyone()
    tag = f"global_step{global_step}"
    with hide_ordinary_parameters_from_zero3_scan(
        core.pipe.dit, old_parameters
    ):
        success = engine.save_checkpoint(
            str(temporary / "deepspeed"),
            tag=tag,
            client_state={"global_step": global_step},
            save_latest=True,
            exclude_frozen_parameters=True,
        )
    save_ok = torch.tensor(
        int(success is not False), device=accelerator.device, dtype=torch.int32
    )
    dist.all_reduce(save_ok, op=dist.ReduceOp.MIN)
    if not bool(save_ok.item()):
        raise RuntimeError("DeepSpeed save_checkpoint returned false")
    accelerator.wait_for_everyone()
    export_error = None
    with gathered_parameter_context(current_parameters):
        with gathered_parameter_context(old_parameters):
            if accelerator.is_main_process:
                try:
                    current_state = single.adapter_export_state(
                        core.pipe.dit, "default"
                    )
                    old_state = single.adapter_export_state(core.pipe.dit, "old")
                    if not current_state or set(current_state) != set(old_state):
                        raise RuntimeError(
                            "Current/old gathered LoRA export is inconsistent"
                        )
                    save_file(
                        current_state,
                        str(temporary / "current_lora.safetensors"),
                    )
                    save_file(old_state, str(temporary / "old_lora.safetensors"))
                    _atomic_json(
                        temporary / "training_state.json",
                        {
                            "format_version": 2,
                            "backend": "accelerate-deepspeed-zero3",
                            "global_step": global_step,
                            "world_size": accelerator.num_processes,
                            "deepspeed_tag": tag,
                            "nft_config": nft_config,
                        },
                    )
                except Exception as exc:  # synchronize instead of hanging peers
                    export_error = f"{type(exc).__name__}: {exc}"
    raise_main_process_error(accelerator, export_error, "LoRA export")
    commit_error = None
    if accelerator.is_main_process:
        try:
            temporary.replace(checkpoint_dir)
        except Exception as exc:
            commit_error = f"{type(exc).__name__}: {exc}"
    raise_main_process_error(accelerator, commit_error, "checkpoint commit")


def load_zero3_checkpoint(
    accelerator: Accelerator,
    engine,
    checkpoint_dir: Path,
    state: dict[str, Any],
) -> None:
    tag = state.get("deepspeed_tag")
    if not isinstance(tag, str):
        raise ValueError("ZeRO-3 checkpoint is missing deepspeed_tag")
    load_path, client_state = engine.load_checkpoint(
        str(checkpoint_dir / "deepspeed"),
        tag=tag,
        load_module_strict=False,
        load_optimizer_states=True,
        load_lr_scheduler_states=False,
        load_module_only=False,
    )
    if load_path is None:
        raise RuntimeError(f"DeepSpeed failed to load checkpoint: {checkpoint_dir}")
    if int(client_state.get("global_step", -1)) != int(state["global_step"]):
        raise RuntimeError("DeepSpeed client_state global_step mismatch")
    accelerator.wait_for_everyone()


def read_zero3_state(checkpoint_dir: Path) -> dict[str, Any]:
    state = single.read_checkpoint_state(checkpoint_dir)
    if state.get("backend") != "accelerate-deepspeed-zero3":
        raise ValueError(f"Not a Phase-9 ZeRO-3 checkpoint: {checkpoint_dir}")
    for name in ("current_lora.safetensors", "old_lora.safetensors"):
        path = checkpoint_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    if not (checkpoint_dir / "deepspeed").is_dir():
        raise FileNotFoundError(checkpoint_dir / "deepspeed")
    tag = state.get("deepspeed_tag")
    if not isinstance(tag, str) or not (checkpoint_dir / "deepspeed" / tag).is_dir():
        raise FileNotFoundError(checkpoint_dir / "deepspeed" / str(tag))
    return state


def rank_log(accelerator: Accelerator, message: str) -> None:
    print(f"[rank {accelerator.process_index}] {message}", flush=True)


def distributed_all_true(value: bool, accelerator: Accelerator) -> bool:
    flag = torch.tensor(int(value), device=accelerator.device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def move_registered_buffers_to_device(
    module: torch.nn.Module, device: torch.device
) -> int:
    moved = 0
    for submodule in module.modules():
        for name, buffer in submodule.named_buffers(recurse=False):
            if buffer.device != device:
                setattr(submodule, name, buffer.to(device=device))
                moved += 1
    return moved


def install_zero3_parameter_containers_for_late_modules(
    module: torch.nn.Module,
) -> int:
    """Give post-prepare PEFT modules DeepSpeed's parameter dictionary type.

    ZeRO-3 replaces every existing module's ``_parameters`` mapping with a
    ``ZeROOrderedDict`` whose ``_in_forward`` flag is used by the engine
    prologue. Late old-adapter injection creates new frozen LoRA submodules
    with ordinary dictionaries. Wrapping only those mappings makes them safe
    engine children without partitioning their FP32 parameters.
    """
    from deepspeed.runtime.zero.parameter_offload import ZeROOrderedDict

    installed = 0
    for child in module.modules():
        if hasattr(child._parameters, "_in_forward"):
            continue
        original = child._parameters
        replacement = ZeROOrderedDict(parent_module=child)
        replacement.update(original)
        child._original_parameters = original
        child._parameters = replacement
        installed += 1
    return installed


@contextmanager
def hide_ordinary_parameters_from_zero3_scan(
    module: torch.nn.Module,
    parameters: list[torch.nn.Parameter],
):
    """Temporarily hide replicated frozen params from ZeRO's global scan.

    Post-prepare old-LoRA modules have no ZeRO per-module hooks and work as
    ordinary FP32 parameters during their frozen forward. At optimizer and
    shard-save boundaries, however, ZeRO recursively assumes every registered
    parameter owns ``ds_*`` partition metadata. Remove only the known old
    parameters for that short boundary and restore the exact mappings after.
    """
    target_ids = {id(parameter) for parameter in parameters}
    removed: list[tuple[torch.nn.Module, str, torch.nn.Parameter]] = []
    for child in module.modules():
        for name, parameter in list(child._parameters.items()):
            if parameter is not None and id(parameter) in target_ids:
                removed.append((child, name, parameter))
                del child._parameters[name]
    if len(removed) != len(parameters):
        for child, name, parameter in removed:
            child._parameters[name] = parameter
        raise RuntimeError(
            f"Expected to hide {len(parameters)} old parameters, hid {len(removed)}"
        )
    try:
        yield
    finally:
        for child, name, parameter in removed:
            child._parameters[name] = parameter


@torch.no_grad()
def move_zero3_partition_storage_to_device(
    module: torch.nn.Module, device: torch.device
) -> tuple[int, int, int]:
    """Move only each rank's ZeRO shard (plus ordinary small params) to CUDA.

    H3's custom safetensors loader materializes checkpoint tensors on the
    requested construction device. Constructing directly on CUDA exceeds an
    80GB H100 before ZeRO can release loader temporaries. Constructing on CPU
    avoids that peak, but leaves ``ds_tensor`` partition storage on CPU even
    when the DeepSpeed config has no offload. Move the already-partitioned
    local shards explicitly; never gather or copy a full ZeRO parameter here.
    """
    zero_parameters = 0
    ordinary_parameters = 0
    moved_bytes = 0
    for parameter in module.parameters():
        partition = getattr(parameter, "ds_tensor", None)
        if partition is not None:
            zero_parameters += 1
            if partition.device != device:
                moved_bytes += partition.numel() * partition.element_size()
                # Preserve the ds_tensor object itself: DeepSpeed attaches
                # dynamic ``status``/``final_location`` metadata to it.
                partition.data = partition.data.to(device=device)
            # Small parameters under the stage-3 persistence threshold may
            # remain materialized; move that local persistent tensor as well.
            if parameter.device != device:
                moved_bytes += parameter.numel() * parameter.element_size()
                parameter.data = parameter.data.to(device=device)
        else:
            ordinary_parameters += 1
            if parameter.device != device:
                moved_bytes += parameter.numel() * parameter.element_size()
                parameter.data = parameter.data.to(device=device)
    return zero_parameters, ordinary_parameters, moved_bytes


def install_zero3_keyword_checkpoint_adapter() -> None:
    """Preserve H3 keyword-only block inputs in DeepSpeed checkpointing.

    DiffSynth's generic DeepSpeed branch flattens kwargs values into positional
    arguments. H3 blocks intentionally declare those inputs keyword-only. This
    process-local adapter retains the existing checkpoint/no-checkpoint choice
    and reconstructs the original call signature inside the DS callback.
    """
    import deepspeed
    from diffsynth.models import minimax_h3_dit as h3_dit_module

    original = h3_dit_module.gradient_checkpoint_forward

    def keyword_safe_checkpoint(
        model,
        use_gradient_checkpointing,
        use_gradient_checkpointing_offload,
        *args,
        **kwargs,
    ):
        if (
            use_gradient_checkpointing
            and deepspeed.checkpointing.is_configured()
        ):
            all_args = args + tuple(kwargs.values())
            requires_grad = any(
                isinstance(value, torch.Tensor) and value.requires_grad
                for value in all_args
            )
            if not requires_grad:
                return model(*args, **kwargs)
            positional_count = len(args)
            keyword_names = tuple(kwargs)

            def custom_forward(*flat_inputs):
                restored_kwargs = dict(
                    zip(keyword_names, flat_inputs[positional_count:], strict=True)
                )
                return model(
                    *flat_inputs[:positional_count], **restored_kwargs
                )

            return deepspeed.checkpointing.checkpoint(
                custom_forward, *all_args
            )
        return original(
            model,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            *args,
            **kwargs,
        )

    h3_dit_module.gradient_checkpoint_forward = keyword_safe_checkpoint


def run_nft_step_zero3(
    accelerator: Accelerator,
    engine,
    core,
    group: list[dict[str, Any]],
    args: argparse.Namespace,
    current_parameters: list[torch.nn.Parameter],
    old_parameters: list[torch.nn.Parameter],
    global_step: int,
    nft_config: dict[str, Any],
) -> int:
    device = accelerator.device
    rewards = torch.tensor([float(record["reward"]) for record in group])
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    advantages = (rewards - reward_mean) / (reward_std + args.adv_eps)
    clipped = advantages.clamp(-args.adv_clip_max, args.adv_clip_max)
    ratios = ((clipped / args.adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)
    contract_hash = verify_global_reward_contract(
        accelerator, rewards, advantages, ratios
    )
    if accelerator.is_main_process:
        print(f"[nft] rewards={rewards.tolist()}", flush=True)
        print(f"[nft] advantages={advantages.tolist()}", flush=True)
        print(f"[nft] clipped_advantages={clipped.tolist()}", flush=True)
        print(f"[nft] r={ratios.tolist()}", flush=True)
        print(f"[nft] global_reward_contract_sha256={contract_hash}", flush=True)

    local_indices = [
        index
        for index in range(len(group))
        if index % accelerator.num_processes == accelerator.process_index
    ]
    if len(group) % accelerator.num_processes:
        raise ValueError(
            f"K={len(group)} must be divisible by world_size={accelerator.num_processes}"
        )
    rank_log(accelerator, f"assigned_sample_indices={local_indices}")
    num_inference_steps = int(nft_config["num_inference_steps"])
    num_train_timesteps = int(num_inference_steps * args.timestep_fraction)
    local_backward_passes = len(local_indices) * num_train_timesteps
    if local_backward_passes <= 0:
        raise RuntimeError("Each rank must receive at least one sample/timestep")

    engine.zero_grad()
    before_step = gathered_adapter_snapshot(accelerator, current_parameters)
    distance_before = gathered_policy_distance(
        accelerator, current_parameters, old_parameters
    )
    if accelerator.is_main_process:
        print(
            f"[step] global_step_before={global_step} samples={len(group)} "
            f"world_size={accelerator.num_processes} "
            f"training_timesteps_per_sample={num_train_timesteps} "
            f"local_backward_passes={local_backward_passes}",
            flush=True,
        )
        print(
            f"[policies] current_old_parameter_distance_before={distance_before:.12f}",
            flush=True,
        )

    metrics = {key: [] for key in (
        "positive", "negative", "policy", "kl", "total", "distance"
    )}
    torch.cuda.reset_peak_memory_stats(device)
    engine_steps_before = int(engine.global_steps)
    for sample_index in local_indices:
        record = group[sample_index]
        video_x0, audio_x0, condition_image = single.load_rollout_clean_state(
            record, device, core.pipe.torch_dtype
        )
        inputs_shared, inputs_posi = load_condition_cache(
            args.condition_cache_dir,
            args.rollout_json.expanduser().resolve(),
            group,
            args.model_id,
            sample_index,
            device,
        )
        rank_log(
            accelerator,
            f"sample={sample_index} video_latent_shape={list(video_x0.shape)} "
            f"audio_latent_shape={list(audio_x0.shape)} "
            "uses_rollout_clean_latent=true uses_exact_rollout_condition=true",
        )
        timestep_indices = single.selected_timestep_indices(
            num_inference_steps,
            args.timestep_fraction,
            shuffle=args.shuffle_timesteps,
            seed=args.timestep_seed + global_step * 100_003 + sample_index,
        )
        rank_log(
            accelerator,
            f"sample={sample_index} seed={record.get('seed')} "
            f"timestep_indices={timestep_indices}",
        )
        for timestep_position, timestep_id in enumerate(timestep_indices):
            single.validate_timestep(core, timestep_id)
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
            ) = single.make_joint_forward_diffusion(
                core, video_x0, audio_x0, timestep_id, noise_seed, device
            )
            with torch.no_grad():
                old_prediction = engine(
                    action="velocity", policy="old",
                    inputs_shared=inputs_shared, inputs_posi=inputs_posi,
                    x_t=video_xt, timestep_video=timestep_video,
                    timestep_audio=timestep_audio, audio_x_t=audio_xt,
                    checkpoint=False,
                ).detach()
                reference_prediction = engine(
                    action="velocity", policy="reference",
                    inputs_shared=inputs_shared, inputs_posi=inputs_posi,
                    x_t=video_xt, timestep_video=timestep_video,
                    timestep_audio=timestep_audio, audio_x_t=audio_xt,
                    checkpoint=False,
                ).detach()
            with accelerator.accumulate(engine):
                current_prediction = engine(
                    action="velocity", policy="current",
                    inputs_shared=inputs_shared, inputs_posi=inputs_posi,
                    x_t=video_xt, timestep_video=timestep_video,
                    timestep_audio=timestep_audio, audio_x_t=audio_xt,
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
                        (positive_x0.double() - video_x0.double()).abs()
                        .mean(dim=reduce_dims, keepdim=True).clamp(min=1e-5)
                    )
                    negative_weight = (
                        (negative_x0.double() - video_x0.double()).abs()
                        .mean(dim=reduce_dims, keepdim=True).clamp(min=1e-5)
                    )
                positive_loss = (
                    (positive_x0 - video_x0).square() / positive_weight
                ).mean(dim=reduce_dims)
                negative_loss = (
                    (negative_x0 - video_x0).square() / negative_weight
                ).mean(dim=reduce_dims)
                ratio = ratios[sample_index].to(device)
                policy_loss = (
                    ratio * positive_loss / args.policy_beta
                    + (1.0 - ratio) * negative_loss / args.policy_beta
                ).mean() * args.adv_clip_max
                kl_loss = F.mse_loss(
                    current_prediction.float(), reference_prediction.float()
                )
                total_loss = policy_loss + args.kl_beta * kl_loss
                if accelerator.sync_gradients:
                    with hide_ordinary_parameters_from_zero3_scan(
                        core.pipe.dit, old_parameters
                    ):
                        accelerator.backward(total_loss)
                else:
                    accelerator.backward(total_loss)
            current_reference_distance = math.sqrt(max(kl_loss.item(), 0.0))
            metrics["positive"].append(positive_loss.detach().mean())
            metrics["negative"].append(negative_loss.detach().mean())
            metrics["policy"].append(policy_loss.detach())
            metrics["kl"].append(kl_loss.detach())
            metrics["total"].append(total_loss.detach())
            metrics["distance"].append(
                torch.tensor(current_reference_distance, device=device)
            )
            rank_log(
                accelerator,
                f"sample={sample_index} timestep_position={timestep_position} "
                f"timestep_id={timestep_id} noise_seed={noise_seed} "
                f"video_sigma={float(sigma_video):.8f} "
                f"audio_sigma={float(sigma_audio):.8f} "
                f"positive_loss={positive_loss.mean().item():.8f} "
                f"negative_loss={negative_loss.mean().item():.8f} "
                f"policy_loss={policy_loss.item():.8f} "
                f"kl_loss={kl_loss.item():.8f} total_loss={total_loss.item():.8f}",
            )

    if int(engine.global_steps) != engine_steps_before + 1:
        raise RuntimeError(
            f"Expected exactly one DeepSpeed optimizer step, got "
            f"{engine.global_steps - engine_steps_before}"
        )
    old_policy_no_grad = distributed_all_true(
        single.no_adapter_has_grad(core.pipe.dit, "old"), accelerator
    )
    reference_policy_no_grad = distributed_all_true(
        not single.frozen_base_has_grad(core.pipe.dit), accelerator
    )
    gradient_norm = float(getattr(engine, "_global_grad_norm", 0.0))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError(f"Invalid ZeRO-3 gradient norm: {gradient_norm}")
    delta = gathered_delta(accelerator, before_step, current_parameters)
    global_step += 1
    decay = single.official_return_decay(global_step, args.old_decay_type)
    update_old_policy_zero3(
        accelerator, current_parameters, old_parameters, decay
    )
    distance_after = gathered_policy_distance(
        accelerator, current_parameters, old_parameters
    )

    means = {
        key: distributed_mean(values, accelerator) for key, values in metrics.items()
    }
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
    rank_log(
        accelerator,
        f"peak_gpu_allocated_mb={peak_allocated:.2f} "
        f"peak_gpu_reserved_mb={peak_reserved:.2f}",
    )
    if accelerator.is_main_process:
        print(
            f"[group] positive_loss={means['positive']:.8f} "
            f"negative_loss={means['negative']:.8f} "
            f"policy_loss={means['policy']:.8f} kl_loss={means['kl']:.8f} "
            f"current_reference_prediction_distance={means['distance']:.8f} "
            f"total_loss={means['total']:.8f}",
            flush=True,
        )
        print(
            f"[grad] gradient_norm={gradient_norm:.8f} "
            f"max_grad_norm={args.max_grad_norm}", flush=True
        )
        print(
            f"[optimizer] current_parameter_delta={delta:.12f}", flush=True
        )
        print(
            f"[policies] old_policy_no_grad={str(old_policy_no_grad).lower()} "
            f"reference_policy_no_grad={str(reference_policy_no_grad).lower()} "
            f"old_decay={decay:.8f} "
            f"current_old_parameter_distance_after={distance_after:.12f}",
            flush=True,
        )
    if not old_policy_no_grad or not reference_policy_no_grad:
        raise RuntimeError("Frozen old/reference policy unexpectedly received gradients")
    if accelerator.is_main_process and (delta is None or delta <= 0):
        raise RuntimeError(f"Optimizer produced invalid parameter delta: {delta}")

    save_zero3_checkpoint(
        accelerator, engine, core, current_parameters, old_parameters,
        args.checkpoint_output, global_step, nft_config,
    )
    if accelerator.is_main_process:
        print(
            f"[step] global_step_after={global_step} optimizer_step=true "
            f"checkpoint={args.checkpoint_output.expanduser().resolve()}",
            flush=True,
        )
    return global_step


def main() -> None:
    args = parse_args()
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rollout_path = args.rollout_json.expanduser().resolve()
    records = single.load_rollout_records(rollout_path)
    group = single.select_prompt_group(
        records, args.rollout_index, args.group_size
    )
    single.validate_nft_rollout_artifacts(group)
    scheduler_config = single.rollout_scheduler_config(group)
    num_train_timesteps = int(
        int(scheduler_config["num_inference_steps"]) * args.timestep_fraction
    )
    if len(group) % env_world_size:
        raise ValueError(
            f"K={len(group)} must be divisible by world_size={env_world_size}"
        )
    local_backward_passes = (
        len(group) // env_world_size * num_train_timesteps
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=local_backward_passes,
        mixed_precision="bf16",
    )
    validate_args(args, accelerator)
    plugin = accelerator.state.deepspeed_plugin
    plugin.deepspeed_config["gradient_clipping"] = args.max_grad_norm
    zero_stage = int(plugin.deepspeed_config["zero_optimization"]["stage"])
    world_size = accelerator.num_processes
    rank_log(
        accelerator,
        f"world_size={world_size} rank={accelerator.process_index} "
        f"local_rank={accelerator.local_process_index} "
        f"device={accelerator.device} deepspeed_enabled=true zero_stage={zero_stage}",
    )
    if world_size != env_world_size:
        raise RuntimeError(
            f"Accelerator world size {world_size} != environment {env_world_size}"
        )
    if len(group) % world_size:
        raise ValueError(
            f"K={len(group)} must be divisible by world_size={world_size}"
        )
    single.selected_timestep_indices(
        int(scheduler_config["num_inference_steps"]),
        args.timestep_fraction, shuffle=False, seed=args.timestep_seed,
    )
    nft_config = single.checkpoint_config(args, scheduler_config)
    nft_config.update({
        "format_version": 2,
        "backend": "accelerate-deepspeed-zero3",
        "world_size": world_size,
    })
    resume_dir = None
    resume_state = None
    global_step = 0
    if args.resume_from is not None:
        resume_dir = args.resume_from.expanduser().resolve()
        resume_state = read_zero3_state(resume_dir)
        single.validate_resume_config(resume_state["nft_config"], nft_config)
        global_step = int(resume_state["global_step"])
    if args.require_policy_provenance:
        single.validate_rollout_policy_provenance(group, resume_dir, global_step)
    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
    torch.manual_seed(args.noise_seed)
    torch.cuda.manual_seed_all(args.noise_seed)

    TrainingModule = single._load_sft_training_module_class()
    # Frozen conditioning tensors come from prepare_zero3_conditions.py.  The
    # ZeRO engine therefore loads and shards only the H3 DiT used by NFT.
    model_paths = f"{args.model_id}:FL2VA/transformer/model*.safetensors"
    rank_log(accelerator, "loading MiniMaxH3 DiT on CPU before ZeRO-3 partitioning")
    core = TrainingModule(
        model_id_with_origin_paths=model_paths,
        processor_path=None,
        lora_base_model="dit",
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        # The custom loader materializes checkpoint tensors on its construction
        # device. CPU construction avoids a full-model CUDA loading peak;
        # accelerator.prepare partitions the DiT and the local pure-GPU shards
        # are moved below. Runtime parameter/optimizer offload remains disabled.
        device="cpu",
        task="sft",
    )
    core.pipe.scheduler.set_timesteps(
        int(scheduler_config["num_inference_steps"]),
        shift=float(scheduler_config["flow_shift"]), training=True,
    )
    core.pipe.scheduler_audio.set_timesteps(
        int(scheduler_config["num_inference_steps"]),
        shift=float(scheduler_config["audio_flow_shift"]), training=True,
    )
    # Only the trainable current adapter exists while DeepSpeed partitions the
    # model. The frozen FP32 old adapter is injected after prepare below, so
    # DeepSpeed BF16 cannot round the official early-step EMA back to current.
    current_parameters = [
        parameter for _, parameter in
        single.named_adapter_parameters(core.pipe.dit, "default")
    ]
    if not current_parameters:
        raise RuntimeError("Current LoRA adapter was not injected")
    if args.initial_lora_path is not None:
        initial_lora = args.initial_lora_path.expanduser().resolve()
        single.load_adapter_state(core.pipe.dit, "default", initial_lora)
        rank_log(accelerator, f"initial_lora_path={initial_lora}")
    elif resume_dir is not None:
        single.load_adapter_state(
            core.pipe.dit, "default", resume_dir / "current_lora.safetensors"
        )
    single.set_policy_adapter(core.pipe.dit, "default", trainable=True)
    wrapper = Zero3NFTModule(core)
    trainable_tensors, trainable_params = single.count_trainable_parameters(wrapper)
    if trainable_tensors != len(current_parameters):
        raise RuntimeError(
            f"Only current LoRA may be trainable: "
            f"model={trainable_tensors} current={len(current_parameters)}"
        )
    optimizer = torch.optim.AdamW(
        current_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    rank_log(
        accelerator,
        f"trainable_parameter_tensors={trainable_tensors} "
        f"trainable_parameters={trainable_params} "
        "old_parameter_tensors_before_prepare=0",
    )
    core.pipe.device = str(accelerator.device)
    engine, optimizer = accelerator.prepare(wrapper, optimizer)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    install_zero3_keyword_checkpoint_adapter()
    unwrapped = accelerator.unwrap_model(engine)
    zero_partition_count, ordinary_parameter_count, moved_partition_bytes = (
        move_zero3_partition_storage_to_device(unwrapped, accelerator.device)
    )
    rank_log(
        accelerator,
        f"zero3_partition_tensors_moved={zero_partition_count} "
        f"ordinary_parameter_tensors_moved={ordinary_parameter_count} "
        f"partition_storage_moved_mb={moved_partition_bytes / 2**20:.2f}",
    )
    moved_buffers = move_registered_buffers_to_device(
        unwrapped, accelerator.device
    )
    rank_log(accelerator, f"registered_buffers_moved_to_device={moved_buffers}")
    core = unwrapped.core
    current_parameters = [
        parameter for _, parameter in
        single.named_adapter_parameters(core.pipe.dit, "default")
    ]
    if resume_dir is not None:
        load_zero3_checkpoint(
            accelerator, engine, resume_dir, resume_state
        )
        rank_log(
            accelerator,
            f"resume_success=true global_step={global_step} path={resume_dir}",
        )
    else:
        rank_log(accelerator, "resume_success=false global_step=0 path=None")

    # Add frozen old only after the ZeRO/BF16 engine is fully initialized.
    # Its forwards still enter through ``engine`` and the same DiT backbone,
    # but the small replicated adapter remains FP32 and outside the optimizer.
    current_parameters_after, old_parameters = single.add_old_adapter(
        core.pipe.dit, args.lora_rank, args.lora_target_modules,
        initialize_from_current=False,
    )
    if [id(p) for p in current_parameters_after] != [id(p) for p in current_parameters]:
        raise RuntimeError("Late old-adapter injection replaced current parameters")
    for parameter in old_parameters:
        parameter.data = parameter.data.to(
            device=accelerator.device, dtype=torch.float32
        )
    if args.initial_lora_path is not None:
        single.load_adapter_state(core.pipe.dit, "old", initial_lora)
    elif resume_dir is not None:
        single.load_adapter_state(
            core.pipe.dit, "old", resume_dir / "old_lora.safetensors"
        )
    else:
        with gathered_parameter_context(current_parameters):
            with torch.no_grad():
                for current, old in zip(
                    current_parameters, old_parameters, strict=True
                ):
                    old.copy_(current.detach().float())
    single.set_policy_adapter(core.pipe.dit, "default", trainable=True)
    for parameter in old_parameters:
        parameter.requires_grad_(False)
    if any(hasattr(parameter, "ds_id") for parameter in old_parameters):
        raise RuntimeError("Frozen old LoRA unexpectedly entered ZeRO partitioning")
    late_parameter_containers = install_zero3_parameter_containers_for_late_modules(
        unwrapped
    )
    rank_log(
        accelerator,
        f"current_lora_dtypes={sorted({str(p.dtype) for p in current_parameters})} "
        f"old_lora_dtypes={sorted({str(p.dtype) for p in old_parameters})} "
        "old_lora_zero_managed=false old_lora_replicated_gpu=true "
        f"late_zero_parameter_containers={late_parameter_containers}",
    )
    run_nft_step_zero3(
        accelerator, engine, core, group, args,
        current_parameters, old_parameters, global_step, nft_config,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
