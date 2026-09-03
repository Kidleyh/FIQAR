#!/usr/bin/env python3
"""Prepare exact single-GPU H3 conditioning for ZeRO-3 NFT training.

The frozen Qwen/VAE condition path is evaluated once with the already verified
single-GPU DiffSynth offload implementation.  ZeRO-3 then consumes these frozen
tensors while sharding only the trainable H3 DiT.  No rollout or core source is
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
SINGLE_TRAIN_PATH = Path(__file__).with_name("train.py")
FORMAT_VERSION = 1

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import OffloadTrainingManager  # noqa: E402


def _load_single_helpers():
    spec = importlib.util.spec_from_file_location(
        "minimax_h3_condition_cache_single", SINGLE_TRAIN_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(SINGLE_TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


single = _load_single_helpers()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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
    raise TypeError(f"Unsupported cached condition value: {type(value)!r}")


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


def model_condition_inputs(
    inputs_shared: dict[str, Any], inputs_posi: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    shared_keys = (
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
    shared = {key: inputs_shared[key] for key in shared_keys if key in inputs_shared}
    positive = {
        key: inputs_posi[key]
        for key in ("prompt_embeds", "packed")
        if key in inputs_posi
    }
    if set(positive) != {"prompt_embeds", "packed"}:
        raise ValueError(f"Condition preparation missed required positive inputs: {positive}")
    if shared.get("keyframe_cond_anchor") is None:
        raise ValueError("Condition preparation missed keyframe_cond_anchor")
    return shared, positive


def group_contract(
    rollout_json: Path,
    group: list[dict[str, Any]],
    model_id: str,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "rollout_json": str(rollout_json.resolve()),
        "rollout_json_sha256": sha256_file(rollout_json),
        "model_id": model_id,
        "records": [
            {
                "sample_index": index,
                "seed": int(record["seed"]),
                "prompt_sha256": sha256_text(record["prompt"]),
                "condition_image_path": str(
                    Path(record["condition_image_path"]).expanduser().resolve()
                ),
                "condition_image_sha256": sha256_file(
                    Path(record["condition_image_path"]).expanduser().resolve()
                ),
                "height": int(record["height"]),
                "width": int(record["width"]),
                "num_frames": int(record["num_frames"]),
            }
            for index, record in enumerate(group)
        ],
    }


def load_condition_cache(
    cache_dir: Path,
    rollout_json: Path,
    group: list[dict[str, Any]],
    model_id: str,
    sample_index: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_dir = cache_dir.expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = group_contract(rollout_json, group, model_id)
    for key in (
        "format_version", "rollout_json", "rollout_json_sha256", "model_id", "records"
    ):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"Condition cache contract mismatch for {key}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(group):
        raise ValueError("Condition cache entry count mismatch")
    entry = entries[sample_index]
    if entry.get("sample_index") != sample_index:
        raise ValueError(f"Condition cache sample index mismatch: {entry}")
    state_path = cache_dir / entry["file"]
    if not state_path.is_file() or state_path.stat().st_size <= 0:
        raise FileNotFoundError(state_path)
    if sha256_file(state_path) != entry.get("sha256"):
        raise ValueError(f"Condition cache SHA256 mismatch: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if set(state) != {"inputs_shared", "inputs_posi"}:
        raise ValueError(f"Invalid condition cache payload keys: {sorted(state)}")
    return (
        tree_to_device(state["inputs_shared"], device),
        tree_to_device(state["inputs_posi"], device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-json", type=Path, required=True)
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="MiniMax/MiniMax-H3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_json = args.rollout_json.expanduser().resolve()
    records = single.load_rollout_records(rollout_json)
    group = single.select_prompt_group(records, args.rollout_index, args.group_size)
    single.validate_nft_rollout_artifacts(group)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = group_contract(rollout_json, group, args.model_id)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == contract[key] for key in contract):
            entries = existing.get("entries", [])
            if len(entries) == len(group) and all(
                (output_dir / entry["file"]).is_file()
                and (output_dir / entry["file"]).stat().st_size > 0
                and sha256_file(output_dir / entry["file"]) == entry.get("sha256")
                for entry in entries
            ):
                print(f"[condition-cache] reuse=true path={output_dir}", flush=True)
                return
        raise FileExistsError(
            f"Existing condition cache is incomplete or belongs to another rollout: {output_dir}"
        )

    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    TrainingModule = single._load_sft_training_module_class()
    # Only frozen models used by prepare_nft_pipeline_inputs are needed.  The
    # DiT and audio VAE never participate in this preprocessing process.
    model_paths = (
        f"{args.model_id}:FL2VA/text_encoder/model*.safetensors,"
        f"{args.model_id}:FL2VA/video_vae/source/model.safetensors"
    )
    print("[condition-cache] loading verified single-GPU condition path", flush=True)
    model = TrainingModule(
        model_id_with_origin_paths=model_paths,
        processor_path=f"{args.model_id}:FL2VA/processor/",
        lora_base_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        device="cpu",
        task="sft",
    )
    model.pipe.device = str(device)
    offload_manager = OffloadTrainingManager(
        model, device, enable_optimizer_cpu_offload=False
    )
    entries = []
    for sample_index, record in enumerate(group):
        _, _, condition_image = single.load_rollout_clean_state(
            record, device, model.pipe.torch_dtype
        )
        with torch.no_grad():
            inputs_shared, inputs_posi, _ = single.prepare_nft_pipeline_inputs(
                model, record, condition_image
            )
        offload_manager.after_backward()
        inputs_shared, inputs_posi = model_condition_inputs(
            inputs_shared, inputs_posi
        )
        state_path = output_dir / f"sample_{sample_index:06d}.pt"
        atomic_torch_save(
            state_path,
            {
                "inputs_shared": tree_to_cpu(inputs_shared),
                "inputs_posi": tree_to_cpu(inputs_posi),
            },
        )
        entries.append(
            {
                "sample_index": sample_index,
                "file": state_path.name,
                "sha256": sha256_file(state_path),
            }
        )
        print(
            f"[condition-cache] sample={sample_index} seed={record['seed']} "
            f"path={state_path}",
            flush=True,
        )
    atomic_json(manifest_path, {**contract, "entries": entries, "complete": True})
    print(
        f"[condition-cache] complete=true samples={len(entries)} path={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
