#!/usr/bin/env python3
"""Compare unchanged single-GPU and ZeRO-3 LoRA optimizer-step outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-lora", type=Path, required=True)
    parser.add_argument("--single-checkpoint", type=Path, required=True)
    parser.add_argument("--zero3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, torch.Tensor]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    state = {key: value.float() for key, value in load_file(str(path)).items()}
    if not state:
        raise ValueError(f"Empty LoRA state: {path}")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f"Non-finite LoRA state: {path}")
    return state


def flatten(state: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    return torch.cat([state[key].reshape(-1) for key in keys])


def compare(
    initial: dict[str, torch.Tensor],
    single: dict[str, torch.Tensor],
    zero3: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    if set(initial) != set(single) or set(single) != set(zero3):
        raise ValueError("Initial/single/ZeRO-3 LoRA key sets differ")
    keys = sorted(initial)
    initial_flat = flatten(initial, keys)
    single_flat = flatten(single, keys)
    zero3_flat = flatten(zero3, keys)
    difference = single_flat - zero3_flat
    single_update = single_flat - initial_flat
    zero3_update = zero3_flat - initial_flat
    element_denominator = torch.maximum(
        torch.maximum(single_flat.abs(), zero3_flat.abs()),
        torch.full_like(single_flat, 1e-8),
    )
    update_denominator = single_update.norm().clamp_min(1e-30)
    cosine = torch.nn.functional.cosine_similarity(
        single_update, zero3_update, dim=0, eps=1e-30
    )
    result = {
        "tensor_count": len(keys),
        "parameter_count": int(single_flat.numel()),
        "max_absolute_difference": float(difference.abs().max()),
        "max_element_relative_difference": float(
            (difference.abs() / element_denominator).max()
        ),
        "relative_l2_difference": float(
            difference.norm() / single_flat.norm().clamp_min(1e-30)
        ),
        "single_update_l2": float(single_update.norm()),
        "zero3_update_l2": float(zero3_update.norm()),
        "update_relative_l2_difference": float(
            (single_update - zero3_update).norm() / update_denominator
        ),
        "update_cosine_similarity": float(cosine),
    }
    if not all(
        math.isfinite(value) for value in result.values() if isinstance(value, float)
    ):
        raise RuntimeError("Comparison produced a non-finite metric")
    return result


def main() -> None:
    args = parse_args()
    initial_path = args.initial_lora.expanduser().resolve()
    initial = load(initial_path)
    output: dict[str, object] = {
        "initial_lora": str(initial_path),
        "initial_lora_sha256": sha256(initial_path),
    }
    for policy in ("current", "old"):
        single_path = (
            args.single_checkpoint.expanduser().resolve()
            / f"{policy}_lora.safetensors"
        )
        zero3_path = (
            args.zero3_checkpoint.expanduser().resolve()
            / f"{policy}_lora.safetensors"
        )
        output[policy] = {
            "single_sha256": sha256(single_path),
            "zero3_sha256": sha256(zero3_path),
            **compare(initial, load(single_path), load(zero3_path)),
        }
    text = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(text, end="")
    if args.output_json is not None:
        path = args.output_json.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


if __name__ == "__main__":
    main()
