#!/usr/bin/env python3
"""MiniMax-H3 Turbo LoRA stage-1 pair renderer.

Uses larryvrh's recommended v4-600 EMA checkpoint at strength 1.0. The default
is 8 sampling steps for best quality; 4–8 are accepted, while 6–8 are the
recommended quality range.
"""

from pathlib import Path

from minimax_h3_stage1_pair_render import main


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TURBO_LORA = (
    REPO_ROOT
    / "lora_models"
    / "larryvrh"
    / "MiniMax-H3-Turbo-Lora"
    / "minimax_h3_turbo_v4_step600_ema.safetensors"
)


if __name__ == "__main__":
    main(
        default_steps=8,
        default_lora_path=DEFAULT_TURBO_LORA,
        default_output_dir=Path("results/minimax_h3_turbo_stage1_pairs"),
        turbo_mode=True,
    )
