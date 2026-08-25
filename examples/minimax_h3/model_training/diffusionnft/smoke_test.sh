#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3

py312/bin/python examples/minimax_h3/model_training/diffusionnft/rollout.py \
  --data-json /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/guangdian_20251114_small_clear_faces/guangdian_20251114_small_clear_faces.json \
  --start 0 \
  --limit 1 \
  --num-samples 2 \
  --seeds 0 1 \
  --output-dir outputs/minimax_h3_diffusionnft_rollout_smoke \
  --height 256 \
  --width 448 \
  --num-frames 22 \
  --num-inference-steps 1 \
  --vram-limit-gb 14 \
  --reward-frame-stride 6
