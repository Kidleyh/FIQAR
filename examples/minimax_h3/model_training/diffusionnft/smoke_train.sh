#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

py312/bin/python examples/minimax_h3/model_training/diffusionnft/train.py \
  --rollout-json outputs/minimax_h3_diffusionnft_rollout_smoke/rollout.json \
  --rollout-index 0 \
  --device cuda:0 \
  --lora-rank 4 \
  --lora-target-modules qkv_proj,out_proj \
  --use-gradient-checkpointing-offload
