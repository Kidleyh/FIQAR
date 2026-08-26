#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

py312/bin/python examples/minimax_h3/model_training/diffusionnft/train.py \
  --mode nft-step \
  --rollout-json outputs/minimax_h3_diffusionnft_rollout_phase4/rollout.json \
  --rollout-index 0 \
  --group-size 2 \
  --device cuda:0 \
  --lora-rank 4 \
  --lora-target-modules qkv_proj,out_proj \
  --adv-clip-max 5 \
  --policy-beta 1.0 \
  --kl-beta 1e-4 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0 \
  --old-decay-type 1 \
  --use-gradient-checkpointing-offload
