#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ROLLOUT_JSON="${ROLLOUT_JSON:-outputs/minimax_h3_diffusionnft_rollout_phase4/rollout.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-outputs/minimax_h3_diffusionnft_phase4_2step/${RUN_ID}}"
CHECKPOINT_1="${RUN_DIR}/checkpoint-1"
CHECKPOINT_2="${RUN_DIR}/checkpoint-2"

COMMON_ARGS=(
  --mode nft-step
  --rollout-json "${ROLLOUT_JSON}"
  --rollout-index 0
  --group-size 2
  --device cuda:0
  --lora-rank 4
  --lora-target-modules qkv_proj,out_proj
  --timestep-fraction 0.99
  --shuffle-timesteps
  --adv-clip-max 5
  --policy-beta 1.0
  --kl-beta 1e-4
  --learning-rate 1e-4
  --max-grad-norm 1.0
  --old-decay-type 1
  --use-gradient-checkpointing-offload
)

echo "[smoke] step=1 expected_global_step=0_to_1 checkpoint=${CHECKPOINT_1}"
py312/bin/python examples/minimax_h3/model_training/diffusionnft/train.py \
  "${COMMON_ARGS[@]}" \
  --checkpoint-output "${CHECKPOINT_1}"

echo "[smoke] step=2 expected_global_step=1_to_2 resume=${CHECKPOINT_1} checkpoint=${CHECKPOINT_2}"
py312/bin/python examples/minimax_h3/model_training/diffusionnft/train.py \
  "${COMMON_ARGS[@]}" \
  --resume-from "${CHECKPOINT_1}" \
  --checkpoint-output "${CHECKPOINT_2}"

echo "[smoke] complete=true checkpoint_1=${CHECKPOINT_1} checkpoint_2=${CHECKPOINT_2}"
