#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-phase5_2iter_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="outputs/minimax_h3_diffusionnft_online/${RUN_ID}"
DATA_JSON="/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/guangdian_20251114_small_clear_faces/guangdian_20251114_small_clear_faces.json"

COMMON_ARGS=(
  --num-samples-per-prompt 2
  --start 0
  --limit 1
  --seeds 0 1
  --data-json "${DATA_JSON}"
  --python py312/bin/python
  --device cuda:0
  --height 256
  --width 448
  --num-frames 22
  --num-inference-steps 2
  --flow-shift 12
  --audio-flow-shift 3
  --vram-limit-gb 14
  --reward-frame-stride 6
  --lora-rank 4
  --lora-target-modules qkv_proj,out_proj
  --timestep-fraction 0.99
  --adv-clip-max 5
  --policy-beta 1.0
  --kl-beta 1e-4
  --learning-rate 1e-4
  --max-grad-norm 1.0
  --old-decay-type 1
  --use-gradient-checkpointing-offload
)

# Process 1 completes iteration 0 and checkpoint-1.
py312/bin/python examples/minimax_h3/model_training/diffusionnft/online_train.py \
  --num-iterations 1 \
  --output-dir "${RUN_DIR}" \
  "${COMMON_ARGS[@]}"

# Process 2 proves run-level recovery, then completes old-policy iteration 1.
py312/bin/python examples/minimax_h3/model_training/diffusionnft/online_train.py \
  --num-iterations 2 \
  --resume-from "${RUN_DIR}" \
  "${COMMON_ARGS[@]}"

echo "[smoke] online_state=${RUN_DIR}/online_state.json"
