#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3

ACCELERATE_BIN="${ACCELERATE_BIN:-py312/bin/accelerate}"
PYTHON_BIN="${PYTHON_BIN:-py312/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-examples/minimax_h3/model_training/diffusionnft/deepspeed/accelerate_zero3.yaml}"
ROLLOUT_JSON="${ROLLOUT_JSON:?Set ROLLOUT_JSON to a K-divisible rollout.json}"
CHECKPOINT_OUTPUT="${CHECKPOINT_OUTPUT:?Set CHECKPOINT_OUTPUT to a new checkpoint directory}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
GROUP_SIZE="${GROUP_SIZE:-4}"
LORA_RANK="${LORA_RANK:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
POLICY_BETA="${POLICY_BETA:-1.0}"
KL_BETA="${KL_BETA:-1e-4}"
ADV_CLIP_MAX="${ADV_CLIP_MAX:-5.0}"
TIMESTEP_FRACTION="${TIMESTEP_FRACTION:-0.99}"
OLD_DECAY_TYPE="${OLD_DECAY_TYPE:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TIMESTEP_SEED="${TIMESTEP_SEED:-1234}"
NOISE_SEED="${NOISE_SEED:-5678}"
RESUME_FROM="${RESUME_FROM:-}"
INITIAL_LORA_PATH="${INITIAL_LORA_PATH:-}"
CONDITION_CACHE_DIR="${CONDITION_CACHE_DIR:-$(dirname "${ROLLOUT_JSON}")/zero3_condition_cache}"

export CUDA_VISIBLE_DEVICES

# Frozen Qwen/VAE conditioning is produced once through the verified
# single-GPU path. The process exits before the pure-GPU ZeRO-3 DiT launch.
"${PYTHON_BIN}" \
  examples/minimax_h3/model_training/diffusionnft/prepare_zero3_conditions.py \
  --rollout-json "${ROLLOUT_JSON}" \
  --group-size "${GROUP_SIZE}" \
  --output-dir "${CONDITION_CACHE_DIR}" \
  --device cuda:0

ARGS=(
  --mode nft-step
  --rollout-json "${ROLLOUT_JSON}"
  --group-size "${GROUP_SIZE}"
  --checkpoint-output "${CHECKPOINT_OUTPUT}"
  --condition-cache-dir "${CONDITION_CACHE_DIR}"
  --lora-rank "${LORA_RANK}"
  --learning-rate "${LEARNING_RATE}"
  --policy-beta "${POLICY_BETA}"
  --kl-beta "${KL_BETA}"
  --adv-clip-max "${ADV_CLIP_MAX}"
  --timestep-fraction "${TIMESTEP_FRACTION}"
  --old-decay-type "${OLD_DECAY_TYPE}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --timestep-seed "${TIMESTEP_SEED}"
  --noise-seed "${NOISE_SEED}"
)
if [[ -n "${RESUME_FROM}" ]]; then
  ARGS+=(--resume-from "${RESUME_FROM}")
fi
if [[ -n "${INITIAL_LORA_PATH}" ]]; then
  ARGS+=(--initial-lora-path "${INITIAL_LORA_PATH}")
fi

"${ACCELERATE_BIN}" launch \
  --config_file "${CONFIG_FILE}" \
  --num_processes "${NUM_PROCESSES}" \
  examples/minimax_h3/model_training/diffusionnft/train_zero3.py \
  "${ARGS[@]}"
