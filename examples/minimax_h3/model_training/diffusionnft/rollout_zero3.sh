#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-py312/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-py312/bin/accelerate}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-examples/minimax_h3/model_training/diffusionnft/deepspeed/accelerate_zero3.yaml}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

DATA_JSON="${DATA_JSON:-/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/guangdian_20251114_small_clear_faces/guangdian_20251114_small_clear_faces.json}"
START="${START:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/minimax_h3_diffusionnft_zero3/phase10_rollout_$(date +%Y%m%d_%H%M%S)}"
WIDTH="${WIDTH:-1088}"
HEIGHT="${HEIGHT:-736}"
NUM_FRAMES="${NUM_FRAMES:-175}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-3}"
SEED="${SEED:-0}"
FLOW_SHIFT="${FLOW_SHIFT:-12.0}"
AUDIO_FLOW_SHIFT="${AUDIO_FLOW_SHIFT:-3.0}"

POLICY_ROLE="${POLICY_ROLE:-base}"
LORA_PATH="${LORA_PATH:-}"
LORA_RANK="${LORA_RANK:-4}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-}"
GLOBAL_STEP_BEFORE="${GLOBAL_STEP_BEFORE:-0}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-0}"

REWARD_FRAME_STRIDE="${REWARD_FRAME_STRIDE:-25}"
REWARD_MAX_FRAMES="${REWARD_MAX_FRAMES:-0}"
REWARD_FRAME_FACE_AGGREGATION="${REWARD_FRAME_FACE_AGGREGATION:-mean}"
MISSING_FACE_REWARD="${MISSING_FACE_REWARD:-0.0}"

if (( NUM_PROCESSES < 2 )); then
  echo "ZeRO-3 rollout requires NUM_PROCESSES>=2" >&2
  exit 2
fi

COMMON_ARGS=(
  --data-json "${DATA_JSON}"
  --start "${START}"
  --output-dir "${OUTPUT_DIR}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --num-frames "${NUM_FRAMES}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --rollout-world-size "${NUM_PROCESSES}"
  --seed "${SEED}"
  --flow-shift "${FLOW_SHIFT}"
  --audio-flow-shift "${AUDIO_FLOW_SHIFT}"
  --policy-role "${POLICY_ROLE}"
  --lora-rank "${LORA_RANK}"
  --global-step-before "${GLOBAL_STEP_BEFORE}"
  --reward-frame-stride "${REWARD_FRAME_STRIDE}"
  --reward-max-frames "${REWARD_MAX_FRAMES}"
  --reward-frame-face-aggregation "${REWARD_FRAME_FACE_AGGREGATION}"
  --missing-face-reward "${MISSING_FACE_REWARD}"
)

if [[ -n "${LORA_PATH}" ]]; then
  COMMON_ARGS+=(--lora-path "${LORA_PATH}")
fi
if [[ -n "${SOURCE_CHECKPOINT}" ]]; then
  COMMON_ARGS+=(--source-checkpoint "${SOURCE_CHECKPOINT}")
fi
if [[ "${SAVE_ROLLOUT_VIDEO}" == "1" ]]; then
  COMMON_ARGS+=(--save-rollout-video)
fi

FIRST_GPU="${CUDA_DEVICES%%,*}"

echo "[phase10-launch] stage=prepare-condition gpu=${FIRST_GPU} output=${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" "${PYTHON_BIN}" \
  examples/minimax_h3/model_training/diffusionnft/rollout_zero3.py \
  --mode prepare-condition "${COMMON_ARGS[@]}"

echo "[phase10-launch] stage=denoise gpus=${CUDA_DEVICES} ranks=${NUM_PROCESSES}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  examples/minimax_h3/model_training/diffusionnft/rollout_zero3.py \
  --mode denoise "${COMMON_ARGS[@]}"

echo "[phase10-launch] stage=finalize gpu=${FIRST_GPU}"
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" "${PYTHON_BIN}" \
  examples/minimax_h3/model_training/diffusionnft/rollout_zero3.py \
  --mode finalize "${COMMON_ARGS[@]}"

echo "[phase10-launch] complete=true rollout=${OUTPUT_DIR}/rollout.json"
