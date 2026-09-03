#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-py312/bin/python}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DATA_JSON="${DATA_JSON:-/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/guangdian_20251114_small_clear_faces/guangdian_20251114_small_clear_faces.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/minimax_h3_diffusionnft_zero3/online_$(date +%Y%m%d_%H%M%S)}"
RESUME_FROM="${RESUME_FROM:-}"

GPU_IDS="${GPU_IDS:-0,1}"
ROLLOUT_WORLD_SIZE="${ROLLOUT_WORLD_SIZE:-2}"
TRAIN_WORLD_SIZE="${TRAIN_WORLD_SIZE:-2}"
GROUP_SIZE="${GROUP_SIZE:-2}"
SEEDS="${SEEDS:-0 1}"

START="${START:-0}"
LIMIT="${LIMIT:-1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
WIDTH="${WIDTH:-1088}"
HEIGHT="${HEIGHT:-736}"
NUM_FRAMES="${NUM_FRAMES:-175}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-3}"
FLOW_SHIFT="${FLOW_SHIFT:-12.0}"
AUDIO_FLOW_SHIFT="${AUDIO_FLOW_SHIFT:-3.0}"

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

REWARD_FRAME_STRIDE="${REWARD_FRAME_STRIDE:-25}"
REWARD_MAX_FRAMES="${REWARD_MAX_FRAMES:-0}"
REWARD_FRAME_FACE_AGGREGATION="${REWARD_FRAME_FACE_AGGREGATION:-mean}"
MISSING_FACE_REWARD="${MISSING_FACE_REWARD:-0.0}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-0}"

read -r -a SEED_ARGS <<<"${SEEDS}"

export PYTORCH_CUDA_ALLOC_CONF

ARGS=(
  --data-json "${DATA_JSON}"
  --output-dir "${OUTPUT_DIR}"
  --start "${START}"
  --limit "${LIMIT}"
  --num-iterations "${NUM_ITERATIONS}"
  --gpu-ids "${GPU_IDS}"
  --rollout-world-size "${ROLLOUT_WORLD_SIZE}"
  --train-world-size "${TRAIN_WORLD_SIZE}"
  --group-size "${GROUP_SIZE}"
  --seeds "${SEED_ARGS[@]}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --num-frames "${NUM_FRAMES}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --flow-shift "${FLOW_SHIFT}"
  --audio-flow-shift "${AUDIO_FLOW_SHIFT}"
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
  --reward-frame-stride "${REWARD_FRAME_STRIDE}"
  --reward-max-frames "${REWARD_MAX_FRAMES}"
  --reward-frame-face-aggregation "${REWARD_FRAME_FACE_AGGREGATION}"
  --missing-face-reward "${MISSING_FACE_REWARD}"
)

if [[ -n "${RESUME_FROM}" ]]; then
  if [[ "${OUTPUT_DIR}" != "${RESUME_FROM}" ]]; then
    echo "For resume, OUTPUT_DIR must equal RESUME_FROM" >&2
    exit 2
  fi
  ARGS+=(--resume-from "${RESUME_FROM}")
fi
if [[ "${SAVE_ROLLOUT_VIDEO}" == "1" ]]; then
  ARGS+=(--save-rollout-video)
fi

exec "${PYTHON_BIN}" \
  examples/minimax_h3/model_training/diffusionnft/online_zero3.py \
  "${ARGS[@]}"
