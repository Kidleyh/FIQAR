#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3

# Formal single-GPU MiniMax-H3 DiffusionNFT online training entrypoint.
# Override any value through the environment; verified Phase-6/7 engineering
# defaults are intentionally conservative and do not alter the NFT objective.
DATA_JSON="${DATA_JSON:-/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/guangdian_20251114_small_clear_faces/guangdian_20251114_small_clear_faces.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/minimax_h3_diffusionnft_online/train_$(date +%Y%m%d_%H%M%S)}"
RESUME_FROM="${RESUME_FROM:-}"
START="${START:-0}"
LIMIT="${LIMIT:-10}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SEEDS="${SEEDS:-0 1 2 3}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-448}"
NUM_FRAMES="${NUM_FRAMES:-22}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-3}"
VRAM_LIMIT_GB="${VRAM_LIMIT_GB:-14}"

LORA_RANK="${LORA_RANK:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
POLICY_BETA="${POLICY_BETA:-1.0}"
KL_BETA="${KL_BETA:-1e-4}"
ADV_CLIP_MAX="${ADV_CLIP_MAX:-5}"
TIMESTEP_FRACTION="${TIMESTEP_FRACTION:-0.99}"
OLD_DECAY_TYPE="${OLD_DECAY_TYPE:-1}"

FLOW_SHIFT="${FLOW_SHIFT:-12}"
AUDIO_FLOW_SHIFT="${AUDIO_FLOW_SHIFT:-3}"
REWARD_FRAME_STRIDE="${REWARD_FRAME_STRIDE:-6}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-qkv_proj,out_proj}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-0}"
PYTHON_BIN="${PYTHON_BIN:-py312/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a SEED_ARGS <<< "${SEEDS}"
if [[ "${#SEED_ARGS[@]}" -ne "${NUM_SAMPLES}" ]]; then
  echo "SEEDS must contain exactly NUM_SAMPLES=${NUM_SAMPLES} values" >&2
  exit 2
fi
if [[ "${SAVE_ROLLOUT_VIDEO}" != "0" && "${SAVE_ROLLOUT_VIDEO}" != "1" ]]; then
  echo "SAVE_ROLLOUT_VIDEO must be 0 or 1" >&2
  exit 2
fi

RUN_ARGS=(
  --num-iterations "${NUM_ITERATIONS}"
  --num-samples-per-prompt "${NUM_SAMPLES}"
  --start "${START}"
  --limit "${LIMIT}"
  --seeds "${SEED_ARGS[@]}"
  --data-json "${DATA_JSON}"
  --python "${PYTHON_BIN}"
  --device "${DEVICE}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num-frames "${NUM_FRAMES}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --flow-shift "${FLOW_SHIFT}"
  --audio-flow-shift "${AUDIO_FLOW_SHIFT}"
  --vram-limit-gb "${VRAM_LIMIT_GB}"
  --reward-frame-stride "${REWARD_FRAME_STRIDE}"
  --lora-rank "${LORA_RANK}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
  --timestep-fraction "${TIMESTEP_FRACTION}"
  --adv-clip-max "${ADV_CLIP_MAX}"
  --policy-beta "${POLICY_BETA}"
  --kl-beta "${KL_BETA}"
  --learning-rate "${LEARNING_RATE}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --old-decay-type "${OLD_DECAY_TYPE}"
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}"
  --use-gradient-checkpointing-offload
)

if [[ -n "${RESUME_FROM}" ]]; then
  RUN_ARGS+=(--resume-from "${RESUME_FROM}")
else
  RUN_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ "${SAVE_ROLLOUT_VIDEO}" == "1" ]]; then
  RUN_ARGS+=(--save-rollout-video)
fi

exec "${PYTHON_BIN}" examples/minimax_h3/model_training/diffusionnft/online_train.py "${RUN_ARGS[@]}"
