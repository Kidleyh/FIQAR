#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3
OUTPUT_DIR="${OUTPUT_DIR:-outputs/minimax_h3_diffusionnft_rollout_resume_phase8_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-py312/bin/python}"
LOG_DIR="${OUTPUT_DIR}/resume_logs"
mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

COMMON=(
  --start 0 --limit 1 --num-samples 2 --seeds 0 1
  --output-dir "${OUTPUT_DIR}" --height 256 --width 448
  --num-frames 22 --num-inference-steps 3 --flow-shift 12
  --audio-flow-shift 3 --vram-limit-gb 14 --reward-frame-stride 6
  --policy-role base --global-step-before 0 --skip-existing
)

# A: seed 0 commits its state, then the process stops. Resume must reuse only seed 0.
set +e
"${PYTHON_BIN}" examples/minimax_h3/model_training/diffusionnft/rollout.py \
  "${COMMON[@]}" --engineering-stop-after-state-seed 0 2>&1 | tee "${LOG_DIR}/a_interrupt.log"
status=${PIPESTATUS[0]}
set -e
[[ "${status}" -ne 0 ]]
"${PYTHON_BIN}" examples/minimax_h3/model_training/diffusionnft/rollout.py \
  "${COMMON[@]}" 2>&1 | tee "${LOG_DIR}/a_resume.log"
grep -q '\[rollout-resume\] reuse seed=0' "${LOG_DIR}/a_resume.log"
grep -q '\[rollout\] generate record=0 seed=1' "${LOG_DIR}/a_resume.log"

PROMPT_DIR="$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'prompt_*' | head -1)"
# B: inject the exact crash artifact state (latent committed, state absent).
mv "${PROMPT_DIR}/seed_1_state.json" "${PROMPT_DIR}/seed_1_state.uncommitted.json"
"${PYTHON_BIN}" examples/minimax_h3/model_training/diffusionnft/rollout.py \
  "${COMMON[@]}" 2>&1 | tee "${LOG_DIR}/b_resume.log"
grep -q '\[rollout-resume\] reuse seed=0' "${LOG_DIR}/b_resume.log"
grep -q '\[rollout\] generate record=0 seed=1' "${LOG_DIR}/b_resume.log"

# C: a complete rollout is reused without generation or reward.
"${PYTHON_BIN}" examples/minimax_h3/model_training/diffusionnft/rollout.py \
  "${COMMON[@]}" 2>&1 | tee "${LOG_DIR}/c_resume.log"
grep -q '\[rollout-resume\] reuse seed=0' "${LOG_DIR}/c_resume.log"
grep -q '\[rollout-resume\] reuse seed=1' "${LOG_DIR}/c_resume.log"
! grep -q '\[rollout\] generate' "${LOG_DIR}/c_resume.log"
! find "${OUTPUT_DIR}" -name 'seed_*.mp4' -print -quit | grep -q .
echo "resume_A=true resume_B=true resume_C=true mp4_required=false"
