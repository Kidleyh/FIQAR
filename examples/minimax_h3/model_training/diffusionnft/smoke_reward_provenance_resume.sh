#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3
OUTPUT_DIR="${OUTPUT_DIR:-outputs/minimax_h3_diffusionnft_reward_provenance_phase81_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-py312/bin/python}"
LOG_DIR="${OUTPUT_DIR}/provenance_logs"
mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

COMMON=(
  --start 0 --limit 1 --num-samples 2 --seeds 0 1
  --output-dir "${OUTPUT_DIR}" --height 256 --width 448
  --num-frames 22 --num-inference-steps 1 --flow-shift 12
  --audio-flow-shift 3 --vram-limit-gb 14
  --policy-role base --global-step-before 0 --skip-existing
)

"${PYTHON_BIN}" examples/minimax_h3/model_training/diffusionnft/rollout.py \
  "${COMMON[@]}" --reward-frame-stride 6 2>&1 | tee "${LOG_DIR}/initial.log"

"${PYTHON_BIN}" examples/minimax_h3/model_training/diffusionnft/rollout.py \
  "${COMMON[@]}" --reward-frame-stride 6 2>&1 | tee "${LOG_DIR}/same_config.log"
[[ "$(grep -c '\[rollout-resume\] reuse seed=' "${LOG_DIR}/same_config.log")" -eq 2 ]]
! grep -q '\[rollout\] generate' "${LOG_DIR}/same_config.log"

"${PYTHON_BIN}" examples/minimax_h3/model_training/diffusionnft/rollout.py \
  "${COMMON[@]}" --reward-frame-stride 5 2>&1 | tee "${LOG_DIR}/changed_stride.log"
[[ "$(grep -c 'reward_frame_stride mismatch' "${LOG_DIR}/changed_stride.log")" -eq 2 ]]
[[ "$(grep -c '\[rollout\] generate record=0 seed=' "${LOG_DIR}/changed_stride.log")" -eq 2 ]]
! grep -q '\[rollout-resume\] reuse seed=' "${LOG_DIR}/changed_stride.log"

"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
states = sorted(root.glob("prompt_*/seed_*_state.json"))
assert len(states) == 2
required = {
    "reward_frame_stride", "reward_max_frames",
    "reward_frame_face_aggregation", "missing_face_reward",
    "scrfd_model_sha256", "magface_checkpoint_sha256",
}
for path in states:
    state = json.loads(path.read_text())
    assert state["complete"] is True and state["reward_frame_stride"] == 5
    assert required <= state.keys()
    assert len(state["scrfd_model_sha256"]) == 64
    assert len(state["magface_checkpoint_sha256"]) == 64
print("same_reward_config_reuse=true changed_reward_config_regenerate=true seeds=2")
PY
