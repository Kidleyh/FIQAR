#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3

OUTPUT_DIR="${OUTPUT_DIR:-outputs/minimax_h3_diffusionnft_online/phase8_inmemory_smoke_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-py312/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" examples/minimax_h3/model_training/diffusionnft/online_train.py \
  --num-iterations 1 \
  --num-samples-per-prompt 2 \
  --start 0 --limit 1 --seeds 0 1 \
  --output-dir "${OUTPUT_DIR}" \
  --python "${PYTHON_BIN}" --device cuda:0 \
  --height 256 --width 448 --num-frames 22 --num-inference-steps 3 \
  --flow-shift 12 --audio-flow-shift 3 --vram-limit-gb 14 \
  --reward-frame-stride 6 --lora-rank 4 \
  --lora-target-modules qkv_proj,out_proj \
  --timestep-fraction 0.99 --adv-clip-max 5 --policy-beta 1.0 \
  --kl-beta 1e-4 --learning-rate 1e-4 --max-grad-norm 1.0 \
  --old-decay-type 1 --use-gradient-checkpointing-offload

"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
rollout_path = root / "iteration_000000" / "rollout" / "rollout.json"
records = json.loads(rollout_path.read_text())
assert len(records) == 2
for row in records:
    assert row["video_path"] is None
    assert pathlib.Path(row["latent_path"]).stat().st_size > 0
    state = pathlib.Path(row["sample_state_path"])
    assert state.stat().st_size > 0 and json.loads(state.read_text())["complete"] is True
assert not list((root / "iteration_000000" / "rollout").rglob("seed_*.mp4"))
checkpoint = root / "checkpoints" / "checkpoint-1"
for name in ("current_lora.safetensors", "old_lora.safetensors", "optimizer.pt", "training_state.json"):
    assert (checkpoint / name).stat().st_size > 0
online = json.loads((root / "online_state.json").read_text())
assert online["global_step"] == 1 and online["iterations"][0]["iteration_success"] is True
print("inmemory_reward=true no_rollout_mp4=true global_step=0->1 iteration_success=true")
PY
