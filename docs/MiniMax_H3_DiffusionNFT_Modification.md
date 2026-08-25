# MiniMax-H3 DiffusionNFT Modification Log

## Repository scope

- Repository alias: **FIQAR**
- Repository path: `/gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3`
- Reward implementation (read-only external dependency): `/gemini/platform/public/aigc/human_guozz2/code/lyh/job/MagFace/inference/eval_video_face_quality.py`
- Objective: extend the working MiniMax-H3 FL2AV multi-seed inference path into DiffusionNFT-style online reinforcement post-training: online rollout -> face-quality reward -> group advantage -> forward diffusion -> advantage-weighted diffusion loss -> model update.
- Explicitly out of scope: DPO, winner/loser pair construction, reward-model reimplementation, and broad DiffSynth-Studio refactoring.

## Change history

### 2026-08-25 — phase 1 online rollout and MagFace reward

- **Modification goal:** implement current-policy MiniMax-H3 multi-seed rollout, external MagFace reward evaluation, and `rollout.json` persistence without any model update.
- **New files:**
  - `examples/minimax_h3/model_training/diffusionnft/reward_face_quality.py`
  - `examples/minimax_h3/model_training/diffusionnft/rollout.py`
  - `examples/minimax_h3/model_training/diffusionnft/smoke_test.sh`
- **Modified files:** `docs/MiniMax_H3_DiffusionNFT_Modification.md` only.
- **Core-code changes:** none. Pipeline, DiT, video/audio VAE, scheduler, generic loss/runner, and SFT trainer remain unchanged.
- **Implementation:**
  - added a subprocess-only adapter for the existing `eval_video_face_quality.py` evaluator;
  - adapter accepts generated mp4 paths, batches them through one evaluator invocation, parses `video_quality.jsonl`, and returns records containing `video_path`, `mean_quality`, `face_visible_ratio`, `num_faces`, and `reward`;
  - a missing/null `mean_quality` maps to configurable `--missing-face-reward` (default `0.0`);
  - added dataset-record and explicit `--prompt`/`--input-video` rollout modes;
  - added `--num-samples`, explicit `--seeds`, `--output-dir`, shape/step/current-policy LoRA options, existing-output reuse, and reward sampling controls;
  - each selected prompt uses one loaded current policy to generate all seeds serially, then the H3 pipeline is released before SCRFD/MagFace start;
  - final `rollout.json` is a list containing prompt, seed, absolute video path, reward, mean quality, visible ratio, and detected face count.
- **Tests:**
  - Python compilation and both CLI help paths passed under the repository `py312` environment;
  - adapter-only real test on two existing H3 renders passed: rewards `21.8002787431` and `22.6031402747`, visible ratio `1.0` for both, 7 detected faces each;
  - full current-policy smoke passed with one existing prompt, seeds 0 and 1, 448x256, 22 frames, 1 denoise step, 14GB VRAM limit, and reward frame stride 6;
  - smoke seed 0: reward `22.3552300135`, mean quality `22.3552300135`, visible ratio `0.75`, 5 detected faces;
  - smoke seed 1: reward `22.7357590199`, mean quality `22.7357590199`, visible ratio `1.0`, 10 detected faces;
  - MagFace scored 15 faces across 8 sampled frames and wrote the final `outputs/minimax_h3_diffusionnft_rollout_smoke/rollout.json`.
- **Current status:** phase 1 complete; online rollout and reward work end to end; no diffusion loss, optimizer, backward pass, or model update exists.
- **Current limitations:**
  - H3 generation is serial and expensive; the smoke uses a deliberately non-production 1-step/22-frame setting only to validate integration;
  - a concurrent long-running H3 job occupied about 60GB GPU memory, so a 39-frame/4-step/14GB attempt was stopped after 12 minutes before seed 0 completed; it did not fail with a code or OOM error;
  - reward evaluation crosses Conda environments and requires mp4 files on disk;
  - reward is currently `mean_quality`; face visibility/count are logged but do not alter valid rewards;
  - no-face behavior is a fixed configurable fallback rather than a learned or composite visibility reward;
  - samples are processed serially because MiniMax-H3 packed DiT currently requires batch size 1;
  - `--skip-existing` trusts non-empty existing files; production resume should validate video dimensions/frame count before reuse.
- **Next step:** add `examples/minimax_h3/model_training/diffusionnft/train.py` as the phase-2 entrypoint for group advantage, generated-sample VAE encoding, forward diffusion, and weighted video diffusion loss. This phase did not create or implement that loss.

### 2026-08-25 — repository and reward-path audit

- **Modification goal:** determine the smallest viable implementation surface before changing core code.
- **Modified files:** `docs/MiniMax_H3_DiffusionNFT_Modification.md` (new documentation only).
- **Core-code changes:** none.
- **Status:** analysis and planning complete; training implementation not started.
- **Next step:** implement and smoke-test the standalone reward adapter, then implement one-prompt/three-seed LoRA online-training smoke path.
- **Problems / risks found:**
  - inference `MiniMaxH3Pipeline.__call__` is decorated with `@torch.no_grad()` and cannot itself provide a training forward;
  - the reward evaluator is a subprocess CLI spanning two Conda environments, not an import-safe single-process API;
  - a video with no scored visible face produces `mean_quality: null`, so the trainer needs an explicit invalid-reward policy;
  - MiniMax-H3 packs video, audio, condition, and text into one sequence and its DiT currently enforces `x.shape[0] == 1`; rollout groups and weighted losses must therefore be processed serially or via gradient accumulation, not stacked as a normal batch;
  - inference calls reset both schedulers to inference timesteps; the training code must reset both to 1000-step training schedules before forward diffusion/loss computation;
  - full-model online training is likely impractical at current model size; LoRA is the minimum-risk first target.

## 1. Current MiniMax-H3 integration audit

### 1.1 Model and pipeline integration points

| Area | File | Confirmed behavior |
|---|---|---|
| Main audio/video pipeline | `diffsynth/pipelines/minimax_h3_audio_video.py` | `MiniMaxH3Pipeline` owns text encoder, DiT, video VAE, audio VAE, video/audio FlowMatch schedulers, preprocessing units, denoise loop, and decode. |
| AV2AV variant | `diffsynth/pipelines/minimax_h3_audio_video_av2av.py` | Separate retake/AV2AV entry; not required by the current FL2AV stage-1 path. |
| DiT | `diffsynth/models/minimax_h3_dit.py` | `MiniMaxH3DiT.forward` consumes one packed joint sequence and returns video and audio velocity rows. It supports gradient checkpointing and offload. |
| Video VAE | `diffsynth/models/minimax_h3_video_vae.py` | `encode_video`/`decode_video` implement normalization, temporal chunking, spatial tiling, and latent scaling. Both public methods are `@torch.no_grad()`, which is acceptable because VAE targets remain frozen. |
| Audio VAE | `diffsynth/models/minimax_h3_audio_vae.py` | `encode_audio`/`decode_audio` provide the corresponding audio latent path. |
| Data helpers | `diffsynth/utils/data/minimax_h3.py` | MiniMax-H3 reference loading for the existing offline training path. |

### 1.2 Working inference path

The confirmed entry is `scripts/test/minimax_h3_stage1_pair_render_multiseed.py`. It imports dataset, prompt, video, alignment, and `build_pipeline` helpers from `scripts/test/minimax_h3_stage1_pair_render.py`.

Current flow:

1. Load a source record and normalize its prompt.
2. Align size to multiples of 32 and frame count to `17n+5` at fixed 24 fps.
3. Use the first GT frame as FL2AV keyframe conditioning.
4. Build `MiniMaxH3Pipeline` with FL2VA text encoder, transformer, video VAE, audio VAE, and processor.
5. Generate multiple seeds serially with the model loaded once.
6. Receive `video, audio` in memory, then write each result with `write_video_audio`.

Within `MiniMaxH3Pipeline.__call__`:

1. Video and audio schedulers are configured independently (`flow_shift`, `audio_flow_shift`).
2. Pipeline units create initial video/audio noise, encode conditions and prompt, and build the packed sequence.
3. The denoise loop calls `model_fn_minimax_h3` and Euler-style `FlowMatchScheduler.step` for both modalities.
4. Video and audio VAEs decode final latents.

This is directly reusable for no-gradient online rollout. It should not be edited for the first implementation.

### 1.3 Model forward

`model_fn_minimax_h3` in `diffsynth/pipelines/minimax_h3_audio_video.py` is the correct trainable forward adapter. It:

- converts scheduler timesteps from `[0,1000]` to the DiT time convention;
- patchifies video latents and packs audio latents;
- merges video, audio, condition anchors, and text positions into one sequence;
- calls `MiniMaxH3DiT.forward`;
- unpatchifies/unpacks outputs and returns `(-v_video, -v_audio)`.

It is already used by both inference and the SFT training loss, so DiffusionNFT training should reuse it rather than introduce another H3 forward implementation.

### 1.4 VAE encode/decode

- Training video targets are produced today by `MiniMaxH3Unit_InputVideoEmbedder`, which preprocesses frames and calls `video_vae.encode_video`, returning both `video_latents` and `input_latents`.
- Training audio targets are produced by `MiniMaxH3Unit_InputAudioEmbedder`, returning `audio_latents` and `audio_input_latents`.
- Keyframe conditioning uses `video_vae.encode_video(..., process_image=True)` and then `patchify_video`.
- Inference decodes video latents through `decode_video`, converts tensors to PIL frames, and decodes audio through `decode_audio`.

For online training, generated samples are detached rollout artifacts. Re-encoding them with the frozen, no-grad VAE is correct and avoids retaining the denoise trajectory graph.

### 1.5 Scheduler and noise process

`diffsynth/diffusion/flow_match.py` already contains the MiniMax-H3 schedule and all operations needed by the requested algorithm:

- `set_timesteps_minimax_h3`: shifted flow-matching sigma schedule;
- `add_noise(x0, noise, timestep)`: `(1-sigma) * x0 + sigma * noise`;
- `training_target(x0, noise, timestep)`: `noise - x0` velocity target;
- `training_weight(timestep)`: existing timestep-dependent SFT weight;
- `step`: inference integration.

Video and audio use separate scheduler objects. Existing H3 training initializes both to 1000 training timesteps.

### 1.6 Existing DiffSynth-Studio training support

`examples/minimax_h3/model_training/train.py` defines `MiniMaxH3TrainingModule` and supports:

- dataset preprocessing and cached two-stage training;
- full DiT or LoRA training;
- gradient checkpointing/offload;
- Accelerate/DDP/ZeRO-3 launch paths;
- optimizer, checkpoint, and logger utilities from `diffsynth/diffusion`;
- FL2AV first/last-frame conditioning;
- `FlowMatchSFTMiniMaxH3AudioVideoLoss`.

The current SFT loss already performs the same low-level forward-diffusion mechanics required by DiffusionNFT, but it samples one timestep and returns an unweighted video-plus-audio MSE. The generic runner assumes one dataset item per step (`collate_fn=lambda x: x[0]`) and does not perform rollout/reward/group advantage. Therefore the existing SFT entrypoint is a reference and a source of reusable setup utilities, not a directly usable online-RL loop.

### 1.7 LoRA support

LoRA is confirmed in both paths:

- Inference scripts can load a state dict through the pipeline LoRA loader and hot-load LoRA into DiT layers under VRAM management.
- Training uses PEFT `inject_adapter_in_model` via `DiffusionTrainingModule.switch_pipe_to_training_mode`.
- Existing H3 launch recipes target DiT `qkv_proj,out_proj`, rank 32, with checkpoint load/export support.
- Full-model training recipes also exist with ZeRO-3.

Recommendation: first DiffusionNFT implementation should train DiT LoRA only, initially using the existing `qkv_proj,out_proj` target set. Full-model training should remain a later opt-in path after correctness and memory are validated.

## 2. Existing face-quality reward model audit

### 2.1 Actual evaluation flow

`eval_video_face_quality.py` is an offline orchestrator with three stages:

1. The top-level Python process parses input pairs/videos and writes a manifest.
2. In Conda env `scrfd-face`, SCRFD samples frames at `--frame-stride`, detects all faces, aligns each face to 112x112 with five landmarks, and writes detection metadata/crops.
3. In Conda env `magface-quality`, MagFace `iresnet100` embeds aligned crops; the L2 norm of each 512-D embedding is the face quality score.
4. Face scores are aggregated per frame (`mean|min|max`), then across scored frames into `mean_quality`, `median_quality`, `p10_quality`, and `min_quality`.

Defaults confirmed from code:

- SCRFD model: `insightface/models/scrfd_10g_bnkps.onnx`
- MagFace checkpoint: `MagFace/checkpoints/magface_iresnet100_quality.pth`
- frame stride: 25
- face aggregation: mean
- pair/video statistic: mean quality

Existing verified outputs under `MagFace/outputs/face_quality_minimax_h3_seed0_124f` confirm that render rows in `video_quality.jsonl` contain `mean_quality`, face counts, sampled/visible-frame counts, and `face_visible_ratio`.

### 2.2 Proposed call interface (without modifying MagFace)

For each online rollout group:

1. Write generated samples to temporary mp4 files using the existing `write_video_audio` utility.
2. Write an input JSON list such as:

   ```json
   [
     {"name": "prompt000_seed0", "render_video_path": "/path/seed0.mp4"},
     {"name": "prompt000_seed1", "render_video_path": "/path/seed1.mp4"}
   ]
   ```

3. Invoke the existing script as a subprocess:

   ```bash
   /root/miniconda3/bin/conda run --no-capture-output -n <orchestrator-env> python \
     /gemini/platform/public/aigc/human_guozz2/code/lyh/job/MagFace/inference/eval_video_face_quality.py \
     --input-json <rollout.json> \
     --output-dir <reward-output-dir> \
     --allow-unpaired \
     --frame-stride 25 \
     --frame-face-aggregation mean \
     --pair-score-stat mean_quality
   ```

   The outer environment only needs to run the standard-library orchestrator; that script itself enters `scrfd-face` and `magface-quality` for the GPU stages. The adapter should expose the Python executable/Conda executable as configuration rather than assume the training environment can import either model.

4. Parse `video_quality.jsonl`, map absolute `video_path` back to rollout ID, and use each render row's `mean_quality` as the scalar reward.

5. Preserve `face_visible_ratio`, `sampled_frame_count`, and `detected_face_count` in training logs. If `mean_quality` is null (no scored face), apply a configured reward floor/penalty or skip the entire invalid group. The recommended default is a fixed penalty so disappearance of faces cannot evade the objective; this behavior must be logged and configurable.

The reward remains non-differentiable and outside the training process. No MagFace source modification or checkpoint conversion is needed.

## 3. Minimal DiffusionNFT design

### 3.1 Online algorithm

For each prompt/FL2AV condition:

1. Generate `G` samples from the current policy using distinct recorded seeds under no-grad.
2. Write samples and evaluate MagFace face quality in one batched reward subprocess call.
3. Compute within-prompt group advantages, initially `A_i = (r_i - mean(r)) / (std(r) + eps)`, with configurable clipping. Do not build winner/loser pairs.
4. Re-encode each sampled video to frozen VAE latent `x0_i` (and generated audio if audio context is retained).
5. Sample training timestep/noise and forward diffuse with the existing scheduler.
6. Call the existing H3 `model_fn` with the same prompt and FL2AV condition.
7. Compute per-sample video velocity MSE, existing scheduler timestep weight, and signed clipped advantage weight. Average across the group and backpropagate once (or accumulate serially because H3 supports packed batch size 1).
8. Update LoRA parameters, log reward/advantage/loss/valid-face metrics, checkpoint, and continue with the updated policy.

The first implementation should optimize the video diffusion error only because the reward measures video face quality. Audio latents may still be encoded/noised and passed as context for the joint DiT forward, but audio-head MSE should not receive the face-quality advantage unless a later audio reward is introduced.

### 3.2 Can the existing training forward be reused directly?

**Partially, at the correct layer.**

- Reuse directly: pipeline units, `model_fn_minimax_h3`, `MiniMaxH3DiT.forward`, VAE encoders, scheduler `add_noise`, `training_target`, `training_weight`, LoRA injection, gradient checkpointing, optimizer/checkpoint/logging helpers.
- Do not reuse as-is: `MiniMaxH3Pipeline.__call__` for training (no-grad); `MiniMaxH3TrainingModule.forward` and `FlowMatchSFTMiniMaxH3AudioVideoLoss` as the complete RL objective (offline single-sample unweighted SFT); generic `launch_training_task` as the complete loop (no online rollout/reward/group semantics).

No change to the DiT forward signature is required. A local DiffusionNFT loss/helper should call the existing `model_fn` and return per-sample video loss before group weighting.

### 3.3 File plan

#### New files required

| File | Purpose |
|---|---|
| `examples/minimax_h3/model_training/diffusionnft/train.py` | Standalone online loop: model/LoRA setup, current-policy rollout, reward call, group advantage, generated-sample VAE encoding, forward diffusion, weighted video loss, Accelerate backward/update, logging, resume/checkpoint. |
| `examples/minimax_h3/model_training/diffusionnft/reward_face_quality.py` | Thin subprocess adapter around the existing MagFace evaluator; creates input JSON, validates outputs, maps scores to rollout IDs, and applies/logs invalid-face policy. Contains no reward model code. |
| `examples/minimax_h3/model_training/diffusionnft/MiniMax-H3-FL2VA-LoRA.sh` | Reproducible conservative LoRA launch configuration for the first smoke and full runs. |
| `scripts/test/minimax_h3_diffusionnft_reward_smoke.py` | Fast adapter-only smoke test against existing/generated mp4 files; validates ordering, null-score behavior, and parsed metadata without a training update. |

#### Existing files to modify

| File | Planned modification |
|---|---|
| `docs/MiniMax_H3_DiffusionNFT_Modification.md` | Mandatory append/update for every later change: date, goal, files, content, status, next step, and problems. |

#### Existing core files not planned for modification

- `diffsynth/pipelines/minimax_h3_audio_video.py`
- `diffsynth/models/minimax_h3_dit.py`
- `diffsynth/models/minimax_h3_video_vae.py`
- `diffsynth/models/minimax_h3_audio_vae.py`
- `diffsynth/diffusion/flow_match.py`
- `diffsynth/diffusion/loss.py`
- `diffsynth/diffusion/runner.py`
- `examples/minimax_h3/model_training/train.py`
- all files in the MagFace repository

Keeping the DiffusionNFT loss local to the new entrypoint avoids changing generic DiffSynth behavior. A core loss helper should be added only later if another pipeline needs the same objective and the interface has stabilized.

## 4. Concise implementation plan

1. Add and smoke-test the read-only MagFace subprocess adapter on known MiniMax-H3 renders; verify exact score-to-path mapping and invalid-face handling.
2. Add a one-prompt, three-seed rollout-only mode using current-policy DiT LoRA and the existing FL2AV conditioning path; persist seeds/rewards/metadata.
3. Add normalized group advantage and a serial per-sample weighted video FlowMatch loss using existing H3 `model_fn` and schedulers; perform one LoRA optimizer step.
4. Verify gradients exist only on intended LoRA parameters; verify high-reward samples contribute positive imitation pressure and low-reward samples negative pressure; verify scheduler training state is restored after every rollout.
5. Add checkpoint/resume, deterministic seed accounting, failure-safe temporary artifact handling, and compact metrics.
6. Run a short multi-step smoke, compare pre/post fixed-seed rewards, and only then scale prompts, frames, resolution, GPUs, or consider full-model training.

## 5. Acceptance checks for the implementation phase

- No winner/loser pairs or DPO loss exist.
- Every optimizer step is traceable to rollout IDs, seeds, raw rewards, advantages, timesteps, and checkpoint version.
- Reward values come only from the existing MagFace evaluator outputs.
- Missing/no-face outputs cannot silently become high reward.
- Generated samples are from the current trainable policy, not a stale separately loaded checkpoint.
- Inference and training scheduler state transitions are explicit and tested.
- Only intended DiT LoRA parameters require gradients in the first milestone.
- The modification log is updated in the same change as every implementation edit.
