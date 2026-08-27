# MiniMax-H3 DiffusionNFT Modification Log

## Repository scope

- Repository alias: **FIQAR**
- Repository path: `/gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3`
- Reward implementation (read-only external dependency): `/gemini/platform/public/aigc/human_guozz2/code/lyh/job/MagFace/inference/eval_video_face_quality.py`
- Objective: extend the working MiniMax-H3 FL2AV multi-seed inference path into DiffusionNFT-style online reinforcement post-training: online rollout -> face-quality reward -> group advantage -> forward diffusion -> advantage-weighted diffusion loss -> model update.
- Explicitly out of scope: DPO, winner/loser pair construction, reward-model reimplementation, and broad DiffSynth-Studio refactoring.

## Change history

### 2026-08-27 — phase 6 training observability and short-horizon stability pilot

- **Modification goal:** add durable per-iteration rollout/training observability to the verified bounded online loop, then run an unchanged-parameter five-iteration, four-sample pilot to inspect short-horizon dynamics. The DiffusionNFT objective, advantage/r mapping, current/old/reference policy semantics, rollout state contract, and H3 pipeline/DiT/VAE/schedulers are unchanged.
- **Modified file:** `examples/minimax_h3/model_training/diffusionnft/online_train.py`.
- **New file:** `examples/minimax_h3/model_training/diffusionnft/smoke_online_stability.sh`.
- **Rollout metrics:** each completed iteration now records raw `rewards`, population mean/std/min/max, per-sample `mean_quality`, `face_visible_ratio`, and `num_faces`, plus aggregate visible ratio, missing-face count, and missing-face ratio. A sample is counted as missing-face when its quality is null or its detected-face count is zero; invalid numeric metadata fails validation.
- **Training metrics:** the orchestration layer parses the already emitted group/step logs and records advantages, positive/negative/policy/KL/total losses, gradient norm, current/reference prediction distance, current/old parameter distance, actual old-policy decay, and global step before/after. Required metrics fail closed rather than silently disappearing from the run history.
- **Durable outputs:** after every completed iteration and on resume, `metrics.jsonl` is deterministically rematerialized with exactly one row per completed iteration. `training_summary.json` is atomically updated with iteration count, reward mean/std series and initial/final change, KL maximum/final, gradient-norm maximum, current/reference distance series, face-visible trend, missing-face ratio, and checkpoint list. The canonical recovery state remains atomic `online_state.json` and embeds each iteration's complete metrics row.
- **Group-size labeling:** K=2 remains supported but is explicitly labeled engineering-smoke-only because normalized two-sample advantages nearly collapse to +/-1. K=4 is labeled as the minimum stability/experiment group used here; no change to the advantage calculation was made.
- **Pilot command:** `RUN_ID=phase6_stability_final_20260826 bash examples/minimax_h3/model_training/diffusionnft/smoke_online_stability.sh` (then the same generated `online_train.py --resume-from outputs/minimax_h3_diffusionnft_online/phase6_stability_final_20260826 ...` command after an SSH transport interruption).
- **Pilot configuration:** one fixed prompt at dataset position 0; fixed seeds `[0, 1, 2, 3]`; 5 iterations; 448x256; 22 frames; 3 inference steps; `timestep_fraction=0.99` (2 trained timesteps/sample, 8 normalized backward passes/iteration); LoRA rank 4 on `qkv_proj,out_proj`; `adv_clip_max=5`, `policy_beta=1`, `kl_beta=1e-4`, learning rate `1e-4`, max gradient norm 1, and the existing type-1 old decay. No hyperparameter was adjusted during the pilot.
- **Real per-iteration results:**

| Iteration | Rewards (seeds 0/1/2/3) | Mean / std | Advantages | Policy loss | KL | Grad norm | Current/ref distance | Current/old distance | Visible / missing |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | `[21.6371, 22.1256, 22.6419, 22.2219]` | `22.1566 / 0.3573` | `[-1.4536, -0.0869, 1.3579, 0.1827]` | 3.114023 | 0 | 0.078539 | 0 | 0.000235 | `1.0 / 0%` |
| 1 | `[22.0472, 21.7636, 21.9236, 21.9575]` | `21.9230 / 0.1025` | `[1.2107, -1.5534, 0.0063, 0.3364]` | 2.871445 | 0.001519 | 0.083888 | 0.038107 | 0.000395 | `1.0 / 0%` |
| 2 | `[21.6522, 21.8224, 22.0270, 22.7744]` | `22.0690 / 0.4284` | `[-0.9728, -0.5756, -0.0981, 1.6465]` | 2.752181 | 0.001432 | 0.160874 | 0.037237 | 0.000528 | `1.0 / 0%` |
| 3 | `[21.8607, 21.1643, 21.6802, 23.2400]` | `21.9863 / 0.7676` | `[-0.1636, -1.0707, -0.3988, 1.6331]` | 3.042433 | 0.001652 | 0.138750 | 0.039852 | 0.000604 | `1.0 / 0%` |
| 4 | `[22.0091, 22.4158, 22.3204, 23.6337]` | `22.5948 / 0.6184` | `[-0.9468, -0.2894, -0.4435, 1.6798]` | 2.820920 | 0.001079 | 0.142658 | 0.032519 | 0.000688 | `1.0 / 0%` |

- **Additional loss/state results:** positive losses were `[0.622805, 0.572350, 0.546901, 0.605421, 0.559571]`; negative losses were `[0.622805, 0.576017, 0.554542, 0.613403, 0.568073]`; total losses were `[3.114023, 2.871446, 2.752181, 3.042433, 2.820920]`. Type-1 old decay used real global steps and progressed `[0.001, 0.002, 0.003, 0.004, 0.005]`. All 20 samples had visible ratio 1.0, zero missing-face samples, and detected-face counts between 7 and 10.
- **Stability conclusion:** this short pilot showed no sustained reward decline: iteration means fluctuated and the final mean exceeded the initial mean by `+0.438129`, but five iterations on one prompt are insufficient evidence of learning effectiveness. KL stayed bounded (`max=0.001652`, `final=0.001079`), gradient norm stayed far below the clip threshold (`max=0.160874`), and current/reference prediction distance stayed in `[0.032519, 0.039852]` after the expected zero-valued initialization step rather than rapidly diverging. Face visibility remained 1.0 with no missing-face outputs. Rewards stayed in the observed range `21.1643–23.6337`; no obvious reward explosion occurred. Current/old parameter distance increased gradually as expected from repeated updates/decay, from `0.000235` to `0.000688`.
- **Recovery observation:** the first SSH client disconnected after checkpoint-2 while iteration 2 was partially rolled out. Resuming the same run reused the already complete seed artifacts, finished only the missing seeds, and continued through checkpoint-5. No completed rollout or checkpoint was regenerated or overwritten.
- **Artifacts/status:** run directory `outputs/minimax_h3_diffusionnft_online/phase6_stability_final_20260826`; `online_state.json` reports `completed_iteration=4`, `global_step=5`; `metrics.jsonl` contains exactly five rows; `training_summary.json` reports five checkpoints; checkpoint-1 through checkpoint-5 each contain current/old LoRA, optimizer state, and training state. Phase 6 is complete.
- **Current limitations / next step:** the result is an engineering stability signal, not a quality conclusion: one prompt, four fixed seeds, low resolution, short clips, three rollout steps, and five updates. The next experiment should expand prompt coverage and duration under a separately bounded configuration while retaining these metrics; multi-GPU, 720p/124-frame training, WandB, reward redesign/visible-ratio weighting, audio reward/loss, new RL objectives, and automated checkpoint cleanup remain deferred.

### 2026-08-26 — phase 5 bounded resumable online DiffusionNFT loop

- **Modification goal:** orchestrate the already verified rollout/reward and NFT update entrypoints into a bounded `old policy rollout -> MagFace -> current update -> old update -> checkpoint -> next old rollout` loop. No rollout, reward, advantage, or NFT loss implementation is duplicated in the orchestration layer.
- **New files:**
  - `examples/minimax_h3/model_training/diffusionnft/online_train.py`;
  - `examples/minimax_h3/model_training/diffusionnft/smoke_online_2iter.sh`.
- **Modified files:**
  - `examples/minimax_h3/model_training/diffusionnft/rollout.py` adds explicit policy provenance fields and validates their LoRA path/hash relationship;
  - `examples/minimax_h3/model_training/diffusionnft/train.py` optionally requires rollout-policy provenance before loading H3 and saves checkpoints through an atomic directory rename while refusing to overwrite any existing checkpoint;
  - this modification log.
- **Online policy rule:** at `global_step=0`, rollout is the adapter-free base H3 and records `policy_role=base` with null LoRA path/hash/source checkpoint. At every later step, online rollout loads only `latest_checkpoint/old_lora.safetensors`; current LoRA is never supplied to rollout.
- **Provenance contract:** every rollout record stores `policy_role`, absolute `policy_lora_path`, actual file SHA-256, absolute `source_checkpoint`, and `global_step_before`. `rollout.py` recomputes and validates the supplied hash, requires an old-policy path to be exactly `<source_checkpoint>/old_lora.safetensors`, and rejects inconsistent base/old arguments. `train.py --require-policy-provenance` independently derives the expected base/old state from `--resume-from`, hashes the checkpoint old LoRA, and rejects any record mismatch before model loading.
- **Bounded orchestration:** `online_train.py` supports `--num-iterations`, `--num-samples-per-prompt`, `--start`, `--limit`, explicit `--seeds`, `--output-dir`, and `--resume-from`. Each iteration selects one dataset position (`start + iteration % limit`) and one same-prompt group, launches existing `rollout.py`, validates all mp4/latent/condition/reward/provenance artifacts, then launches existing `train.py --mode nft-step` for exactly one optimizer step.
- **Run state/layout:** each run contains atomic `online_state.json`, `iteration_<index>/rollout`, per-stage logs, and `checkpoints/checkpoint-<global_step>`. State records completed iteration, latest checkpoint, global step, dataset position, seeds, stable run configuration, per-iteration rollout path, policy provenance, rewards and population statistics, parsed KL/current-old distance, checkpoint path, and success status.
- **Recovery behavior:** a planned iteration with a fully valid rollout reuses it; a partial rollout restarts `rollout.py --skip-existing` under the policy locked in state; a complete checkpoint whose state update was interrupted is validated and adopted without retraining. A completed run resumed at the same iteration bound exits immediately. Stable configuration or policy mismatches fail closed. Checkpoints are written to unique temporary directories and atomically renamed only after current/old LoRA, optimizer, and training state all succeed; an existing formal checkpoint path is never overwritten.
- **Real two-process smoke:** `RUN_ID=phase5_final_20260826 bash examples/minimax_h3/model_training/diffusionnft/smoke_online_2iter.sh`. The first process completed iteration 0/checkpoint-1; a second `--resume-from` process reported `resume_success=true`, then completed iteration 1/checkpoint-2. Configuration used one prompt, seeds 0/1, 448x256, 22 frames, 2 rollout inference steps, and one selected NFT timestep per sample.
- **Iteration 0 (base rollout, step 0 -> 1):** rewards `[22.84569120, 21.15675545]`, population mean/std/min/max `22.00122333 / 0.84446788 / 21.15675545 / 22.84569120`; provenance was base with null LoRA path/hash and `policy_match=true`. Group policy loss `3.02616882`, initial KL `0`, gradient norm `0.14696176`, current/old distance after update `0.000235897996`, and checkpoint-1 was atomically completed with `iteration_success=true`.
- **Iteration 1 (checkpoint-1 old rollout, step 1 -> 2):** rollout loaded exactly `checkpoint-1/old_lora.safetensors` with SHA-256 `56bdad99af7a65d51770697dcbb6d1ccd99d1942ed9c5c47c251de30d280ccb1`; the existing loader patched exactly 104 H3 DiT modules and generated both mp4 and clean video/audio latent artifacts. Every rollout record hash matched the file and referenced checkpoint-1, while training independently printed `policy_match=true` and `resume_success=true global_step=1`.
- **Iteration 1 metrics:** rewards `[22.64419484, 20.94056980]`, population mean/std/min/max `21.79238232 / 0.85181252 / 20.94056980 / 22.64419484`; group policy loss `3.00572443`, KL `0.00176368`, current/reference prediction distance `0.04182750`, gradient norm `2.50354202` before clipping, and current/old distance after the step `0.000392955543`. Checkpoint-2 contains current/old LoRA, optimizer, and `global_step=2`; final `iteration_success=true`.
- **Final recovery verification:** `online_state.json` records `completed_iteration=1`, `global_step=2`, and checkpoint-2 as latest. Re-running the same `--resume-from` target printed `target already complete` and performed no rollout or training. Both checkpoints contain exactly the four required files and no temporary checkpoint directory remained.
- **Core/loss changes:** none. MiniMax-H3 pipeline, DiT, VAEs, schedulers, common training framework, MagFace evaluator, and the verified DiffusionNFT objective are unchanged in Phase 5.
- **Current status:** phase 5 complete. Two real online iterations passed across a process restart, including base rollout, old-LoRA rollout, exact policy-hash validation, reward, resumed NFT update, old-policy update, atomic checkpointing, and no-op recovery of a completed run.
- **Current limitations / next step:** one process/GPU, one prompt group per iteration, serial samples, video-only reward/NFT policy loss, no WandB, and no large-scale checkpoint retention. The next step can run a longer bounded experiment with larger `num_samples_per_prompt` and more inference steps, then evaluate reward/quality trends before adding multi-GPU or retention policy.

### 2026-08-26 — phase 4.5 rollout/training state and conditioning consistency

- **Modification goal:** make formal NFT training consume the exact clean joint latent state and exact FL2AV first-frame condition used by rollout, while retaining mp4 only for MagFace reward and retaining the Phase-2 mp4/VAE backward smoke as a legacy diagnostic.
- **Modified files:**
  - `diffsynth/pipelines/minimax_h3_audio_video.py`;
  - `examples/minimax_h3/model_training/diffusionnft/rollout.py`;
  - `examples/minimax_h3/model_training/diffusionnft/train.py`;
  - `examples/minimax_h3/model_training/diffusionnft/smoke_nft_step.sh`;
  - `examples/minimax_h3/model_training/diffusionnft/smoke_nft_2step.sh`.
- **New file:** `examples/minimax_h3/model_training/diffusionnft/smoke_nft_state_consistency.sh`.
- **Minimal pipeline extension:** `MiniMaxH3Pipeline.__call__` now accepts backward-compatible `return_latents=False`. When enabled, it returns a third dictionary containing detached CPU copies of final `video_latents` and `audio_latents`, captured after denoising and before either VAE decode. The default two-value return and all default generation behavior are unchanged; DiT, VAE, scheduler, and common training code are unchanged.
- **Rollout state persistence:** each seed writes `seed_<seed>_latents.safetensors` with exactly `video_latents` and `audio_latents`, and records its absolute `latent_path`. `--skip-existing` now reuses a sample only when both mp4 and latent file exist. The resized rollout first frame is saved losslessly as `condition_image.png`, reloaded, and that exact reloaded image is passed to rollout as `keyframes=[condition_image]`, `keyframe_indices=[0]`. Each record stores `condition_image_path`, its SHA-256, width, height, frame count, fps, and scheduler metadata.
- **Formal NFT input contract:** `--mode nft-step` requires every selected record to contain an existing `latent_path` and `condition_image_path`, matching geometry, and valid seed/scheduler metadata. Missing state raises immediately; it never falls back to decoding or VAE-encoding the mp4. `--mode backward-smoke` retains the legacy mp4 decode -> VAE encode path.
- **Exact FL2AV conditioning:** formal NFT loads the saved RGB PNG, verifies its optional SHA-256 and dimensions, and runs the existing H3 condition units with `keyframes=[condition_image]`, `keyframe_indices=[0]`, and the rollout seed. A frame list is supplied only for the existing geometry contract; `input_video=None` is set before pipeline units run, and training rejects any resulting clean `input_latents`, preventing accidental mp4/VAE reuse.
- **Joint forward state:** for each selected schedule index, training samples deterministic video/audio noise and uses each existing scheduler independently:
  - `video_xt = scheduler.add_noise(video_x0, video_noise, video_t)` = `(1-sigma_video) * video_x0 + sigma_video * video_noise`;
  - `audio_xt = scheduler_audio.add_noise(audio_x0, audio_noise, audio_t)` = `(1-sigma_audio) * audio_x0 + sigma_audio * audio_noise`.
  Both noisy states enter every current/old/reference joint H3 DiT forward. NFT reconstruction, policy loss, reward, and KL remain video-prediction-only; no audio reward or audio policy loss was added.
- **Real rollout smoke:** prompt 0, seeds 0/1, 448x256, 22 frames, 3 inference steps, video shift 12 and audio shift 3. Each safetensors file reloaded successfully with video shape `[1, 24, 7, 16, 28]` and audio shape `[2, 32, 37]`; the shared condition PNG is 448x256 and its SHA-256 matches both rollout records. MagFace rewards were `21.79515616` and `22.16554221`, with face-visible ratio `1.0` and 8 detected faces for each sample.
- **Real consistency optimizer smoke:** `bash examples/minimax_h3/model_training/diffusionnft/smoke_nft_state_consistency.sh` completed 2 samples x 2 selected timesteps = 4 normalized backward passes and one optimizer step. Both samples printed `uses_rollout_clean_latent=true` and `uses_exact_rollout_condition=true`. The non-endpoint schedule index printed `video_sigma=0.95999998` and `audio_sigma=0.85714281`, proving separate scheduler shifts/noise states were used; the endpoint printed 1.0/1.0.
- **Optimizer result:** group positive/negative loss `0.60300767 / 0.60300767`, policy loss and total loss `3.01503801`, KL `0` on the initial current=reference step, gradient norm `0.10306433`, current parameter delta `0.235702204011`, frozen old/reference policies had no gradients, and final `optimizer_step=true`, `global_step 0 -> 1`.
- **Current status:** phase 4.5 complete. Formal NFT training now uses rollout-native clean video/audio latent state and exact rollout FL2AV conditioning, with a real joint-forward LoRA optimizer step verified.
- **Current limitations / next step:** latent artifacts are per-sample and not yet managed by retention/cleanup policy; only video prediction participates in reward and NFT policy loss. The next phase may build a bounded online generate -> reward -> train orchestration loop around this now-consistent state contract. Multi-GPU and audio reward/loss remain deferred.

### 2026-08-26 — phase 4 repeated multi-timestep DiffusionNFT updates

- **Modification goal:** replace the Phase-3 random 1000-step single-timestep NFT path with rollout-schedule multi-timestep training, then persist and resume current/old LoRA, optimizer, and `global_step` across real updates. Online rollout-training iteration remains out of scope.
- **Modified files:**
  - `examples/minimax_h3/model_training/diffusionnft/train.py`;
  - `examples/minimax_h3/model_training/diffusionnft/rollout.py`;
  - `examples/minimax_h3/model_training/diffusionnft/smoke_nft_step.sh` (removed the obsolete fixed decay-step argument and pointed it at scheduler-aware rollout data).
- **New file:** `examples/minimax_h3/model_training/diffusionnft/smoke_nft_2step.sh`.
- **Rollout scheduler metadata:** every new rollout record now stores `num_inference_steps`, `flow_shift`, and `audio_flow_shift`. NFT training requires these values to exist and agree across the same-prompt group; it does not silently guess values for legacy rollout files.
- **Multi-timestep implementation:**
  - video/audio `FlowMatchScheduler` instances are rebuilt with the rollout's actual inference step count and respective shifts;
  - the resulting inference `timesteps`/`sigmas` are unchanged, while `training=True` is set through the existing scheduler API so the existing H3 input units produce clean VAE latents;
  - formal NFT training uses the leading `int(num_inference_steps * timestep_fraction)` schedule entries, matching official DiffusionNFT; `--timestep-fraction` defaults to `0.99` and a zero selected count is rejected;
  - timestep order is independently shuffleable per sample and global step (`--shuffle-timesteps` / `--no-shuffle-timesteps`);
  - H3 remains batch size 1: each sample and selected timestep is forwarded serially, every loss is divided by `group_size * selected_timestep_count`, and all gradients are accumulated before one optimizer step;
  - the previous 1000-step random single-timestep route remains only in `--mode backward-smoke`, not the NFT policy path.
- **Checkpoint format:** each checkpoint directory contains:
  - `current_lora.safetensors`: rollout-compatible current LoRA keys such as `blocks.0.attn.out_proj.lora_A.weight`;
  - `old_lora.safetensors`: frozen old-policy LoRA;
  - `optimizer.pt`: AdamW state;
  - `training_state.json`: `global_step` plus model, LoRA, reward/loss, scheduler, timestep, optimizer, clipping, and decay configuration.
- **Resume behavior:** `--resume-from` validates the saved NFT configuration, loads current and old from separate files, restores optimizer state and `global_step`, and does not execute the first-step old=current initialization. The reference remains the adapter-disabled base H3 and is not saved.
- **Old-policy precision and update:** frozen old LoRA is kept in fp32 and EMA arithmetic casts current to fp32 before applying `old = decay * old + (1-decay) * current`. This preserves the official small early decay values that would otherwise round away in bf16. Decay is computed from the incremented real `global_step`; no manually fixed decay step remains.
- **Real test rollout:** same source prompt and seeds 0/1, 22 frames at 448x256, 3 inference steps, video flow shift 12, audio flow shift 3. MagFace rewards were `21.59431569` and `21.51016061`. With fraction `0.99`, each sample trained schedule indices 0 and 1 (`sigma=1.0` and `0.96`), giving 2 timesteps per sample and 4 normalized backward passes per optimizer step.
- **Real two-process smoke command:** `RUN_ID=phase4_final_20260826 bash examples/minimax_h3/model_training/diffusionnft/smoke_nft_2step.sh`.
- **Step 1 result (`global_step 0 -> 1`):**
  - shuffled timestep order `[1, 0]` for both samples;
  - group positive/negative/policy loss `2.23898125 / 2.23898125 / 11.19490623`;
  - KL and current/reference prediction distance `0 / 0`, expected before the first update;
  - gradient norm `0.12542670`; all 208 current LoRA tensors received gradients;
  - current parameter delta `0.235796341397`;
  - official type-1 decay at real step 1: `0.001`; post-update current/old distance `0.000235790816`;
  - saved `outputs/minimax_h3_diffusionnft_phase4_2step/phase4_final_20260826/checkpoint-1`.
- **Step 2 resume result (`global_step 1 -> 2`):**
  - `resume_success=true`; restored pre-step current/old distance exactly `0.000235790816`, proving current and old were not reinitialized or overwritten;
  - group positive/negative/policy loss `2.28660250 / 2.27989721 / 11.41664600`;
  - group KL `0.00579755` and current/reference prediction distance `0.06821136`, so reference regularization is no longer identically zero;
  - gradient norm `0.11655332`; current parameter delta `0.197328707721`;
  - official type-1 decay at real step 2: `0.002`; post-update current/old distance `0.000394891737`;
  - saved `outputs/minimax_h3_diffusionnft_phase4_2step/phase4_final_20260826/checkpoint-2`; final `optimizer_step=true` and `global_step=2`.
- **Checkpoint verification:** checkpoint 1/2 record global steps 1/2 and optimizer state steps 1/2. Each current and old file contains the same 208 LoRA keys; current is bf16 and old EMA is fp32. Loading checkpoint-2 `current_lora.safetensors` through the exact existing rollout pipeline path patched 104 H3 DiT modules (`load_success=true`).
- **Problems encountered:**
  - rebuilding the scheduler without its training flag made the existing H3 video input unit skip clean-latent encoding; enabling `training=True` through `set_timesteps` retained the exact inference schedule and fixed the input path without core changes;
  - keeping old LoRA in bf16 rounded decay `0.001` to current exactly; moving only frozen old LoRA and EMA arithmetic to fp32 preserved policy lag;
  - multiplying bf16 current directly by Python scalar `0.999` also rounded before promotion; explicitly converting current to fp32 before both EMA products restored the exact convex update.
- **Core-code changes:** none. MiniMax-H3 pipeline, DiT, VAE, scheduler implementation, SFT trainer, and common training framework remain unchanged.
- **Current status:** phase 4 complete. Two real multi-timestep optimizer updates passed across a process restart with persistent current/old/reference state relationships, nonzero second-step KL, and rollout-compatible LoRA export.
- **Next step:** build a bounded online generate/reward/train orchestration layer around the checkpointable offline update, then add production checkpoint retention and failure recovery. Multi-GPU and audio reward/loss remain deferred.

### 2026-08-26 — phase 3 one-step LoRA DiffusionNFT update

- **Modification goal:** consume an existing same-prompt rollout group and complete one real `reward -> group advantage -> positive/implicit-negative DiffusionNFT objective -> reference regularization -> backward -> optimizer.step()` update on MiniMax-H3 LoRA.
- **Official reference:** `/gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffusionNFT/scripts/train_nft_sd3.py`. The reward normalization, clipping/mixing ratio, positive/implicit-negative construction, adaptive reconstruction weighting, policy-loss scaling, reference MSE, gradient clipping, and old-policy decay follow that implementation. H3 scheduler algebra replaces the SD3-specific normalized-time expression.
- **Modified file:** `examples/minimax_h3/model_training/diffusionnft/train.py`.
  - retained Phase 2 as the default `--mode backward-smoke` path;
  - added `--mode nft-step`, same-prompt group selection, population-standard-deviation advantage normalization, serial batch-size-1 accumulation, current/old/reference policy switching, official NFT losses, AdamW, gradient clipping, one optimizer step, and configurable old-policy decay;
  - current policy is the trainable `default` LoRA; `old` is a second frozen PEFT adapter copied from current before the step; reference is the same frozen MiniMax-H3 DiT with all adapters disabled;
  - old and reference forwards run under `no_grad`; only the current adapter is passed to the optimizer;
  - added validations for finite rewards, at least two same-prompt videos, compatible video shapes, all current LoRA tensors receiving gradients, frozen-policy gradient isolation, finite nonzero gradient norm, and nonzero parameter delta.
- **New file:** `examples/minimax_h3/model_training/diffusionnft/smoke_nft_step.sh`, fixed to the existing two-seed smoke rollout, rank-4 `qkv_proj,out_proj` LoRA, one H100, and one optimizer step.
- **H3 FlowMatch formula mapping:** for the scheduler's actual sigma at the selected training timestep,
  - `x_t = scheduler.add_noise(x0, noise, t) = (1-sigma) * x0 + sigma * noise`;
  - `v_target = scheduler.training_target(x0, noise, t) = noise - x0`;
  - therefore `x_t = x0 + sigma * v_target` and the reconstruction used by NFT is `x0_prediction = x_t - sigma * v_prediction`;
  - sigma is looked up from `scheduler.sigmas` using the selected `scheduler.timesteps` entry. No SD3 `t/1000` assumption is used.
- **DiffusionNFT objective mapping:**
  - `advantage = (reward - group_mean) / (population_std + eps)`;
  - `adv_clip = clamp(advantage, -adv_clip_max, adv_clip_max)` and `r = clamp((adv_clip / adv_clip_max) / 2 + 0.5, 0, 1)`;
  - `positive_pred = policy_beta * current_pred + (1-policy_beta) * old_pred`;
  - `implicit_negative_pred = (1+policy_beta) * old_pred - policy_beta * current_pred`;
  - each reconstructed-x0 squared error is divided by its detached mean absolute reconstruction error, matching the official adaptive weighting;
  - `policy_loss = adv_clip_max * mean(r * positive_loss / policy_beta + (1-r) * negative_loss / policy_beta)`;
  - `kl_loss = MSE(current_pred, reference_pred)` and `total_loss = policy_loss + kl_beta * kl_loss`.
- **Old-policy update:** after the optimizer step, `old = decay * old + (1-decay) * current`. `--old-decay-type {0,1,2}` and `--old-decay-step` reproduce the official `return_decay` schedules; smoke defaults to type 1 at step 1, giving decay `0.001`.
- **Real smoke command:** `bash examples/minimax_h3/model_training/diffusionnft/smoke_nft_step.sh`.
- **Real smoke result:**
  - rewards: `[22.35523033, 22.73575974]`;
  - advantages: `[-0.99946970, 0.99947971]`; `r`: `[0.40005302, 0.59994799]`;
  - sample sigmas: `0.39191842` at timestep id 775 and `0.96082568` at timestep id 83;
  - group positive loss `1.10106587`, negative loss `1.10106587`, policy loss `5.50532961`, KL loss `0.00000000`, total loss `5.50532961`;
  - all `208` current LoRA tensors received gradients; pre-clip gradient norm `0.15455209`; clip return norm `0.15429688` with max norm `1.0`;
  - current LoRA parameter delta after AdamW step: `0.235854022010`;
  - `old_policy_no_grad=true`, `reference_policy_no_grad=true`, `optimizer_step=true`.
- **Interpretation/current limitation:** on the first step, old is an exact copy of current and PEFT LoRA B matrices start at zero, so positive and implicit-negative predictions have equal forward values and the reference KL is zero. Their derivatives with respect to current differ, and reward-derived `r` still controls the policy gradient. Later steps will separate current, old, and reference predictions. This phase intentionally executes exactly one offline group step and does not add online rollout-training iteration, multi-GPU, checkpoint/resume, audio loss, or repeated optimizer steps.
- **Problem encountered:** the first shell launch stopped before model loading because `set -u` conflicts with an unset variable in Conda's CUDA activation hook. The smoke script now directly invokes the repository's verified `py312/bin/python`, matching Phase 2. No H3 core or common framework file was changed.
- **Current status:** phase 3 complete; a real two-seed MiniMax-H3 LoRA DiffusionNFT optimizer step passed every requested gradient, isolation, and parameter-update check.
- **Next step:** build a bounded online rollout/reward/train loop around this verified single-group step, then add LoRA checkpoint/resume and repeated-step old-policy scheduling without changing the H3 model or scheduler internals.

### 2026-08-26 — phase 2 minimal LoRA training forward/backward

- **Modification goal:** verify one generated MiniMax-H3 video can traverse VAE encode -> forward diffusion -> H3 DiT prediction -> diffusion MSE -> `backward()` without reward weighting or a model update.
- **New files:**
  - `examples/minimax_h3/model_training/diffusionnft/train.py`
  - `examples/minimax_h3/model_training/diffusionnft/smoke_train.sh`
- **Modified files:** `docs/MiniMax_H3_DiffusionNFT_Modification.md` only.
- **Core-code changes:** none. Pipeline, DiT, video/audio VAE, scheduler, generic loss/runner, SFT trainer, rollout, and reward adapter remain unchanged.
- **`train.py` structure:**
  1. Read one record from a list-style `rollout.json` (also accepts an object with a `rollouts` list).
  2. Decode the selected mp4 with OpenCV into RGB PIL frames and validate one video, 24fps, spatial divisibility by 32, and `17n+5` frame count.
  3. Dynamically load the existing `MiniMaxH3TrainingModule` from `examples/minimax_h3/model_training/train.py` rather than duplicating its model/LoRA setup.
  4. Instantiate the existing module on CPU with FL2VA text encoder, transformer, video VAE, audio VAE, processor, gradient checkpointing, and PEFT LoRA on DiT `qkv_proj,out_proj`.
  5. Use the existing `OffloadTrainingManager` in its default leaf-module mode to onload frozen weights around forward/recomputation/backward while LoRA parameters remain on one GPU.
  6. Run the existing MiniMax-H3 pipeline units, including `MiniMaxH3Unit_InputVideoEmbedder`/`video_vae.encode_video`, prompt embedding, noise initialization, and packed-sequence construction.
  7. Reset both existing FlowMatch schedulers to 1000-step training mode, sample one timestep and video noise, then call `scheduler.add_noise()` and `scheduler.training_target()`.
  8. Call the existing `model_fn_minimax_h3` through `model.pipe.model_fn`, compute video MSE times the existing scheduler training weight, call `loss.backward()`, and report LoRA gradient norm.
  9. Explicitly stop without constructing an optimizer or calling `optimizer.step()`.
- **Existing components reused:** `MiniMaxH3TrainingModule`, `model_fn_minimax_h3`, MiniMax-H3 pipeline units, frozen video VAE encode, video/audio `FlowMatchScheduler`, PEFT LoRA injection from `DiffusionTrainingModule`, gradient checkpointing/offload, and `OffloadTrainingManager`.
- **Backward smoke command:** `examples/minimax_h3/model_training/diffusionnft/smoke_train.sh` using rollout index 0, single H100 GPU, batch size 1, rank-4 LoRA, 22 frames at 448x256.
- **Backward test result:**
  - latent shape/dtype: `[1, 24, 7, 16, 28]`, `torch.bfloat16`;
  - sampled timestep id/value: `775` / `391.918427`;
  - raw video MSE: `0.24644278`;
  - scheduler training weight: `1.59255123`;
  - final diffusion loss: `0.39247274`;
  - trainable LoRA tensors/parameters: `208` / `8,200,192`;
  - tensors receiving gradients: `208`;
  - global LoRA gradient norm: `0.00840276`;
  - final status: `backward_success=true`, `optimizer_step=false`.
- **Problems encountered:**
  - an initial smoke used `cpu_offload_split_threshold=1024MB`; the current `OffloadTrainingManager` grouped a parent VAE module, while `UnitWiseParamManager.onload_module()` only onloaded direct (`recurse=False`) parameters, leaving a child convolution weight as an empty placeholder and producing `RuntimeError: weight should have at least three dimensions`;
  - the new entrypoint was changed to the manager's default leaf-module offload (`split_threshold=None`), which completed VAE encode, DiT forward, and backward without modifying the offload framework;
  - the environment prints a non-fatal torchao warning because torch 2.6.0+cu124 and torchao 0.16.0 extensions are incompatible; the smoke completed successfully without those extensions.
- **Current status:** phase 2 complete. A real MiniMax-H3 LoRA diffusion loss produced finite nonzero gradients on every trainable LoRA tensor.
- **Current limitations:** exactly one rollout video, batch size 1, one process/one CUDA GPU, LoRA only, one sampled timestep, no audio loss, no reward/advantage weighting, no online rollout, no optimizer/scheduler step, and no checkpoint save.
- **Next step:** phase 3 may wrap this verified forward in group-wise advantage weighting and an explicit optimizer/update loop, while keeping rollout/reward generation separate until the offline weighted-loss path is validated.

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
