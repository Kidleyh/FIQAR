# MiniMax-H3 DiffusionNFT Modification Log

## Repository scope

- Repository alias: **FIQAR**
- Repository path: `/gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3`
- Reward implementation (read-only external dependency): `/gemini/platform/public/aigc/human_guozz2/code/lyh/job/MagFace/inference/eval_video_face_quality.py`
- Objective: extend the working MiniMax-H3 FL2AV multi-seed inference path into DiffusionNFT-style online reinforcement post-training: online rollout -> face-quality reward -> group advantage -> forward diffusion -> advantage-weighted diffusion loss -> model update.
- Explicitly out of scope: DPO, winner/loser pair construction, reward-model reimplementation, and broad DiffSynth-Studio refactoring.

## Change history

### 2026-09-03 — phase 10 two-GPU DeepSpeed ZeRO-3 single-seed rollout (complete)

- **Modification goal and scope:** add a separate two-rank rollout path in which both H100s jointly execute every denoising DiT forward for one 1088x736, 175-frame, three-step I2V sample. Existing `rollout.py`, single/ZeRO training entrypoints, online orchestration, reward, NFT loss, scheduler, DiT, and VAE sources are unchanged. The first version is deliberately bounded to seed 0, one sample, one machine, two GPUs, CFG 1, and no resume/online loop.
- **New files:** `examples/minimax_h3/model_training/diffusionnft/rollout_zero3.py` and `examples/minimax_h3/model_training/diffusionnft/rollout_zero3.sh`. The launcher defaults to `CUDA_VISIBLE_DEVICES=0,1`, two processes, 1088x736, 175 frames, three inference steps, seed 0, base policy, and no mp4; it also accepts `POLICY_ROLE`, `LORA_PATH`, `LORA_RANK`, `SOURCE_CHECKPOINT`, and `GLOBAL_STEP_BEFORE` for a later old-policy integration. The existing Phase-9 Accelerate and DeepSpeed configuration files are reused unchanged.
- **Implementation:** the launcher has three isolated stages. GPU0 first runs the verified frozen Qwen/video-VAE I2V condition path and atomically caches the initial video/audio noise, exact resized condition anchor, prompt embeddings, and packed positions. Two Accelerate ranks then load only the H3 DiT on CPU, enter the Phase-9 ZeRO-3 partition path, move only each rank's shard to CUDA, and synchronously call the same DeepSpeed engine at all three scheduler timesteps. After denoising, rank0 atomically records temporary clean latents and both ranks' gathered memory metrics. A final GPU0-only process loads the inference video VAE, decodes PIL frames, initializes the existing in-memory SCRFD/MagFace reward, and commits the unchanged Phase-8.1 latent/state/rollout contract. Rank1 never decodes, scores, or writes rollout artifacts.
- **True two-rank participation:** the real run logged `world_size=2`, `zero_stage=3`, ranks/local-ranks `0/0` and `1/1`, identical condition-contract hashes, and identical input shapes on both ranks. Both rank logs contain `dit_forward=true` and `dit_forward_complete=true` for timestep indices 0, 1, and 2. ZeRO moved 536 parameter partitions and about `31,588.55 MiB` of shard storage per rank; rank consistency checks passed after every scheduler step.
- **Required real result:** `outputs/minimax_h3_diffusionnft_zero3/phase10_real_1088x736x175_seed0_20260903`. Base-policy seed 0 completed full denoising without OOM. Clean video latent shape is `[1,24,52,46,68]`; clean audio latent shape is `[2,32,292]`. Rank-0 and rank-1 peak allocated memory were both `39,301.95 MiB`, with `43,184.00 MiB` peak reserved. Rank0 video-VAE decode peak allocation was `10,285.39 MiB`.
- **Reward and artifacts:** GPU0 decoded exactly 175 PIL frames and in-memory MagFace returned reward/mean quality `21.865512916019984`, face-visible ratio `1.0`, and 27 detected faces. `condition_image.png`, `seed_0_latents.safetensors`, `seed_0_state.json`, and `rollout.json` are present and non-empty; the final latent is 7,845,040 bytes, all latent/condition hashes and Phase-8.1 reward provenance fields validate, and no mp4 exists. The existing `train.py` rollout loader and NFT artifact validator accepted the one-record `rollout.json` unchanged. Actual two-GPU NFT training still requires a group size divisible by world size, as established in Phase 9.
- **Finalize integration correction:** the first post-denoise finalize attempt used the SFT training wrapper to load the VAE and failed before frame decode because its training-time `WarpedTensor` representation did not preserve the inference decoder's callable register-token shape. Clean latents were unaffected. The new file was corrected to use the already verified inference `MiniMaxH3Pipeline.from_pretrained` VRAM-managed VAE loader; rerunning only rank0 finalize then decoded, rewarded, and committed successfully. This was not an OOM and required no core/source change.
- **Status and limits:** Phase 10 is complete for the requested base-policy real sample. Parameter sharding and the target activation size both fit on two 80GB H100s, so sequence/context parallelism was not introduced. Multi-seed/group rollout, resume, online orchestration, CPU offload, sequence/context parallelism, multi-node execution, and multi-GPU reward remain out of scope.

### 2026-08-31–2026-09-03 — phase 9 Accelerate/DeepSpeed ZeRO-3 NFT step (complete)

- **Modification goal:** add a separate Accelerate + DeepSpeed ZeRO-3 implementation of the already verified `nft-step`, without changing `train.py`, `online_train.py`, `train_online.sh`, `rollout.py`, `reward_face_quality.py`, or any H3/policy/loss/scheduler code. Rollout and in-memory reward remain the Phase-8.1 single-GPU path.
- **New files:** `examples/minimax_h3/model_training/diffusionnft/train_zero3.py`, `examples/minimax_h3/model_training/diffusionnft/train_zero3.sh`, `examples/minimax_h3/model_training/diffusionnft/prepare_zero3_conditions.py`, `examples/minimax_h3/model_training/diffusionnft/compare_zero3_checkpoints.py`, `examples/minimax_h3/model_training/diffusionnft/deepspeed/accelerate_zero3.yaml`, and `examples/minimax_h3/model_training/diffusionnft/deepspeed/ds_zero3.json`. The condition-preparation utility atomically caches the exact frozen text/keyframe tensors through the verified single-GPU DiffSynth offload path; the comparison utility reports file hashes, max absolute/element-relative and relative-L2 state differences, plus optimizer-update L2 and cosine similarity from a shared initialization.
- **DeepSpeed configuration:** single-machine launch defaults to two processes; ZeRO stage 3 and BF16 are enabled, CPU parameter/optimizer offload is absent, gradients are contiguous, communication overlap is conservatively disabled, 16-bit weights are gathered on save, gradient clipping is supplied from the existing NFT default (`1.0`), and gradient accumulation/batch sizes are resolved by Accelerate. DiffSynth's DeepSpeed activation-checkpointing initializer is called after `accelerator.prepare()`.
- **Integration and policy semantics:** the new entrypoint imports the existing rollout/scheduler/noise/advantage/adapter/loss/decay/checkpoint helpers without changing them. One `MiniMaxH3TrainingModule` backbone is wrapped and prepared by Accelerate; every old/current/reference joint-DiT forward enters through the DeepSpeed engine. Adapter roles remain `old` frozen, `default` current/trainable, and all adapters disabled for the frozen base reference. A process-local checkpoint-call adapter preserves H3 block keyword-only arguments when using DeepSpeed activation checkpointing; no core source file is changed.
- **Global group and sample sharding:** every rank reads the complete prompt group and computes the same rewards, population-standardized advantages, clipping, and `r`; the serialized reward/advantage/r SHA-256 contract is gathered and checked across ranks. Samples are assigned by `group_index % world_size`, and `K % world_size == 0` is enforced before model load. Timestep order and video/audio noise use the exact existing deterministic seeds derived from global sample index, timestep id, and global step.
- **Loss normalization:** each rank processes its equal number of local samples times all selected inference timesteps. The DeepSpeed gradient-accumulation length is exactly that local count, each micro-loss is passed once to `accelerator.backward()`, and DeepSpeed's data-parallel averaging supplies the world-size mean; there is no second division by global `K * N_t`.
- **EMA and checkpoint/resume:** all ranks enter `deepspeed.zero.GatheredParameters`; rank 0 applies the unchanged FP32 old-policy EMA and the modifier-rank broadcast repartitions it. Each checkpoint atomically commits complete gathered `current_lora.safetensors` and `old_lora.safetensors`, `training_state.json`, plus ZeRO-3 model/optimizer state under `deepspeed/`. Resume first restores complete current/old adapters and then restores the DeepSpeed optimizer/client state and global step; old is not recopied from current.
- **Real one-process ZeRO-3 functional result:** on the existing 448x256, 22-frame, three-inference-step K=2 rollout, ZeRO stage 3 was active, four deterministic sample/timestep micro-passes produced one engine optimizer step, `global_step 0 -> 1`, gradient norm was `0.15224105`, current parameter delta was `0.235893576091`, old/reference had no gradients, and current/old distance after decay was `0.000235886838`. Per-step reported peak allocation/reservation was 2647.56/4688.00 MiB. A restart from this checkpoint restored optimizer/current/old/global step and completed `1 -> 2`; step-2 KL was `0.00116796`, current/reference distance `0.03364596`, gradient norm `0.12568062`, and current/old distance `0.000389702707`.
- **LoRA export compatibility:** both gathered files contain 208 adapter tensors. The Phase-9 exported `old_lora.safetensors` was loaded by the unchanged rollout pipeline and printed `104 tensors are patched by LoRA` and `[lora] patched_modules=104`.
- **Current mathematical comparison result (not accepted yet):** a common rank-4 initialization was supplied to unchanged single-GPU `train.py` and the one-process ZeRO-3 engine with identical rollout, timestep/noise seeds, LR, and NFT settings. Single/ZeRO policy losses were `2.92733955` / `2.91339302`, gradient norms were `0.11074358` / `0.16302166`, and parameter deltas were `0.235718971608` / `0.235889496473`. The new comparison utility reported current-LoRA max absolute difference `0.0002012253`, full-state relative L2 difference `0.0231733`, update-vector relative L2 difference `1.15623`, and update cosine `0.329389`; output is stored at `outputs/minimax_h3_diffusionnft_zero3/phase9_equivalence_comparison.json`. Disabling DeepSpeed activation checkpointing or `zero3_init_flag` did not change the ZeRO result. This exceeds the requested structural-equivalence bar and remains under investigation; it is not recorded as an accepted BF16 tolerance.
- **Real K=4 unchanged-single baseline for the two-GPU comparison:** `outputs/minimax_h3_diffusionnft_zero3/phase9_user_2gpu_20260831_155552/single/checkpoint-1` completed all eight sample/timestep backward passes with the shared initialization. Rewards were `[21.31691360, 22.07617188, 22.43754959, 22.63247490]`, advantages `[-1.58927429, -0.07879303, 0.64013791, 1.02792561]`, group policy/total loss `3.11402273`, gradient norm `0.07303665`, parameter delta `0.235386187809`, and current/old distance `0.000235380880`. It committed current/old LoRA, optimizer state, and training state with `global_step 0 -> 1`; no OOM, NaN, Inf, or traceback occurred. This is the fixed baseline to compare against the pending true two-rank run.
- **First true two-rank integration issue and fix (2026-09-01):** both ranks enabled ZeRO stage 3, agreed on the global reward contract, and assigned samples correctly (`rank0=[0,2]`, `rank1=[1,3]`), then failed before forward in the diagnostic current/old distance gather with `TypeError: output tensor must have the same type as input tensor`. The manual `GatheredParameters` call had coalesced BF16 current LoRA and intentionally FP32 old LoRA into one collective; this is invisible at world size 1. Distance, EMA, and export now use nested dtype-homogeneous gathers (current BF16 first, old FP32 second). The old-policy dtype, EMA formula, policy roles, loss, and single-GPU files are unchanged. A fresh two-rank retry is pending.
- **Second two-rank integration issue and fix (2026-09-01):** the dtype-homogeneous gather completed and reported zero initial current/old distance, then both ranks failed in condition preparation because CUDA token IDs reached a Qwen embedding whose frozen weight remained on CPU. The new entrypoint had constructed `MiniMaxH3TrainingModule` with `device="cpu"`; this differs from DiffSynth's native pure-GPU DeepSpeed runner, which constructs/moves the model on `accelerator.device` before `prepare`. Phase 9 now passes each rank's `accelerator.device` into H3 construction so ZeRO.Init partitions frozen condition/backbone parameters from the correct CUDA device. No CPU offload was added and no single-GPU/core file changed. A third two-rank retry is pending.
- **Third two-rank integration issue and revised device fix (2026-09-01):** direct CUDA construction avoided the CPU/CUDA condition mismatch but H3's custom safetensors loader materialized almost the complete model state on every rank before DeepSpeed partitioning; both 80GB H100s reached about 79.18 GiB and failed on the next 498 MiB allocation. Phase 9 therefore keeps checkpoint loading on CPU, lets `accelerator.prepare` create ZeRO-3 partitions, then explicitly moves only each rank's `ds_tensor` partition storage (plus small persistent/ordinary parameters) to its CUDA device. This preserves a pure-GPU runtime with no parameter/optimizer offload while avoiding the full-model CUDA initialization peak. The next run logs partition tensor counts and moved shard MiB for verification. A fourth two-rank retry is pending.
- **Fifth two-rank integration issue and fix (2026-09-01):** after the in-place partition-storage move preserved DeepSpeed's `ds_tensor.status` metadata, both ranks completed condition preparation and reached all four local deterministic sample/timestep backward passes. The optimizer post-step then failed when ZeRO-3 coalesced its persistent-parameter list containing BF16 trainable current-LoRA tensors and FP32 frozen old-LoRA tensors into one all-gather (`output tensor must have the same type as input tensor`). The ZeRO-3 config now sets `stage3_param_persistence_threshold=0`, so heterogeneous adapters remain normally partitioned and are gathered by their dtype-safe adapter contexts instead of entering the mixed-dtype persistent post-step gather. This does not change LoRA dtypes, optimizer membership, NFT loss, gradient normalization, or old-policy EMA semantics; it only disables the optional small-parameter persistence optimization. A sixth two-rank retry is pending.
- **First successful true two-rank ZeRO-3 step (2026-09-01):** `zero3_attempt6` used the shared K=4 initialization and completed with actual `world_size=2`, ZeRO stage 3, `rank0=[0,2]`, `rank1=[1,3]`, two selected timesteps per sample, and four local backward passes per rank. It produced group policy/total loss `3.10672188`, ZeRO global gradient norm `0.14725783`, current delta `0.235663858219`, old decay `0.001`, current/old distance `0.000235657404`, `optimizer_step=true`, and `global_step 0 -> 1`; old/reference remained gradient-free. Peak allocated/reserved VRAM was `61788.56/64784.00 MiB` on rank 0 and `61788.07/64842.00 MiB` on rank 1. The atomic checkpoint contains complete current/old LoRA files, `training_state.json`, and model/optimizer shards for both ranks under `deepspeed/global_step1/`.
- **True two-rank mathematical comparison (not accepted yet):** against the unchanged K=4 single baseline, group policy losses were `3.11402273` / `3.10672188` and current update L2 norms were `0.23563457` / `0.23593366`. The gathered current LoRA has max absolute difference `0.0002012253` and full-state relative L2 difference `0.0240601`; however, update-vector relative L2 difference is `1.20243` and cosine is only `0.275748` (old-LoRA cosine `0.277946`). The per-sample divergence is already present before optimizer step despite identical rewards, advantage/r, latent shapes, timestep IDs, scheduler sigmas, and deterministic noise seeds. It is therefore not attributed to sample sharding or group-loss normalization and is not accepted as ordinary BF16 tolerance; the ZeRO backbone/condition forward and optimizer-precision paths remain under investigation. Report: `outputs/minimax_h3_diffusionnft_zero3/phase9_user_2gpu_20260831_155552/equivalence_attempt6.json`.
- **Real two-rank checkpoint resume (2026-09-01):** restarting Accelerate/DeepSpeed from attempt6 checkpoint-1 restored both ranks with `resume_success=true`, the complete current/old adapters, optimizer shards, and `global_step=1`; the pre-step current/old distance exactly matched the prior committed value `0.000235657404`, proving old was not overwritten from current. The resumed update completed `1 -> 2` with group KL `0.00125311`, current/reference prediction distance `0.03495320`, gradient norm `0.09423846`, current delta `0.196538887586`, step-aware old decay `0.002`, and post-EMA current/old distance `0.000393341334`. Checkpoint-2 contains complete current/old LoRA plus rank-0 and rank-1 model/optimizer states under `deepspeed/global_step2/`; no OOM, NaN, Inf, or traceback occurred. The saved engine config confirms gradient accumulation `4`, global train batch `8`, DP world size `2`, no FP16/BF16 autocast mode, and no parameter/optimizer offload; H3 and current LoRA tensors themselves retain their explicitly loaded BF16 dtype.
- **DeepSpeed BF16 configuration correction (2026-09-01):** inspection of the attempt6/7 serialized engine state showed that `bf16.enabled="auto"` had resolved to `false` because the entrypoint explicitly constructed Accelerate with `mixed_precision="no"`; the parameters were BF16, but DeepSpeed's BF16 execution mode itself was not active. This does not satisfy the Phase-9 BF16 requirement. The independent ZeRO-3 entrypoint now selects `Accelerator(mixed_precision="bf16")`, and the external DeepSpeed JSON explicitly enables BF16. A fresh shared-initialization K=4 comparison is required; attempt6 remains evidence for two-rank sharding/checkpoint mechanics, not the final BF16 numerical acceptance result.
- **Accelerate external-config compatibility correction (2026-09-01):** the first BF16 retry exited before model loading because this installed Accelerate version rejects `mixed_precision` inside an Accelerate YAML that also points to a complete external `deepspeed_config_file`. Following its required ownership model, the redundant YAML field was removed, `ds_zero3.json` now directly sets `bf16.enabled=true`, and the Python entrypoint retains `Accelerator(mixed_precision="bf16")`. Thus DeepSpeed owns its BF16 engine setting while Accelerate uses the matching runtime precision, without an ignored/ambiguous `auto` field.
- **BF16 old-policy preservation fix (2026-09-01; pending retry):** the corrected BF16 engine completed the K=4 step, and its serialized config confirmed `bf16.enabled=true`, gradient accumulation 4, and global batch 8. Forward losses remained identical to attempt6 and peak allocated VRAM fell slightly to `61676.26 MiB` per rank. However, DeepSpeed had cast the frozen old adapter to BF16 together with the managed model; the official step-1 EMA (`decay=0.001`) then rounded old exactly to current (`current_old_parameter_distance_after=0`, identical current/old SHA-256). This violates the required policy lag and the attempt is rejected. The revised independent entrypoint now presents only base/current LoRA to `accelerator.prepare`; after ZeRO-3/BF16 initialization and any engine resume, it injects one replicated frozen FP32 old adapter on each GPU. Old/current/reference forwards still enter the same DeepSpeed engine and one H3 backbone; only trainable current is ZeRO-managed, as required. Every rank applies the same FP32 EMA after a collective current gather, while checkpoint export remains full current/old safetensors. The next retry must prove old is FP32, not ZeRO-managed, nonzero after-EMA distance, and restartable.
- **Late-module ZeRO hook compatibility fix (2026-09-01; pending retry):** the first late-old attempt confirmed on both ranks that current remained BF16/ZeRO-managed while old was GPU FP32, frozen, replicated, and not ZeRO-managed. It then failed at the first engine call because PEFT-created post-prepare LoRA submodules had ordinary Python `_parameters` dictionaries, whereas the ZeRO-3 engine prologue sets `_in_forward` on every module's DeepSpeed `ZeROOrderedDict`. The entrypoint now wraps only parameter mappings created after prepare in DeepSpeed's `ZeROOrderedDict`; this supplies the required forward-hook metadata but does not assign `ds_id`, partition, cast, or register old with the optimizer. Core files and pre-existing ZeRO-managed module containers remain untouched.
- **ZeRO optimizer-boundary old exclusion (2026-09-01; pending retry):** after installing compatible parameter containers, both ranks completed condition preparation and all old/current/reference policy forwards and reached the final accumulated backward. ZeRO's optimizer-boundary `partition_all_parameters()` then recursively scanned the late ordinary old parameters and assumed they had `ds_active_sub_modules`, which only partitioned parameters own. The entrypoint now temporarily removes exactly the 208 known frozen old parameters from their module registration mappings only during the final `accelerator.backward()`/implicit engine step and DeepSpeed shard save, restoring the exact objects immediately afterward. Old is inactive during the current-policy backward, and DeepSpeed checkpoint state intentionally excludes frozen old; therefore this does not change any forward, gradient, optimizer membership, EMA, or full old-LoRA export. A strict count guard prevents hiding any unexpected parameter.
- **Successful BF16 ZeRO-3 + FP32 old K=4 step (2026-09-01):** attempt11 completed the full two-rank optimizer step with true DeepSpeed BF16, rank assignment `[0,2]`/`[1,3]`, four local backward passes per rank, and 312 late-module ZeRO parameter containers. Current was BF16 and ZeRO-managed; old was replicated GPU FP32, frozen, and not ZeRO-managed. Group policy loss was `3.10672188`, ZeRO gradient norm `0.14728298`, current delta `0.235628537870`, and the official `0.001` EMA retained a nonzero current/old distance `0.000235622078`. The committed checkpoint has a 16,422,896-byte current LoRA, 32,823,216-byte FP32 old LoRA, true BF16 optimizer shards for both ranks, and `global_step 0 -> 1`; per-rank peak allocated VRAM was about `61.70 GiB` (reserved `63.68-64.72 GiB`). No OOM, NaN, Inf, or traceback occurred.
- **Gradient-level mathematical comparison remains rejected:** the attempt11 state/update comparison remains essentially unchanged (current max absolute difference `0.0002012253`, relative state L2 `0.0240575`, update cosine `0.275805`). To avoid Adam first-step sign amplification as an explanation, the two rank BF16 optimizer shards were reconstructed parameter-by-parameter and their first-step `exp_avg` compared directly with the unchanged single optimizer. The reconstruction was independently verified by reproducing the exported ZeRO LoRA with zero error from the shard AdamW state; the analogous single reconstruction matched within max `9.54e-7`. The raw exp-avg norms were `0.00730376` single and `0.01472830` ZeRO (ratio `2.01654`), but cosine was only `0.103563` and same-sign fraction `0.638206`. Thus a real gradient-direction discrepancy remains; it is not checkpoint ordering or Adam export noise. Report: `outputs/minimax_h3_diffusionnft_zero3/phase9_user_2gpu_20260831_155552/gradient_equivalence_attempt11.json`.
- **Successful BF16/FP32-old two-rank resume (2026-09-01):** attempt12 restarted from attempt11 checkpoint-1 and both ranks reported `resume_success=true`, restored `global_step=1`, and recovered the exact committed pre-step current/old distance `0.000235622078`; old was not recopied from current and remained replicated GPU FP32/non-ZeRO. The resumed update completed `1 -> 2` with KL `0.00146809`, current/reference prediction distance `0.03728507`, gradient norm `0.08579068`, current delta `0.196067645515`, step-aware old decay `0.002`, and post-EMA current/old distance `0.000392397530`. Checkpoint-2 contains both BF16 ZeRO optimizer shards, both model-state shards, full BF16 current LoRA, full FP32 old LoRA, and training state. No OOM, NaN, Inf, duplicate step, or traceback occurred.
- **Repeated unchanged single-GPU baseline (2026-09-01):** rerunning the exact checkpoint-0/K=4 command on GPU 0 reproduced every per-sample/timestep loss exactly and produced policy loss `3.11402273`, gradient norm `0.07347074`, and current delta `0.235388886053`. Against the original single run, current-state relative L2 was `0.00267865`, update cosine was `0.988926`, Adam `exp_avg` cosine was `0.99975296`, and its same-sign fraction was `0.99491`. This rejects ordinary single-path/BF16 run-to-run variation as the cause of the ZeRO gradient-direction discrepancy.
- **Forward-boundary diagnostic (2026-09-01):** temporary, non-production diagnostic wrappers captured clean/noisy video and audio latents, all tensor-valued H3 conditioning inputs, timesteps, and old/reference/current predictions. They localized the discrepancy described below and were removed from the final source tree; the production entrypoint contains no diagnostic override path.
- **Forward discrepancy localized and formal fix prepared (2026-09-03):** the first diagnostic showed exact equality for clean/noisy video and audio latents, timesteps, packed positions, and the FL2AV keyframe anchor. The only differing model input was `prompt_embeds` (single versus ZeRO relative L2 `0.01035236`); the three policy predictions then differed by relative L2 `0.05548777`. Both ZeRO ranks produced bit-identical prompt embeddings, excluding rank divergence. Replacing only the ZeRO prompt embedding with the captured single tensor made old/reference/current predictions bit-identical to single (`max_abs=0`, `relative_l2=0`). The remaining discrepancy therefore originates solely in the frozen Qwen condition forward under partitioning, while the ZeRO-managed DiT and adapter switching are exact for identical inputs.
- **Exact frozen-condition cache (2026-09-03):** added `prepare_zero3_conditions.py`. It evaluates each rollout record's frozen text/keyframe preprocessing once through the verified single-GPU DiffSynth offload path, atomically caches only the exact `model_fn` condition tensors, and commits a SHA256-validated manifest bound to the rollout JSON, prompt, seed, condition image, geometry, and model ID. `train_zero3.py` now requires this strict cache and registers only the trainable H3 DiT with the pure-GPU ZeRO-3 engine; frozen text encoder/video VAE are not part of the training engine and are not resident during NFT updates. `train_zero3.sh` creates or reuses the cache in a separate process before launching Accelerate. The formal two-GPU equivalence result is recorded below.
- **K=4 formal mathematical-equivalence result (2026-09-03):** the condition-cache run completed all eight sample×timestep forwards/backwards and one optimizer step under real two-rank ZeRO-3. Every individual positive/negative/policy loss printed exactly the same value as the unchanged single baseline; distributed group policy loss was `3.11402297` versus single `3.11402273`. The ZeRO gradient norm was `0.07323800`, within the two unchanged single runs (`0.07303665` and `0.07347074`), confirming correct global-mean normalization and eliminating the previous `~2x` artifact. Current update L2 was `0.23546632` versus single `0.23563457`, cosine `0.99085033`; comparison against the repeated single run gave cosine `0.99126422`. This is at least as close as the two unchanged single BF16 runs to each other (cosine `0.988926`), so no structural distributed discrepancy remains. The exported old update cosine was `0.99791497` and the official step-1 EMA retained distance `0.000235318386`.
- **ZeRO-3 sharding/VRAM improvement:** after caching frozen conditions and sharding only the training DiT, each rank held 743 ZeRO parameter tensors and about `31.60 GiB` of partition storage. Peak allocated VRAM was `34.346 GiB` on rank 0 and `34.345 GiB` on rank 1 (reserved `35.868/35.802 GiB`), down from about `61.70 GiB` per rank when frozen condition models were unnecessarily included. Checkpoint-1 contains two BF16 optimizer shards, two model-state shards, full 16,422,896-byte current LoRA, full 32,823,216-byte FP32 old LoRA, and committed `global_step=1` training state.
- **Fourth two-rank integration issue and metadata-preserving fix (2026-09-01):** both ranks successfully moved 3499 ZeRO partition tensors (about 58,799.63 MiB per rank) to CUDA without OOM, proving that post-prepare local-shard migration avoids the full-model load peak. The first manual LoRA gather then failed because assigning `parameter.ds_tensor = partition.to(cuda)` replaced DeepSpeed's tensor object and discarded its dynamic `status` metadata. Migration now updates `partition.data` in place, preserving the original `ds_tensor.status`, `final_location`, and other ZeRO bookkeeping. A fifth two-rank retry is pending.
- **Reduced-engine checkpoint/resume result (2026-09-03):** restarting from the formal K=4 checkpoint restored both ranks with `resume_success=true`, `global_step=1`, and the exact committed current/old distance `0.000235318386`; old was not recopied from current. The second optimizer step completed `1 -> 2` with KL `0.00194539`, current/reference prediction distance `0.04094686`, gradient norm `0.09565747`, current delta `0.197833799171`, step-aware decay `0.002`, and post-EMA current/old distance `0.000395874058`. Checkpoint-2 contains complete current/old LoRA and both ranks' model/optimizer shards. Peak allocated memory remained stable at about `34.370 GiB` per rank.
- **Exported old-policy rollout compatibility (2026-09-03):** the unchanged Phase-8.1 rollout entrypoint loaded `checkpoint-2/old_lora.safetensors`, printed `104 tensors are patched by LoRA` and `[lora] patched_modules=104`, then generated a 448x256x22 mp4-free rollout with clean video/audio latent, exact condition, in-memory reward, complete seed state, and policy provenance bound to checkpoint-2/global-step-2. This verifies that the gathered ZeRO-3 old LoRA remains directly consumable by the existing single-GPU rollout path.
- **768x448x124 two-GPU result (2026-09-03):** a real K=2, three-inference-step rollout produced clean video latent shape `[1,24,37,28,48]` and audio latent shape `[2,32,207]`. The two-rank ZeRO-3 step processed both selected timesteps for each sample, produced policy loss `3.27790213`, gradient norm `0.06854518`, current delta `0.235170711168`, nonzero post-EMA current/old distance `0.000235165435`, and committed checkpoint-1 with `optimizer_step=true` and `global_step 0 -> 1`. Rank-0/rank-1 peak allocated memory was `45.981/45.978 GiB` (reserved `50.094/51.064 GiB`); no OOM, NaN, or Inf occurred.
- **Large-shape single/ZeRO comparison:** the unchanged single-GPU `train.py` also completed the identical 768x448x124 rollout and produced the same four per-sample/timestep losses and group policy loss. Its gradient norm/current delta were `0.06870616` / `0.235227028526`, versus ZeRO `0.06854518` / `0.235170711168`. Current-update cosine was `0.99143732`; current/old relative state L2 differences were `0.00236608` / `0.00236365`. Live `nvidia-smi` observation for the aggressively CPU-offloaded single path was about `44.113 GiB`; therefore this pure-GPU ZeRO-3 implementation did not lower per-GPU memory below that specialized single offload path at this geometry, although it reduced the earlier incorrect all-model ZeRO residency from about `61.70 GiB` to `45.98 GiB`. No claim is made about activation-memory scaling.
- **Unchanged single-GPU regression (2026-09-03):** the original `train_online.sh` ran without Accelerate/DeepSpeed for one K=2 iteration at 448x256x22 and three inference steps. Both rollouts and in-memory rewards completed, the NFT step reported gradient norm `0.10576884`, current delta `0.235663344934`, `optimizer_step=true`, checkpoint-1/final export were complete, and global step advanced `0 -> 1` with `iteration_success=true`. This confirms Phase 9 did not change or regress the protected single-GPU entrypoints.
- **Final status and limitations:** Phase 9 is complete on one machine with two H100 80GB GPUs. True ZeRO stage 3, global group advantage, modulo sample sharding, single-equivalent global-mean loss, deterministic timestep/noise, current/old/reference roles, FP32 old EMA, full-LoRA gather/export, optimizer-shard resume, old-LoRA rollout loading, the large-shape step, and the single-GPU regression are all verified. DeepSpeed training has no CPU parameter or optimizer offload; the separate frozen-condition preprocessing process deliberately uses the existing DiffSynth single-GPU offload path. Rollout/reward remain single-GPU, K must be divisible by world size, and multi-node, sequence/context parallelism, CPU offload, multi-GPU rollout, and online multi-process orchestration remain out of scope.

### 2026-08-31 — phase 8.1 reward provenance for seed resume

- **Modification goal:** make the mp4-free per-seed resume contract sensitive to the complete face-reward configuration and the exact SCRFD/MagFace weights. Reward computation, DiffusionNFT loss, policy/scheduler/checkpoint behavior, and the Phase-8 in-memory rollout path are unchanged.
- **Modified files:** `examples/minimax_h3/model_training/diffusionnft/rollout.py` and `examples/minimax_h3/model_training/diffusionnft/online_train.py`.
- **New test script:** `examples/minimax_h3/model_training/diffusionnft/smoke_reward_provenance_resume.sh`.
- **Committed seed contract:** every `seed_<N>_state.json` and corresponding `rollout.json` record now stores `reward_frame_stride`, `reward_max_frames`, `reward_frame_face_aggregation`, `missing_face_reward`, `scrfd_model_sha256`, and `magface_checkpoint_sha256`. `rollout.py` resolves and SHA-256 hashes both checkpoints once during process initialization, before the K-seed loop.
- **Strict seed reuse:** the six fields are included in the existing `_valid_complete_state()` expected contract. A missing field, reward-configuration difference, or checkpoint-content hash difference rejects the state and regenerates/rewards that seed; old Phase-8 states without reward provenance are intentionally not reusable. Latent, condition, prompt, geometry, scheduler, and policy checks remain unchanged.
- **Online validation/config:** `online_train.py` computes the same reward provenance once, includes it in `stable_config`, forwards every reward setting/model path to `rollout.py`, and verifies both record-to-sample-state equality and equality with the current expected provenance. The prior `reward_frame_stride` stable-config entry was replaced by the complete contract, not duplicated.
- **Real K=2 resume test:** `outputs/minimax_h3_diffusionnft_reward_provenance_phase81_20260828`, prompt 0, seeds `[0,1]`, 448x256, 22 frames, one inference step, base policy, no mp4. The initial stride-6 run generated both seeds. Repeating stride 6 produced exactly two `reuse` events and zero generation events. Changing only stride from 6 to 5 produced exactly two `reward_frame_stride mismatch` rejections, zero reuse events, and two fresh generation/reward events. Final rewards were `22.79337025` and `23.37336432`; both states contain all six fields with stride 5.
- **Checkpoint hashes verified in artifacts:** SCRFD SHA-256 is `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`; MagFace SHA-256 is `30e138480978d430120c669dd95ce6786d7c87803433ec91b6c811abb331167c`. Both final seed states and rollout records agree, and `video_path` remains null.
- **Verification:** `py312/bin/python -m py_compile` passed for both modified Python files, `bash -n` passed for the new smoke script, and `git diff --check` passed. `online_train.validate_rollout()` accepted the real two-record stride-5 rollout and rejected the same artifacts when the expected stride was changed to 6. Phase 8.1 is complete.

### 2026-08-28 — phase 8 same-environment in-memory reward and mp4-free rollout contract

- **Modification goal:** move the existing SCRFD + MagFace reward inference into the MiniMax-H3 `py312` process, score H3's returned PIL frames directly, make formal online rollout independent of mp4, and replace the old `mp4 + latent` resume predicate with an atomic per-seed state contract. The verified DiffusionNFT loss, advantage/r mapping, current/old/reference policy logic, H3 schedulers/core models, old-policy rollout provenance, optimizer, checkpoint, and final-export logic are unchanged.
- **Existing evaluator semantics confirmed from source:** `frame_stride=N` samples decoded frame indices `0,N,2N,...`; `max_frames_per_video=0` is unlimited and a positive value limits sampled-frame attempts. SCRFD `scrfd_10g_bnkps.onnx` runs at 640x640, threshold 0.5, `max_num=0`, on OpenCV BGR frames. Every detected face is five-landmark ArcFace-aligned to 112x112 by `insightface.utils.face_align.norm_crop`; the evaluator writes JPEG quality 95 and rereads BGR. MagFace is `iresnet100`, 512-D, receives contiguous BGR CHW float divided only by 255 (no RGB conversion and no mean/std normalization), and face quality is embedding L2 norm. For multiple faces, per-frame quality is configured mean/min/max (default mean). `mean_quality` is the arithmetic mean of numeric per-frame aggregates only, not a global per-face mean. Visible ratio is scored-frame count divided by sampled-frame count; detected faces are summed across sampled frames. With no scored face, `mean_quality=null`, ratio 0, and the adapter returns the configured `missing_face_reward` (default 0).
- **Environment merge:** no new Conda environment was created and existing `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`, CUDA, OpenCV 4.13, NumPy 2.4.4, and SciPy 1.17.1 were left unchanged. Packages installed into `py312` with `pip --no-deps` were `onnx==1.22.0`, `onnxruntime-gpu==1.23.2`, `insightface==1.0.1`, `scikit-image==0.25.2`, `termcolor==3.3.0`, plus the import-required transitive packages `ml-dtypes==0.5.4`, `lazy-loader==0.5`, `tifffile==2025.5.10`, `coloredlogs==15.0.1`, `flatbuffers==25.12.19`, and `humanfriendly==10.0`. `py312` directly imports all reward modules and exposes `CUDAExecutionProvider`; it initialized both SCRFD and MagFace and scored real frames successfully.
- **New in-memory API:** `reward_face_quality.py` now provides one reusable `FaceQualityReward` instance and `score_frames(...)`. It accepts H3 PIL RGB frames (and compatible RGB NumPy/Torch frames), converts to BGR, reuses the existing evaluator's `load_magface_model`, preserves the legacy JPEG-95 crop roundtrip and exact aggregation, and returns only the existing reward semantics (`reward == mean_quality`, or missing-face reward). `evaluate_face_quality_from_video_files(...)` and the old `evaluate_face_quality` alias remain as compatibility/debug subprocess paths; formal rollout never calls them.
- **Rollout/artifact changes:** `rollout.py` initializes H3 once and reward once per group, then for each seed performs generation -> in-memory reward -> atomic video/audio latent save -> optional debug mp4 -> atomic `seed_<N>_state.json`. Formal default `--save-rollout-video=false` sets `video_path=null`. Each committed format-version-1 state stores reward metadata, latent/condition/prompt hashes, exact policy provenance, geometry, and scheduler configuration. `--skip-existing` reuses only a complete state whose latent/condition hashes and every prompt/seed/geometry/scheduler/policy field match; an orphan latent is never accepted. `rollout.json` is rebuilt only after the full group completes. Reward model setup plus scoring time is emitted separately for online observability.
- **Training/orchestrator compatibility:** `train.py --mode nft-step` now accepts `video_path=null` and still requires clean video/audio latent plus exact condition; legacy `backward-smoke` continues to fail clearly without mp4. `online_train.py` validates every latent, condition, sample-state JSON, hash, reward field, prompt isolation, scheduler setting, and policy provenance; it validates mp4 only when `--save-rollout-video` is explicitly set. `train_online.sh` exposes `SAVE_ROLLOUT_VIDEO=0` by default and passes the opt-in flag only for value 1.
- **New tests/scripts:** `test_reward_inmemory_compat.py`, `smoke_inmemory_reward.sh`, and `smoke_rollout_resume_contract.sh`.
- **Compatibility result:** three real Phase-7 seed 0/1/2 videos at 448x256, 22 frames, stride 6 were compared through the original two-Conda evaluator and the new `py312` API. Legacy rewards were `[21.31691273, 22.07617140, 22.43755023]`; in-memory results were `[21.29497012, 22.03704055, 22.44718424]`. Face counts `[8,9,8]` and visible ratios `[1,1,1]` matched exactly. Maximum absolute/relative reward deltas were `0.03913085` / `0.177254%`, within the explicit 0.05 compatibility tolerance. Diagnosis on the same frame found byte-identical decoded BGR pixels, but CUDA SCRFD landmark drift up to `0.001764 px` between the isolated and merged runtimes; this changes aligned crop pixels. Feeding the legacy crop to py312 MagFace changed its face score by only `1.7e-5`, confirming that color, JPEG, MagFace preprocessing, model, and aggregation were not redefined. The residual is therefore a located CUDA-provider floating-point effect, not an API semantic mismatch.
- **Real no-mp4 online smoke:** `outputs/minimax_h3_diffusionnft_online/phase8_inmemory_smoke_20260828`, one prompt, K=2, seeds 0/1, 448x256, 22 frames, 3 inference steps, one iteration. Rewards were `21.87992883` and `22.32871930`; score calls took 0.495 s and 0.299 s. Both latent files and committed states exist, `rollout.json` has null video paths, no `seed_*.mp4` exists, formal NFT used saved clean video/audio latents and exact condition, optimizer step succeeded, checkpoint-1/final export exist, and global step is `0 -> 1` with `iteration_success=true`. Subprocess-boundary GPU memory was 1819 MB before and after rollout; orchestrator RSS changed 19.07 -> 17.33 MB, with no OOM or single-iteration leak indication.
- **Resume result:** real A/B/C contract test directory `outputs/minimax_h3_diffusionnft_rollout_resume_phase8_20260828`. (A) stopping after seed-0 state commit reused seed 0 and generated/rewarded only seed 1; (B) injecting the exact orphan-latent/no-state crash state caused that seed to regenerate while the valid seed was reused; (C) a complete group reused both states, ran zero generation/reward calls, reported reward time 0, and rebuilt `rollout.json`. No mp4 was present or consulted in any case.
- **Performance/I/O comparison:** on the same three existing videos and identical stride/models/aggregation, legacy reward wall time was 17.044 s versus 9.267 s for the new in-process path including model initialization (45.6% lower), while formal K=2 scoring after the one-time initialization totaled 0.795 s. Phase-7 iteration-0 old K=4 rollout used 3,047,592 bytes including 2,114,908 bytes of mp4; Phase-8 K=2 used 537,641 bytes and zero mp4. For the directly comparable seed-0/1 artifact core, the old condition + two mp4 + two latents occupied 1,562,929 bytes, while the complete new condition + two latents + two states + rollout JSON occupied 521,257 bytes (66.6% lower). Overall rollout wall time was 890.58 s for the available old K=4 Phase-7 baseline and 736.21 s for the new K=2 smoke; because group sizes and machine load differ, this wall-time pair is recorded only as engineering context, not as a speedup claim.
- **Current limitations / next step:** reward is CUDA-0 single-process only; the verified tolerance accounts for isolated-vs-merged CUDA detector drift; corrupted-file decode edge cases remain the legacy adapter's domain. Optional mp4 remains debug-only. Multi-GPU/DDP/FSDP, reward redesign/visible weighting, audio reward/loss, new RL loss, 720p/124-frame scaling, WandB, and checkpoint-retention redesign remain deferred. Phase 8 is complete.

### 2026-08-27–28 — phase 7 production online entrypoint and long-run engineering hardening

- **Modification goal:** turn the verified bounded online loop into a formal, configurable, long-running single-GPU entrypoint and validate operational continuity only. This phase does not evaluate reward quality and does not change the DiffusionNFT loss, advantage/r mapping, current/old/reference policy logic, MagFace reward, rollout clean latent/condition contract, or H3 core components.
- **New formal entrypoint:** `examples/minimax_h3/model_training/diffusionnft/train_online.sh`. Major data, range, iteration/group, seed, geometry, inference, VRAM, LoRA, optimizer, NFT, decay, retention, Python, and device settings are environment-overridable. Its defaults are the verified low-cost engineering configuration: 20 iterations, 10 rotating prompts, K=4 seeds 0/1/2/3, 448x256, 22 frames, 3 inference steps, LoRA rank 4, learning rate `1e-4`, `policy_beta=1`, `kl_beta=1e-4`, `adv_clip_max=5`, `timestep_fraction=0.99`, and type-1 old decay.
- **Modified orchestrator:** `examples/minimax_h3/model_training/diffusionnft/online_train.py` now verifies every mp4, latent safetensors, condition PNG, rollout JSON, and checkpoint file is present and non-empty. It enforces that every sample in a group shares one `prompt_<dataset_position>_*` directory and condition, that seed/file names agree, and that no video/latent path is duplicated or crosses prompt directories.
- **Operational telemetry:** every completed metrics row records rollout/train subprocess exit codes, executed-vs-reused state, rollout/reward/train/iteration active wall time, and resource snapshots before/after both subprocesses. Resource snapshots include orchestrator CPU RSS and per-GPU used/total memory from `nvidia-smi`; reward wall time is parsed from the existing adapter output without modifying MagFace.
- **Retention/export:** `--keep-last-checkpoints N` defaults to 0 (retain all). For positive N, only fully validated older checkpoint directories are removed after the new checkpoint and atomic online state commit; latest is always protected. On successful target completion, `final/current_lora.safetensors`, `final/old_lora.safetensors`, and `final/training_state.json` are atomically copied from latest, hash-recorded in state, and idempotently validated on completed-run resume.
- **Resume test hooks:** hidden engineering-only stop hooks expose the two existing state-machine boundaries needed for deterministic testing; they do not alter normal training. Real tests on the current run verified: (A) interruption during rollout after seed 0 caused resume to print `reuse seed=0` and generate only missing seeds; (B) stopping after a complete/validated rollout resumed without regeneration and started training once; (C) stopping after atomic checkpoint-1 creation but before iteration completion resumed with `reuse_valid_checkpoint`, committed `global_step 0 -> 1`, and did not repeat the optimizer step.
- **Completed long engineering run:** `outputs/minimax_h3_diffusionnft_online/phase7_engineering_20iter_20260827`, using the formal defaults above with `KEEP_LAST_CHECKPOINTS=0`. All 20 iterations completed with `global_step 0 -> 20`; dataset positions are exactly `[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]`. Each position has one stable prompt hash across both cycles, every iteration contains one four-seed prompt group, and no cross-prompt artifact or condition was found.
- **Process/artifact result:** all 20 rollout/train subprocess exit-code pairs are `0/0`; every MagFace subprocess completed; all monitored losses, distances, and gradient norms are finite; no OOM, CUDA error, traceback, killed process, NaN, or Inf occurred. All 80 mp4 files, 80 clean latent files, per-prompt condition PNGs, 20 rollout JSON files, and checkpoint-1 through checkpoint-20 were found non-empty. Every old-policy LoRA path/hash matched its source checkpoint, all steps resumed the prior state without current/old/reference reinitialization, and the largest observed gradient norm was `0.589813`, below the clip threshold 1.0.
- **Timing/memory result:** total active subprocess time was `37038.75 s` (about 10.29 h): rollout `29132.85 s`, training `7905.91 s`, with reward evaluation accounting for `373.94 s` inside rollout. Per-iteration rollout time ranged `890.58–1926.85 s`, training `384.80–405.98 s`, and reward `17.31–20.14 s`. Post-subprocess GPU boundary usage was exactly 1819 MB for all 20 iterations. Orchestrator post-training RSS rose from 14.73 MB at startup and settled at 23.30 MB; over the final ten iterations the fitted endpoint change was only about `0.07 MB/iteration`, with no operational evidence of a CUDA or CPU memory leak.
- **Final export/load verification:** run state reports `completed_iteration=19`, `global_step=20`, and checkpoint-20 as latest. Atomic `final/` contains non-empty `current_lora.safetensors` (16,422,896 bytes), `old_lora.safetensors` (32,823,216 bytes), and `training_state.json`; all three hashes exactly match checkpoint-20. The final current LoRA SHA-256 is `f3a0d91c40934acc61563b3fc5136c5a56c26ea811a298768c342504b3d2a463`. Loading this file through `rollout.py`'s existing `build_pipeline`/`--lora-path` path completed successfully and printed `patched_modules=104`; no video generation or reward-quality evaluation was performed for this load-only check.
- **Current status / next step:** phase 7 is complete. The formal entrypoint, 20-step prompt rotation, subprocess/resource observability, artifact enforcement, A/B/C resume robustness, optional safe retention, final export, and rollout-compatible LoRA loading are verified. This is only an engineering-continuity result; it makes no claim about reward-model validity or trained quality. Multi-GPU, DDP, 720p/124-frame scale-up, audio reward/loss, reward redesign, new RL loss, and WandB remain deferred.

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
