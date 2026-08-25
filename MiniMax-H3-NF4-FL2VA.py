import torch
import gc
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth.core.vram.layers import AutoTorchModule
from modelscope import dataset_snapshot_download
from PIL import Image
from PIL import ImageOps
import pandas as pd
import glob, json, re, os
from operators_multi_compat import LoadMagiPromptFile


def print_vram(tag):
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    total, free = torch.cuda.mem_get_info()
    print(f"[VRAM {tag}] allocated={allocated:.1f}G reserved={reserved:.1f}G free={free / 1024 ** 3:.1f}G")


def release_vram(pipe):
    """彻底释放本轮推理占用的显存。

    1. `load_models_to_device([])` 卸载 pipeline 管理的全部模型；
    2. 兜底：递归找出仍处于非 offload 状态(state != 0)的 AutoTorchModule，
       逐个强制 offload（含 buffer 跳过导致的漏网模块）；
    3. 回收引用 + 清空 PyTorch 缓存池。
    """
    pipe.load_models_to_device([])
    for module in pipe.modules():
        if isinstance(module, AutoTorchModule) and getattr(module, "state", 0) != 0:
            try:
                module.offload()
            except Exception as e:
                print(f"[warn] offload failed for {type(module).__name__}: {e}")
    gc.collect()
    torch.cuda.empty_cache()

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

# Image -> Video + Audio (loop inference)
height = 720
width = 1280
num_frames = 175

json_dir = "/gemini/platform/public/aigc/human_guozz2/code/xqp/code/LLM/caption_change_thinking_4/"
img_dir = "/gemini/platform/public/aigc/human_guozz2/code/wzh/lmx/i2av/dataset/downloaded_images/"

for index, j_file in enumerate(sorted(glob.glob(f"{json_dir}/*.json"))):
    # if (index + 1) not in [9, 14, 21, 25, 36, 55, 60, 62, 71, 77, 81, 98, 100, 113, 121, 122, 133, 137, 179, 180, 182, 185, 186, 190, 196, 197, 203, 205, 207]:
    #     continue
    name_without_ext = os.path.splitext(os.path.basename(j_file))[0]
    img_matches = glob.glob(f"{img_dir}/{name_without_ext}.*")
    print("img_path:", img_matches)
    prompt = LoadMagiPromptFile()(j_file)["prompt"]
    prompt = re.sub(r"\[time_range:[^\]]*\]", "", prompt)
    print("prompt:", prompt)
    image = Image.open(img_matches[0])
    first_frame = ImageOps.fit(image, (width, height), centering=(0.5, 0.5)).convert("RGB")

    print_vram("before inference")
    try:
        video, audio = pipe(
            prompt=prompt,
            height=height, width=width, num_frames=num_frames,
            num_inference_steps=50, seed=0,
            keyframes=[first_frame],
            keyframe_indices=[0],
        )
    finally:
        print_vram("after inference")
    save_path = f'results/H3_nf4_test/H3_nf4_i2av_{name_without_ext}.mp4'
    folder_path = os.path.dirname(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    write_video_audio(
        video=video,
        audio=audio,
        output_path=save_path,
        fps=24,
        audio_sample_rate=pipe.audio_vae.sample_rate,
    )
    print("saved", save_path, "frames:", len(video), "audio:", tuple(audio.shape))

    # 释放本轮推理占用的显存：将 dit / video_vae / audio_vae 全部 offload 回 disk，
    # 否则下一轮推理时权重仍常驻 GPU，会触发 CUDA OOM。
    del video, audio
    release_vram(pipe)
    print_vram("after release")
