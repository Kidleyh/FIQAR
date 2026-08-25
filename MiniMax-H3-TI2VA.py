import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download
from PIL import Image
from PIL import ImageOps
import pandas as pd
import glob, json, re, os
from operators_multi_compat import LoadMagiPromptFile

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
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
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="minimax_h3/MiniMax-H3-TI2VA/*",
)

height = 720
width = 1280 
num_frames = 175

##通用评测
# json_dir = "/gemini/platform/public/aigc/human_guozz2/code/xqp/code/LLM/caption_change_thinking_4/" # 替换为你的文件夹路径和 JSON 文件模式
# img_dir = "/gemini/platform/public/aigc/human_guozz2/code/wzh/lmx/i2av/dataset/downloaded_images/"

##motion评测
# json_dir = "/gemini/platform/public/aigc/human_guozz2/code/xqp/code/LLM/caption_change_motion/" # 替换为你的文件夹路径和 JSON 文件模式
# img_dir = "/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DS_LTX23/docs/VideoMotion/"

# ##专项评测
json_dir = "/gemini/platform/public/aigc/human_guozz2/code/xqp/code/LLM/caption_plus_change"
img_dir = "/gemini/platform/public/aigc/human_guozz2/code/wzh/lmx/dataset/"
#递归查找 json_dir 下所有子目录中的 JSON 文件
for index, j_file in enumerate(sorted(glob.glob(f"{json_dir}/**/*.json", recursive=True))):
    name_without_ext = os.path.splitext(os.path.basename(j_file))[0]
    # JSON 所在子目录名，在 img_dir 下找同名子目录中的图片
    sub_dir_name = os.path.basename(os.path.dirname(j_file))
    img_matches = glob.glob(f"{img_dir}/{sub_dir_name}/{name_without_ext}.*")
    save_path = f'results/H3_ori_spec_nfe_20/{sub_dir_name}/ltx2.3_twostage_i2av_{name_without_ext}.mp4'


# for index, j_file in enumerate(sorted(glob.glob(f"{json_dir}/*.json"))):
#     # if (index+1) not in [9,14,21,25,36,55,60,62,71,77,81,98,100,113,121,122,133,137,179,180,182,185,186,190,196,197,203,205,207]:  # 只处理前 3 个文件
#     #     continue
#     # if index+1 < 57:
#     #     continue
#     name_without_ext = os.path.splitext(os.path.basename(j_file))[0]
#     img_matches = glob.glob(f"{img_dir}/*{name_without_ext}.*")
#     print("img_path:", img_matches)
#     save_path = f'results/H3_ori_motion_nfe_50/H3_i2av_{name_without_ext}.mp4'

    prompt = LoadMagiPromptFile()(j_file)["prompt"]
    prompt = re.sub(r"\[time_range:[^\]]*\]", "", prompt)
    print("prompt:", prompt)
    image = Image.open(img_matches[0])
    first_frame = ImageOps.fit(image, (width, height), centering=(0.5, 0.5)).convert("RGB")
    
    video, audio = pipe(
        prompt=prompt,
        height=height, width=width, num_frames=num_frames,
        num_inference_steps=20, seed=0,
        keyframes=[first_frame],
        keyframe_indices=[0],
    )
    
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
