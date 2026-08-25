import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download
from PIL import Image
from PIL import ImageOps
import pandas as pd
import glob, json, re, os, time, base64, io
from operators_multi_compat import LoadMagiPromptFile
import requests

IR_API_URL = "https://api.minimaxi.com/v2/h3_context_ir"
IR_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-api--SggWiLkonzPkiGlGLVi_4LG9k0NpX7qTBUH5ZRP9mXlZBn_SJ-_egrCjj43i9Rdad27e40j3HPT0YO0UAM0H74C7kON3o3CMuE2rWda3IxSJPWfjMukzhI",
}

def h3_context_ir(prompt_text, image, duration=5, ratio="adaptive"):
    """调用 MiniMax H3-Context-IR：输入文本+首帧图片，返回增强后的视频提示词文本。"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": "MiniMax-H3",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}, "role": "first_frame"},
        ],
        "duration": duration,
        "ratio": ratio,
    }
    resp = requests.post(IR_API_URL, json=payload, headers=IR_HEADERS, timeout=120)
    task_id = resp.json()["task_id"]
    query_url = f"https://api.minimaxi.com/v2/query/video_generation/{task_id}"
    while True:
        result = requests.get(query_url, headers=IR_HEADERS, timeout=120).json()
        status = result["task"]["status"]
        print("  ir status:", status)
        if status == "succeeded":
            return result["task"]["content"]["prompt"]
        elif status == "failed":
            raise RuntimeError(f"h3_context_ir failed: {result}")
        time.sleep(3)

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

first_frame = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-TI2VA/first.png")
last_frame = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-TI2VA/last.png")
prompt = "室内家庭争吵短剧场景，竖屏短剧质感，真实真人表演，中式家庭/小饭馆室内环境，暖色灯光，背景有红色装饰和书法字幅，浅景深，情绪强烈，剪辑节奏紧凑。表演要求：真实短剧表演风格，不要夸张舞台腔。男人的语气是愤怒、委屈、急切的反驳，他说“你到底想干什么？”；中老年女性的语气是尖锐、强势、咄咄逼人的质问，她说“你必须赔钱！”。两人之间有强烈对峙感，节奏逐步升级。画面风格：竖屏9:16，手机短剧质感，真人实拍感，浅景深，室内暖光，中近景为主，频繁正反打剪辑，背景保持生活化，不要科幻、不要古装、不要动画感。画面中不要出现任何字幕、文字、平台水印或贴片。 "
height = 720
width = 1280 
num_frames = 175

json_dir = "/gemini/platform/public/aigc/human_guozz2/code/xqp/code/LLM/caption_change_thinking_4/" # 替换为你的文件夹路径和 JSON 文件模式
img_dir = "/gemini/platform/public/aigc/human_guozz2/code/wzh/lmx/i2av/dataset/downloaded_images/"
for index, j_file in enumerate(sorted(glob.glob(f"{json_dir}/*.json"))):
    if (index+1) not in [9,14,21,25,36,55,60,62,71,77,81,98,100,113,121,122,133,137,179,180,182,185,186,190,196,197,203,205,207]:  # 只处理前 3 个文件
        continue
    name_without_ext = os.path.splitext(os.path.basename(j_file))[0]
    img_matches = glob.glob(f"{img_dir}/{name_without_ext}.*")
    print("img_path:", img_matches)
    prompt = LoadMagiPromptFile()(j_file)["prompt"]
    prompt = re.sub(r"\[time_range:[^\]]*\]", "", prompt)
    
    image = Image.open(img_matches[0])
    first_frame = ImageOps.fit(image, (width, height), centering=(0.5, 0.5)).convert("RGB")

    # 调用 H3-Context-IR：输入文本+首帧图片，返回增强后的提示词文本
    enhanced_prompt = h3_context_ir(prompt, first_frame)
    print("enhanced_prompt:", enhanced_prompt)
    prompt = enhanced_prompt

    video, audio = pipe(
        prompt=prompt,
        height=height, width=width, num_frames=num_frames,
        num_inference_steps=50, seed=0,
        keyframes=[first_frame],
        keyframe_indices=[0],
    )
    save_path = f'results/H3_test/H3_i2av_{name_without_ext}.mp4'
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
