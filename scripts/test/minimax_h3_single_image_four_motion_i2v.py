#!/usr/bin/env python3
"""Generate four camera-motion variants from one image with one H3 load.

The script reuses the base MiniMax-H3 FL2AV implementation from
minimax_h3_single_image_i2v.py.  It prepares one keyframe, loads the model once,
then renders all requested motion prompts sequentially in the same process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from minimax_h3_single_image_i2v import (  # noqa: E402
    DEFAULT_DIFFSYNTH_ROOT,
    DEFAULT_MODEL_ID,
    H3_FPS,
    NEGATIVE_PROMPT,
    align_up,
    build_pipeline,
    direct_resize_frame,
    fit_pad_resize_frame,
    validate_frame_count,
)


MOTION_TEXTS = {
    "left_to_right": (
        "The camera performs a pronounced, continuous left-to-right lateral dolly movement "
        "over a substantial physical distance. Begin from a clearly left-side viewpoint and "
        "finish at a distinctly different right-side viewpoint. Maintain a steady speed and "
        "produce strong, physically correct parallax. This is a wide tracking movement, not a subtle drift."
    ),
    "right_to_left": (
        "The camera performs a pronounced, continuous right-to-left lateral dolly movement "
        "over a substantial physical distance. Begin from a clearly right-side viewpoint and "
        "finish at a distinctly different left-side viewpoint. Maintain a steady speed and "
        "produce strong, physically correct parallax. This is a wide tracking movement, not a subtle drift."
    ),
    "left_to_right_closer": (
        "The camera performs a pronounced left-to-right lateral dolly while simultaneously moving "
        "forward toward the landmark. Travel across a wide lateral baseline and close the physical "
        "distance substantially, so the landmark becomes noticeably larger through real camera "
        "translation rather than zooming. Produce strong, physically correct parallax."
    ),
    "right_to_left_closer": (
        "The camera performs a pronounced right-to-left lateral dolly while simultaneously moving "
        "forward toward the landmark. Travel across a wide lateral baseline and close the physical "
        "distance substantially, so the landmark becomes noticeably larger through real camera "
        "translation rather than zooming. Produce strong, physically correct parallax."
    ),
}


SUBJECTS = {
    "palace_of_westminster": (
        "The Palace of Westminster in London, including its monumental Gothic Revival facade, "
        "ornate stonework, towers, spires, and Elizabeth Tower with the Big Ben clock, shown with "
        "accurate architectural proportions and finely resolved details"
    ),
    "brandenburg_gate": (
        "The Brandenburg Gate in Berlin, a monumental neoclassical sandstone landmark with six "
        "large columns, accurate architectural proportions, and finely resolved stone details"
    ),
    "sacre_coeur": (
        "The Basilica of Sacre-Coeur in Paris, a monumental white-stone basilica with accurate "
        "domes, arches, steps, architectural proportions, and finely resolved masonry details"
    ),
    "taj_mahal": (
        "The Taj Mahal in Agra, a monumental white-marble mausoleum with an accurate central dome, "
        "minarets, arches, symmetry, architectural proportions, and finely resolved marble details"
    ),
    "temple_nara_japan": (
        "A historic Buddhist temple in Nara, Japan, with accurate traditional wooden architecture, "
        "roof geometry, structural proportions, fine timber details, and surrounding grounds"
    ),
}


def load_rgb_on_white(path: Path) -> tuple[Image.Image, int, int, str]:
    with Image.open(path) as image:
        source_width, source_height = image.size
        source_mode = image.mode
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(white, rgba).convert("RGB")
    return rgb, source_width, source_height, source_mode


def default_output_dir(input_image: Path) -> Path:
    return Path("outputs/repair") / input_image.parent.name / input_image.stem


def build_prompt(subject: str, motion_text: str) -> str:
    return (
        f"<SUBJECT>: {subject}. "
        "<SCENE>: A photorealistic daytime view of the landmark and its surroundings. Preserve the "
        "exact landmark identity and keep the architecture rigid, sharp, geometrically accurate, "
        "and temporally consistent throughout the entire video. Architectural edges, windows, "
        "columns, towers, rooflines, stone textures, and clock details must remain fine, coherent, "
        "and stable. The foreground and surrounding surfaces must remain highly detailed, with "
        "consistent perspective, realistic textures, natural contact shadows, and no texture "
        "flickering or swimming. Maintain stable daylight, exposure, colors, materials, and fine "
        f"details across all frames. <EVENT>: {motion_text} Keep the landmark recognizable, sharp, "
        "and structurally stable for the full shot. No artificial zoom, no static camera, no tiny "
        "motion, no camera shake, no sudden acceleration, no jumps, no blur, no flickering, no "
        "texture swimming, no warped architecture, no structural deformation, no duplicated "
        "architectural elements, and no objects appearing or disappearing. The video is silent."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render four I2V camera motions from one image while loading MiniMax-H3 only once."
    )
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subject", help="Override the automatic landmark subject description.")
    parser.add_argument(
        "--motions",
        nargs="+",
        choices=tuple(MOTION_TEXTS),
        default=list(MOTION_TEXTS),
    )
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=175)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=H3_FPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram-limit-gb", type=float, default=60.0)
    parser.add_argument("--vram-reserve-gb", type=float, default=2.0)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--keep-generated-audio", action="store_true")
    parser.add_argument(
        "--exact-requested-output",
        action="store_true",
        help="Fit-pad aligned generation to exact --width/--height; default writes native aligned size.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input_image.is_file():
        raise FileNotFoundError(args.input_image)
    if not args.diffsynth_root.is_dir():
        raise FileNotFoundError(args.diffsynth_root)
    if args.fps != H3_FPS:
        raise ValueError(f"MiniMax-H3 timing is fixed at {H3_FPS} fps")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("Output dimensions must be positive even numbers")
    validate_frame_count(args.num_frames)
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")

    output_dir = args.output_dir or default_output_dir(args.input_image)
    scene_name = args.input_image.parent.name
    subject = args.subject or SUBJECTS.get(
        scene_name,
        "The landmark shown in the input image, preserving its exact identity, architecture, geometry, and materials",
    )
    source_image, source_width, source_height, source_mode = load_rgb_on_white(args.input_image)
    generation_width = align_up(args.width, 32)
    generation_height = align_up(args.height, 32)
    keyframe = direct_resize_frame(source_image, generation_width, generation_height)
    output_width = args.width if args.exact_requested_output else generation_width
    output_height = args.height if args.exact_requested_output else generation_height

    jobs: list[tuple[str, Path, Path, str]] = []
    for motion in args.motions:
        video_path = output_dir / f"{motion}_{args.num_frames}f_{output_width}x{output_height}.mp4"
        metadata_path = video_path.with_suffix(".json")
        prompt = build_prompt(subject, MOTION_TEXTS[motion])
        if (
            video_path.is_file()
            and video_path.stat().st_size > 0
            and metadata_path.is_file()
            and not args.overwrite_existing
        ):
            print(f"[skip-existing] motion={motion} video={video_path}")
            continue
        if video_path.exists() and not args.overwrite_existing:
            raise FileExistsError(
                f"Incomplete output exists; pass --overwrite-existing to replace it: {video_path}"
            )
        jobs.append((motion, video_path, metadata_path, prompt))

    print(f"[model] {args.model_id} (base checkpoint, no LoRA, one model load)")
    print(
        f"[input] {args.input_image} source={source_width}x{source_height} mode={source_mode} "
        "alpha_background=white input_resize=direct"
    )
    print(
        f"[render] generation={generation_width}x{generation_height}@{args.fps} "
        f"output={output_width}x{output_height} frames={args.num_frames} "
        f"steps={args.num_inference_steps} seed={args.seed} pending={len(jobs)}/{len(args.motions)}"
    )
    for motion, video_path, _, prompt in jobs:
        print(f"[job] motion={motion} output={video_path}")
        print(f"[prompt] {prompt}")
    if args.dry_run or not jobs:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(args)
    from diffsynth.utils.data.audio_video import write_video_audio

    completed = 0
    failed = 0
    for motion, video_path, metadata_path, prompt in jobs:
        print(f"[render] motion={motion} output={video_path}")
        try:
            video, audio = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=generation_height,
                width=generation_width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                flow_shift=args.flow_shift,
                audio_flow_shift=args.audio_flow_shift,
                tiled=not args.no_tiled,
                keyframes=[keyframe],
                keyframe_indices=[0],
            )
            if len(video) != args.num_frames:
                raise RuntimeError(f"Pipeline returned {len(video)} frames, expected {args.num_frames}")
            if args.exact_requested_output:
                video = [fit_pad_resize_frame(frame, args.width, args.height) for frame in video]
            write_video_audio(
                video=video,
                audio=audio if args.keep_generated_audio else None,
                output_path=str(video_path),
                fps=args.fps,
                audio_sample_rate=pipe.audio_vae.sample_rate,
                video_quality=8,
            )
            metadata = {
                "input_image": str(args.input_image),
                "output_video": str(video_path),
                "motion": motion,
                "subject": subject,
                "source_width": source_width,
                "source_height": source_height,
                "source_mode": source_mode,
                "alpha_background": "white",
                "input_resize_mode": "direct_resize_to_h3_aligned_generation_size",
                "generation_width": generation_width,
                "generation_height": generation_height,
                "output_width": output_width,
                "output_height": output_height,
                "output_resize_mode": "aspect_fit_then_pad" if args.exact_requested_output else "aligned_direct",
                "num_frames": len(video),
                "fps": args.fps,
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed,
                "cfg_scale": args.cfg_scale,
                "flow_shift": args.flow_shift,
                "audio_flow_shift": args.audio_flow_shift,
                "has_audio": bool(args.keep_generated_audio),
                "model_id": args.model_id,
                "lora_path": None,
                "prompt": prompt,
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            completed += 1
            print(f"[saved] motion={motion} video={video_path} metadata={metadata_path}")
            del video, audio
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            failed += 1
            print(f"[failed] motion={motion}: {type(exc).__name__}: {exc}")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[done] completed={completed} failed={failed} requested={len(jobs)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
