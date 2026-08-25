#!/usr/bin/env python3
"""Render shuffled text-only MiniMax-H3 samples with multiple seeds per GT.

This uses the base checkpoint and pure-noise text-to-audio-video flow, with no
keyframe/reference image passed to the model. It writes one folder per source:

    <video_stem>/gt.mp4
    <video_stem>/0.mp4
    <video_stem>/1.mp4
    <video_stem>/2.mp4
    <video_stem>/sample.json

GT is retained only for comparison and frame-count selection; it is never used
as generation conditioning. Records and per-record seed order are shuffled
deterministically. The model is loaded once and completed outputs are resumable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from minimax_h3_stage1_pair_render import (  # noqa: E402
    DEFAULT_DIFFSYNTH_ROOT,
    DEFAULT_MODEL_ID,
    H3_FPS,
    NEGATIVE_PROMPT,
    align_up,
    build_pipeline,
    get_video_info,
    h3_supported_frame_count,
    iter_records,
    load_prompt_from_record,
    load_video_path_from_record,
    normalize_prompt,
    safe_folder_name,
    trim_gt_video,
    validate_requested_frames,
)


DEFAULT_DATA_JSON = Path(
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/"
    "guangdian_20251114_small_clear_faces/guangdian_20251114_small_clear_faces.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/guangdian_20251114_small_clear_faces_minimax_h3_t2v_3seed_shuffled")


def validate_existing_render(
    path: Path,
    expected_frames: int,
    expected_fps: int,
    expected_width: int,
    expected_height: int,
) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        return False, "missing_or_empty"
    try:
        frames, fps, width, height = get_video_info(path)
    except Exception as exc:
        return False, f"unreadable:{type(exc).__name__}:{exc}"
    actual = f"frames={frames},fps={fps:.6f},size={width}x{height}"
    if frames != expected_frames:
        return False, f"frame_mismatch:{actual},expected={expected_frames}"
    if abs(fps - expected_fps) >= 0.01:
        return False, f"fps_mismatch:{actual},expected={expected_fps}"
    if width != expected_width or height != expected_height:
        return False, f"size_mismatch:{actual},expected={expected_width}x{expected_height}"
    return True, actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render shuffled text-only MiniMax-H3 samples with multiple seeds per GT."
    )
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--num-frames",
        default="175",
        help="auto or an integer satisfying 17n+5; default 175 avoids 464-frame renders for 20-second inputs.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=H3_FPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle-seed", type=int, default=20260821)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram-limit-gb", type=float, default=60.0)
    parser.add_argument("--vram-reserve-gb", type=float, default=2.0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_requested_frames(args.num_frames)
    if args.fps != H3_FPS:
        raise ValueError(f"MiniMax-H3 audio/video timing is fixed at {H3_FPS} fps")
    if args.start < 0 or args.limit is not None and args.limit < 0:
        raise ValueError("--start and --limit must be non-negative")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Seeds must be unique: {args.seeds}")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError(f"Seeds must be non-negative: {args.seeds}")
    if not args.data_json.is_file():
        raise FileNotFoundError(args.data_json)
    if not args.diffsynth_root.is_dir():
        raise FileNotFoundError(args.diffsynth_root)

    # build_pipeline from the original script supports optional LoRA. Keep it
    # explicitly disabled here so this remains the original base-H3 flow.
    args.lora_path = None
    args.lora_strength = 1.0

    render_width = align_up(args.width, 32)
    render_height = align_up(args.height, 32)
    all_records = list(iter_records(args.data_json, 0, None))
    if not args.no_shuffle:
        random.Random(args.shuffle_seed).shuffle(all_records)
    end = len(all_records) if args.limit is None else min(len(all_records), args.start + args.limit)
    records = all_records[args.start:end]
    print(f"[model] {args.model_id} (base checkpoint, text-only, no image/reference, no LoRA)")
    print(f"[data] {args.data_json}")
    print(f"[output] {args.output_dir}")
    print(
        f"[render] requested={args.width}x{args.height} aligned={render_width}x{render_height} "
        f"fps={args.fps} frames={args.num_frames} steps={args.num_inference_steps} seeds={args.seeds}"
    )
    print(
        f"[order] shuffled={not args.no_shuffle} shuffle_seed={args.shuffle_seed} "
        f"slice_start={args.start} slice_size={len(records)} total_records={len(all_records)}"
    )

    if args.dry_run:
        valid = 0
        used_names: set[str] = set()
        for idx, record in records:
            try:
                video_path = load_video_path_from_record(record)
                prompt = normalize_prompt(load_prompt_from_record(record))
                raw_frames, raw_fps, raw_w, raw_h = get_video_info(video_path)
                frame_count = h3_supported_frame_count(raw_frames, raw_fps, args.num_frames, args.fps)
                folder = args.output_dir / safe_folder_name(video_path, used_names)
                print(
                    f"[dry-run] idx={idx} video={video_path} source={raw_w}x{raw_h}@{raw_fps:.3f} "
                    f"source_frames={raw_frames} render={render_width}x{render_height}@{args.fps} "
                    f"render_frames={frame_count} seeds={args.seeds} folder={folder} prompt_chars={len(prompt)}"
                )
                valid += 1
            except Exception as exc:
                print(f"[invalid] idx={idx}: {type(exc).__name__}: {exc}")
        print(f"[done] dry_run_valid={valid} total={len(records)}")
        if records and valid == 0:
            raise SystemExit(1)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(args)
    from diffsynth.utils.data.audio_video import write_video_audio

    used_names: set[str] = set()
    rendered = 0
    skipped = 0
    failed = 0
    invalid_records = 0

    for idx, record in records:
        try:
            video_path = load_video_path_from_record(record)
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            prompt = normalize_prompt(load_prompt_from_record(record))
            raw_frames, raw_fps, raw_w, raw_h = get_video_info(video_path)
            frame_count = h3_supported_frame_count(raw_frames, raw_fps, args.num_frames, args.fps)
            folder = args.output_dir / safe_folder_name(video_path, used_names)
            gt_path = folder / "gt.mp4"
            sample_path = folder / "sample.json"
            folder.mkdir(parents=True, exist_ok=True)
            trim_gt_video(video_path, gt_path, frame_count, args.fps, render_width, render_height)
        except Exception as exc:
            invalid_records += 1
            print(f"[invalid] idx={idx}: {type(exc).__name__}: {exc}")
            continue

        print(f"[case] idx={idx} name={folder.name}")
        print(f"  source={video_path}")
        print(
            f"  source_shape={raw_w}x{raw_h}@{raw_fps:.3f} source_frames={raw_frames} "
            f"render={render_width}x{render_height}@{args.fps} render_frames={frame_count}"
        )
        seed_status: dict[str, dict[str, object]] = {}
        seed_order = list(args.seeds)
        if not args.no_shuffle:
            random.Random(args.shuffle_seed + idx * 1_000_003).shuffle(seed_order)
        print(f"  seed_order={seed_order}")
        for seed in seed_order:
            render_path = folder / f"{seed}.mp4"
            if render_path.exists() and not args.overwrite_existing:
                is_valid, validation = validate_existing_render(
                    render_path,
                    expected_frames=frame_count,
                    expected_fps=args.fps,
                    expected_width=render_width,
                    expected_height=render_height,
                )
                if is_valid:
                    print(f"  [skip-valid] seed={seed} path={render_path} {validation}")
                    seed_status[str(seed)] = {
                        "status": "skipped_valid",
                        "path": str(render_path),
                        "validation": validation,
                    }
                    skipped += 1
                    continue
                print(f"  [rerender-invalid] seed={seed} path={render_path} reason={validation}")
            try:
                print(f"  [render] seed={seed} path={render_path}")
                video, audio = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=render_height,
                    width=render_width,
                    num_frames=frame_count,
                    num_inference_steps=args.num_inference_steps,
                    seed=seed,
                    cfg_scale=args.cfg_scale,
                    flow_shift=args.flow_shift,
                    audio_flow_shift=args.audio_flow_shift,
                    tiled=not args.no_tiled,
                )
                if len(video) != frame_count:
                    raise RuntimeError(f"Pipeline returned {len(video)} frames, expected {frame_count}")
                write_video_audio(
                    video=video,
                    audio=audio,
                    output_path=str(render_path),
                    fps=args.fps,
                    audio_sample_rate=pipe.audio_vae.sample_rate,
                    video_quality=8,
                )
                seed_status[str(seed)] = {
                    "status": "rendered",
                    "path": str(render_path),
                    "frame_count": len(video),
                }
                rendered += 1
                print(f"  [saved] seed={seed} path={render_path}")
            except Exception as exc:
                failed += 1
                seed_status[str(seed)] = {
                    "status": "failed",
                    "path": str(render_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  [failed] seed={seed}: {type(exc).__name__}: {exc}")

        sample_record = dict(record)
        sample_record.update(
            {
                "gt_path": str(gt_path),
                "render_seed_status": seed_status,
                "render_seeds": args.seeds,
                "render_frame_count": frame_count,
                "render_fps": args.fps,
                "render_width": render_width,
                "render_height": render_height,
                "source_frame_count": raw_frames,
                "source_fps": raw_fps,
                "source_width": raw_w,
                "source_height": raw_h,
                "render_model": args.model_id,
                "render_num_inference_steps": args.num_inference_steps,
                "render_cfg_scale": args.cfg_scale,
                "render_flow_shift": args.flow_shift,
                "render_audio_flow_shift": args.audio_flow_shift,
                "render_lora_path": None,
                "render_conditioning": "text_only_no_keyframe_no_reference",
                "record_order_shuffled": not args.no_shuffle,
                "shuffle_seed": args.shuffle_seed,
                "render_seed_order": seed_order,
                "normalized_prompt": prompt,
            }
        )
        sample_path.write_text(json.dumps(sample_record, ensure_ascii=False, indent=2), encoding="utf-8")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[done] rendered={rendered} skipped={skipped} failed={failed} "
        f"invalid_records={invalid_records} records={len(records)}"
    )
    if records and rendered == 0 and skipped == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
