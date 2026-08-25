#!/usr/bin/env python3
"""Stream-filter a huge JSON array by motion classification.

The input is parsed incrementally with the Python standard library so scanning
stops immediately once the requested number of matching records is collected.
The output is written atomically and contains only the requested source fields
plus the full motion_classification object.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO


OUTPUT_FIELDS = ("video_caption_path", "file_path", "label_path")


def iter_json_array(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Incrementally decode values from one top-level UTF-8 JSON array."""
    decoder = json.JSONDecoder()
    utf8_decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    position = 0
    eof = False
    state = "open"  # open -> value_or_end -> comma_or_end

    def skip_whitespace(pos: int) -> int:
        while pos < len(buffer) and buffer[pos].isspace():
            pos += 1
        return pos

    while True:
        need_more = False
        position = skip_whitespace(position)

        if state == "open":
            if position >= len(buffer):
                need_more = True
            elif buffer[position] != "[":
                raise ValueError("Input JSON must have a top-level array")
            else:
                position += 1
                state = "value_or_end"

        elif state == "value_or_end":
            if position >= len(buffer):
                need_more = True
            elif buffer[position] == "]":
                return
            else:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    need_more = True
                else:
                    yield value
                    position = end
                    state = "comma_or_end"

        elif state == "comma_or_end":
            if position >= len(buffer):
                need_more = True
            elif buffer[position] == ",":
                position += 1
                state = "value_or_end"
            elif buffer[position] == "]":
                return
            else:
                preview = buffer[position : position + 80]
                raise ValueError(f"Expected ',' or ']' in input array near: {preview!r}")

        if not need_more:
            continue
        if eof:
            raise ValueError(f"Unexpected end of JSON input while parsing state={state}")

        # Preserve an incomplete value but discard text already consumed.
        if position:
            buffer = buffer[position:]
            position = 0
        chunk = stream.read(chunk_size)
        if chunk:
            buffer += utf8_decoder.decode(chunk, final=False)
        else:
            buffer += utf8_decoder.decode(b"", final=True)
            eof = True


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def compact_record(item: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_caption_path": item["video_caption_path"],
        "file_path": item["file_path"],
        "label_path": item["label_path"],
        "lipsync": item.get("lipsync", {}),
        "motion_classification": classification,
    }


def write_indented_item(stream, item: dict[str, Any], first: bool) -> None:
    if not first:
        stream.write(",\n")
    serialized = json.dumps(item, ensure_ascii=False, indent=2)
    stream.write("  " + serialized.replace("\n", "\n  "))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-filter motion_classification records and stop at --limit matches."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--limit", type=positive_int, required=True)
    parser.add_argument("--l2", default="多人互动")
    parser.add_argument("--exclude-motion-intensity", default="静态")
    parser.add_argument("--progress-every", type=positive_int, default=100_000)
    parser.add_argument("--chunk-size-mb", type=positive_int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_json.is_file():
        raise FileNotFoundError(args.input_json)
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output_json}")
    if args.input_json.resolve() == args.output_json.resolve():
        raise ValueError("Input and output paths must be different")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output_json.with_name(f".{args.output_json.name}.tmp.{os.getpid()}")
    if temp_path.exists():
        raise FileExistsError(f"Temporary output unexpectedly exists: {temp_path}")

    scanned = 0
    selected = 0
    rejected_not_object = 0
    rejected_no_classification = 0
    rejected_l2 = 0
    rejected_missing_intensity = 0
    rejected_excluded_intensity = 0
    rejected_missing_fields = 0
    reached_limit = False

    print(f"[input] {args.input_json}")
    print(f"[output] {args.output_json}")
    print(
        f"[filter] l2={args.l2!r} motion_intensity!={args.exclude_motion_intensity!r} "
        f"limit={args.limit}"
    )

    try:
        with args.input_json.open("rb") as source, temp_path.open("x", encoding="utf-8") as output:
            output.write("[\n")
            for item in iter_json_array(source, chunk_size=args.chunk_size_mb * 1024 * 1024):
                scanned += 1
                if scanned % args.progress_every == 0:
                    print(f"[progress] scanned={scanned} selected={selected}")
                if not isinstance(item, dict):
                    rejected_not_object += 1
                    continue

                classification = item.get("motion_classification")
                if not isinstance(classification, dict):
                    rejected_no_classification += 1
                    continue
                if classification.get("l2") != args.l2:
                    rejected_l2 += 1
                    continue

                intensity = classification.get("motion_intensity")
                if not isinstance(intensity, str) or not intensity.strip():
                    rejected_missing_intensity += 1
                    continue
                if intensity.strip() == args.exclude_motion_intensity:
                    rejected_excluded_intensity += 1
                    continue
                if any(not isinstance(item.get(field), str) or not item[field] for field in OUTPUT_FIELDS):
                    rejected_missing_fields += 1
                    continue

                write_indented_item(output, compact_record(item, classification), first=selected == 0)
                selected += 1
                if selected >= args.limit:
                    reached_limit = True
                    print(f"[stop] reached limit after scanning {scanned} records")
                    break

            output.write("\n]\n")
            output.flush()
            os.fsync(output.fileno())

        if args.output_json.exists() and not args.overwrite:
            raise FileExistsError(f"Output appeared during processing: {args.output_json}")
        os.replace(temp_path, args.output_json)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise

    summary = {
        "scanned": scanned,
        "selected": selected,
        "requested": args.limit,
        "reached_limit": reached_limit,
        "rejected_not_object": rejected_not_object,
        "rejected_no_classification": rejected_no_classification,
        "rejected_l2": rejected_l2,
        "rejected_missing_intensity": rejected_missing_intensity,
        "rejected_excluded_intensity": rejected_excluded_intensity,
        "rejected_missing_fields": rejected_missing_fields,
    }
    print(f"[done] {json.dumps(summary, ensure_ascii=False)}")
    print(f"[saved] {args.output_json}")
    if not reached_limit:
        print(
            f"[warning] input exhausted before requested limit: selected={selected}, requested={args.limit}",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
