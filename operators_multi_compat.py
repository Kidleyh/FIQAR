import math
import torch, torchvision, imageio, os
import imageio.v3 as iio
import json
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torchaudio
import torchaudio.transforms as T
try:
    import decord
except Exception:
    decord = None
from torchvision.transforms import v2
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None
import random

def _operator_profile_steps() -> int:
    return int(os.environ.get("DATA_OPERATOR_PROFILE_STEPS", "0") or 0)


def _operator_profile_interval() -> int:
    return int(os.environ.get("DATA_OPERATOR_PROFILE_INTERVAL", "20") or 20)


def _should_profile(counter: int) -> bool:
    steps = _operator_profile_steps()
    return steps != 0 and (steps < 0 or counter < steps)


def _format_ms(value: float) -> str:
    return f"{value * 1000.0:.2f}ms"


def _update_totals(totals: dict, values: dict) -> None:
    for key, value in values.items():
        totals[key] = totals.get(key, 0.0) + value


def _summary_parts(totals: dict, count: int) -> str:
    return " ".join(f"avg_{key}={_format_ms(total / count)}" for key, total in totals.items())


def _video_process_workers() -> int:
    return max(1, int(os.environ.get("DATA_VIDEO_PROCESS_WORKERS", "1") or 1))


def _video_process_inflight() -> int:
    workers = _video_process_workers()
    return max(workers, int(os.environ.get("DATA_VIDEO_PROCESS_INFLIGHT", str(max(2, workers * 2)))) or max(2, workers * 2))


def _video_decode_use_batch() -> bool:
    return str(os.environ.get("DATA_VIDEO_DECODE_USE_BATCH", "1")).strip().lower() in ("1", "true", "yes", "on")


def _video_frame_process_mode() -> str:
    return str(os.environ.get("DATA_VIDEO_FRAME_PROCESS_MODE", "auto")).strip().lower()


def _video_output_mode() -> str:
    return str(os.environ.get("DATA_VIDEO_OUTPUT_MODE", "pil")).strip().lower()


def _local_rank_for_decode() -> int:
    for env_name in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID"):
        value = os.environ.get(env_name)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except Exception:
            continue
    return 0


def _decord_ctx():
    spec = str(os.environ.get("DATA_VIDEO_DECORD_CTX", "cpu")).strip().lower()
    if spec == "cpu":
        return decord.cpu()
    if spec.startswith("gpu"):
        device_spec = spec.split(":", 1)[1] if ":" in spec else "0"
        if device_spec in ("auto", "local", "local_rank", "rank"):
            device_id = _local_rank_for_decode()
            try:
                visible_count = int(torch.cuda.device_count())
            except Exception:
                visible_count = 0
            if visible_count > 0:
                device_id = device_id % visible_count
        else:
            try:
                device_id = int(device_spec)
            except Exception:
                device_id = 0
        return decord.gpu(device_id)
    return decord.cpu()


def _duplicate_processed_frame(frame):
    if hasattr(frame, "copy"):
        return frame.copy()
    if isinstance(frame, torch.Tensor):
        return frame.clone()
    return frame


def resolve_video_operator_impl(video_operator_impl=None) -> str:
    if video_operator_impl is None:
        video_operator_impl = os.environ.get("DATA_VIDEO_OPERATOR_IMPL", "default")
    return str(video_operator_impl).strip().lower()


def resolve_video_backend(video_backend=None) -> str:
    if video_backend is None:
        # [核心修改] 默认强制使用 decord，如果没装就会在 _DecordReaderAdapter 报错
        video_backend = os.environ.get("DATA_VIDEO_BACKEND", "decord")
    return str(video_backend).strip().lower()


def _decord_to_numpy(value):
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class _DecordReaderAdapter:
    def __init__(self, path: str):
        if decord is None:
            raise ImportError("decord is not installed but DATA_VIDEO_BACKEND=decord was requested.")
        ctx = _decord_ctx()
        self._reader = decord.VideoReader(path, ctx=ctx)
        self._fps = float(self._reader.get_avg_fps())

    def get_meta_data(self):
        return {"fps": self._fps}

    def count_frames(self):
        return len(self._reader)

    def get_data(self, frame_id: int):
        return _decord_to_numpy(self._reader[frame_id])

    def get_batch(self, frame_ids):
        return _decord_to_numpy(self._reader.get_batch(frame_ids))

    def __iter__(self):
        for frame in self._reader:
            yield _decord_to_numpy(frame)

    def close(self):
        return None


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, *args, **kwargs):
        data = None
        first = True
        for operator in self.operators:
            if first:
                data = operator(*args, **kwargs)
                first = False
            else:
                data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image

    def _frame_to_numpy(self, frame):
        if isinstance(frame, np.ndarray):
            return frame
        if isinstance(frame, torch.Tensor):
            tensor = frame.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            return tensor.numpy()
        if isinstance(frame, Image.Image):
            return np.array(frame)
        return np.array(frame)

    def _numpy_to_output(self, frame_np):
        output_mode = _video_output_mode()
        if output_mode == "tensor":
            return torch.from_numpy(frame_np).permute(2, 0, 1).contiguous()
        return Image.fromarray(frame_np)

    def _resize_numpy_frame(self, frame_np, target_height, target_width):
        if cv2 is None:
            image = Image.fromarray(frame_np)
            return np.array(self.crop_and_resize(image, target_height, target_width))
        src_height, src_width = frame_np.shape[:2]
        if src_height == target_height and src_width == target_width:
            return frame_np
        scale = max(target_width / src_width, target_height / src_height)
        resized_width = max(1, round(src_width * scale))
        resized_height = max(1, round(src_height * scale))
        resized = cv2.resize(frame_np, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        top = max(0, (resized_height - target_height) // 2)
        left = max(0, (resized_width - target_width) // 2)
        return resized[top:top + target_height, left:left + target_width]

    def process_array(self, frame_array):
        frame_np = self._frame_to_numpy(frame_array)
        target_height, target_width = self.get_height_width(Image.fromarray(frame_np))
        resized = self._resize_numpy_frame(frame_np, target_height, target_width)
        return self._numpy_to_output(resized)

    def process_array_batch(self, frame_arrays):
        if len(frame_arrays) == 0:
            return []

        sample = frame_arrays[0]
        sample_np = self._frame_to_numpy(sample)
        height, width = self.get_height_width(Image.fromarray(sample_np))

        resized_frames = []
        for frame in frame_arrays:
            frame_np = self._frame_to_numpy(frame)
            resized_np = self._resize_numpy_frame(frame_np, height, width)
            resized_frames.append(self._numpy_to_output(resized_np))
        return resized_frames


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class FrameSamplerByRateMixin:
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_rate=24, fix_frame_rate=False):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_rate = frame_rate
        self.fix_frame_rate = fix_frame_rate

    def get_reader(self, data: str):
        backend = resolve_video_backend()
        if backend == "decord":
            return _DecordReaderAdapter(data)
        return imageio.get_reader(data)

    def get_raw_video_info(self, reader, st_time=None, ed_time=None):
        meta_data = reader.get_meta_data()
        fps = meta_data.get("fps")
        fps = float(fps) if fps is not None else 0.0
        total_raw_frames = int(reader.count_frames())
        backend = resolve_video_backend()
        return fps, total_raw_frames, f"{backend}_count_frames"

    def get_available_num_frames(self, reader):
        if not self.fix_frame_rate:
            return reader.count_frames()
        meta_data = reader.get_meta_data()
        total_original_frames = int(reader.count_frames())
        duration = meta_data["duration"] if "duration" in meta_data else total_original_frames / meta_data['fps']
        total_available_frames = math.floor(duration * self.frame_rate)
        return int(total_available_frames)

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        total_frames = self.get_available_num_frames(reader)
        if int(total_frames) < num_frames:
            num_frames = total_frames
            num_frames = self.adjust_num_frames(num_frames)
        return num_frames

    def adjust_num_frames(self, num_frames: int) -> int:
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def map_single_frame_id(self, new_sequence_id: int, raw_frame_rate: float, total_raw_frames: int) -> int:
        if not self.fix_frame_rate:
            return new_sequence_id
        target_time_in_seconds = new_sequence_id / self.frame_rate
        raw_frame_index_float = target_time_in_seconds * raw_frame_rate
        frame_id = int(round(raw_frame_index_float))        
        frame_id = min(frame_id, total_raw_frames - 1)
        return frame_id


class LoadVideo(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def __call__(self, data: str):
        reader = self.get_reader(data)
        raw_frame_rate, total_raw_frames, _ = self.get_raw_video_info(reader)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, total_raw_frames)
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class LoadVideoCut(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        self.frame_processor = frame_processor
        self._profile_counter = 0
        self._profiled_steps = 0
        self._profile_totals = {}

    def _resolve_crop_pixels(self, crop_box, width, height):
        if crop_box is None:
            return None
        if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4:
            return None

        x1, y1, x2, y2 = crop_box
        try:
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        except (TypeError, ValueError):
            return None

        if width <= 0 or height <= 0:
            return None

        # Support normalized xyxy boxes from label json, and tolerate absolute boxes.
        if 0.0 <= min(x1, y1, x2, y2) and max(x1, y1, x2, y2) <= 1.0:
            left = math.floor(x1 * width)
            top = math.floor(y1 * height)
            right = math.ceil(x2 * width)
            bottom = math.ceil(y2 * height)
        else:
            left = math.floor(x1)
            top = math.floor(y1)
            right = math.ceil(x2)
            bottom = math.ceil(y2)

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))

        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    def _crop_frame(self, frame: Image.Image, crop_box=None):
        crop_pixels = self._resolve_crop_pixels(crop_box, *frame.size)
        if crop_pixels is None:
            return frame
        return frame.crop(crop_pixels)

    def _crop_array(self, frame, crop_box=None):
        height, width = frame.shape[:2]
        crop_pixels = self._resolve_crop_pixels(crop_box, width, height)
        if crop_pixels is None:
            return frame
        left, top, right, bottom = crop_pixels
        return frame[top:bottom, left:right]

    def _process_frame_array(self, frame, crop_box=None):
        frame = self._crop_array(frame, crop_box)
        if hasattr(self.frame_processor, "process_array"):
            return self.frame_processor.process_array(frame)
        image = Image.fromarray(frame)
        return self.frame_processor(image)

    def _process_frame_array_profiled(self, frame, crop_box=None):
        crop_start = time.perf_counter()
        frame = self._crop_array(frame, crop_box)
        crop_end = time.perf_counter()
        if hasattr(self.frame_processor, "process_array"):
            processed = self.frame_processor.process_array(frame)
            pil_end = crop_end
        else:
            image = Image.fromarray(frame)
            pil_end = time.perf_counter()
            processed = self.frame_processor(image)
        process_end = time.perf_counter()
        return processed, crop_end - crop_start, pil_end - crop_end, process_end - pil_end

    def _can_batch_process_frames(self):
        mode = _video_frame_process_mode()
        if mode in ("off", "disable", "disabled", "0", "false", "no"):
            return False
        return hasattr(self.frame_processor, "process_array_batch")

    def _process_frame_batch(self, frame_batch, crop_box=None):
        if len(frame_batch) == 0:
            return []
        crop_start = time.perf_counter()
        if crop_box is not None:
            cropped = [self._crop_array(frame, crop_box) for frame in frame_batch]
            frame_batch = cropped
        crop_end = time.perf_counter()
        processed = self.frame_processor.process_array_batch(frame_batch)
        process_end = time.perf_counter()
        return processed, crop_end - crop_start, 0.0, process_end - crop_end

    def __call__(self, data: str, st_time=None, ed_time=None, offset=None, crop_box=None):
        profile_this_step = _should_profile(self._profile_counter)
        open_start = time.perf_counter() if profile_this_step else None
        reader = self.get_reader(data)
        open_end = time.perf_counter() if profile_this_step else None
        meta_start = open_end if profile_this_step else None
        raw_frame_rate, total_raw_frames, meta_source = self.get_raw_video_info(reader, st_time, ed_time)
        meta_end = time.perf_counter() if profile_this_step else None
        frames = []

        offset = 0 if offset is None else int(round(float(offset)))
        st_frame = 0
        ed_frame = total_raw_frames
        if st_time is not None and ed_time is not None:
            st_frame = max(0, round(float(st_time) * raw_frame_rate))
            ed_frame = min(total_raw_frames, round(float(ed_time) * raw_frame_rate))

        # Negative offset means audio lags behind video, so drop video head frames.
        if offset < 0:
            st_frame = min(ed_frame, st_frame + abs(offset))

        available_raw_frames = max(0, ed_frame - st_frame)
        if available_raw_frames == 0:
            reader.close()
            return frames

        if self.fix_frame_rate:
            available_target_frames = math.floor(available_raw_frames / raw_frame_rate * self.frame_rate)
        else:
            available_target_frames = available_raw_frames
        num_frames = min(self.num_frames, int(available_target_frames))
        num_frames = self.adjust_num_frames(num_frames)

        decode_total = 0.0
        pil_total = 0.0
        crop_total = 0.0
        process_total = 0.0

        target_frame_ids = []
        for frame_id in range(num_frames):
            mapped_frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, available_raw_frames)
            target_frame_ids.append(st_frame + mapped_frame_id)
        requested_frames = len(target_frame_ids)
        target_counts = {}
        for frame_id in target_frame_ids:
            target_counts[frame_id] = target_counts.get(frame_id, 0) + 1

        workers = _video_process_workers()
        inflight_limit = _video_process_inflight()
        use_batch_decode = _video_decode_use_batch() and hasattr(reader, "get_batch") and len(target_counts) > 1
        use_batch_frame_process = self._can_batch_process_frames()
        prefer_full_batch_path = use_batch_decode and use_batch_frame_process

        if prefer_full_batch_path:
            unique_target_ids = list(target_counts.keys())
            decode_start = time.perf_counter() if profile_this_step else None
            batch_frames = reader.get_batch(unique_target_ids)
            decode_end = time.perf_counter() if profile_this_step else None
            if profile_this_step:
                decode_total += decode_end - decode_start
                processed_batch, crop_duration, pil_duration, process_duration = self._process_frame_batch(batch_frames, crop_box)
                crop_total += crop_duration
                pil_total += pil_duration
                process_total += process_duration
            else:
                processed_batch = self.frame_processor.process_array_batch(
                    [self._crop_array(frame, crop_box) for frame in batch_frames] if crop_box is not None else batch_frames
                )
            for idx, target_frame_id in enumerate(unique_target_ids):
                    processed = processed_batch[idx]
                    repeat_count = target_counts[target_frame_id]
                    for repeat_idx in range(repeat_count):
                        if repeat_idx < repeat_count - 1:
                            frames.append(_duplicate_processed_frame(processed))
                        else:
                            frames.append(processed)
        elif workers <= 1:
            unique_target_ids = list(target_counts.keys())
            batch_frames = None
            if use_batch_decode:
                decode_start = time.perf_counter() if profile_this_step else None
                batch_frames = reader.get_batch(unique_target_ids)
                decode_end = time.perf_counter() if profile_this_step else None
                if profile_this_step:
                    decode_total += decode_end - decode_start
            for idx, target_frame_id in enumerate(unique_target_ids):
                repeat_count = target_counts[target_frame_id]
                if batch_frames is None:
                    decode_start = time.perf_counter() if profile_this_step else None
                    frame = reader.get_data(target_frame_id)
                    decode_end = time.perf_counter() if profile_this_step else None
                    if profile_this_step:
                        decode_total += decode_end - decode_start
                else:
                    frame = batch_frames[idx]
                if profile_this_step:
                    processed, crop_duration, pil_duration, process_duration = self._process_frame_array_profiled(frame, crop_box)
                    crop_total += crop_duration
                    pil_total += pil_duration
                    process_total += process_duration
                else:
                    processed = self._process_frame_array(frame, crop_box)
                for repeat_idx in range(repeat_count):
                    if repeat_idx < repeat_count - 1:
                        frames.append(_duplicate_processed_frame(processed))
                    else:
                        frames.append(processed)
        else:
            pending = deque()

            def _drain_one():
                nonlocal crop_total, pil_total, process_total
                target_frame_id, future = pending.popleft()
                result = future.result()
                if profile_this_step:
                    processed, crop_duration, pil_duration, process_duration = result
                    crop_total += crop_duration
                    pil_total += pil_duration
                    process_total += process_duration
                else:
                    processed = result
                repeat_count = target_counts[target_frame_id]
                for repeat_idx in range(repeat_count):
                    if repeat_idx < repeat_count - 1:
                        frames.append(_duplicate_processed_frame(processed))
                    else:
                        frames.append(processed)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                unique_target_ids = list(target_counts.keys())
                if use_batch_decode:
                    decode_start = time.perf_counter() if profile_this_step else None
                    batch_frames = reader.get_batch(unique_target_ids)
                    decode_end = time.perf_counter() if profile_this_step else None
                    if profile_this_step:
                        decode_total += decode_end - decode_start
                    frame_iter = zip(unique_target_ids, batch_frames)
                else:
                    frame_iter = []
                    for target_frame_id in unique_target_ids:
                        decode_start = time.perf_counter() if profile_this_step else None
                        frame = reader.get_data(target_frame_id)
                        decode_end = time.perf_counter() if profile_this_step else None
                        if profile_this_step:
                            decode_total += decode_end - decode_start
                        frame_iter.append((target_frame_id, frame))

                for target_frame_id, frame in frame_iter:
                    frame = frame.copy() if hasattr(frame, "copy") else frame
                    future = executor.submit(
                        self._process_frame_array_profiled if profile_this_step else self._process_frame_array,
                        frame,
                        crop_box,
                    )
                    pending.append((target_frame_id, future))
                    if len(pending) >= inflight_limit:
                        _drain_one()

                while pending:
                    _drain_one()

        close_start = time.perf_counter() if profile_this_step else None
        reader.close()
        close_end = time.perf_counter() if profile_this_step else None
        if profile_this_step:
            stage_values = {
                "reader_open": open_end - open_start,
                "meta": meta_end - meta_start,
                "decode": decode_total,
                "pil": pil_total,
                "crop": crop_total,
                "frame_process": process_total,
                "close": close_end - close_start,
            }
            self._profile_counter += 1
            self._profiled_steps += 1
            _update_totals(self._profile_totals, stage_values)
            print(
                f"[OP][video] path={data} frames={num_frames} requested_frames={requested_frames} unique_targets={len(target_counts)} raw_span={st_frame}-{max(st_frame, ed_frame - 1)} meta_source={meta_source} "
                + " ".join(f"{key}={_format_ms(value)}" for key, value in stage_values.items())
            )
            interval = _operator_profile_interval()
            if interval > 0 and self._profiled_steps % interval == 0:
                print(f"[OP][video-summary] steps={self._profiled_steps} " + _summary_parts(self._profile_totals, self._profiled_steps))
        else:
            self._profile_counter += 1
        return frames


class LoadVideoCutSequential(LoadVideoCut):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        super().__init__(num_frames, time_division_factor, time_division_remainder, frame_processor, frame_rate, fix_frame_rate)

    def __call__(self, data: str, st_time=None, ed_time=None, offset=None, crop_box=None):
        profile_this_step = _should_profile(self._profile_counter)
        open_start = time.perf_counter() if profile_this_step else None
        reader = self.get_reader(data)
        open_end = time.perf_counter() if profile_this_step else None
        meta_start = open_end if profile_this_step else None
        raw_frame_rate, total_raw_frames, meta_source = self.get_raw_video_info(reader, st_time, ed_time)
        meta_end = time.perf_counter() if profile_this_step else None
        frames = []

        offset = 0 if offset is None else int(round(float(offset)))
        st_frame = 0
        ed_frame = total_raw_frames
        if st_time is not None and ed_time is not None:
            st_frame = max(0, round(float(st_time) * raw_frame_rate))
            ed_frame = min(total_raw_frames, round(float(ed_time) * raw_frame_rate))

        if offset < 0:
            st_frame = min(ed_frame, st_frame + abs(offset))

        available_raw_frames = max(0, ed_frame - st_frame)
        if available_raw_frames == 0:
            reader.close()
            self._profile_counter += 1
            return frames

        if self.fix_frame_rate:
            available_target_frames = math.floor(available_raw_frames / raw_frame_rate * self.frame_rate)
        else:
            available_target_frames = available_raw_frames
        num_frames = min(self.num_frames, int(available_target_frames))
        num_frames = self.adjust_num_frames(num_frames)
        if num_frames <= 0:
            reader.close()
            self._profile_counter += 1
            return frames

        target_frame_ids = []
        for frame_id in range(num_frames):
            mapped_frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, available_raw_frames)
            target_frame_ids.append(st_frame + mapped_frame_id)
        target_frame_ids = sorted(target_frame_ids)
        requested_frames = len(target_frame_ids)
        target_counts = {}
        for frame_id in target_frame_ids:
            target_counts[frame_id] = target_counts.get(frame_id, 0) + 1

        first_target = target_frame_ids[0]
        last_target = target_frame_ids[-1]

        decode_total = 0.0
        pil_total = 0.0
        crop_total = 0.0
        process_total = 0.0
        iter_total = 0.0

        iterator_start = time.perf_counter() if profile_this_step else None
        workers = _video_process_workers()
        inflight_limit = _video_process_inflight()
        use_batch_frame_process = self._can_batch_process_frames()
        if use_batch_frame_process:
            sequential_frames = []
            sequential_needs = []
            for raw_idx, frame in enumerate(reader):
                if raw_idx < first_target:
                    continue
                if raw_idx > last_target:
                    break
                needed = target_counts.get(raw_idx, 0)
                if needed <= 0:
                    continue
                sequential_frames.append(frame)
                sequential_needs.append(needed)
            if profile_this_step:
                processed_batch, crop_duration, pil_duration, process_duration = self._process_frame_batch(sequential_frames, crop_box)
                crop_total += crop_duration
                pil_total += pil_duration
                process_total += process_duration
            else:
                processed_batch = self.frame_processor.process_array_batch(
                    [self._crop_array(frame, crop_box) for frame in sequential_frames] if crop_box is not None else sequential_frames
                )
            for processed, needed in zip(processed_batch, sequential_needs):
                for repeat_idx in range(needed):
                    if repeat_idx < needed - 1:
                        frames.append(_duplicate_processed_frame(processed))
                    else:
                        frames.append(processed)
        elif workers <= 1:
            sequential_frames = []
            sequential_needs = []
            for raw_idx, frame in enumerate(reader):
                if raw_idx < first_target:
                    continue
                if raw_idx > last_target:
                    break
                needed = target_counts.get(raw_idx, 0)
                if needed <= 0:
                    continue
                sequential_frames.append(frame)
                sequential_needs.append(needed)
            for frame, needed in zip(sequential_frames, sequential_needs):
                if profile_this_step:
                    processed, crop_duration, pil_duration, process_duration = self._process_frame_array_profiled(frame, crop_box)
                    crop_total += crop_duration
                    pil_total += pil_duration
                    process_total += process_duration
                else:
                    processed = self._process_frame_array(frame, crop_box)

                for repeat_idx in range(needed):
                    if repeat_idx < needed - 1:
                        frames.append(_duplicate_processed_frame(processed))
                    else:
                        frames.append(processed)
        else:
            pending = deque()

            def _drain_one():
                nonlocal crop_total, pil_total, process_total
                needed, future = pending.popleft()
                result = future.result()
                if profile_this_step:
                    processed, crop_duration, pil_duration, process_duration = result
                    crop_total += crop_duration
                    pil_total += pil_duration
                    process_total += process_duration
                else:
                    processed = result
                for repeat_idx in range(needed):
                    if repeat_idx < needed - 1:
                        frames.append(_duplicate_processed_frame(processed))
                    else:
                        frames.append(processed)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for raw_idx, frame in enumerate(reader):
                    if raw_idx < first_target:
                        continue
                    if raw_idx > last_target:
                        break
                    needed = target_counts.get(raw_idx, 0)
                    if needed <= 0:
                        continue
                    frame = frame.copy() if hasattr(frame, "copy") else frame
                    future = executor.submit(
                        self._process_frame_array_profiled if profile_this_step else self._process_frame_array,
                        frame,
                        crop_box,
                    )
                    pending.append((needed, future))
                    if len(pending) >= inflight_limit:
                        _drain_one()

                while pending:
                    _drain_one()

        iter_end = time.perf_counter() if profile_this_step else None
        close_start = time.perf_counter() if profile_this_step else None
        reader.close()
        close_end = time.perf_counter() if profile_this_step else None

        if profile_this_step:
            iter_total = iter_end - iterator_start
            decode_total = max(0.0, iter_total - pil_total - crop_total - process_total)
            stage_values = {
                "reader_open": open_end - open_start,
                "meta": meta_end - meta_start,
                "iter_total": iter_total,
                "decode": decode_total,
                "pil": pil_total,
                "crop": crop_total,
                "frame_process": process_total,
                "close": close_end - close_start,
            }
            self._profile_counter += 1
            self._profiled_steps += 1
            _update_totals(self._profile_totals, stage_values)
            print(
                f"[OP][video-seq] path={data} frames={len(frames)} requested_frames={requested_frames} unique_targets={len(target_counts)} raw_span={first_target}-{last_target} meta_source={meta_source} "
                + " ".join(f"{key}={_format_ms(value)}" for key, value in stage_values.items())
            )
            interval = _operator_profile_interval()
            if interval > 0 and self._profiled_steps % interval == 0:
                print(f"[OP][video-seq-summary] steps={self._profiled_steps} " + _summary_parts(self._profile_totals, self._profiled_steps))
        else:
            self._profile_counter += 1
        return frames


def build_video_cut_operator(
    *,
    num_frames=81,
    time_division_factor=4,
    time_division_remainder=1,
    frame_processor=lambda x: x,
    frame_rate=24,
    fix_frame_rate=False,
    video_operator_impl=None,
):
    impl = resolve_video_operator_impl(video_operator_impl)
    operator_cls = LoadVideoCut
    if impl in ("sequential", "seq", "fast"):
        operator_cls = LoadVideoCutSequential
    elif impl not in ("default", "legacy", "random_access", "random-access"):
        print(f"[DATASET] unknown DATA_VIDEO_OPERATOR_IMPL={impl}, fallback to {LoadVideoCut.__name__}")
        impl = "default"
    print(f"[DATASET] video operator impl={impl} class={operator_cls.__name__}")
    return operator_cls(
        num_frames,
        time_division_factor,
        time_division_remainder,
        frame_processor=frame_processor,
        frame_rate=frame_rate,
        fix_frame_rate=fix_frame_rate,
    )


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio

import torchaudio.functional as F
import whisper
class LoadAudioWithTorchaudio(DataProcessingOperator, FrameSamplerByRateMixin):

    def __init__(self, num_frames=121, time_division_factor=8, time_division_remainder=1, frame_rate=24, fix_frame_rate=True):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)

    def __call__(self, data: str):
        target_sr = 48000
        reader = self.get_reader(data)
        num_frames = self.get_num_frames(reader)
        duration = num_frames / self.frame_rate
        # waveform, sample_rate = torchaudio.load(data)
        # target_samples = int(duration * sample_rate)
        # current_samples = waveform.shape[-1]
        # if current_samples > target_samples:
        #     waveform = waveform[..., :target_samples]
        # elif current_samples < target_samples:
        #     padding = target_samples - current_samples
        #     waveform = torch.nn.functional.pad(waveform, (0, padding))

        # #  定义目标采样率
        # if sample_rate != target_sr:
        #     waveform = F.resample(waveform, sample_rate, target_sr)
        #     sample_rate = target_sr
        # if waveform.shape[0] == 1:
        #     waveform = waveform.repeat(1, 2, 1)

        audio_full = whisper.load_audio(data, sr=target_sr)
        audio_full = audio_full[: min(int(duration * target_sr), audio_full.shape[0])]
        audio = torch.from_numpy(audio_full)
        waveform = audio.unsqueeze(0).expand(2, -1)
        
        return waveform, target_sr



class LoadAudioCutWithTorchaudio(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=121, time_division_factor=8, time_division_remainder=1, frame_rate=24, fix_frame_rate=True):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        self.target_sr = 48000 #51200
        self._profile_counter = 0
        self._profiled_steps = 0
        self._profile_totals = {}

    def __call__(self, data: str, st_time=None, ed_time=None, offset=None):
        profile_this_step = _should_profile(self._profile_counter)
        meta_start = time.perf_counter() if profile_this_step else None
        reader = self.get_reader(data)
        raw_frame_rate, total_raw_frames, meta_source = self.get_raw_video_info(reader, st_time, ed_time)
        reader.close()
        meta_end = time.perf_counter() if profile_this_step else None

        audio_load_start = meta_end if profile_this_step else None
        waveform = torch.from_numpy(whisper.load_audio(data, sr=self.target_sr)).unsqueeze(0).expand(2, -1)
        audio_load_end = time.perf_counter() if profile_this_step else None
        current_samples = waveform.shape[-1]
        offset = 0 if offset is None else int(round(float(offset)))

        raw_start_frame = 0
        raw_end_frame = total_raw_frames
        if st_time is not None and ed_time is not None:
            raw_start_frame = max(0, round(float(st_time) * raw_frame_rate))
            raw_end_frame = min(total_raw_frames, round(float(ed_time) * raw_frame_rate))

        raw_start_frame = min(raw_start_frame, raw_end_frame)
        audio_start_frame = raw_start_frame + max(0, offset)
        available_raw_frames = max(0, raw_end_frame - raw_start_frame - max(0, offset) - max(0, -offset))
        if available_raw_frames <= 0:
            empty = waveform[..., :0]
            return empty, self.target_sr

        if self.fix_frame_rate:
            available_target_frames = math.floor(available_raw_frames / raw_frame_rate * self.frame_rate)
        else:
            available_target_frames = available_raw_frames
        num_frames = min(self.num_frames, int(available_target_frames))
        num_frames = self.adjust_num_frames(num_frames)
        target_duration = 0.0 if num_frames <= 0 else num_frames / self.frame_rate

        new_st_samples = math.ceil(audio_start_frame / raw_frame_rate * self.target_sr)
        new_ed_samples = new_st_samples + round(target_duration * self.target_sr)

        slice_start = time.perf_counter() if profile_this_step else None
        if new_st_samples >= 0 and new_ed_samples <= current_samples:
            waveform = waveform[..., new_st_samples:new_ed_samples]
        elif new_st_samples < 0 and new_ed_samples <= current_samples:
            waveform = torch.nn.functional.pad(waveform[..., :new_ed_samples], (-new_st_samples, 0))
        elif new_st_samples >= 0 and new_ed_samples > current_samples:
            waveform = torch.nn.functional.pad(waveform[..., new_st_samples:], (0, new_ed_samples - current_samples))
        else:
            waveform = torch.nn.functional.pad(waveform, (-new_st_samples, new_ed_samples - current_samples))
        slice_end = time.perf_counter() if profile_this_step else None

        if profile_this_step:
            stage_values = {
                "meta": meta_end - meta_start,
                "load_audio": audio_load_end - audio_load_start,
                "slice_pad": slice_end - slice_start,
            }
            self._profile_counter += 1
            self._profiled_steps += 1
            _update_totals(self._profile_totals, stage_values)
            print(
                f"[OP][audio] path={data} samples={waveform.shape[-1]} meta_source={meta_source} "
                + " ".join(f"{key}={_format_ms(value)}" for key, value in stage_values.items())
            )
            interval = _operator_profile_interval()
            if interval > 0 and self._profiled_steps % interval == 0:
                print(f"[OP][audio-summary] steps={self._profiled_steps} " + _summary_parts(self._profile_totals, self._profiled_steps))
        else:
            self._profile_counter += 1

        return waveform, self.target_sr
    
class LoadMagiPromptFile(DataProcessingOperator):
    def __init__(self, default_prompt="The preson is talking.",drop_rate=0.0):
        self.default_prompt = default_prompt
        self.drop_rate = drop_rate

    def __call__(self, data: str):
        with open(data, "r", encoding="utf-8") as f:
            payload = json.load(f)

        result = {}
        text_with_speech = (
            payload.get("audio_video_description")
            or payload.get("audiovisual_caption")
            or payload.get("video_caption")
            or self.default_prompt
        )
        if random.random() < self.drop_rate:
            text_with_speech = re.sub(r"\[time_range:[^\]]*\]", "", text_with_speech)
        speech_content = payload.get("speech_content")
        if isinstance(speech_content, dict):
            for placeholder, speech_info in speech_content.items():
                if isinstance(speech_info, dict):
                    speech_text = speech_info.get("content", "")
                else:
                    speech_text = str(speech_info)
                speech_text = speech_text.strip()
                if len(speech_text) == 0:
                    continue
                text_with_speech = text_with_speech.replace(f"[{placeholder}]", f"“{speech_text}”")

        audio_content = payload.get("audio_content")
        if isinstance(audio_content, dict):
            match_rule = r"\[.*?\]\[.*?\]:\s*\"?([^\"]+)\"?"
            for placeholder, raw_text in audio_content.items():
                if not isinstance(raw_text, str):
                    continue
                match = re.search(match_rule, raw_text)
                if match is None:
                    continue
                speech_text = match.group(1).strip()
                if len(speech_text) == 0:
                    continue
                text_with_speech = text_with_speech.replace(placeholder, f"“{speech_text}”")

        result["prompt"] = text_with_speech

        return result
