# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming video frame DataSource using decord.

Produces video frames as a DuckDB DataSource with ~10MB streaming partitions.
Each video file is an independent DataSourceTask that decodes frames via decord
and yields them as Arrow fixed-shape tensor columns.

Usage::

    from duckdb.datasource.video_reader import VideoFrameSource

    source = VideoFrameSource(["video1.avi", "video2.avi"], height=640, width=480)
    rel = con.read_datasource(source)
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import islice, repeat
from types import ModuleType
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from duckdb.datasource import DataSource, DataSourceTask


def _import_video_dependency(module_name: str, package_name: str) -> ModuleType:
    """Import an optional video dependency, reporting the extra to install on failure."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = "Please `pip install 'vane-ai[video]'` to use the video data source."
        if package_name == "decord":
            hint += " The video extra installs decord on Linux x86-64, Vane's currently supported video platform."
        raise ImportError(f"The video data source requires the '{package_name}' package. {hint}") from exc


# The generic datasource keeps its historical 10 MiB default. The aligned
# video benchmark explicitly supplies Ray Data's 128 MiB soft block target.
_DEFAULT_MAX_PARTITION_BYTES = int(os.environ.get("VANE_VIDEO_MAX_PARTITION_BYTES", str(10 * 1024 * 1024)))
_DEFAULT_VIDEO_SOURCE_UDF_MEMORY_BYTES = 512 * 1024**2
_DEFAULT_MAX_REMOTE_VIDEO_BYTES = int(os.environ.get("VANE_VIDEO_MAX_REMOTE_BYTES", str(8 * 1024**3)))
_DEFAULT_MAX_SOURCE_FRAME_BYTES = int(os.environ.get("VANE_VIDEO_MAX_SOURCE_FRAME_BYTES", str(128 * 1024**2)))
_REMOTE_VIDEO_READ_CHUNK_BYTES = 8 * 1024**2
_VIDEO_DECODER_MEMORY_HEADROOM_BYTES = 128 * 1024**2
_DEFAULT_MAX_SOURCE_PATH_BYTES = 4096
_SOURCE_ID_BYTES = hashlib.sha256().digest_size * 2
_ARROW_STRING_OFFSET_BYTES = np.dtype(np.int32).itemsize
_FRAME_INDEX_BYTES = np.dtype(np.int64).itemsize
_ARROW_STRING_DATA_MAX_BYTES = int(np.iinfo(np.int32).max)
# Ray Data read/map tasks use one CPU by default. Keep the decode/resize
# worker at that width unless the caller explicitly requests otherwise.
_VIDEO_RESIZE_THREADS = max(1, int(os.environ.get("VANE_VIDEO_RESIZE_THREADS", "1")))
# A fixed single Decord thread keeps its native buffering inside the declared
# source-task working-set headroom.
_VIDEO_DECODE_THREADS = 1
# Decord 0.6 exposes FFmpeg's padded RGB row as extra channels when the row
# width is not 32-byte aligned. RGB rows are aligned when pixels are a
# multiple of 32, after which Pillow applies the exact requested width.
_VIDEO_DECODE_WIDTH_ALIGNMENT = 32
_VIDEO_ERROR_MODES = frozenset({"raise", "skip"})
_LOGGER = logging.getLogger(__name__)


class VideoReadError(RuntimeError):
    """A failure associated with one input video."""

    def __init__(self, video_path: str, message: str):
        self.video_path = str(video_path)
        self.message = str(message)
        super().__init__(self.video_path, self.message)

    def __str__(self) -> str:
        return f"Failed to read video {self.video_path!r}: {self.message}"


class RemoteVideoTooLargeError(VideoReadError):
    """A remote video exceeded its configured per-file disk bound."""


class SourceFrameTooLargeError(VideoReadError):
    """A decoded source frame exceeded its configured working-set bound."""


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _read_int_env(name: str, default: int, minimum: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        result = default
    else:
        result = int(value)
    if minimum is not None:
        result = max(minimum, result)
    return result


def _read_optional_text_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        text = value.strip()
        if text:
            return text
    return None


def _read_optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _read_optional_positive_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _read_positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        result = int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}") from None
    if result < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return result


_VIDEO_READER_TIMING_LOG = _read_bool_env(
    "VANE_VIDEO_READER_TIMING_LOG",
    _read_bool_env("VIDEO_READER_TIMING_LOG", _read_bool_env("VIDEO_UDF_TIMING_LOG", False)),
)
_VIDEO_READER_TIMING_SAMPLE_RATE = _read_int_env("VANE_VIDEO_READER_TIMING_SAMPLE_RATE", 1, minimum=1)
_VIDEO_READER_TIMING_LOG_PATH = _read_optional_text_env(
    ("VANE_VIDEO_READER_TIMING_LOG_PATH", "VIDEO_UDF_TIMING_LOG_PATH", "UDF_TIMING_LOG_PATH")
)
_VIDEO_READER_TIMING_STDOUT = _read_bool_env("VANE_VIDEO_READER_TIMING_STDOUT", True)
_VIDEO_READER_RUN_ID = os.environ.get("VIDEO_BENCHMARK_RUN_ID", "-").strip() or "-"
_VIDEO_READER_TIMING_CALLS = 0
_VIDEO_READER_TIMING_LOCK = threading.Lock()


def _format_seconds_and_ms(name: str, seconds: float) -> str:
    value = max(0.0, float(seconds))
    return f"{name}_s={value:.6f} {name}_ms={value * 1000.0:.3f}"


def _format_reader_timing_fields(*, total_s: float, rows_per_s: float, **stage_seconds: float) -> str:
    fields = [
        _format_seconds_and_ms(name[:-2] if name.endswith("_s") else name, seconds)
        for name, seconds in stage_seconds.items()
    ]
    fields.append(_format_seconds_and_ms("total", total_s))
    fields.append(f"rows_per_s={float(rows_per_s):.2f}")
    return " ".join(fields)


def _next_reader_timing_call() -> int:
    global _VIDEO_READER_TIMING_CALLS

    with _VIDEO_READER_TIMING_LOCK:
        _VIDEO_READER_TIMING_CALLS += 1
        return _VIDEO_READER_TIMING_CALLS


def _emit_reader_timing_line(line: str) -> None:
    if _VIDEO_READER_TIMING_STDOUT:
        print(line, flush=True)
    if not _VIDEO_READER_TIMING_LOG_PATH:
        return
    try:
        log_path = os.path.abspath(os.path.expanduser(_VIDEO_READER_TIMING_LOG_PATH))
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
    except Exception as exc:
        print(
            "[vane_video][reader_timing_log_error] "
            f"run_id={_VIDEO_READER_RUN_ID} pid={os.getpid()} path={_VIDEO_READER_TIMING_LOG_PATH!r} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )


def _emit_reader_timing(
    *,
    video_path: str,
    call: int,
    rows: int,
    total_s: float,
    **stage_seconds: float,
) -> None:
    if not _VIDEO_READER_TIMING_LOG:
        return
    if call % _VIDEO_READER_TIMING_SAMPLE_RATE != 0:
        return
    rows_per_s = rows / total_s if total_s > 0 else 0.0
    timing_fields = _format_reader_timing_fields(total_s=total_s, rows_per_s=rows_per_s, **stage_seconds)
    _emit_reader_timing_line(
        "[vane_video][reader_timing] "
        f"run_id={_VIDEO_READER_RUN_ID} pid={os.getpid()} call={call} "
        f"video={os.path.basename(video_path)} rows={rows} {timing_fields}"
    )


# Limit concurrent video decodes within a single process.
_MAX_CONCURRENT_DECODES = _read_positive_int_env("VANE_MAX_CONCURRENT_DECODES", 1)
_decode_semaphore = threading.Semaphore(_MAX_CONCURRENT_DECODES)

# Memory-based backpressure: block NEW decode tasks (not mid-decode!) when
# system memory exceeds this percentage.  Only checked at task boundaries
# to avoid deadlocking the push-based pipeline.
# Default 80% → ~75GB on a 94GB machine (leaves room for Ray + torch).
_MEM_HIGH_WATERMARK = float(os.environ.get("VANE_DECODE_MEM_HIGH_PCT", "80"))
_MEM_LOW_WATERMARK = float(os.environ.get("VANE_DECODE_MEM_LOW_PCT", "70"))
_MEM_CHECK_INTERVAL = 2.0  # seconds between memory checks when blocked
try:
    _MEM_MIN_AVAILABLE_MB = max(0, int(os.environ.get("VANE_DECODE_MIN_AVAILABLE_MB", "4096")))
except Exception:
    _MEM_MIN_AVAILABLE_MB = 4096
_MEM_MIN_AVAILABLE_BYTES = _MEM_MIN_AVAILABLE_MB * 1024 * 1024


def _wait_for_memory() -> None:
    """Block until system memory usage drops below the low watermark.

    IMPORTANT: Only call at task START (before a new video). Never call
    mid-decode or mid-yield — that deadlocks the DuckDB push pipeline.
    """
    psutil = _import_video_dependency("psutil", "psutil")

    def has_capacity(mem: Any) -> bool:
        if _MEM_MIN_AVAILABLE_BYTES > 0 and mem.available >= _MEM_MIN_AVAILABLE_BYTES:
            return True
        return mem.percent < _MEM_HIGH_WATERMARK

    def has_recovered(mem: Any) -> bool:
        if _MEM_MIN_AVAILABLE_BYTES > 0 and mem.available >= _MEM_MIN_AVAILABLE_BYTES:
            return True
        return mem.percent < _MEM_LOW_WATERMARK

    mem = psutil.virtual_memory()
    if has_capacity(mem):
        return
    while True:
        time.sleep(_MEM_CHECK_INTERVAL)
        mem = psutil.virtual_memory()
        if has_recovered(mem):
            return


def _s3_filesystem_kwargs() -> dict[str, str | bool]:
    """Build explicit S3 overrides without bypassing the AWS SDK defaults."""
    from urllib.parse import urlparse

    kwargs: dict[str, str | bool] = {}

    endpoint_url = _read_optional_text_env(("AWS_ENDPOINT_URL",))
    if endpoint_url is not None:
        # Treat a scheme-less value as an authority so paths are discarded
        # consistently from endpoint_override.
        parse_target = endpoint_url
        if "://" not in parse_target and not parse_target.startswith("//"):
            parse_target = f"//{parse_target}"
        parsed = urlparse(parse_target)
        endpoint = parsed.netloc or parsed.path.rstrip("/")
        if parsed.scheme:
            kwargs["scheme"] = parsed.scheme
        kwargs["endpoint_override"] = endpoint

    region = _read_optional_text_env(("AWS_REGION", "AWS_DEFAULT_REGION"))
    if region is not None:
        kwargs["region"] = region

    if _read_bool_env("S3FS_ANON", False):
        kwargs["anonymous"] = True

    return kwargs


def _video_source_identity(video_path: str) -> tuple[str, str]:
    """Return a stable provenance path and collision-resistant source ID."""
    if video_path.startswith("s3://"):
        object_path = video_path[len("s3://") :]
        bucket, separator, key = object_path.partition("/")
        if not bucket or not separator or not key:
            raise ValueError(f"invalid S3 video path: {video_path!r}")
        canonical_path = f"s3://{bucket.lower()}/{key}"
    else:
        # Resolve components in filesystem order.  Lexically collapsing
        # ``link/..`` before following ``link`` can identify a different file
        # from the one the decoder opens.
        canonical_path = os.path.realpath(video_path)
    source_id = hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()
    return canonical_path, source_id


def _video_source_path_bytes(video_path: str) -> int:
    canonical_path, _ = _video_source_identity(video_path)
    return len(canonical_path.encode("utf-8"))


def _max_video_source_path_bytes(video_paths: list[str]) -> int:
    return max((_video_source_path_bytes(path) for path in video_paths), default=0)


def _safe_video_suffix(video_path: str) -> str:
    suffix = os.path.splitext(video_path)[1]
    if len(suffix) > 16 or not suffix[1:].isalnum():
        return ""
    return suffix


def _s3_filesystem() -> Any:
    import pyarrow.fs as pa_fs

    return pa_fs.S3FileSystem(**_s3_filesystem_kwargs())


@contextmanager
def _materialize_video_path(
    video_path: str,
    *,
    max_remote_video_bytes: int,
    remote_temp_dir: str | None,
) -> Iterator[str]:
    """Yield a local decoder path while keeping remote heap use constant."""
    if not video_path.startswith("s3://"):
        yield video_path
        return
    if max_remote_video_bytes <= 0:
        raise ValueError("max_remote_video_bytes must be positive")

    canonical_path, _ = _video_source_identity(video_path)
    object_path = canonical_path[len("s3://") :]
    fs = _s3_filesystem()
    file_info = fs.get_file_info(object_path)
    object_size_value = file_info.size
    object_size = None if object_size_value is None else int(object_size_value)
    if object_size is not None and object_size >= 0 and object_size > max_remote_video_bytes:
        raise RemoteVideoTooLargeError(
            canonical_path,
            f"remote object size {object_size} exceeds limit {max_remote_video_bytes}",
        )

    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="vane-video-",
        suffix=_safe_video_suffix(object_path),
        dir=remote_temp_dir,
        delete=False,
    )
    local_path = temporary.name
    try:
        with temporary, fs.open_input_file(object_path) as source:
            copied_bytes = 0
            while True:
                chunk = source.read(_REMOTE_VIDEO_READ_CHUNK_BYTES)
                if not chunk:
                    break
                copied_bytes += len(chunk)
                if copied_bytes > max_remote_video_bytes:
                    raise RemoteVideoTooLargeError(
                        canonical_path,
                        f"remote object exceeded limit {max_remote_video_bytes} while downloading",
                    )
                temporary.write(chunk)
        yield local_path
    finally:
        try:
            os.unlink(local_path)
        except FileNotFoundError:
            pass


def _build_frame_array(frames: npt.ArrayLike) -> pa.ExtensionArray:
    frame_array: npt.NDArray[np.uint8] = np.ascontiguousarray(frames, dtype=np.uint8)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(f"expected RGB frame batch with shape (N, H, W, 3), got {frame_array.shape!r}")
    return pa.FixedShapeTensorArray.from_numpy_ndarray(frame_array)


def _int64_array(values: npt.ArrayLike) -> pa.Array:
    array = np.ascontiguousarray(values, dtype=np.int64)
    return pa.Array.from_buffers(pa.int64(), len(array), [None, pa.py_buffer(array)])


def _constant_string_array(value: str, count: int) -> pa.Array:
    if count < 0:
        raise ValueError("constant string array count must be non-negative")
    encoded = value.encode("utf-8")
    data_bytes = len(encoded) * count
    if count > _ARROW_STRING_DATA_MAX_BYTES or data_bytes > _ARROW_STRING_DATA_MAX_BYTES:
        raise ValueError(
            f"constant string array requires {data_bytes} data bytes, "
            f"exceeding Arrow's {_ARROW_STRING_DATA_MAX_BYTES}-byte UTF-8 offset limit"
        )
    offsets: npt.NDArray[np.int32] = np.arange(count + 1, dtype=np.int32) * len(encoded)
    data = encoded * count
    return pa.Array.from_buffers(pa.string(), count, [None, pa.py_buffer(offsets), pa.py_buffer(data)])


def _open_decord_reader(video_path: str, *, width: int, height: int) -> Any:
    VideoReader = _import_video_dependency("decord", "decord").VideoReader
    return VideoReader(
        video_path,
        width=int(width),
        height=int(height),
        num_threads=_VIDEO_DECODE_THREADS,
    )


def _video_decoder_dimensions(*, width: int, height: int) -> tuple[int, int]:
    decoder_width = ((int(width) + _VIDEO_DECODE_WIDTH_ALIGNMENT - 1) // _VIDEO_DECODE_WIDTH_ALIGNMENT) * (
        _VIDEO_DECODE_WIDTH_ALIGNMENT
    )
    return decoder_width, int(height)


def _resize_rgb_frame(
    frame: npt.NDArray[np.uint8],
    width: int,
    height: int,
) -> npt.NDArray[np.uint8]:
    pil_image_module = _import_video_dependency("PIL.Image", "pillow")
    pil_image = pil_image_module.fromarray(frame)
    return np.array(pil_image.resize((width, height)))


def _resize_frame_batch(
    frames: list[npt.NDArray[np.uint8]],
    *,
    width: int,
    height: int,
    executor: ThreadPoolExecutor | None = None,
) -> list[npt.NDArray[np.uint8]]:
    if not frames:
        return []
    if _VIDEO_RESIZE_THREADS <= 1 or len(frames) == 1:
        return [_resize_rgb_frame(frame, width, height) for frame in frames]
    if executor is not None:
        return list(executor.map(_resize_rgb_frame, frames, repeat(width), repeat(height)))
    with ThreadPoolExecutor(max_workers=_VIDEO_RESIZE_THREADS) as executor:
        return list(executor.map(_resize_rgb_frame, frames, repeat(width), repeat(height)))


def _source_frame_bytes_from_shape(frame: Any) -> int | None:
    shape = getattr(frame, "shape", None)
    if shape is None:
        return None
    frame_bytes = 1
    for dimension in shape:
        frame_bytes *= int(dimension)
    return frame_bytes


def _decode_video_batches(
    video_path: str,
    *,
    height: int,
    width: int,
    max_partition_bytes: int,
    max_frames: int | None = None,
    max_remote_video_bytes: int = _DEFAULT_MAX_REMOTE_VIDEO_BYTES,
    max_source_frame_bytes: int = _DEFAULT_MAX_SOURCE_FRAME_BYTES,
    max_source_path_bytes: int | None = None,
    remote_temp_dir: str | None = None,
) -> Iterator[pa.RecordBatch]:
    if max_frames is not None and max_frames <= 0:
        return
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if max_remote_video_bytes <= 0:
        raise ValueError("max_remote_video_bytes must be positive")
    if max_source_frame_bytes <= 0:
        raise ValueError("max_source_frame_bytes must be positive")

    source_path, source_id = _video_source_identity(video_path)
    source_path_bytes = len(source_path.encode("utf-8"))
    if max_source_path_bytes is None:
        max_source_path_bytes = source_path_bytes
    else:
        max_source_path_bytes = int(max_source_path_bytes)
        if max_source_path_bytes < source_path_bytes:
            raise ValueError(
                f"max_source_path_bytes {max_source_path_bytes} is smaller than canonical path size {source_path_bytes}"
            )

    decoder_width, decoder_height = _video_decoder_dimensions(width=width, height=height)
    decoded_frame_bytes = decoder_height * decoder_width * 3
    if decoded_frame_bytes > max_source_frame_bytes:
        raise SourceFrameTooLargeError(
            source_path,
            f"bounded decoder frame size {decoded_frame_bytes} exceeds limit {max_source_frame_bytes}",
        )

    batch_size = _video_source_udf_output_batch_size(
        height,
        width,
        max_partition_bytes,
        max_source_path_bytes=max_source_path_bytes,
    )
    resized: npt.NDArray[np.uint8] = np.empty((batch_size, height, width, 3), dtype=np.uint8)
    pending_frames: list[npt.NDArray[np.uint8]] = []
    pending_indices: list[int] = []
    indices: npt.NDArray[np.int64] = np.empty(batch_size, dtype=np.int64)
    count = 0
    decode_s = 0.0
    resize_s = 0.0
    open_s = 0.0
    batch_start = time.perf_counter()
    resize_executor: ThreadPoolExecutor | None = None
    resize_window_size = max(1, min(_VIDEO_RESIZE_THREADS, batch_size))

    def resize_pending_frames() -> None:
        nonlocal count, pending_frames, pending_indices, resize_s
        if not pending_frames:
            return

        resize_start = time.perf_counter()
        resized_frames = _resize_frame_batch(
            pending_frames,
            width=width,
            height=height,
            executor=resize_executor,
        )
        resize_s += time.perf_counter() - resize_start
        if len(resized_frames) != len(pending_indices):
            raise RuntimeError("video resize returned a different number of frames")

        for frame_index, resized_frame in zip(pending_indices, resized_frames, strict=True):
            resized[count] = resized_frame
            indices[count] = frame_index
            count += 1
        pending_frames = []
        pending_indices = []

    def emit_resized_batch(*, allocate_next: bool) -> Iterator[pa.RecordBatch]:
        nonlocal resized, count, indices, decode_s, resize_s, batch_start, open_s
        current_count = count
        flush_start = time.perf_counter()
        batch = _flush_frame_batch(source_id, source_path, resized, indices, current_count)
        flush_s = time.perf_counter() - flush_start
        total_s = time.perf_counter() - batch_start
        _emit_reader_timing(
            video_path=video_path,
            call=_next_reader_timing_call(),
            rows=current_count,
            total_s=total_s,
            open_s=open_s,
            decode_s=decode_s,
            resize_s=resize_s,
            flush_s=flush_s,
        )
        yield batch
        del batch
        if allocate_next:
            # The emitted Arrow batch owns the old zero-copy NumPy buffer.
            resized = np.empty((batch_size, height, width, 3), dtype=np.uint8)
            indices = np.empty(batch_size, dtype=np.int64)
        count = 0
        decode_s = 0.0
        resize_s = 0.0
        open_s = 0.0
        batch_start = time.perf_counter()

    open_start = time.perf_counter()
    with _materialize_video_path(
        source_path,
        max_remote_video_bytes=max_remote_video_bytes,
        remote_temp_dir=remote_temp_dir,
    ) as decoder_path:
        # Bound Decord's materialized frame shape at reader construction rather
        # than discovering an oversized frame after iteration begins. The
        # aligned width also avoids Decord exposing FFmpeg row padding as
        # additional channels for narrow RGB output.
        reader = _open_decord_reader(
            decoder_path,
            width=decoder_width,
            height=decoder_height,
        )
        open_s = time.perf_counter() - open_start
        if _VIDEO_RESIZE_THREADS > 1:
            resize_executor = ThreadPoolExecutor(max_workers=_VIDEO_RESIZE_THREADS)
        try:
            frames = reader if max_frames is None else islice(reader, max_frames)
            for frame_idx, frame in enumerate(frames):
                shape_frame_bytes = _source_frame_bytes_from_shape(frame)
                if shape_frame_bytes is not None and shape_frame_bytes > max_source_frame_bytes:
                    raise SourceFrameTooLargeError(
                        source_path,
                        f"source frame size {shape_frame_bytes} exceeds limit {max_source_frame_bytes}",
                    )
                decode_start = time.perf_counter()
                array = frame.asnumpy()
                decode_s += time.perf_counter() - decode_start
                if array.nbytes > max_source_frame_bytes:
                    raise SourceFrameTooLargeError(
                        source_path,
                        f"source frame size {array.nbytes} exceeds limit {max_source_frame_bytes}",
                    )
                pending_frames.append(array)
                pending_indices.append(frame_idx)
                del array
                del frame

                remaining_batch_rows = batch_size - count
                pending_limit = min(resize_window_size, remaining_batch_rows)
                if len(pending_frames) >= pending_limit:
                    resize_pending_frames()
                if count >= batch_size:
                    yield from emit_resized_batch(allocate_next=True)

            resize_pending_frames()
            if count > 0:
                yield from emit_resized_batch(allocate_next=False)
        finally:
            if resize_executor is not None:
                resize_executor.shutdown()
            del reader


def _flush_frame_batch(
    source_id: str,
    video_path: str,
    resized: npt.NDArray[np.uint8],
    indices: npt.ArrayLike,
    count: int,
) -> pa.RecordBatch:
    frame_indices = np.ascontiguousarray(indices, dtype=np.int64)
    if frame_indices.ndim != 1 or len(frame_indices) < count:
        raise ValueError("frame indices must be a one-dimensional array covering every frame")
    if count < len(frame_indices):
        frame_indices = frame_indices[:count].copy()
    else:
        frame_indices = frame_indices[:count]
    frame_values = resized[:count]
    if count < len(resized):
        # Arrow retains the NumPy owner of a zero-copy slice. Compact a short
        # tail so coalescing tails from many files cannot pin one full output
        # allocation per file.
        frame_values = frame_values.copy()
    frame_arr = _build_frame_array(frame_values)
    return pa.record_batch(
        {
            "source_id": _constant_string_array(source_id, count),
            "video_path": _constant_string_array(video_path, count),
            "frame_index": _int64_array(frame_indices),
            "frame": frame_arr,
        }
    )


def _validate_video_error_mode(on_error: str) -> str:
    if on_error not in _VIDEO_ERROR_MODES:
        choices = ", ".join(sorted(_VIDEO_ERROR_MODES))
        raise ValueError(f"on_error must be one of: {choices}; got {on_error!r}")
    return on_error


def _is_decord_error(exc: Exception) -> bool:
    """Return whether *exc* is one of Decord's direct Exception subclasses."""
    try:
        decord_base = importlib.import_module("decord._ffi.base")
    except ImportError:
        return False
    return any(
        isinstance(error_type, type) and isinstance(exc, error_type)
        for error_type in (
            getattr(decord_base, "DECORDError", None),
            getattr(decord_base, "DECORDLimitReachedError", None),
        )
    )


def _decode_video_with_policy(
    video_path: str,
    *,
    height: int,
    width: int,
    max_partition_bytes: int,
    max_frames: int | None,
    max_remote_video_bytes: int,
    max_source_frame_bytes: int,
    remote_temp_dir: str | None,
    on_error: str,
    max_source_path_bytes: int | None = None,
) -> Iterator[pa.RecordBatch]:
    _validate_video_error_mode(on_error)
    try:
        yield from _decode_video_batches(
            video_path,
            height=height,
            width=width,
            max_partition_bytes=max_partition_bytes,
            max_frames=max_frames,
            max_remote_video_bytes=max_remote_video_bytes,
            max_source_frame_bytes=max_source_frame_bytes,
            max_source_path_bytes=max_source_path_bytes,
            remote_temp_dir=remote_temp_dir,
        )
    except Exception as exc:
        if not isinstance(exc, (OSError, RuntimeError, ValueError)) and not _is_decord_error(exc):
            raise
        if isinstance(exc, VideoReadError):
            error = exc
        else:
            try:
                source_path, _ = _video_source_identity(video_path)
            except ValueError:
                source_path = video_path
            error = VideoReadError(source_path, f"{type(exc).__name__}: {exc}")
        if on_error == "raise":
            if error is exc:
                raise
            raise error from exc
        _LOGGER.warning(
            "Skipping unreadable video source=%r error_type=%s error=%s",
            error.video_path,
            type(exc).__name__,
            exc,
        )


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _video_source_udf_backend() -> str:
    backend = os.environ.get("VANE_VIDEO_SOURCE_UDF_BACKEND", "").strip().lower()
    if not backend:
        runner = os.environ.get("VANE_RUNNER", "").strip().lower() or "ray"
        backend = "ray_task" if runner == "ray" else "subprocess_task"
    if backend not in {"ray_task", "subprocess_task"}:
        raise ValueError(f"VANE_VIDEO_SOURCE_UDF_BACKEND must be one of: ray_task, subprocess_task; got {backend!r}")
    return backend


def _video_source_udf_videos_per_task(frame_limit: int | None, path_count: int) -> int:
    if frame_limit is not None:
        return max(1, path_count)
    return _read_int_env("VANE_VIDEO_SOURCE_UDF_VIDEOS_PER_TASK", 1, minimum=1)


def _video_max_output_rows(max_source_path_bytes: int) -> int:
    max_source_path_bytes = int(max_source_path_bytes)
    if max_source_path_bytes < 0:
        raise ValueError("max_source_path_bytes must be non-negative")
    if max_source_path_bytes > _ARROW_STRING_DATA_MAX_BYTES:
        raise ValueError(
            f"canonical video path requires {max_source_path_bytes} bytes, "
            f"exceeding Arrow's {_ARROW_STRING_DATA_MAX_BYTES}-byte UTF-8 offset limit"
        )
    path_limit = (
        _ARROW_STRING_DATA_MAX_BYTES
        if max_source_path_bytes == 0
        else _ARROW_STRING_DATA_MAX_BYTES // max_source_path_bytes
    )
    return min(
        _ARROW_STRING_DATA_MAX_BYTES // _SOURCE_ID_BYTES,
        path_limit,
    )


def _video_output_row_bytes(*, height: int, width: int, max_source_path_bytes: int) -> int:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if max_source_path_bytes < 0:
        raise ValueError("max_source_path_bytes must be non-negative")
    frame_bytes = int(height) * int(width) * 3
    return (
        frame_bytes
        + _SOURCE_ID_BYTES
        + int(max_source_path_bytes)
        # Coalescing file tails can leave one Arrow chunk per row. Charge both
        # the row offset and each chunk's terminal offset for both strings.
        + 4 * _ARROW_STRING_OFFSET_BYTES
        + _FRAME_INDEX_BYTES
    )


def _video_output_batch_bytes(
    row_count: int,
    *,
    height: int,
    width: int,
    max_source_path_bytes: int,
) -> int:
    row_count = int(row_count)
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    max_rows = _video_max_output_rows(max_source_path_bytes)
    if row_count > max_rows:
        raise ValueError(
            f"video output batch has {row_count} rows, exceeding Arrow's UTF-8 offset limit of {max_rows} rows"
        )
    row_bytes = _video_output_row_bytes(
        height=height,
        width=width,
        max_source_path_bytes=max_source_path_bytes,
    )
    return row_count * row_bytes


def _video_source_udf_output_batch_size(
    height: int,
    width: int,
    max_partition_bytes: int,
    *,
    max_source_path_bytes: int = _DEFAULT_MAX_SOURCE_PATH_BYTES,
) -> int:
    if max_partition_bytes <= 0:
        raise ValueError("max_partition_bytes must be positive")
    row_bytes = _video_output_row_bytes(
        height=height,
        width=width,
        max_source_path_bytes=max_source_path_bytes,
    )
    # Ray Data's block target is soft: the row that first crosses the target
    # is retained in the emitted block.
    target_rows = max(1, int(max_partition_bytes) // row_bytes + 1)
    return min(target_rows, _video_max_output_rows(max_source_path_bytes))


def _video_source_peak_memory_bytes(
    *,
    height: int,
    width: int,
    max_partition_bytes: int,
    max_source_frame_bytes: int,
    max_source_path_bytes: int = _DEFAULT_MAX_SOURCE_PATH_BYTES,
    output_batch_size: int | None = None,
) -> int:
    """Return the bounded source-task working set used for Ray admission."""
    frame_bytes = max(1, int(height) * int(width) * 3)
    decode_batch_size = _video_source_udf_output_batch_size(
        height,
        width,
        max_partition_bytes,
        max_source_path_bytes=max_source_path_bytes,
    )
    transport_batch_size = decode_batch_size if output_batch_size is None else max(1, int(output_batch_size))
    output_bytes = _video_output_batch_bytes(
        max(decode_batch_size, transport_batch_size),
        height=height,
        width=width,
        max_source_path_bytes=max_source_path_bytes,
    )
    resize_window_size = max(1, min(_VIDEO_RESIZE_THREADS, decode_batch_size))
    resize_window_bytes = resize_window_size * (int(max_source_frame_bytes) + frame_bytes)
    # Decord still owns the current decoded frame while ``asnumpy()`` copies it.
    # That overlap is separate from the NumPy frames retained by the resize
    # window and scales with the configured source-frame limit.
    decoder_copy_overlap_bytes = int(max_source_frame_bytes)
    return (
        _REMOTE_VIDEO_READ_CHUNK_BYTES
        + _VIDEO_DECODER_MEMORY_HEADROOM_BYTES
        + 2 * output_bytes
        + resize_window_bytes
        + decoder_copy_overlap_bytes
    )


def _coalesce_video_frame_batches(
    batches: Iterator[pa.RecordBatch],
    *,
    target_rows: int,
) -> Iterator[pa.Table]:
    """Shape a read task's stream without flushing at individual file tails."""
    target_rows = max(1, int(target_rows))
    pending: list[pa.Table] = []
    pending_rows = 0

    for batch in batches:
        table = pa.Table.from_batches([batch])
        offset = 0
        while offset < table.num_rows:
            take_rows = min(target_rows - pending_rows, table.num_rows - offset)
            pending.append(table.slice(offset, take_rows))
            pending_rows += take_rows
            offset += take_rows
            if pending_rows == target_rows:
                yield pa.concat_tables(pending)
                pending = []
                pending_rows = 0

    if pending_rows:
        yield pa.concat_tables(pending)


def _video_source_udf_cpus() -> float:
    """Return the resource-accounted CPU width of one decode/resize task."""
    configured = _read_optional_float_env("VANE_VIDEO_SOURCE_UDF_CPUS")
    if configured is not None:
        return configured
    # Each source invocation owns a ThreadPoolExecutor of this width. Leaving
    # the payload at Ray's one-CPU default oversubscribes the node and can
    # intermittently starve downstream GPU actors.
    return float(_VIDEO_RESIZE_THREADS)


def _video_source_udf_kwargs(
    *,
    height: int = 640,
    width: int = 480,
    max_partition_bytes: int = _DEFAULT_MAX_PARTITION_BYTES,
    max_source_frame_bytes: int = _DEFAULT_MAX_SOURCE_FRAME_BYTES,
    max_source_path_bytes: int = _DEFAULT_MAX_SOURCE_PATH_BYTES,
    frame_limit: int | None = None,
    path_count: int = 1,
    pre_grouped_paths: bool = False,
    schema: dict[str, object] | None = None,
) -> dict[str, object]:
    videos_per_task = 1 if pre_grouped_paths else _video_source_udf_videos_per_task(frame_limit, path_count)
    execution_backend = _video_source_udf_backend()
    output_batch_size = _read_int_env(
        "VANE_VIDEO_SOURCE_UDF_OUTPUT_BATCH_SIZE",
        _video_source_udf_output_batch_size(
            height,
            width,
            max_partition_bytes,
            max_source_path_bytes=max_source_path_bytes,
        ),
        minimum=1,
    )
    output_batch_bytes = _video_output_batch_bytes(
        output_batch_size,
        height=height,
        width=width,
        max_source_path_bytes=max_source_path_bytes,
    )
    udf_kwargs: dict[str, object] = {
        "execution_backend": execution_backend,
        "batch_size": videos_per_task,
        "output_batch_size": output_batch_size,
        # Ray Data's target_max_block_size is a soft threshold: the row that
        # crosses it remains in the block.  Vane's runtime byte limit is a
        # hard pre-append limit, so leaving both values at 128 MiB would split
        # a Ray-compatible soft-target block at its hard byte boundary. Row
        # count is the semantic flush boundary here; this larger guard only
        # prevents a second byte-based split of that already bounded output
        # batch.
        "output_target_max_bytes": max(int(max_partition_bytes) * 2, output_batch_bytes, 1),
        # One compute batch is one Ray-compatible read-task descriptor.  Its
        # final short block must be published before the next descriptor is
        # evaluated, just like Ray Data finalizes a block builder per read
        # task.
        "preserve_compute_batch_boundaries": True,
        "streaming_breaker": True,
        "cpus": _video_source_udf_cpus(),
    }
    if execution_backend == "ray_task":
        configured_memory_bytes = (
            _read_optional_positive_int_env("VANE_VIDEO_SOURCE_UDF_MEMORY_BYTES")
            or _DEFAULT_VIDEO_SOURCE_UDF_MEMORY_BYTES
        )
        peak_memory_bytes = _video_source_peak_memory_bytes(
            height=height,
            width=width,
            max_partition_bytes=max_partition_bytes,
            max_source_frame_bytes=max_source_frame_bytes,
            max_source_path_bytes=max_source_path_bytes,
            output_batch_size=output_batch_size,
        )
        udf_kwargs["memory_bytes"] = max(configured_memory_bytes, peak_memory_bytes)
    if schema is not None:
        udf_kwargs["schema"] = schema
    return udf_kwargs


def _split_video_path_groups(paths: list[str], task_count: int) -> list[list[str]]:
    if not paths:
        return []
    task_count = min(len(paths), int(task_count))
    if task_count <= 0:
        raise ValueError("read_task_count must be positive")
    base_size, larger_group_count = divmod(len(paths), task_count)
    groups: list[list[str]] = []
    offset = 0
    for task_index in range(task_count):
        group_size = base_size + (1 if task_index < larger_group_count else 0)
        groups.append(paths[offset : offset + group_size])
        offset += group_size
    return groups


def _video_source_path_groups(source: VideoFrameSource) -> list[list[str]]:
    if not source.paths:
        return []
    if source.frame_limit is not None:
        return [list(source.paths)]
    if source.read_task_count is None:
        return [[path] for path in source.paths]
    return _split_video_path_groups(source.paths, source.read_task_count)


def _video_frame_source_manifest_sql(source: VideoFrameSource) -> str:
    if not source.paths:
        return (
            "select []::VARCHAR[] as video_paths, 0::BIGINT as height, 0::BIGINT as width, "
            "0::BIGINT as max_partition_bytes, NULL::BIGINT as frame_limit, "
            "0::BIGINT as max_remote_video_bytes, 0::BIGINT as max_source_frame_bytes, "
            "0::BIGINT as max_source_path_bytes, "
            "'raise'::VARCHAR as on_error, NULL::VARCHAR as remote_temp_dir where false"
        )

    frame_limit_sql = "NULL" if source.frame_limit is None else str(max(0, int(source.frame_limit)))
    remote_temp_dir_sql = "NULL" if source.remote_temp_dir is None else _sql_string_literal(source.remote_temp_dir)
    path_groups = _video_source_path_groups(source)
    max_source_path_bytes = _max_video_source_path_bytes(source.paths)
    rows = ", ".join(
        "("
        f"list_value({', '.join(_sql_string_literal(_video_source_identity(path)[0]) for path in paths)}), "
        f"{int(source.height)}, "
        f"{int(source.width)}, "
        f"{int(source.max_partition_bytes)}, "
        f"{frame_limit_sql}, "
        f"{int(source.max_remote_video_bytes)}, "
        f"{int(source.max_source_frame_bytes)}, "
        f"{max_source_path_bytes}, "
        f"{_sql_string_literal(source.on_error)}, "
        f"{remote_temp_dir_sql}"
        ")"
        for paths in path_groups
    )
    return (
        "select "
        "video_paths::VARCHAR[] as video_paths, "
        "height::BIGINT as height, "
        "width::BIGINT as width, "
        "max_partition_bytes::BIGINT as max_partition_bytes, "
        "frame_limit::BIGINT as frame_limit, "
        "max_remote_video_bytes::BIGINT as max_remote_video_bytes, "
        "max_source_frame_bytes::BIGINT as max_source_frame_bytes, "
        "max_source_path_bytes::BIGINT as max_source_path_bytes, "
        "on_error::VARCHAR as on_error, "
        "remote_temp_dir::VARCHAR as remote_temp_dir "
        f"from (values {rows}) as manifest("
        "video_paths, height, width, max_partition_bytes, frame_limit, "
        "max_remote_video_bytes, max_source_frame_bytes, max_source_path_bytes, "
        "on_error, remote_temp_dir)"
    )


def _manifest_int_value(values: list[object], index: int, name: str) -> int:
    value = values[index]
    if value is None:
        raise ValueError(f"VideoFrameSource manifest column {name!r} cannot be NULL")
    return int(cast(Any, value))


def _manifest_frame_limit(values: list[object]) -> int | None:
    for value in values:
        if value is not None:
            return max(0, int(cast(Any, value)))
    return None


def _manifest_path_groups(table: pa.Table) -> list[list[str | None]]:
    if "video_paths" in table.column_names:
        groups = table.column("video_paths").to_pylist()
        return [list(group) if group is not None else [] for group in groups]
    return [[path] for path in table.column("video_path").to_pylist()]


def _video_frame_source_map_batches(table: pa.Table) -> Iterator[pa.Table]:
    path_groups = _manifest_path_groups(table)
    heights = table.column("height").to_pylist()
    widths = table.column("width").to_pylist()
    max_partition_bytes_values = table.column("max_partition_bytes").to_pylist()
    frame_limits = table.column("frame_limit").to_pylist()
    max_remote_video_bytes_values = table.column("max_remote_video_bytes").to_pylist()
    max_source_frame_bytes_values = table.column("max_source_frame_bytes").to_pylist()
    max_source_path_bytes_values = table.column("max_source_path_bytes").to_pylist()
    on_error_values = table.column("on_error").to_pylist()
    remote_temp_dir_values = table.column("remote_temp_dir").to_pylist()
    remaining = _manifest_frame_limit(frame_limits)

    for row_idx, paths in enumerate(path_groups):
        height = _manifest_int_value(heights, row_idx, "height")
        width = _manifest_int_value(widths, row_idx, "width")
        max_partition_bytes = _manifest_int_value(max_partition_bytes_values, row_idx, "max_partition_bytes")
        max_remote_video_bytes = _manifest_int_value(
            max_remote_video_bytes_values,
            row_idx,
            "max_remote_video_bytes",
        )
        max_source_frame_bytes = _manifest_int_value(
            max_source_frame_bytes_values,
            row_idx,
            "max_source_frame_bytes",
        )
        max_source_path_bytes = _manifest_int_value(
            max_source_path_bytes_values,
            row_idx,
            "max_source_path_bytes",
        )
        on_error_value = on_error_values[row_idx]
        if on_error_value is None:
            raise ValueError("VideoFrameSource manifest column 'on_error' cannot be NULL")
        on_error = _validate_video_error_mode(str(on_error_value))
        remote_temp_dir_value = remote_temp_dir_values[row_idx]
        remote_temp_dir = None if remote_temp_dir_value is None else str(remote_temp_dir_value)
        target_rows = _video_source_udf_output_batch_size(
            height,
            width,
            max_partition_bytes,
            max_source_path_bytes=max_source_path_bytes,
        )

        def decode_group(
            paths: list[str | None] = paths,
            height: int = height,
            width: int = width,
            max_partition_bytes: int = max_partition_bytes,
            max_remote_video_bytes: int = max_remote_video_bytes,
            max_source_frame_bytes: int = max_source_frame_bytes,
            max_source_path_bytes: int = max_source_path_bytes,
            on_error: str = on_error,
            remote_temp_dir: str | None = remote_temp_dir,
        ) -> Iterator[pa.RecordBatch]:
            nonlocal remaining
            for path in paths:
                if path is None:
                    raise ValueError("VideoFrameSource manifest path cannot be NULL")
                if remaining is not None and remaining <= 0:
                    break
                max_frames = remaining if remaining is not None else None

                _wait_for_memory()
                _decode_semaphore.acquire()
                try:
                    for batch in _decode_video_with_policy(
                        path,
                        height=height,
                        width=width,
                        max_partition_bytes=max_partition_bytes,
                        max_frames=max_frames,
                        max_remote_video_bytes=max_remote_video_bytes,
                        max_source_frame_bytes=max_source_frame_bytes,
                        max_source_path_bytes=max_source_path_bytes,
                        remote_temp_dir=remote_temp_dir,
                        on_error=on_error,
                    ):
                        if remaining is not None:
                            if batch.num_rows > remaining:
                                batch = batch.slice(0, remaining)
                            remaining -= batch.num_rows
                        yield batch
                        if remaining is not None and remaining <= 0:
                            break
                finally:
                    _decode_semaphore.release()

        yield from _coalesce_video_frame_batches(decode_group(), target_rows=target_rows)
        if remaining is not None and remaining <= 0:
            break


class VideoFrameTask(DataSourceTask):
    """DataSourceTask for decoding a single video file into frame batches."""

    def __init__(
        self,
        video_path: str,
        height: int,
        width: int,
        max_partition_bytes: int = _DEFAULT_MAX_PARTITION_BYTES,
        max_remote_video_bytes: int = _DEFAULT_MAX_REMOTE_VIDEO_BYTES,
        max_source_frame_bytes: int = _DEFAULT_MAX_SOURCE_FRAME_BYTES,
        on_error: str = "raise",
        remote_temp_dir: str | None = None,
    ):
        self.video_path = video_path
        self.height = height
        self.width = width
        self.max_partition_bytes = max_partition_bytes
        self.max_remote_video_bytes = max_remote_video_bytes
        self.max_source_frame_bytes = max_source_frame_bytes
        self.on_error = _validate_video_error_mode(on_error)
        self.remote_temp_dir = remote_temp_dir

    def execute(self) -> Iterator[pa.RecordBatch]:
        """Decode video frames with decord, yielding batches of ~10MB.

        Two levels of backpressure:
        1. Memory-based: block when system memory > high watermark (cross-process)
        2. Semaphore: limit concurrent decodes within this process
        """
        _wait_for_memory()
        _decode_semaphore.acquire()
        try:
            yield from self._execute_inner()
        finally:
            _decode_semaphore.release()

    def _execute_inner(self) -> Iterator[pa.RecordBatch]:
        yield from _decode_video_with_policy(
            self.video_path,
            height=self.height,
            width=self.width,
            max_partition_bytes=self.max_partition_bytes,
            max_frames=None,
            max_remote_video_bytes=self.max_remote_video_bytes,
            max_source_frame_bytes=self.max_source_frame_bytes,
            max_source_path_bytes=None,
            remote_temp_dir=self.remote_temp_dir,
            on_error=self.on_error,
        )

    def _flush(
        self,
        source_id: str,
        video_path: str,
        resized: npt.NDArray[np.uint8],
        indices: npt.ArrayLike,
        count: int,
    ) -> pa.RecordBatch:
        """Build a RecordBatch from the accumulated frame buffer."""
        return _flush_frame_batch(source_id, video_path, resized, indices, count)


class LimitedVideoFrameTask(DataSourceTask):
    """Decode videos in manifest order until a global frame limit is reached."""

    def __init__(
        self,
        paths: list[str],
        height: int,
        width: int,
        max_frames: int,
        max_partition_bytes: int = _DEFAULT_MAX_PARTITION_BYTES,
        max_remote_video_bytes: int = _DEFAULT_MAX_REMOTE_VIDEO_BYTES,
        max_source_frame_bytes: int = _DEFAULT_MAX_SOURCE_FRAME_BYTES,
        on_error: str = "raise",
        remote_temp_dir: str | None = None,
    ):
        self.paths = paths
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.max_partition_bytes = max_partition_bytes
        self.max_remote_video_bytes = max_remote_video_bytes
        self.max_source_frame_bytes = max_source_frame_bytes
        self.on_error = _validate_video_error_mode(on_error)
        self.remote_temp_dir = remote_temp_dir

    def execute(self) -> Iterator[pa.RecordBatch]:
        _wait_for_memory()
        _decode_semaphore.acquire()
        try:
            yield from self._execute_inner()
        finally:
            _decode_semaphore.release()

    def _execute_inner(self) -> Iterator[pa.RecordBatch]:
        remaining = self.max_frames
        for path in self.paths:
            if remaining <= 0:
                break
            for batch in _decode_video_with_policy(
                path,
                height=self.height,
                width=self.width,
                max_partition_bytes=self.max_partition_bytes,
                max_frames=remaining,
                max_remote_video_bytes=self.max_remote_video_bytes,
                max_source_frame_bytes=self.max_source_frame_bytes,
                max_source_path_bytes=None,
                remote_temp_dir=self.remote_temp_dir,
                on_error=self.on_error,
            ):
                remaining -= batch.num_rows
                yield batch
                if remaining <= 0:
                    break


class VideoFrameSource(DataSource):
    """DataSource that streams video frames from local or S3 video files.

    Each video file becomes an independent task. Frames are decoded using
    decord and resized to (height, width). Output rows contain the canonical
    ``video_path``, its SHA-256 ``source_id``, ``frame_index``, and the frame
    tensor. S3 objects are copied to a bounded task-local temporary file before
    decoding.

    ``on_error="raise"`` fails the query on an unreadable file.
    ``on_error="skip"`` keeps frames emitted before the error, skips the
    remainder of that file, and continues with the next file. Frames that
    decord recovers without raising are considered successfully decoded.
    """

    def __init__(
        self,
        paths: list[str],
        height: int = 640,
        width: int = 480,
        max_partition_bytes: int = _DEFAULT_MAX_PARTITION_BYTES,
        frame_limit: int | None = None,
        read_task_count: int | None = None,
        max_remote_video_bytes: int = _DEFAULT_MAX_REMOTE_VIDEO_BYTES,
        max_source_frame_bytes: int = _DEFAULT_MAX_SOURCE_FRAME_BYTES,
        on_error: str = "raise",
        remote_temp_dir: str | None = None,
    ):
        self.paths = paths
        self.height = int(height)
        self.width = int(width)
        self.max_partition_bytes = int(max_partition_bytes)
        self.frame_limit = frame_limit
        self.max_remote_video_bytes = int(max_remote_video_bytes)
        self.max_source_frame_bytes = int(max_source_frame_bytes)
        self.on_error = _validate_video_error_mode(on_error)
        self.remote_temp_dir = None if remote_temp_dir is None else os.fspath(remote_temp_dir)
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.max_partition_bytes <= 0:
            raise ValueError("max_partition_bytes must be positive")
        if self.max_remote_video_bytes <= 0:
            raise ValueError("max_remote_video_bytes must be positive")
        if self.max_source_frame_bytes <= 0:
            raise ValueError("max_source_frame_bytes must be positive")
        decoder_width, decoder_height = _video_decoder_dimensions(
            width=self.width,
            height=self.height,
        )
        decoded_frame_bytes = decoder_height * decoder_width * 3
        if decoded_frame_bytes > self.max_source_frame_bytes:
            raise ValueError(
                f"bounded decoder frame size {decoded_frame_bytes} exceeds "
                f"max_source_frame_bytes {self.max_source_frame_bytes}"
            )
        if read_task_count is not None and int(read_task_count) <= 0:
            raise ValueError("read_task_count must be positive")
        self.read_task_count = None if read_task_count is None else int(read_task_count)

    @property
    def schema(self) -> dict[str, object]:
        return {
            "source_id": "VARCHAR",
            "video_path": "VARCHAR",
            "frame_index": "BIGINT",
            "frame": {"kind": "tensor", "dtype": "UINT8", "shape": [self.height, self.width, 3]},
        }

    def get_tasks(self) -> Iterator[DataSourceTask]:
        if self.frame_limit is not None:
            yield LimitedVideoFrameTask(
                paths=self.paths,
                height=self.height,
                width=self.width,
                max_frames=max(0, self.frame_limit),
                max_partition_bytes=self.max_partition_bytes,
                max_remote_video_bytes=self.max_remote_video_bytes,
                max_source_frame_bytes=self.max_source_frame_bytes,
                on_error=self.on_error,
                remote_temp_dir=self.remote_temp_dir,
            )
            return
        for path in self.paths:
            yield VideoFrameTask(
                video_path=path,
                height=self.height,
                width=self.width,
                max_partition_bytes=self.max_partition_bytes,
                max_remote_video_bytes=self.max_remote_video_bytes,
                max_source_frame_bytes=self.max_source_frame_bytes,
                on_error=self.on_error,
                remote_temp_dir=self.remote_temp_dir,
            )

    def to_udf_relation(self, con: Any) -> Any:
        import duckdb

        manifest = con.sql(_video_frame_source_manifest_sql(self))
        udf_kwargs = _video_source_udf_kwargs(
            height=self.height,
            width=self.width,
            max_partition_bytes=self.max_partition_bytes,
            max_source_frame_bytes=self.max_source_frame_bytes,
            max_source_path_bytes=_max_video_source_path_bytes(self.paths),
            frame_limit=self.frame_limit,
            path_count=len(_video_source_path_groups(self)),
            pre_grouped_paths=True,
            schema={
                "source_id": duckdb.sqltypes.VARCHAR,
                "video_path": duckdb.sqltypes.VARCHAR,
                "frame_index": duckdb.sqltypes.BIGINT,
                "frame": duckdb.tensor_type(duckdb.sqltypes.UTINYINT, (self.height, self.width, 3)),
            },
        )
        return manifest.map_batches(_video_frame_source_map_batches, **udf_kwargs)
