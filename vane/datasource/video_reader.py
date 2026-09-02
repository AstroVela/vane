# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed, bounded frame streaming for governed VIDEOFILE values.

``VideoFile.frames()`` owns encoded-video semantics: FILE resolution, strict
logical ranges, PyAV decoding, selection, provenance, cancellation, and error
classification. This module only turns that row-wise stream into bounded Arrow
batches and schedules independent VIDEOFILE values across UDF workers.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from vane.datasource import DataSource, DataSourceTask

if TYPE_CHECKING:
    import vane


_DEFAULT_MAX_PARTITION_BYTES = int(os.environ.get("VANE_VIDEO_MAX_PARTITION_BYTES", str(10 * 1024 * 1024)))
_DEFAULT_VIDEO_SOURCE_UDF_MEMORY_BYTES = 512 * 1024**2
_DEFAULT_VIDEO_BUFFER_SIZE = 1024 * 1024
_DEFAULT_MAX_INPUT_BYTES = 8 * 1024**3
_DEFAULT_MAX_DECODED_FRAMES = 1_000_000
_DEFAULT_MAX_PIXELS = 32 * 1024**2
_VIDEO_DECODER_MEMORY_HEADROOM_BYTES = 128 * 1024**2
_ARROW_STRING_DATA_MAX_BYTES = int(np.iinfo(np.int32).max)
_ARROW_STRING_OFFSET_BYTES = np.dtype(np.int32).itemsize
_INT64_BYTES = np.dtype(np.int64).itemsize
_DOUBLE_BYTES = np.dtype(np.float64).itemsize
_BOOLEAN_BYTES = np.dtype(np.bool_).itemsize
_FILE_STRING_COLUMNS = 3
_FILE_INTEGER_COLUMNS = 2
_FILE_VALIDITY_BYTES_PER_ROW = 5
_PROVENANCE_NULLABLE_COLUMNS = 7
_VIDEO_ERROR_MODES = frozenset({"raise", "skip"})
_FILE_FIELDS = ("url", "content_type", "position", "size", "checksum")
_LOGGER = logging.getLogger(__name__)


def _import_video_dependency(module_name: str, package_name: str) -> ModuleType:
    """Import an optional dependency and report the supported installation extra."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"The video data source requires the {package_name!r} package. Please `pip install 'vane-ai[video]'`."
        ) from error


def _read_optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    result = float(value)
    if result <= 0.0 or not math.isfinite(result):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
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


# Admission control is intentionally process-local. Ray accounts for the same
# task working set through ``memory_bytes`` below, while this guard prevents a
# local worker from starting another decoder under host-memory pressure.
_MAX_CONCURRENT_DECODES = _read_positive_int_env("VANE_MAX_CONCURRENT_DECODES", 1)
_decode_semaphore = threading.Semaphore(_MAX_CONCURRENT_DECODES)
_MEM_HIGH_WATERMARK = float(os.environ.get("VANE_DECODE_MEM_HIGH_PCT", "80"))
_MEM_LOW_WATERMARK = float(os.environ.get("VANE_DECODE_MEM_LOW_PCT", "70"))
_MEM_CHECK_INTERVAL = 2.0
try:
    _MEM_MIN_AVAILABLE_MB = max(0, int(os.environ.get("VANE_DECODE_MIN_AVAILABLE_MB", "4096")))
except Exception:
    _MEM_MIN_AVAILABLE_MB = 4096
_MEM_MIN_AVAILABLE_BYTES = _MEM_MIN_AVAILABLE_MB * 1024**2


def _wait_for_memory() -> None:
    """Wait before opening a decoder when the host is above its admission watermark."""
    psutil = _import_video_dependency("psutil", "psutil")

    def has_capacity(memory: Any) -> bool:
        if _MEM_MIN_AVAILABLE_BYTES > 0 and memory.available >= _MEM_MIN_AVAILABLE_BYTES:
            return True
        return bool(memory.percent < _MEM_HIGH_WATERMARK)

    def has_recovered(memory: Any) -> bool:
        if _MEM_MIN_AVAILABLE_BYTES > 0 and memory.available >= _MEM_MIN_AVAILABLE_BYTES:
            return True
        return bool(memory.percent < _MEM_LOW_WATERMARK)

    memory = psutil.virtual_memory()
    if has_capacity(memory):
        return
    while True:
        time.sleep(_MEM_CHECK_INTERVAL)
        memory = psutil.virtual_memory()
        if has_recovered(memory):
            return


class VideoReadError(RuntimeError):
    """A classified encoded-media failure associated with one VIDEOFILE."""

    def __init__(self, video_url: str, message: str):
        self.video_url = str(video_url)
        self.message = str(message)
        super().__init__(self.video_url, self.message)

    def __str__(self) -> str:
        return f"Failed to read video {self.video_url!r}: {self.message}"


@dataclass(frozen=True, slots=True)
class _FileStorageBounds:
    max_string_bytes: int
    max_row_bytes: int


@dataclass(frozen=True, slots=True)
class _VideoDecodeOptions:
    height: int
    width: int
    max_partition_bytes: int
    start_time: int | float
    end_time: int | float | None
    is_key_frame: bool | None
    sample_interval_seconds: int | float | None
    buffer_size: int
    max_input_bytes: int
    max_decoded_frames: int
    max_pixels: int
    on_error: str
    file_bounds: _FileStorageBounds


def _positive_int(value: object, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, not {type(value).__name__!r}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _optional_nonnegative_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int or None, not {type(value).__name__!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _time_value(value: object, *, name: str, optional: bool = False) -> int | float | None:
    if value is None:
        if optional:
            return None
        raise TypeError(f"{name} must be int or float, not 'NoneType'")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        expected = "int, float, or None" if optional else "int or float"
        raise TypeError(f"{name} must be {expected}, not {type(value).__name__!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_video_error_mode(on_error: object) -> str:
    if not isinstance(on_error, str) or on_error not in _VIDEO_ERROR_MODES:
        choices = ", ".join(sorted(_VIDEO_ERROR_MODES))
        raise ValueError(f"on_error must be one of: {choices}; got {on_error!r}")
    return on_error


def _normalize_video_file(value: object) -> vane.VideoFile:
    import vane

    if isinstance(value, vane.VideoFile):
        return value
    if isinstance(value, (str, os.PathLike)):
        url = os.fspath(value)
        if not isinstance(url, str):
            raise TypeError("VideoFrameSource paths must resolve to str, not bytes")
        return vane.VideoFile(url)
    if type(value) is vane.File:
        file_value = value
        return vane.VideoFile(
            file_value.url,
            file_value.content_type,
            file_value.position,
            file_value.size,
            file_value.checksum,
        )
    if isinstance(value, vane.File):
        raise TypeError(f"VideoFrameSource accepts VIDEOFILE, generic FILE, or path values; got {type(value).__name__}")
    raise TypeError(
        f"VideoFrameSource entries must be VIDEOFILE, generic FILE, str, or os.PathLike; got {type(value).__name__}"
    )


def _normalize_video_files(values: Iterable[object]) -> tuple[vane.VideoFile, ...]:
    import vane

    if isinstance(values, (str, os.PathLike, vane.File)):
        raise TypeError("VideoFrameSource files must be an iterable of values, not a single value")
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError("VideoFrameSource files must be an iterable of file values") from error
    return tuple(_normalize_video_file(value) for value in iterator)


def _encoded_length(value: str | None) -> int:
    return 0 if value is None else len(value.encode("utf-8"))


def _file_storage_bounds(files: Sequence[vane.VideoFile]) -> _FileStorageBounds:
    max_string_bytes = 0
    max_row_bytes = (
        _FILE_INTEGER_COLUMNS * _INT64_BYTES
        + _FILE_STRING_COLUMNS * 2 * _ARROW_STRING_OFFSET_BYTES
        + _FILE_VALIDITY_BYTES_PER_ROW
    )
    for value in files:
        lengths = (
            _encoded_length(value.url),
            _encoded_length(value.content_type),
            _encoded_length(value.checksum),
        )
        max_string_bytes = max(max_string_bytes, *lengths)
        max_row_bytes = max(
            max_row_bytes,
            sum(lengths)
            + _FILE_INTEGER_COLUMNS * _INT64_BYTES
            + _FILE_STRING_COLUMNS * 2 * _ARROW_STRING_OFFSET_BYTES
            + _FILE_VALIDITY_BYTES_PER_ROW,
        )
    if max_string_bytes > _ARROW_STRING_DATA_MAX_BYTES:
        raise ValueError(
            f"VIDEOFILE string field requires {max_string_bytes} bytes, exceeding Arrow's "
            f"{_ARROW_STRING_DATA_MAX_BYTES}-byte UTF-8 offset limit"
        )
    return _FileStorageBounds(max_string_bytes=max_string_bytes, max_row_bytes=max_row_bytes)


def _make_decode_options(
    *,
    height: object,
    width: object,
    max_partition_bytes: object,
    start_time: object,
    end_time: object,
    is_key_frame: object,
    sample_interval_seconds: object,
    buffer_size: object,
    max_input_bytes: object,
    max_decoded_frames: object,
    max_pixels: object,
    on_error: object,
    file_bounds: _FileStorageBounds,
) -> _VideoDecodeOptions:
    normalized_height = _positive_int(height, name="height")
    normalized_width = _positive_int(width, name="width")
    normalized_partition_bytes = _positive_int(max_partition_bytes, name="max_partition_bytes")
    normalized_start = _time_value(start_time, name="start_time")
    assert normalized_start is not None
    normalized_end = _time_value(end_time, name="end_time", optional=True)
    if normalized_end is not None and normalized_end < normalized_start:
        raise ValueError("end_time must be greater than or equal to start_time")
    normalized_interval = _time_value(
        sample_interval_seconds,
        name="sample_interval_seconds",
        optional=True,
    )
    if normalized_interval == 0:
        raise ValueError("sample_interval_seconds must be greater than zero")
    if is_key_frame is not None and not isinstance(is_key_frame, bool):
        raise TypeError(f"is_key_frame must be bool or None, not {type(is_key_frame).__name__!r}")
    normalized_buffer_size = _positive_int(buffer_size, name="buffer_size", maximum=(1 << 31) - 1)
    normalized_max_input_bytes = _positive_int(max_input_bytes, name="max_input_bytes", maximum=(1 << 64) - 1)
    normalized_max_decoded_frames = _positive_int(
        max_decoded_frames,
        name="max_decoded_frames",
        maximum=(1 << 63) - 1,
    )
    normalized_max_pixels = _positive_int(
        max_pixels,
        name="max_pixels",
        maximum=_DEFAULT_MAX_PIXELS,
    )
    output_pixels = normalized_height * normalized_width
    if output_pixels > normalized_max_pixels:
        import vane

        raise vane.VideoFileLimitError(
            f"requested video frames contain {output_pixels} pixels, exceeding max_pixels={normalized_max_pixels}"
        )
    return _VideoDecodeOptions(
        height=normalized_height,
        width=normalized_width,
        max_partition_bytes=normalized_partition_bytes,
        start_time=normalized_start,
        end_time=normalized_end,
        is_key_frame=is_key_frame,
        sample_interval_seconds=normalized_interval,
        buffer_size=normalized_buffer_size,
        max_input_bytes=normalized_max_input_bytes,
        max_decoded_frames=normalized_max_decoded_frames,
        max_pixels=normalized_max_pixels,
        on_error=_validate_video_error_mode(on_error),
        file_bounds=file_bounds,
    )


def _build_frame_array(frames: npt.ArrayLike) -> pa.ExtensionArray:
    frame_array: npt.NDArray[np.uint8] = np.ascontiguousarray(frames, dtype=np.uint8)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(f"expected RGB frame batch with shape (N, H, W, 3), got {frame_array.shape!r}")
    return pa.FixedShapeTensorArray.from_numpy_ndarray(frame_array)


def _constant_string_array(value: str, count: int) -> pa.Array:
    if count < 0:
        raise ValueError("constant string array count must be non-negative")
    encoded = value.encode("utf-8")
    data_bytes = len(encoded) * count
    if count > _ARROW_STRING_DATA_MAX_BYTES or data_bytes > _ARROW_STRING_DATA_MAX_BYTES:
        raise ValueError(
            f"constant string array requires {data_bytes} data bytes, exceeding Arrow's "
            f"{_ARROW_STRING_DATA_MAX_BYTES}-byte UTF-8 offset limit"
        )
    offsets: npt.NDArray[np.int32] = np.arange(count + 1, dtype=np.int32) * len(encoded)
    return pa.Array.from_buffers(
        pa.string(),
        count,
        [None, pa.py_buffer(offsets), pa.py_buffer(encoded * count)],
    )


def _optional_constant_string_array(value: str | None, count: int) -> pa.Array:
    if value is None:
        return pa.nulls(count, type=pa.string())
    return _constant_string_array(value, count)


def _optional_constant_int64_array(value: int | None, count: int) -> pa.Array:
    if value is None:
        return pa.nulls(count, type=pa.int64())
    return pa.array(np.full(count, value, dtype=np.int64), type=pa.int64())


def _video_file_array(value: vane.VideoFile, count: int) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            _constant_string_array(value.url, count),
            _optional_constant_string_array(value.content_type, count),
            _optional_constant_int64_array(value.position, count),
            _optional_constant_int64_array(value.size, count),
            _optional_constant_string_array(value.checksum, count),
        ],
        names=list(_FILE_FIELDS),
    )


def _video_max_output_rows(max_file_string_bytes: int) -> int:
    max_file_string_bytes = int(max_file_string_bytes)
    if max_file_string_bytes < 0:
        raise ValueError("max_file_string_bytes must be non-negative")
    if max_file_string_bytes > _ARROW_STRING_DATA_MAX_BYTES:
        raise ValueError(
            f"VIDEOFILE string field requires {max_file_string_bytes} bytes, exceeding Arrow's "
            f"{_ARROW_STRING_DATA_MAX_BYTES}-byte UTF-8 offset limit"
        )
    if max_file_string_bytes == 0:
        return _ARROW_STRING_DATA_MAX_BYTES
    return _ARROW_STRING_DATA_MAX_BYTES // max_file_string_bytes


def _video_output_row_bytes(*, height: int, width: int, max_file_row_bytes: int) -> int:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if max_file_row_bytes < 0:
        raise ValueError("max_file_row_bytes must be non-negative")
    frame_bytes = int(height) * int(width) * 3
    provenance_bytes = 6 * _INT64_BYTES + _DOUBLE_BYTES + _BOOLEAN_BYTES + _PROVENANCE_NULLABLE_COLUMNS
    return frame_bytes + int(max_file_row_bytes) + provenance_bytes


def _video_output_batch_bytes(
    row_count: int,
    *,
    height: int,
    width: int,
    max_file_string_bytes: int,
    max_file_row_bytes: int,
) -> int:
    row_count = int(row_count)
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    max_rows = _video_max_output_rows(max_file_string_bytes)
    if row_count > max_rows:
        raise ValueError(
            f"video output batch has {row_count} rows, exceeding Arrow's UTF-8 offset limit of {max_rows} rows"
        )
    return row_count * _video_output_row_bytes(
        height=height,
        width=width,
        max_file_row_bytes=max_file_row_bytes,
    )


def _video_source_udf_output_batch_size(
    height: int,
    width: int,
    max_partition_bytes: int,
    *,
    max_file_string_bytes: int = 0,
    max_file_row_bytes: int = 0,
) -> int:
    if max_partition_bytes <= 0:
        raise ValueError("max_partition_bytes must be positive")
    row_bytes = _video_output_row_bytes(
        height=height,
        width=width,
        max_file_row_bytes=max_file_row_bytes,
    )
    # Ray's block target is soft: the row that first crosses it remains in the
    # block. Keep the same boundary so this source composes with Ray Data.
    target_rows = max(1, int(max_partition_bytes) // row_bytes + 1)
    return min(target_rows, _video_max_output_rows(max_file_string_bytes))


def _video_source_peak_memory_bytes(
    *,
    height: int,
    width: int,
    max_partition_bytes: int,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    buffer_size: int = _DEFAULT_VIDEO_BUFFER_SIZE,
    max_file_string_bytes: int = 0,
    max_file_row_bytes: int = 0,
    output_batch_size: int | None = None,
) -> int:
    """Return the bounded source-task working set used for Ray admission."""
    decode_batch_size = _video_source_udf_output_batch_size(
        height,
        width,
        max_partition_bytes,
        max_file_string_bytes=max_file_string_bytes,
        max_file_row_bytes=max_file_row_bytes,
    )
    transport_batch_size = decode_batch_size if output_batch_size is None else max(1, int(output_batch_size))
    output_bytes = _video_output_batch_bytes(
        max(decode_batch_size, transport_batch_size),
        height=height,
        width=width,
        max_file_string_bytes=max_file_string_bytes,
        max_file_row_bytes=max_file_row_bytes,
    )
    current_rgb_frame_bytes = int(max_pixels) * 3
    return _VIDEO_DECODER_MEMORY_HEADROOM_BYTES + int(buffer_size) + current_rgb_frame_bytes + 2 * output_bytes


def _flush_frame_batch(
    value: vane.VideoFile,
    resized: npt.NDArray[np.uint8],
    count: int,
    *,
    frame_indices: Sequence[int | None],
    frame_times: Sequence[float | None],
    time_base_numerators: Sequence[int | None],
    time_base_denominators: Sequence[int | None],
    frame_pts: Sequence[int | None],
    frame_dts: Sequence[int | None],
    frame_durations: Sequence[int | None],
    key_frame_flags: Sequence[bool],
) -> pa.RecordBatch:
    provenance = (
        frame_indices,
        frame_times,
        time_base_numerators,
        time_base_denominators,
        frame_pts,
        frame_dts,
        frame_durations,
        key_frame_flags,
    )
    if count < 0 or any(len(column) != count for column in provenance):
        raise ValueError("video provenance columns must cover exactly every output frame")
    frame_values = resized[:count]
    if count < len(resized):
        # Do not let a short file tail pin a full target-sized NumPy allocation
        # after it is coalesced with tails from other files.
        frame_values = frame_values.copy()
    return pa.record_batch(
        [
            _video_file_array(value, count),
            pa.array(frame_indices, type=pa.int64()),
            pa.array(frame_times, type=pa.float64()),
            pa.array(time_base_numerators, type=pa.int64()),
            pa.array(time_base_denominators, type=pa.int64()),
            pa.array(frame_pts, type=pa.int64()),
            pa.array(frame_dts, type=pa.int64()),
            pa.array(frame_durations, type=pa.int64()),
            pa.array(key_frame_flags, type=pa.bool_()),
            _build_frame_array(frame_values),
        ],
        names=[
            "file",
            "frame_index",
            "frame_time",
            "frame_time_base_numerator",
            "frame_time_base_denominator",
            "frame_pts",
            "frame_dts",
            "frame_duration",
            "is_key_frame",
            "frame",
        ],
    )


def _decode_video_batches(
    value: vane.VideoFile,
    *,
    options: _VideoDecodeOptions,
    max_output_frames: int | None,
) -> Iterator[pa.RecordBatch]:
    if max_output_frames is not None and max_output_frames <= 0:
        return

    batch_size = _video_source_udf_output_batch_size(
        options.height,
        options.width,
        options.max_partition_bytes,
        max_file_string_bytes=options.file_bounds.max_string_bytes,
        max_file_row_bytes=options.file_bounds.max_row_bytes,
    )
    resized: npt.NDArray[np.uint8] = np.empty(
        (batch_size, options.height, options.width, 3),
        dtype=np.uint8,
    )
    frame_indices: list[int | None] = []
    frame_times: list[float | None] = []
    time_base_numerators: list[int | None] = []
    time_base_denominators: list[int | None] = []
    frame_pts: list[int | None] = []
    frame_dts: list[int | None] = []
    frame_durations: list[int | None] = []
    key_frame_flags: list[bool] = []
    count = 0
    output_count = 0

    def flush() -> pa.RecordBatch:
        return _flush_frame_batch(
            value,
            resized,
            count,
            frame_indices=frame_indices,
            frame_times=frame_times,
            time_base_numerators=time_base_numerators,
            time_base_denominators=time_base_denominators,
            frame_pts=frame_pts,
            frame_dts=frame_dts,
            frame_durations=frame_durations,
            key_frame_flags=key_frame_flags,
        )

    frames = value.frames(
        start_time=options.start_time,
        end_time=options.end_time,
        width=options.width,
        height=options.height,
        is_key_frame=options.is_key_frame,
        sample_interval_seconds=options.sample_interval_seconds,
        buffer_size=options.buffer_size,
        max_input_bytes=options.max_input_bytes,
        max_frames=options.max_decoded_frames,
        max_pixels=options.max_pixels,
    )
    try:
        for record in frames:
            try:
                pixels = np.asarray(record.data)
                expected_shape = (options.height, options.width, 3)
                if pixels.dtype != np.uint8 or pixels.shape != expected_shape:
                    raise RuntimeError(
                        "VideoFile.frames() violated its detached RGB UINT8 contract: "
                        f"expected {expected_shape}, found dtype={pixels.dtype} shape={pixels.shape}"
                    )
                np.copyto(resized[count], pixels, casting="no")
                frame_indices.append(record.frame_index)
                frame_times.append(record.frame_time)
                if record.frame_time_base is None:
                    time_base_numerators.append(None)
                    time_base_denominators.append(None)
                else:
                    time_base_numerators.append(record.frame_time_base.numerator)
                    time_base_denominators.append(record.frame_time_base.denominator)
                frame_pts.append(record.frame_pts)
                frame_dts.append(record.frame_dts)
                frame_durations.append(record.frame_duration)
                key_frame_flags.append(record.is_key_frame)
                count += 1
                output_count += 1
            except BaseException:
                try:
                    record.data.close()
                except BaseException:
                    pass
                raise
            else:
                record.data.close()

            if count == batch_size:
                yield flush()
                if max_output_frames is not None and output_count >= max_output_frames:
                    frames.close()
                    return
                resized = np.empty(
                    (batch_size, options.height, options.width, 3),
                    dtype=np.uint8,
                )
                frame_indices = []
                frame_times = []
                time_base_numerators = []
                time_base_denominators = []
                frame_pts = []
                frame_dts = []
                frame_durations = []
                key_frame_flags = []
                count = 0

            if max_output_frames is not None and output_count >= max_output_frames:
                break

        if count:
            yield flush()
    except BaseException:
        _close_iterator_preserving_active_error(frames)
        raise
    else:
        frames.close()


def _close_iterator_preserving_active_error(iterator: Iterator[object]) -> None:
    close = getattr(iterator, "close", None)
    if not callable(close):
        return
    try:
        close()
    except BaseException:
        # A consumer exception already owns this boundary. Decoder cleanup must
        # not replace cancellation, GeneratorExit, or a downstream failure.
        pass


def _decode_video_with_policy(
    value: vane.VideoFile,
    *,
    options: _VideoDecodeOptions,
    max_output_frames: int | None,
) -> Iterator[pa.RecordBatch]:
    import vane

    batches = _decode_video_batches(value, options=options, max_output_frames=max_output_frames)
    while True:
        try:
            batch = next(batches)
        except StopIteration:
            return
        except vane.VideoFileFormatError as error:
            if options.on_error == "raise":
                raise VideoReadError(value.url, str(error)) from error
            _LOGGER.warning(
                "Skipping unreadable VIDEOFILE url=%r error_type=%s error=%s",
                value.url,
                type(error).__name__,
                error,
            )
            return
        try:
            yield batch
        except BaseException:
            _close_iterator_preserving_active_error(batches)
            raise


def _decode_video_guarded(
    value: vane.VideoFile,
    *,
    options: _VideoDecodeOptions,
    max_output_frames: int | None,
) -> Iterator[pa.RecordBatch]:
    _wait_for_memory()
    _decode_semaphore.acquire()
    try:
        yield from _decode_video_with_policy(
            value,
            options=options,
            max_output_frames=max_output_frames,
        )
    finally:
        _decode_semaphore.release()


def _coalesce_video_frame_batches(
    batches: Iterator[pa.RecordBatch],
    *,
    target_rows: int,
) -> Iterator[pa.Table]:
    """Coalesce short file tails without crossing one read-task row target."""
    target_rows = max(1, int(target_rows))
    pending: list[pa.Table] = []
    pending_rows = 0
    try:
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
    except BaseException:
        _close_iterator_preserving_active_error(batches)
        raise


def _video_source_udf_backend() -> str:
    backend = os.environ.get("VANE_VIDEO_SOURCE_UDF_BACKEND", "").strip().lower()
    if not backend:
        runner = os.environ.get("VANE_RUNNER", "").strip().lower() or "ray"
        backend = "ray_task" if runner == "ray" else "subprocess_task"
    if backend not in {"ray_task", "subprocess_task"}:
        raise ValueError(f"VANE_VIDEO_SOURCE_UDF_BACKEND must be one of: ray_task, subprocess_task; got {backend!r}")
    return backend


def _video_source_udf_files_per_task(frame_limit: int | None, file_count: int) -> int:
    if frame_limit is not None:
        return max(1, file_count)
    return _read_positive_int_env("VANE_VIDEO_SOURCE_UDF_FILES_PER_TASK", 1)


def _video_source_udf_cpus() -> float:
    configured = _read_optional_float_env("VANE_VIDEO_SOURCE_UDF_CPUS")
    return 1.0 if configured is None else configured


def _video_source_udf_kwargs(
    *,
    height: int = 640,
    width: int = 480,
    max_partition_bytes: int = _DEFAULT_MAX_PARTITION_BYTES,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    buffer_size: int = _DEFAULT_VIDEO_BUFFER_SIZE,
    max_file_string_bytes: int = 0,
    max_file_row_bytes: int = 0,
    frame_limit: int | None = None,
    file_count: int = 1,
    pre_grouped_files: bool = False,
    schema: dict[str, object] | None = None,
) -> dict[str, object]:
    files_per_task = 1 if pre_grouped_files else _video_source_udf_files_per_task(frame_limit, file_count)
    execution_backend = _video_source_udf_backend()
    output_batch_size = _read_positive_int_env(
        "VANE_VIDEO_SOURCE_UDF_OUTPUT_BATCH_SIZE",
        _video_source_udf_output_batch_size(
            height,
            width,
            max_partition_bytes,
            max_file_string_bytes=max_file_string_bytes,
            max_file_row_bytes=max_file_row_bytes,
        ),
    )
    output_batch_bytes = _video_output_batch_bytes(
        output_batch_size,
        height=height,
        width=width,
        max_file_string_bytes=max_file_string_bytes,
        max_file_row_bytes=max_file_row_bytes,
    )
    result: dict[str, object] = {
        "execution_backend": execution_backend,
        "batch_size": files_per_task,
        "output_batch_size": output_batch_size,
        "output_target_max_bytes": max(int(max_partition_bytes) * 2, output_batch_bytes, 1),
        "preserve_compute_batch_boundaries": True,
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
            max_pixels=max_pixels,
            buffer_size=buffer_size,
            max_file_string_bytes=max_file_string_bytes,
            max_file_row_bytes=max_file_row_bytes,
            output_batch_size=output_batch_size,
        )
        result["memory_bytes"] = max(configured_memory_bytes, peak_memory_bytes)
    if schema is not None:
        result["schema"] = schema
    return result


def _split_video_file_groups(
    files: Sequence[vane.VideoFile],
    task_count: int,
) -> list[tuple[vane.VideoFile, ...]]:
    if not files:
        return []
    task_count = min(len(files), int(task_count))
    if task_count <= 0:
        raise ValueError("read_task_count must be positive")
    base_size, larger_group_count = divmod(len(files), task_count)
    groups: list[tuple[vane.VideoFile, ...]] = []
    offset = 0
    for task_index in range(task_count):
        group_size = base_size + (1 if task_index < larger_group_count else 0)
        groups.append(tuple(files[offset : offset + group_size]))
        offset += group_size
    return groups


def _video_source_file_groups(source: VideoFrameSource) -> list[tuple[vane.VideoFile, ...]]:
    if not source.files:
        return []
    if source.frame_limit is not None:
        return [source.files]
    if source.read_task_count is None:
        return [(value,) for value in source.files]
    return _split_video_file_groups(source.files, source.read_task_count)


def _sql_string_literal(value: str) -> str:
    """Encode arbitrary Python Unicode without placing it in a SQL string token."""
    return f"decode(from_hex('{value.encode('utf-8').hex()}'))"


def _sql_optional_string(value: str | None) -> str:
    return "NULL::VARCHAR" if value is None else _sql_string_literal(value)


def _sql_optional_bigint(value: int | None) -> str:
    return "NULL::BIGINT" if value is None else f"{int(value)}::BIGINT"


def _sql_optional_time(value: int | float | None) -> str:
    if value is None:
        return "NULL::VARCHAR"
    if isinstance(value, int):
        token = f"i:{value}"
    else:
        token = f"f:{value.hex()}"
    return _sql_string_literal(token)


def _sql_optional_boolean(value: bool | None) -> str:
    if value is None:
        return "NULL::BOOLEAN"
    return "TRUE::BOOLEAN" if value else "FALSE::BOOLEAN"


def _video_file_sql(value: vane.VideoFile) -> str:
    generic_file = (
        "file("
        f"{_sql_string_literal(value.url)}, "
        f"{_sql_optional_string(value.content_type)}, "
        f"{_sql_optional_bigint(value.position)}, "
        f"{_sql_optional_bigint(value.size)}, "
        f"{_sql_optional_string(value.checksum)}"
        ")"
    )
    return f"video_file({generic_file})"


def _video_frame_source_manifest_sql(source: VideoFrameSource) -> str:
    if not source.files:
        return (
            "select []::VIDEOFILE[] as video_files, 0::BIGINT as height, 0::BIGINT as width, "
            "0::BIGINT as max_partition_bytes, NULL::BIGINT as frame_limit, "
            "'i:0'::VARCHAR as start_time, NULL::VARCHAR as end_time, NULL::BOOLEAN as is_key_frame, "
            "NULL::VARCHAR as sample_interval_seconds, 0::BIGINT as buffer_size, "
            "0::UBIGINT as max_input_bytes, 0::BIGINT as max_decoded_frames, "
            "0::BIGINT as max_pixels, 0::BIGINT as max_file_string_bytes, "
            "0::BIGINT as max_file_row_bytes, 'raise'::VARCHAR as on_error where false"
        )

    frame_limit_sql = "NULL::BIGINT" if source.frame_limit is None else f"{source.frame_limit}::BIGINT"
    rows = ", ".join(
        "("
        f"list_value({', '.join(_video_file_sql(value) for value in group)}), "
        f"{source.height}, {source.width}, {source.max_partition_bytes}, {frame_limit_sql}, "
        f"{_sql_optional_time(source.start_time)}, {_sql_optional_time(source.end_time)}, "
        f"{_sql_optional_boolean(source.is_key_frame)}, "
        f"{_sql_optional_time(source.sample_interval_seconds)}, "
        f"{source.buffer_size}, {source.max_input_bytes}, {source.max_decoded_frames}, "
        f"{source.max_pixels}, {source.file_bounds.max_string_bytes}, "
        f"{source.file_bounds.max_row_bytes}, {_sql_string_literal(source.on_error)}"
        ")"
        for group in _video_source_file_groups(source)
    )
    return (
        "select video_files::VIDEOFILE[] as video_files, height::BIGINT as height, "
        "width::BIGINT as width, max_partition_bytes::BIGINT as max_partition_bytes, "
        "frame_limit::BIGINT as frame_limit, start_time::VARCHAR as start_time, "
        "end_time::VARCHAR as end_time, is_key_frame::BOOLEAN as is_key_frame, "
        "sample_interval_seconds::VARCHAR as sample_interval_seconds, buffer_size::BIGINT as buffer_size, "
        "max_input_bytes::UBIGINT as max_input_bytes, max_decoded_frames::BIGINT as max_decoded_frames, "
        "max_pixels::BIGINT as max_pixels, max_file_string_bytes::BIGINT as max_file_string_bytes, "
        "max_file_row_bytes::BIGINT as max_file_row_bytes, on_error::VARCHAR as on_error "
        f"from (values {rows}) as manifest("
        "video_files, height, width, max_partition_bytes, frame_limit, start_time, end_time, "
        "is_key_frame, sample_interval_seconds, buffer_size, max_input_bytes, max_decoded_frames, "
        "max_pixels, max_file_string_bytes, max_file_row_bytes, on_error)"
    )


def _manifest_column(table: pa.Table, name: str) -> list[object]:
    if name not in table.column_names:
        raise ValueError(f"VideoFrameSource manifest is missing column {name!r}")
    return table.column(name).to_pylist()


def _manifest_int(values: Sequence[object], index: int, name: str) -> int:
    value = values[index]
    if value is None:
        raise ValueError(f"VideoFrameSource manifest column {name!r} cannot be NULL")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"VideoFrameSource manifest column {name!r} must be an integer")
    return value


def _manifest_frame_limit(values: Sequence[object]) -> int | None:
    present = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("VideoFrameSource manifest frame_limit must be an integer or NULL")
        if value < 0:
            raise ValueError("VideoFrameSource manifest frame_limit must be non-negative")
        present.append(value)
    if not present:
        return None
    if len(present) != len(values) or any(value != present[0] for value in present):
        raise ValueError("VideoFrameSource manifest frame_limit must be constant")
    return present[0]


def _manifest_time(value: object, *, name: str, optional: bool = False) -> int | float | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"VideoFrameSource manifest column {name!r} cannot be NULL")
    if not isinstance(value, str):
        raise ValueError(f"VideoFrameSource manifest column {name!r} must be VARCHAR")
    kind, separator, payload = value.partition(":")
    if not separator or not payload:
        raise ValueError(f"VideoFrameSource manifest column {name!r} has an invalid time token")
    try:
        if kind == "i":
            return int(payload)
        if kind == "f":
            return float.fromhex(payload)
    except ValueError:
        pass
    raise ValueError(f"VideoFrameSource manifest column {name!r} has an invalid time token")


def _video_file_from_arrow(value: object) -> vane.VideoFile:
    import vane

    if not isinstance(value, dict) or tuple(value) != _FILE_FIELDS:
        raise ValueError("VideoFrameSource manifest contains a malformed VIDEOFILE value")
    url = value["url"]
    if not isinstance(url, str):
        raise ValueError("VideoFrameSource manifest VIDEOFILE url must be non-NULL VARCHAR")
    return vane.VideoFile(
        url,
        cast("str | None", value["content_type"]),
        cast("int | None", value["position"]),
        cast("int | None", value["size"]),
        cast("str | None", value["checksum"]),
    )


def _video_frame_source_map_batches(table: pa.Table) -> Iterator[pa.Table]:
    file_groups_raw = _manifest_column(table, "video_files")
    file_groups = [
        [] if group is None else [_video_file_from_arrow(value) for value in cast("list[object]", group)]
        for group in file_groups_raw
    ]
    heights = _manifest_column(table, "height")
    widths = _manifest_column(table, "width")
    partition_bytes = _manifest_column(table, "max_partition_bytes")
    frame_limits = _manifest_column(table, "frame_limit")
    start_times = _manifest_column(table, "start_time")
    end_times = _manifest_column(table, "end_time")
    key_frame_filters = _manifest_column(table, "is_key_frame")
    sample_intervals = _manifest_column(table, "sample_interval_seconds")
    buffer_sizes = _manifest_column(table, "buffer_size")
    max_input_bytes_values = _manifest_column(table, "max_input_bytes")
    max_decoded_frames_values = _manifest_column(table, "max_decoded_frames")
    max_pixels_values = _manifest_column(table, "max_pixels")
    max_file_string_bytes_values = _manifest_column(table, "max_file_string_bytes")
    max_file_row_bytes_values = _manifest_column(table, "max_file_row_bytes")
    error_modes = _manifest_column(table, "on_error")
    remaining = _manifest_frame_limit(frame_limits)

    for row_index, files in enumerate(file_groups):
        bounds = _FileStorageBounds(
            max_string_bytes=_manifest_int(
                max_file_string_bytes_values,
                row_index,
                "max_file_string_bytes",
            ),
            max_row_bytes=_manifest_int(max_file_row_bytes_values, row_index, "max_file_row_bytes"),
        )
        actual_bounds = _file_storage_bounds(files)
        if (
            bounds.max_string_bytes < actual_bounds.max_string_bytes
            or bounds.max_row_bytes < actual_bounds.max_row_bytes
        ):
            raise ValueError("VideoFrameSource manifest FILE storage bounds are smaller than its VIDEOFILE values")
        options = _make_decode_options(
            height=_manifest_int(heights, row_index, "height"),
            width=_manifest_int(widths, row_index, "width"),
            max_partition_bytes=_manifest_int(partition_bytes, row_index, "max_partition_bytes"),
            start_time=_manifest_time(start_times[row_index], name="start_time"),
            end_time=_manifest_time(end_times[row_index], name="end_time", optional=True),
            is_key_frame=key_frame_filters[row_index],
            sample_interval_seconds=_manifest_time(
                sample_intervals[row_index],
                name="sample_interval_seconds",
                optional=True,
            ),
            buffer_size=_manifest_int(buffer_sizes, row_index, "buffer_size"),
            max_input_bytes=_manifest_int(max_input_bytes_values, row_index, "max_input_bytes"),
            max_decoded_frames=_manifest_int(
                max_decoded_frames_values,
                row_index,
                "max_decoded_frames",
            ),
            max_pixels=_manifest_int(max_pixels_values, row_index, "max_pixels"),
            on_error=error_modes[row_index],
            file_bounds=bounds,
        )
        target_rows = _video_source_udf_output_batch_size(
            options.height,
            options.width,
            options.max_partition_bytes,
            max_file_string_bytes=bounds.max_string_bytes,
            max_file_row_bytes=bounds.max_row_bytes,
        )

        def decode_group() -> Iterator[pa.RecordBatch]:
            nonlocal remaining
            for value in files:
                if remaining is not None and remaining <= 0:
                    return
                for batch in _decode_video_guarded(
                    value,
                    options=options,
                    max_output_frames=remaining,
                ):
                    if remaining is not None:
                        if batch.num_rows > remaining:
                            batch = batch.slice(0, remaining)
                        remaining -= batch.num_rows
                    yield batch
                    if remaining is not None and remaining <= 0:
                        return

        yield from _coalesce_video_frame_batches(decode_group(), target_rows=target_rows)
        if remaining is not None and remaining <= 0:
            return


class VideoFrameTask(DataSourceTask):
    """Decode one governed VIDEOFILE into bounded Arrow frame batches."""

    def __init__(
        self,
        video_file: object,
        *,
        options: _VideoDecodeOptions,
    ) -> None:
        self.video_file = _normalize_video_file(video_file)
        self.options = options

    def execute(self) -> Iterator[pa.RecordBatch]:
        yield from _decode_video_guarded(
            self.video_file,
            options=self.options,
            max_output_frames=None,
        )


class LimitedVideoFrameTask(DataSourceTask):
    """Decode VIDEOFILE values in manifest order up to one global output limit."""

    def __init__(
        self,
        files: Iterable[object],
        *,
        options: _VideoDecodeOptions,
        max_frames: int,
    ) -> None:
        self.files = _normalize_video_files(files)
        self.options = options
        normalized_max_frames = _optional_nonnegative_int(max_frames, name="max_frames")
        assert normalized_max_frames is not None
        self.max_frames = normalized_max_frames

    def execute(self) -> Iterator[pa.RecordBatch]:
        remaining = self.max_frames
        for value in self.files:
            if remaining <= 0:
                return
            for batch in _decode_video_guarded(
                value,
                options=self.options,
                max_output_frames=remaining,
            ):
                remaining -= batch.num_rows
                yield batch
                if remaining <= 0:
                    return


class _VideoFrameGroupTask(DataSourceTask):
    def __init__(self, files: Sequence[vane.VideoFile], options: _VideoDecodeOptions) -> None:
        self.files = tuple(files)
        self.options = options

    def execute(self) -> Iterator[pa.RecordBatch]:
        for value in self.files:
            yield from _decode_video_guarded(
                value,
                options=self.options,
                max_output_frames=None,
            )


class VideoFrameSource(DataSource):
    """Stream selected VIDEOFILE frames as bounded distributed rows.

    Strings and path-like values are convenience inputs and become VIDEOFILE
    values without I/O. Generic FILE values preserve all five fields while
    acquiring video semantics. IMAGEFILE and AUDIOFILE values are rejected.

    ``on_error="skip"`` suppresses only :class:`vane.VideoFileFormatError`.
    Filesystem, permission, Secret, cancellation, resource-limit, dependency,
    and internal failures always propagate.
    """

    def __init__(
        self,
        files: Iterable[object],
        height: int = 640,
        width: int = 480,
        max_partition_bytes: int = _DEFAULT_MAX_PARTITION_BYTES,
        frame_limit: int | None = None,
        read_task_count: int | None = None,
        *,
        start_time: int | float = 0,
        end_time: int | float | None = None,
        is_key_frame: bool | None = None,
        sample_interval_seconds: int | float | None = None,
        buffer_size: int = _DEFAULT_VIDEO_BUFFER_SIZE,
        max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
        max_decoded_frames: int = _DEFAULT_MAX_DECODED_FRAMES,
        max_pixels: int = _DEFAULT_MAX_PIXELS,
        on_error: str = "raise",
    ) -> None:
        self.files = _normalize_video_files(files)
        self.file_bounds = _file_storage_bounds(self.files)
        self.frame_limit = _optional_nonnegative_int(frame_limit, name="frame_limit")
        if read_task_count is not None:
            read_task_count = _positive_int(read_task_count, name="read_task_count")
        self.read_task_count = read_task_count
        self.options = _make_decode_options(
            height=height,
            width=width,
            max_partition_bytes=max_partition_bytes,
            start_time=start_time,
            end_time=end_time,
            is_key_frame=is_key_frame,
            sample_interval_seconds=sample_interval_seconds,
            buffer_size=buffer_size,
            max_input_bytes=max_input_bytes,
            max_decoded_frames=max_decoded_frames,
            max_pixels=max_pixels,
            on_error=on_error,
            file_bounds=self.file_bounds,
        )

    @property
    def height(self) -> int:
        return self.options.height

    @property
    def width(self) -> int:
        return self.options.width

    @property
    def max_partition_bytes(self) -> int:
        return self.options.max_partition_bytes

    @property
    def start_time(self) -> int | float:
        return self.options.start_time

    @property
    def end_time(self) -> int | float | None:
        return self.options.end_time

    @property
    def is_key_frame(self) -> bool | None:
        return self.options.is_key_frame

    @property
    def sample_interval_seconds(self) -> int | float | None:
        return self.options.sample_interval_seconds

    @property
    def buffer_size(self) -> int:
        return self.options.buffer_size

    @property
    def max_input_bytes(self) -> int:
        return self.options.max_input_bytes

    @property
    def max_decoded_frames(self) -> int:
        return self.options.max_decoded_frames

    @property
    def max_pixels(self) -> int:
        return self.options.max_pixels

    @property
    def on_error(self) -> str:
        return self.options.on_error

    @property
    def schema(self) -> dict[str, object]:
        return {
            "file": "VIDEOFILE",
            "frame_index": "BIGINT",
            "frame_time": "DOUBLE",
            "frame_time_base_numerator": "BIGINT",
            "frame_time_base_denominator": "BIGINT",
            "frame_pts": "BIGINT",
            "frame_dts": "BIGINT",
            "frame_duration": "BIGINT",
            "is_key_frame": "BOOLEAN",
            "frame": {
                "kind": "tensor",
                "dtype": "UINT8",
                "shape": [self.height, self.width, 3],
            },
        }

    def get_tasks(self) -> Iterator[DataSourceTask]:
        if self.frame_limit is not None:
            yield LimitedVideoFrameTask(
                self.files,
                options=self.options,
                max_frames=self.frame_limit,
            )
            return
        for group in _video_source_file_groups(self):
            if len(group) == 1:
                yield VideoFrameTask(group[0], options=self.options)
            else:
                yield _VideoFrameGroupTask(group, self.options)

    def to_udf_relation(self, con: Any) -> Any:
        import vane

        manifest = con.sql(_video_frame_source_manifest_sql(self))
        udf_kwargs = _video_source_udf_kwargs(
            height=self.height,
            width=self.width,
            max_partition_bytes=self.max_partition_bytes,
            max_pixels=self.max_pixels,
            buffer_size=self.buffer_size,
            max_file_string_bytes=self.file_bounds.max_string_bytes,
            max_file_row_bytes=self.file_bounds.max_row_bytes,
            frame_limit=self.frame_limit,
            file_count=len(_video_source_file_groups(self)),
            pre_grouped_files=True,
            schema={
                "file": vane.file_type(vane.MediaType.video()),
                "frame_index": vane.sqltypes.BIGINT,
                "frame_time": vane.sqltypes.DOUBLE,
                "frame_time_base_numerator": vane.sqltypes.BIGINT,
                "frame_time_base_denominator": vane.sqltypes.BIGINT,
                "frame_pts": vane.sqltypes.BIGINT,
                "frame_dts": vane.sqltypes.BIGINT,
                "frame_duration": vane.sqltypes.BIGINT,
                "is_key_frame": vane.sqltypes.BOOLEAN,
                "frame": vane.tensor_type(
                    vane.sqltypes.UTINYINT,
                    (self.height, self.width, 3),
                ),
            },
        )
        return manifest.map_batches(_video_frame_source_map_batches, **udf_kwargs)


__all__ = [
    "LimitedVideoFrameTask",
    "VideoFrameSource",
    "VideoFrameTask",
    "VideoReadError",
]
