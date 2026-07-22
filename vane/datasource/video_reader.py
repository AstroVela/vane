# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming video frame DataSource using decord.

Produces video frames as a DuckDB DataSource with ~10MB streaming partitions.
Each video file is an independent DataSourceTask that decodes frames via decord
and yields them as Arrow fixed-shape tensor columns.

Usage::

    from vane.datasource.video_reader import VideoFrameSource

    source = VideoFrameSource(["video1.avi", "video2.avi"], height=640, width=480)
    rel = con.read_datasource(source)
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
import pyarrow as pa
from PIL import Image

from vane.datasource import DataSource, DataSourceTask

# The generic datasource keeps its historical 10 MiB default. The aligned
# video benchmark explicitly supplies Ray Data's 128 MiB soft block target.
_DEFAULT_MAX_PARTITION_BYTES = int(os.environ.get("VANE_VIDEO_MAX_PARTITION_BYTES", str(10 * 1024 * 1024)))
_DEFAULT_VIDEO_SOURCE_UDF_MEMORY_BYTES = 512 * 1024**2
# Ray Data read/map tasks use one CPU by default. Keep the decode/resize
# worker at that width unless the caller explicitly requests otherwise.
_VIDEO_RESIZE_THREADS = max(1, int(os.environ.get("VANE_VIDEO_RESIZE_THREADS", "1")))


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
_MAX_CONCURRENT_DECODES = int(os.environ.get("VANE_MAX_CONCURRENT_DECODES", "1"))
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
    import psutil

    def has_capacity(mem) -> bool:
        if _MEM_MIN_AVAILABLE_BYTES > 0 and mem.available >= _MEM_MIN_AVAILABLE_BYTES:
            return True
        return mem.percent < _MEM_HIGH_WATERMARK

    def has_recovered(mem) -> bool:
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


def _read_s3_bytes(s3_path: str) -> bytes:
    """Read a file from S3 into memory via PyArrow's S3FileSystem."""
    import pyarrow.fs as pa_fs

    ak = os.environ.get("AWS_ACCESS_KEY_ID", "")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    ep = os.environ.get("AWS_ENDPOINT_URL", "")
    region = os.environ.get("AWS_REGION", "us-east-1")
    from urllib.parse import urlparse

    parsed = urlparse(ep)
    endpoint = parsed.netloc or parsed.path
    scheme = parsed.scheme or "http"
    kwargs = {"endpoint_override": endpoint, "region": region, "scheme": scheme}
    if ak or sk:
        kwargs["access_key"] = ak
        kwargs["secret_key"] = sk
        kwargs["anonymous"] = False
    else:
        kwargs["anonymous"] = True
    fs = pa_fs.S3FileSystem(**kwargs)
    obj_path = s3_path[len("s3://") :]
    with fs.open_input_file(obj_path) as f:
        return f.read()


def _build_frame_array(frames: np.ndarray) -> pa.ExtensionArray:
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected RGB frame batch with shape (N, H, W, 3), got {frames.shape!r}")
    return pa.FixedShapeTensorArray.from_numpy_ndarray(frames)


def _int64_array(values: list[int]) -> pa.Array:
    array = np.ascontiguousarray(values, dtype=np.int64)
    return pa.Array.from_buffers(pa.int64(), len(array), [None, pa.py_buffer(array)])


def _constant_string_array(value: str, count: int) -> pa.Array:
    encoded = value.encode("utf-8")
    offsets = np.arange(count + 1, dtype=np.int32) * len(encoded)
    data = encoded * count
    return pa.Array.from_buffers(pa.string(), count, [None, pa.py_buffer(offsets), pa.py_buffer(data)])


def _open_decord_reader(video_path: str):
    from decord import VideoReader

    if video_path.startswith("s3://"):
        import io as _io

        return VideoReader(_io.BytesIO(_read_s3_bytes(video_path)))
    return VideoReader(video_path)


def _resize_rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(frame)
    return np.array(pil_image.resize((width, height)))


def _resize_frame_batch(
    frames: list[np.ndarray],
    *,
    width: int,
    height: int,
    executor: ThreadPoolExecutor | None = None,
) -> list[np.ndarray]:
    if not frames:
        return []
    if _VIDEO_RESIZE_THREADS <= 1 or len(frames) == 1:
        return [_resize_rgb_frame(frame, width, height) for frame in frames]
    if executor is not None:
        return list(executor.map(_resize_rgb_frame, frames, repeat(width), repeat(height)))
    with ThreadPoolExecutor(max_workers=_VIDEO_RESIZE_THREADS) as executor:
        return list(executor.map(_resize_rgb_frame, frames, repeat(width), repeat(height)))


def _decode_video_batches(
    video_path: str,
    *,
    height: int,
    width: int,
    max_partition_bytes: int,
    max_frames: int | None = None,
) -> Iterator[pa.RecordBatch]:
    if max_frames is not None and max_frames <= 0:
        return

    open_start = time.perf_counter()
    reader = _open_decord_reader(video_path)
    open_s = time.perf_counter() - open_start
    vname = os.path.basename(video_path)
    batch_size = _video_source_udf_output_batch_size(height, width, max_partition_bytes)

    resized = np.empty((batch_size, height, width, 3), dtype=np.uint8)
    raw_frames: list[np.ndarray] = []
    indices: list[int] = []
    count = 0
    decode_s = 0.0
    batch_start = time.perf_counter()
    resize_executor = ThreadPoolExecutor(max_workers=_VIDEO_RESIZE_THREADS) if _VIDEO_RESIZE_THREADS > 1 else None

    def emit_resized_batch() -> Iterator[pa.RecordBatch]:
        nonlocal resized, count, raw_frames, indices, decode_s, batch_start, open_s
        current_count = count

        resize_start = time.perf_counter()
        resized_frames = _resize_frame_batch(raw_frames, width=width, height=height, executor=resize_executor)
        resize_s = time.perf_counter() - resize_start

        for out_idx, resized_frame in enumerate(resized_frames):
            resized[out_idx] = resized_frame

        flush_start = time.perf_counter()
        batch = _flush_frame_batch(vname, resized, indices, current_count)
        # FixedShapeTensorArray retains a zero-copy reference to ``resized``.
        # Transfer ownership of that buffer to the emitted Arrow batch before
        # the generator can resume; reusing it would mutate already published
        # blocks while Ray serializes or downstream operators consume them.
        resized = np.empty((batch_size, height, width, 3), dtype=np.uint8)
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

        raw_frames = []
        indices = []
        count = 0
        decode_s = 0.0
        open_s = 0.0
        batch_start = time.perf_counter()

    try:
        for frame_idx, frame in enumerate(reader):
            if max_frames is not None and frame_idx >= max_frames:
                break
            decode_start = time.perf_counter()
            arr = frame.asnumpy()
            decode_s += time.perf_counter() - decode_start
            raw_frames.append(arr)
            indices.append(frame_idx)
            count += 1

            if count >= batch_size:
                yield from emit_resized_batch()
        if count > 0:
            yield from emit_resized_batch()
    finally:
        if resize_executor is not None:
            resize_executor.shutdown()


def _flush_frame_batch(
    vname: str,
    resized: np.ndarray,
    indices: list[int],
    count: int,
) -> pa.RecordBatch:
    frame_arr = _build_frame_array(resized[:count])
    return pa.record_batch(
        {
            "video_path": _constant_string_array(vname, count),
            "frame_index": _int64_array(indices),
            "frame": frame_arr,
        }
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


def _video_source_udf_output_batch_size(height: int, width: int, max_partition_bytes: int) -> int:
    frame_bytes = max(1, int(height) * int(width) * 3)
    # Ray Data's block target is soft: the row that first crosses the target
    # is retained in the emitted block.
    return max(1, int(max_partition_bytes) // frame_bytes + 1)


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
    frame_limit: int | None = None,
    path_count: int = 1,
    pre_grouped_paths: bool = False,
    schema: dict[str, object] | None = None,
) -> dict[str, object]:
    videos_per_task = 1 if pre_grouped_paths else _video_source_udf_videos_per_task(frame_limit, path_count)
    execution_backend = _video_source_udf_backend()
    output_batch_size = _read_int_env(
        "VANE_VIDEO_SOURCE_UDF_OUTPUT_BATCH_SIZE",
        _video_source_udf_output_batch_size(height, width, max_partition_bytes),
        minimum=1,
    )
    udf_kwargs: dict[str, object] = {
        "execution_backend": execution_backend,
        "batch_size": videos_per_task,
        "output_batch_size": output_batch_size,
        # Ray Data's target_max_block_size is a soft threshold: the row that
        # crosses it remains in the block.  Vane's runtime byte limit is a
        # hard pre-append limit, so leaving both values at 128 MiB would split
        # a Ray-compatible 110-frame block back to 109 frames.  Row count is
        # the semantic flush boundary here; this larger guard only prevents a
        # second byte-based split of that already bounded output batch.
        "output_target_max_bytes": max(int(max_partition_bytes) * 2, 1),
        # One compute batch is one Ray-compatible read-task descriptor.  Its
        # final short block must be published before the next descriptor is
        # evaluated, just like Ray Data finalizes a block builder per read
        # task.
        "preserve_compute_batch_boundaries": True,
        "streaming_breaker": True,
        "cpus": _video_source_udf_cpus(),
    }
    if execution_backend == "ray_task":
        udf_kwargs["memory_bytes"] = (
            _read_optional_positive_int_env("VANE_VIDEO_SOURCE_UDF_MEMORY_BYTES")
            or _DEFAULT_VIDEO_SOURCE_UDF_MEMORY_BYTES
        )
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
            "0::BIGINT as max_partition_bytes, NULL::BIGINT as frame_limit where false"
        )

    frame_limit_sql = "NULL" if source.frame_limit is None else str(max(0, int(source.frame_limit)))
    path_groups = _video_source_path_groups(source)
    rows = ", ".join(
        "("
        f"list_value({', '.join(_sql_string_literal(path) for path in paths)}), "
        f"{int(source.height)}, "
        f"{int(source.width)}, "
        f"{int(source.max_partition_bytes)}, "
        f"{frame_limit_sql}"
        ")"
        for paths in path_groups
    )
    return (
        "select "
        "video_paths::VARCHAR[] as video_paths, "
        "height::BIGINT as height, "
        "width::BIGINT as width, "
        "max_partition_bytes::BIGINT as max_partition_bytes, "
        "frame_limit::BIGINT as frame_limit "
        f"from (values {rows}) as manifest(video_paths, height, width, max_partition_bytes, frame_limit)"
    )


def _manifest_int_value(values: list[object], index: int, name: str) -> int:
    value = values[index]
    if value is None:
        raise ValueError(f"VideoFrameSource manifest column {name!r} cannot be NULL")
    return int(value)


def _manifest_frame_limit(values: list[object]) -> int | None:
    for value in values:
        if value is not None:
            return max(0, int(value))
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
    remaining = _manifest_frame_limit(frame_limits)

    for row_idx, paths in enumerate(path_groups):
        height = _manifest_int_value(heights, row_idx, "height")
        width = _manifest_int_value(widths, row_idx, "width")
        max_partition_bytes = _manifest_int_value(max_partition_bytes_values, row_idx, "max_partition_bytes")
        target_rows = _video_source_udf_output_batch_size(height, width, max_partition_bytes)

        def decode_group(
            paths: list[str | None] = paths,
            height: int = height,
            width: int = width,
            max_partition_bytes: int = max_partition_bytes,
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
                    for batch in _decode_video_batches(
                        path,
                        height=height,
                        width=width,
                        max_partition_bytes=max_partition_bytes,
                        max_frames=max_frames,
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
    ):
        self.video_path = video_path
        self.height = height
        self.width = width
        self.max_partition_bytes = max_partition_bytes

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
        yield from _decode_video_batches(
            self.video_path,
            height=self.height,
            width=self.width,
            max_partition_bytes=self.max_partition_bytes,
        )

    def _flush(
        self,
        vname: str,
        resized: np.ndarray,
        indices: list[int],
        count: int,
    ) -> pa.RecordBatch:
        """Build a RecordBatch from the accumulated frame buffer."""
        return _flush_frame_batch(vname, resized, indices, count)


class LimitedVideoFrameTask(DataSourceTask):
    """Decode videos in manifest order until a global frame limit is reached."""

    def __init__(
        self,
        paths: list[str],
        height: int,
        width: int,
        max_frames: int,
        max_partition_bytes: int = _DEFAULT_MAX_PARTITION_BYTES,
    ):
        self.paths = paths
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.max_partition_bytes = max_partition_bytes

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
            for batch in _decode_video_batches(
                path,
                height=self.height,
                width=self.width,
                max_partition_bytes=self.max_partition_bytes,
                max_frames=remaining,
            ):
                remaining -= batch.num_rows
                yield batch
                if remaining <= 0:
                    break


class VideoFrameSource(DataSource):
    """DataSource that streams video frames from local or S3 video files.

    Each video file becomes an independent task. Frames are decoded using
    decord and resized to (height, width). Output batches are ~10MB each.
    """

    def __init__(
        self,
        paths: list[str],
        height: int = 640,
        width: int = 480,
        max_partition_bytes: int = _DEFAULT_MAX_PARTITION_BYTES,
        frame_limit: int | None = None,
        read_task_count: int | None = None,
    ):
        self.paths = paths
        self.height = height
        self.width = width
        self.max_partition_bytes = max_partition_bytes
        self.frame_limit = frame_limit
        if read_task_count is not None and int(read_task_count) <= 0:
            raise ValueError("read_task_count must be positive")
        self.read_task_count = None if read_task_count is None else int(read_task_count)

    @property
    def schema(self) -> dict[str, object]:
        return {
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
            )
            return
        for path in self.paths:
            yield VideoFrameTask(
                video_path=path,
                height=self.height,
                width=self.width,
                max_partition_bytes=self.max_partition_bytes,
            )

    def to_udf_relation(self, con):
        import vane

        manifest = con.sql(_video_frame_source_manifest_sql(self))
        udf_kwargs = _video_source_udf_kwargs(
            height=self.height,
            width=self.width,
            max_partition_bytes=self.max_partition_bytes,
            frame_limit=self.frame_limit,
            path_count=len(_video_source_path_groups(self)),
            pre_grouped_paths=True,
            schema={
                "video_path": vane.sqltypes.VARCHAR,
                "frame_index": vane.sqltypes.BIGINT,
                "frame": vane.tensor_type(vane.sqltypes.UTINYINT, (self.height, self.width, 3)),
            },
        )
        return manifest.map_batches(_video_frame_source_map_batches, **udf_kwargs)
