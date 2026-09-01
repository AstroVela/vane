# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Encoded VIDEOFILE metadata helpers."""

from __future__ import annotations

import importlib
import io
import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import vane
from vane._expressions import as_expression

DEFAULT_VIDEO_METADATA_BYTES = 8 * 1024 * 1024
MAX_VIDEO_METADATA_BYTES = 64 * 1024 * 1024
_MAX_VIDEO_METADATA_FETCH_BYTES = 64 * 1024
_VIDEO_METADATA_FETCHES_PER_BUDGET = 8
_MAX_VIDEO_METADATA_FETCHES = 1024
_VIDEO_METADATA_TIMEOUT_SECONDS = 5.0
# Permit 8K UHD metadata while bounding decoder allocation before dimensions
# become available to Vane's post-probe validation.
_MAX_VIDEO_FRAME_PIXELS = 32 * 1024 * 1024
_MAX_VIDEO_AUDIO_SAMPLES = 1024 * 1024
_MAX_VIDEO_STREAMS = 64
_MAX_VIDEO_PROBE_PACKETS = 256
_MAX_VIDEO_INDEX_BYTES = 256 * 1024
_MAX_VIDEO_ANALYZE_DURATION_US = 5 * 1_000_000
_MAX_VIDEO_FPS_PROBE_FRAMES = 32
_MAX_BIGINT = (1 << 63) - 1
_MAX_UINTEGER = (1 << 32) - 1

_PYAV_UNSAFE_STREAM_OPTIONS_ERROR = (
    "stream_options were provided, but this format does not expose its streams before avformat_find_stream_info"
)

_MIME_ALIASES = {
    "application/ogg": "video/ogg",
    "video/avi": "video/x-msvideo",
    "video/mkv": "video/x-matroska",
    "video/mp2t": "video/mp2t",
    "video/x-m4v": "video/mp4",
    "video/x-matroska": "video/x-matroska",
}
_GENERIC_MIME_TYPES = frozenset({"application/octet-stream", "binary/octet-stream", "video/*"})
_FORMAT_MIME_TYPES: dict[str, frozenset[str]] = {
    "3g2": frozenset({"video/3gpp2"}),
    "3gp": frozenset({"video/3gpp"}),
    "asf": frozenset({"video/x-ms-asf", "video/x-ms-wmv"}),
    "avi": frozenset({"video/x-msvideo"}),
    "flv": frozenset({"video/x-flv"}),
    "matroska": frozenset({"video/x-matroska", "video/webm"}),
    "mj2": frozenset({"video/mj2"}),
    "mov": frozenset({"video/mp4", "video/quicktime"}),
    "mp4": frozenset({"video/mp4", "video/quicktime"}),
    "mpeg": frozenset({"video/mpeg"}),
    "mpegts": frozenset({"video/mp2t"}),
    "ogg": frozenset({"video/ogg"}),
    "webm": frozenset({"video/webm", "video/x-matroska"}),
}


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Container metadata for the first non-attached video stream of a VideoFile."""

    width: int
    height: int
    fps: float | None
    duration: float | None
    frame_count: int | None
    time_base: Fraction


class VideoFileError(RuntimeError):
    """Base class for failures caused by encoded video content."""


class VideoFileFormatError(VideoFileError):
    """The logical FILE view is not a supported, self-contained video."""


class VideoFileLimitError(VideoFileError):
    """A VideoFile operation exceeded an explicit resource limit."""


@contextmanager
def _close_container(container: Any) -> Iterator[Any]:
    """Close a PyAV container without replacing its operation's exception."""
    try:
        yield container
    except BaseException:
        try:
            container.close()
        except BaseException:
            pass
        raise
    else:
        container.close()


class _VideoMetadataView:
    """Expose bounded, range-aware reads over a logical FILE view."""

    def __init__(
        self,
        read_at: Callable[[int, int], bytes],
        *,
        logical_size: int,
        max_bytes: int,
    ) -> None:
        self._read_at = read_at
        self._logical_size = logical_size
        self._max_bytes = max_bytes
        self._fetch_size = min(
            _MAX_VIDEO_METADATA_FETCH_BYTES,
            max(1, max_bytes // _VIDEO_METADATA_FETCHES_PER_BUDGET),
        )
        self._position = 0
        self._bytes_read = 0
        self._fetches = 0
        self._cache_starts: list[int] = []
        self._cache: list[tuple[int, bytes]] = []
        self._error: BaseException | None = None
        self.budget_exhausted = False
        self.fetch_limit_exhausted = False

    def _cached_at(self, position: int) -> tuple[int, bytes] | None:
        index = bisect_right(self._cache_starts, position) - 1
        if index >= 0:
            start, data = self._cache[index]
            if position < start + len(data):
                return start, data
        return None

    def _next_cached_start(self, position: int) -> int:
        index = bisect_right(self._cache_starts, position)
        return self._cache_starts[index] if index < len(self._cache_starts) else self._logical_size

    def _previous_cached_end(self, position: int) -> int:
        index = bisect_right(self._cache_starts, position) - 1
        if index < 0:
            return 0
        start, data = self._cache[index]
        return min(position, start + len(data))

    def _cache_bytes(self, start: int, data: bytes) -> None:
        insert_at = bisect_left(self._cache_starts, start)
        if insert_at > 0:
            previous_start, previous_data = self._cache[insert_at - 1]
            if previous_start + len(previous_data) > start:
                raise RuntimeError("video metadata cache received overlapping ranges")
        if insert_at < len(self._cache) and start + len(data) > self._cache[insert_at][0]:
            raise RuntimeError("video metadata cache received overlapping ranges")
        self._cache_starts.insert(insert_at, start)
        self._cache.insert(insert_at, (start, data))

    def _remember(self, error: BaseException) -> None:
        if self._error is None:
            self._error = error

    def _read(self, size: int | None) -> bytes:
        if size == 0 or self._position >= self._logical_size:
            return b""
        if size is None or size < 0:
            requested_end = self._logical_size
        else:
            requested_end = min(self._logical_size, self._position + size)

        result = bytearray()
        position = self._position
        while position < requested_end:
            cached = self._cached_at(position)
            if cached is not None:
                cached_start, cached_data = cached
                cached_offset = position - cached_start
                copy_size = min(requested_end - position, len(cached_data) - cached_offset)
                result.extend(cached_data[cached_offset : cached_offset + copy_size])
                position += copy_size
                continue

            remaining_budget = self._max_bytes - self._bytes_read
            if remaining_budget <= 0:
                self.budget_exhausted = True
                break
            if self._fetches >= _MAX_VIDEO_METADATA_FETCHES:
                self.fetch_limit_exhausted = True
                break

            block_start = position - position % self._fetch_size
            fetch_start = max(block_start, self._previous_cached_end(position))
            fetch_end = min(
                self._logical_size,
                self._next_cached_start(position),
                max(requested_end, block_start + self._fetch_size),
            )
            if remaining_budget <= position - fetch_start:
                fetch_start = position
            fetch_size = min(fetch_end - fetch_start, remaining_budget)
            if fetch_size <= 0:
                self.budget_exhausted = True
                break
            fetched = self._read_at(fetch_start, fetch_size)
            if not isinstance(fetched, bytes) or len(fetched) != fetch_size:
                raise OSError(
                    f"video metadata source returned {len(fetched) if isinstance(fetched, bytes) else 0} bytes "
                    f"after requesting {fetch_size}"
                )
            self._bytes_read += len(fetched)
            self._fetches += 1
            self._cache_bytes(fetch_start, fetched)

        self._position = position
        return bytes(result)

    def read(self, size: int | None = -1, /) -> bytes:
        if self._error is not None:
            return b""
        try:
            return self._read(size)
        except BaseException as error:
            self._remember(error)
            return b""

    def _seek(self, offset: int, whence: int) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._logical_size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
        if self._error is not None:
            return -1
        try:
            return self._seek(offset, whence)
        except BaseException as error:
            self._remember(error)
            return -1

    def tell(self) -> int:
        return -1 if self._error is not None else self._position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error.with_traceback(self._error.__traceback__)


class _NestedIOBlocker:
    """Prevent a container manifest from bypassing the FILE resolver."""

    def __init__(self) -> None:
        self.error: VideoFileFormatError | None = None

    def __call__(self, url: str, flags: int, options: dict[str, str]) -> Any:
        # Do not echo a nested URL because manifests can contain signed query
        # parameters or embedded credentials.
        del url, flags, options
        self.error = VideoFileFormatError("video metadata does not permit nested external resources")
        raise self.error

    def raise_if_error(self) -> None:
        if self.error is not None:
            raise self.error.with_traceback(self.error.__traceback__)


def _load_av() -> Any:
    try:
        av_module = importlib.import_module("av")
        av_module.open
        av_module.error.ExitError
        av_module.error.FFmpegError
        av_module.stream.Disposition.attached_pic
        av_module.time_base
    except (AttributeError, ImportError, OSError) as error:
        raise ImportError(
            "VideoFile metadata requires the 'av' package and its bundled FFmpeg libraries. "
            "Please `pip install 'vane-ai[video]'`."
        ) from error
    return av_module


def _positive_limit(value: object, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, not {type(value).__name__!r}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _limit_expression(value: int | vane.Expression, *, name: str, maximum: int) -> vane.Expression:
    if isinstance(value, vane.Expression):
        return value
    return as_expression(_positive_limit(value, name=name, maximum=maximum))


def _canonical_mime_type(value: str) -> str:
    normalized = value.partition(";")[0].strip().lower()
    return _MIME_ALIASES.get(normalized, normalized)


def _container_format_names(container: Any) -> frozenset[str]:
    container_format = container.format
    format_name = None if container_format is None else container_format.name
    if not isinstance(format_name, str) or not format_name:
        raise VideoFileFormatError("video parser did not report a container format")
    names = frozenset(part.strip().lower() for part in format_name.split(",") if part.strip())
    if not names:
        raise VideoFileFormatError("video parser reported an invalid container format")
    return names


def _validate_content_type(content_type: str | None, format_names: frozenset[str]) -> None:
    compatible: set[str] = set()
    for format_name in format_names:
        compatible.update(_FORMAT_MIME_TYPES.get(format_name, ()))
    if not compatible:
        raise VideoFileFormatError(f"unsupported video container format {','.join(sorted(format_names))!r}")

    if content_type is None:
        return
    declared = _canonical_mime_type(content_type)
    if declared in _GENERIC_MIME_TYPES:
        return
    if not declared.startswith("video/"):
        raise VideoFileFormatError(f"VIDEOFILE content_type {content_type!r} contradicts the video content")
    if declared not in compatible:
        raise VideoFileFormatError(
            f"VIDEOFILE content_type {content_type!r} contradicts detected container format "
            f"{','.join(sorted(format_names))!r}"
        )


def _positive_fraction(value: Any, *, name: str) -> Fraction:
    try:
        result = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise VideoFileFormatError(f"video parser reported an invalid {name}") from error
    if result <= 0 or result.numerator > _MAX_BIGINT or result.denominator > _MAX_BIGINT:
        raise VideoFileFormatError(f"video parser reported an out-of-range {name}")
    return result


def _optional_rate(video: Any) -> float | None:
    for attribute in ("average_rate", "guessed_rate"):
        rate = getattr(video, attribute)
        if rate is None:
            continue
        try:
            exact_rate = Fraction(rate)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise VideoFileFormatError("video parser reported an invalid frame rate") from error
        # FFmpeg uses a zero rational when this rate is unavailable. Let the
        # guessed rate fill an unavailable average rate, then preserve NULL if
        # neither source provides a positive value.
        if exact_rate == 0:
            continue
        if exact_rate < 0 or exact_rate.numerator > _MAX_BIGINT or exact_rate.denominator > _MAX_BIGINT:
            raise VideoFileFormatError("video parser reported an out-of-range frame rate")
        result = float(exact_rate)
        if not math.isfinite(result):
            raise VideoFileFormatError("video parser reported a non-finite frame rate")
        return result
    return None


def _optional_duration(container: Any, video: Any, time_base: Fraction, av_module: Any) -> float | None:
    stream_duration = video.duration
    if stream_duration is not None:
        stream_duration = int(stream_duration)
        if stream_duration < 0:
            raise VideoFileFormatError("video parser reported a negative stream duration")
        if stream_duration > 0:
            exact_duration = stream_duration * time_base
            result = float(exact_duration)
            if not math.isfinite(result):
                raise VideoFileFormatError("video parser reported a non-finite duration")
            return result

    if container.duration is not None:
        container_duration = int(container.duration)
        if container_duration < 0:
            raise VideoFileFormatError("video parser reported a negative container duration")
        if container_duration > 0:
            exact_duration = Fraction(container_duration, int(av_module.time_base))
            result = float(exact_duration)
            if not math.isfinite(result):
                raise VideoFileFormatError("video parser reported a non-finite duration")
            return result

    # Zero is also FFmpeg's unknown-duration sentinel. Returning NULL avoids
    # claiming that an unindexed stream is an exact zero-length video.
    return None


def _optional_frame_count(video: Any) -> int | None:
    if video.frames is None:
        return None
    frame_count = int(video.frames)
    if frame_count < 0 or frame_count > _MAX_BIGINT:
        raise VideoFileFormatError("video parser reported an out-of-range frame count")
    # FFmpeg uses zero when the container does not carry an exact frame count.
    return frame_count or None


def _metadata_from_container(container: Any, content_type: str | None, av_module: Any) -> VideoMetadata:
    attached_pic = av_module.stream.Disposition.attached_pic
    video = next(
        (stream for stream in container.streams if stream.type == "video" and not (stream.disposition & attached_pic)),
        None,
    )
    if video is None:
        raise VideoFileFormatError("logical FILE view does not contain a video stream")

    format_names = _container_format_names(container)
    _validate_content_type(content_type, format_names)
    if getattr(video, "codec_context", None) is None:
        raise VideoFileFormatError("video stream does not have an available decoder")

    width = int(video.width)
    height = int(video.height)
    if width <= 0 or width > _MAX_UINTEGER or height <= 0 or height > _MAX_UINTEGER:
        raise VideoFileFormatError(f"video dimensions must fit positive UINTEGER values, found {width}x{height}")
    if width * height > _MAX_VIDEO_FRAME_PIXELS:
        raise VideoFileLimitError(
            f"video dimensions {width}x{height} exceed the metadata pixel limit of {_MAX_VIDEO_FRAME_PIXELS}"
        )
    time_base = _positive_fraction(video.time_base, name="time base")
    return VideoMetadata(
        width=width,
        height=height,
        fps=_optional_rate(video),
        duration=_optional_duration(container, video, time_base, av_module),
        frame_count=_optional_frame_count(video),
        time_base=time_base,
    )


def _probe_video_metadata(
    read_at: Callable[[int, int], bytes],
    logical_size: int,
    content_type: str | None,
    max_bytes: int,
) -> tuple[int, int, float | None, float | None, int | None, int, int]:
    """Bounded PyAV helper called by the native SQL scalar function."""
    av_module = _load_av()
    stream = _VideoMetadataView(read_at, logical_size=logical_size, max_bytes=max_bytes)
    nested_io = _NestedIOBlocker()
    decoder_options = {
        "max_pixels": str(_MAX_VIDEO_FRAME_PIXELS),
        "max_samples": str(_MAX_VIDEO_AUDIO_SAMPLES),
        "skip_frame": "all",
        "threads": "1",
    }
    probe_options = {
        "analyzeduration": str(_MAX_VIDEO_ANALYZE_DURATION_US),
        "formatprobesize": str(max(2048, max_bytes)),
        "fpsprobesize": str(_MAX_VIDEO_FPS_PROBE_FRAMES),
        "indexmem": str(_MAX_VIDEO_INDEX_BYTES),
        "max_probe_packets": str(_MAX_VIDEO_PROBE_PACKETS),
        "max_streams": str(_MAX_VIDEO_STREAMS),
        "probesize": str(max(32, max_bytes)),
        "skip_estimate_duration_from_pts": "1",
    }
    try:
        container = av_module.open(
            stream,
            mode="r",
            options=decoder_options,
            container_options=probe_options,
            # PyAV 17.1 fails closed when streams are not available before
            # avformat_find_stream_info(), because per-stream decoder limits
            # cannot otherwise be applied safely.
            stream_options=[decoder_options.copy()],
            metadata_encoding="utf-8",
            metadata_errors="replace",
            buffer_size=min(_MAX_VIDEO_METADATA_FETCH_BYTES, max_bytes),
            timeout=(_VIDEO_METADATA_TIMEOUT_SECONDS, _VIDEO_METADATA_TIMEOUT_SECONDS),
            io_open=nested_io,
        )
        with _close_container(container):
            stream.raise_if_error()
            nested_io.raise_if_error()
            metadata = _metadata_from_container(container, content_type, av_module)
            stream.raise_if_error()
            nested_io.raise_if_error()
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        stream.raise_if_error()
        nested_io.raise_if_error()
        if isinstance(error, VideoFileError):
            raise
        if isinstance(error, av_module.error.ExitError):
            raise VideoFileLimitError(
                f"a video metadata probe phase exceeded its {_VIDEO_METADATA_TIMEOUT_SECONDS:g}-second timeout"
            ) from error
        if isinstance(error, ValueError) and str(error).startswith(_PYAV_UNSAFE_STREAM_OPTIONS_ERROR):
            raise VideoFileFormatError(
                "video container cannot be inspected safely within metadata resource limits"
            ) from error
        if not isinstance(error, av_module.error.FFmpegError):
            raise
        if stream.budget_exhausted:
            raise VideoFileLimitError(f"video metadata requires more than max_bytes={max_bytes}") from error
        if stream.fetch_limit_exhausted:
            raise VideoFileLimitError(
                f"video metadata requires more than {_MAX_VIDEO_METADATA_FETCHES} source ranges"
            ) from error
        raise VideoFileFormatError("logical FILE view is not a supported encoded video") from error
    if stream.budget_exhausted:
        raise VideoFileLimitError(f"video metadata requires more than max_bytes={max_bytes}")
    if stream.fetch_limit_exhausted:
        raise VideoFileLimitError(f"video metadata requires more than {_MAX_VIDEO_METADATA_FETCHES} source ranges")
    return (
        metadata.width,
        metadata.height,
        metadata.fps,
        metadata.duration,
        metadata.frame_count,
        metadata.time_base.numerator,
        metadata.time_base.denominator,
    )


def _video_file_metadata_value(
    value: vane.VideoFile,
    *,
    max_bytes: int = DEFAULT_VIDEO_METADATA_BYTES,
    connection: vane.DuckDBPyConnection | None = None,
) -> VideoMetadata:
    normalized_max_bytes = _positive_limit(max_bytes, name="max_bytes", maximum=MAX_VIDEO_METADATA_BYTES)
    _load_av()
    with value.open(buffer_size=1, connection=connection) as reader:
        logical_size = reader.size()

        def read_at(offset: int, size: int) -> bytes:
            reader.seek(offset)
            return reader.read(size)

        fields = _probe_video_metadata(read_at, logical_size, value.content_type, normalized_max_bytes)
    return VideoMetadata(
        width=fields[0],
        height=fields[1],
        fps=fields[2],
        duration=fields[3],
        frame_count=fields[4],
        time_base=Fraction(fields[5], fields[6]),
    )


def video_metadata(
    value: vane.VideoFile | vane.Expression,
    *,
    max_bytes: int | vane.Expression = DEFAULT_VIDEO_METADATA_BYTES,
) -> vane.Expression:
    """Inspect the first non-attached video stream with bounded stream-info probing."""
    return vane.FunctionExpression(
        "video_metadata",
        as_expression(value),
        _limit_expression(max_bytes, name="max_bytes", maximum=MAX_VIDEO_METADATA_BYTES),
    )


__all__ = [
    "VideoFileError",
    "VideoFileFormatError",
    "VideoFileLimitError",
    "VideoMetadata",
    "video_metadata",
]
