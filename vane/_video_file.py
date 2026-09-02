# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Encoded VIDEOFILE metadata and streaming frame helpers."""

from __future__ import annotations

import importlib
import io
import math
import operator
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import vane
from vane._expressions import as_expression
from vane._file import _positive_buffer_size

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage  # type: ignore[import-not-found]
else:
    PILImage = Any

DEFAULT_VIDEO_METADATA_BYTES = 8 * 1024 * 1024
DEFAULT_VIDEO_BUFFER_SIZE = 1024 * 1024
DEFAULT_VIDEO_MAX_INPUT_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_VIDEO_MAX_FRAMES = 1_000_000
DEFAULT_VIDEO_MAX_PIXELS = 32 * 1024 * 1024
MAX_VIDEO_METADATA_BYTES = 64 * 1024 * 1024
_MAX_VIDEO_METADATA_FETCH_BYTES = 64 * 1024
_VIDEO_METADATA_FETCHES_PER_BUDGET = 8
_MAX_VIDEO_METADATA_FETCHES = 1024
_VIDEO_METADATA_TIMEOUT_SECONDS = 5.0
# Permit 8K UHD metadata and visible decoded frames by default.
_MAX_VIDEO_FRAME_PIXELS = 32 * 1024 * 1024
# FFmpeg applies AVCodecContext.max_pixels to coded/aligned dimensions. Leave
# bounded headroom above the public visible-frame ceiling so ordinary alignment
# and cropping cannot make a valid visible frame fail the wrong contract.
_MAX_VIDEO_CODED_FRAME_PIXELS = 64 * 1024 * 1024
_MAX_VIDEO_AUDIO_SAMPLES = 1024 * 1024
_MAX_VIDEO_STREAMS = 64
_MAX_VIDEO_PROBE_PACKETS = 256
_MAX_VIDEO_INDEX_BYTES = 256 * 1024
_MAX_VIDEO_ANALYZE_DURATION_US = 5 * 1_000_000
_MAX_VIDEO_FPS_PROBE_FRAMES = 32
_MAX_PYAV_BUFFER_SIZE = (1 << 31) - 1
_MIN_BIGINT = -(1 << 63)
_MAX_BIGINT = (1 << 63) - 1
_MAX_UINTEGER = (1 << 32) - 1
_MAX_UBIGINT = (1 << 64) - 1

_PYAV_UNSAFE_STREAM_OPTIONS_ERROR = (
    "stream_options were provided, but this format does not expose its streams before avformat_find_stream_info"
)

_PYAV_MEDIA_ERROR_NAMES = (
    "DecoderNotFoundError",
    "DemuxerNotFoundError",
    "EOFError",
    "InvalidDataError",
)

# PyAV 17.1 exposes a non-copying presence check only for a named packet side-
# data kind. Keep this synchronized with its public ``PktSideDataT`` literals;
# ``iter_sidedata()`` copies each payload merely to discover whether one exists.
_PYAV_PACKET_SIDE_DATA_TYPES = (
    "palette",
    "new_extradata",
    "param_change",
    "h263_mb_info",
    "replay_gain",
    "display_matrix",
    "stereo_3d",
    "audio_service_type",
    "quality_stats",
    "fallback_track",
    "cpb_properties",
    "skip_samples",
    "jp_dual_mono",
    "strings_metadata",
    "subtitle_position",
    "matroska_block_additional",
    "webvtt_identifier",
    "webvtt_settings",
    "metadata_update",
    "mpegts_stream_id",
    "mastering_display_metadata",
    "spherical",
    "content_light_level",
    "a53_cc",
    "encryption_init_info",
    "encryption_info",
    "afd",
    "prft",
    "icc_profile",
    "dovi_conf",
    "s12m_timecode",
    "dynamic_hdr10_plus",
    "iamf_mix_gain_param",
    "iamf_info_param",
    "iamf_recon_gain_info_param",
    "ambient_viewing_environment",
    "frame_cropping",
    "lcevc",
    "3d_reference_displays",
    "rtcp_sr",
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


@dataclass(frozen=True, slots=True)
class VideoFrameData:
    """One detached RGB frame and its decoder-reported temporal provenance.

    ``frame_time`` is elapsed presentation time from the stream's declared
    timestamp origin. ``frame_time_base`` is the exact unit for the raw PTS,
    DTS, and duration fields. The zero-based ``frame_index`` records the frame's
    position in the sequential decode stream when the decoder can establish it.
    """

    frame_index: int | None
    frame_time: float | None
    frame_time_base: Fraction | None
    frame_pts: int | None
    frame_dts: int | None
    frame_duration: int | None
    is_key_frame: bool
    data: PILImage


class VideoFileError(RuntimeError):
    """Base class for failures caused by encoded video content."""


class VideoFileFormatError(VideoFileError):
    """The logical FILE view is not a supported, self-contained video."""


class VideoFileLimitError(VideoFileError):
    """A VideoFile operation exceeded an explicit resource limit."""


@dataclass(frozen=True, slots=True)
class _VideoFrameOptions:
    start_time: Fraction
    end_time: Fraction | None
    width: int | None
    height: int | None
    is_key_frame: bool | None
    sample_interval_seconds: Fraction | None
    buffer_size: int
    max_input_bytes: int
    max_frames: int
    max_pixels: int
    target_frame_index: int | None = None


@dataclass(frozen=True, slots=True)
class _DecodedFrameInfo:
    width: int
    height: int
    frame_index: int
    frame_pts: int | None
    frame_dts: int | None
    frame_duration: int | None
    time_base: Fraction | None
    exact_time: Fraction | None
    frame_time: float | None
    is_key_frame: bool


@dataclass(slots=True)
class _DecodedPacketBatch:
    frames: list[Any | None]
    infos: tuple[_DecodedFrameInfo, ...]

    def take_frame(self, index: int) -> Any:
        frame = self.frames[index]
        if frame is None:
            raise RuntimeError("decoded video frame was already released")
        self.frames[index] = None
        return frame

    def release(self) -> None:
        self.frames.clear()


@dataclass(slots=True)
class _PreparedFrameBatch:
    results: list[VideoFrameData | None]
    next_sample_time: Fraction | None
    last_frame_time: Fraction | None

    def take_result(self, index: int) -> VideoFrameData:
        result = self.results[index]
        if result is None:
            raise RuntimeError("prepared video frame was already released")
        self.results[index] = None
        return result

    def release(self) -> None:
        for result in self.results:
            if result is not None:
                _close_image(result.data)
        self.results.clear()


@contextmanager
def _close_container(container: Any) -> Iterator[Any]:
    """Close a PyAV container without replacing its operation's exception."""
    try:
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
    finally:
        # A retained exception traceback must not keep the container's stream
        # objects and their native codec contexts alive.
        container = None


@contextmanager
def _close_video_reader(reader: vane.VaneFileReader | None) -> Iterator[vane.VaneFileReader]:
    """Check the retained open generation only after a successful video run."""

    if reader is None:
        raise TypeError("video reader must not be None")
    try:
        try:
            yield reader
        except BaseException:
            try:
                reader.close()
            except BaseException:
                pass
            raise
        else:
            reader._close_and_check_interrupted()
    finally:
        reader = None


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
        self.error = VideoFileFormatError("VideoFile does not permit nested external resources")
        raise self.error

    def raise_if_error(self) -> None:
        if self.error is not None:
            raise self.error.with_traceback(self.error.__traceback__)


class _VideoReaderProxy:
    """Keep FILE resolver failures distinct from FFmpeg media failures."""

    def __init__(self, reader: vane.VaneFileReader) -> None:
        self._reader = reader
        self._error: BaseException | None = None

    def _remember(self, error: BaseException) -> None:
        if self._error is None:
            self._error = error

    def read(self, size: int = -1, /) -> bytes:
        if self._error is not None:
            return b""
        try:
            self._reader._check_interrupted()
            result = self._reader._read_and_check_interrupted(size)
            self._reader._check_interrupted()
        except BaseException as error:
            self._remember(error)
            return b""
        return result

    def check_interrupted(self) -> None:
        control_flow_error = self._error if self._error is not None and not isinstance(self._error, Exception) else None
        try:
            self._reader._check_interrupted()
        except BaseException as error:
            self._remember(error)
            if control_flow_error is not None:
                # Once a callback has captured caller control flow (for example
                # KeyboardInterrupt or SystemExit), a later query interrupt must
                # not replace it at another checked FFmpeg boundary.
                raise control_flow_error.with_traceback(control_flow_error.__traceback__)
            raise
        if control_flow_error is not None:
            raise control_flow_error.with_traceback(control_flow_error.__traceback__)

    def readinto(self, buffer: Any, /) -> int:
        if self._error is not None:
            return 0
        try:
            self._reader._check_interrupted()
            result = self._reader._readinto_and_check_interrupted(buffer)
            self._reader._check_interrupted()
        except BaseException as error:
            self._remember(error)
            return 0
        return result

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
        if self._error is not None:
            return -1
        try:
            self._reader._check_interrupted()
            result = self._reader.seek(offset, whence)
            self._reader._check_interrupted()
        except BaseException as error:
            self._remember(error)
            return -1
        return result

    def tell(self) -> int:
        if self._error is not None:
            return -1
        try:
            self._reader._check_interrupted()
            result = self._reader.tell()
            self._reader._check_interrupted()
        except BaseException as error:
            self._remember(error)
            return -1
        return result

    def readable(self) -> bool:
        if self._error is not None:
            return False
        try:
            self._reader._check_interrupted()
            result = self._reader.readable()
            self._reader._check_interrupted()
        except BaseException as error:
            self._remember(error)
            return False
        return result

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        if self._error is not None:
            return False
        try:
            self._reader._check_interrupted()
            result = self._reader.seekable()
            self._reader._check_interrupted()
        except BaseException as error:
            self._remember(error)
            return False
        return result

    @property
    def closed(self) -> bool:
        return self._reader.closed

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error.with_traceback(self._error.__traceback__)


def _load_av() -> Any:
    try:
        av_module = importlib.import_module("av")
        av_module.open
        av_module.error.ExitError
        av_module.error.FFmpegError
        av_module.stream.Disposition.attached_pic
        av_module.time_base
        av_module.video.reformatter.VideoReformatter
    except (AttributeError, ImportError, OSError) as error:
        raise ImportError(
            "VideoFile operations require the 'av' package and its bundled FFmpeg libraries. "
            "Please `pip install 'vane-ai[video]'`."
        ) from error
    return av_module


def _load_pillow() -> Any:
    try:
        image_module = importlib.import_module("PIL.Image")
        image_module.Image
    except (AttributeError, ImportError, OSError) as error:
        raise ImportError(
            "VideoFile frame decoding requires the 'Pillow' package. Please `pip install 'vane-ai[video]'`."
        ) from error
    return image_module


def _positive_limit(value: object, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, not {type(value).__name__!r}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _nonnegative_frame_index(value: object, *, name: str = "idx") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, not {type(value).__name__!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value > _MAX_BIGINT:
        raise ValueError(f"{name} must be at most {_MAX_BIGINT}")
    return value


def _nonnegative_time(value: object, *, name: str, optional: bool = False) -> Fraction | None:
    if value is None:
        if optional:
            return None
        raise TypeError(f"{name} must be int or float, not 'NoneType'")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        expected = "int, float, or None" if optional else "int or float"
        raise TypeError(f"{name} must be {expected}, not {type(value).__name__!r}")
    if isinstance(value, int):
        result = Fraction(value)
    else:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        result = Fraction(str(value))
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_keyframe_filter(value: object) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"is_key_frame must be bool or None, not {type(value).__name__!r}")


def _optional_dimensions(width: object, height: object, *, max_pixels: int) -> tuple[int | None, int | None]:
    if (width is None) != (height is None):
        raise ValueError("width and height must be provided together")
    if width is None:
        return None, None
    normalized_width = _positive_limit(width, name="width", maximum=_MAX_UINTEGER)
    normalized_height = _positive_limit(height, name="height", maximum=_MAX_UINTEGER)
    output_pixels = normalized_width * normalized_height
    if output_pixels > max_pixels:
        raise VideoFileLimitError(
            f"requested video frames contain {output_pixels} pixels, exceeding max_pixels={max_pixels}"
        )
    return normalized_width, normalized_height


def _normalize_frame_options(
    *,
    start_time: object,
    end_time: object,
    width: object,
    height: object,
    is_key_frame: object,
    sample_interval_seconds: object,
    buffer_size: object,
    max_input_bytes: object,
    max_frames: object,
    max_pixels: object,
    target_frame_index: object | None = None,
) -> _VideoFrameOptions:
    normalized_start = _nonnegative_time(start_time, name="start_time")
    assert normalized_start is not None
    normalized_end = _nonnegative_time(end_time, name="end_time", optional=True)
    if normalized_end is not None and normalized_end < normalized_start:
        raise ValueError("end_time must be greater than or equal to start_time")
    normalized_interval = _nonnegative_time(
        sample_interval_seconds,
        name="sample_interval_seconds",
        optional=True,
    )
    if normalized_interval == 0:
        raise ValueError("sample_interval_seconds must be greater than zero")
    normalized_buffer = _positive_buffer_size(buffer_size, none_default=None, name="buffer_size")
    if normalized_buffer > _MAX_PYAV_BUFFER_SIZE:
        raise OverflowError("buffer_size must fit in the C int accepted by PyAV")
    normalized_max_input = _positive_limit(max_input_bytes, name="max_input_bytes", maximum=_MAX_UBIGINT)
    normalized_max_frames = _positive_limit(max_frames, name="max_frames", maximum=_MAX_BIGINT)
    normalized_max_pixels = _positive_limit(
        max_pixels,
        name="max_pixels",
        maximum=_MAX_VIDEO_FRAME_PIXELS,
    )
    normalized_width, normalized_height = _optional_dimensions(
        width,
        height,
        max_pixels=normalized_max_pixels,
    )
    return _VideoFrameOptions(
        start_time=normalized_start,
        end_time=normalized_end,
        width=normalized_width,
        height=normalized_height,
        is_key_frame=_optional_keyframe_filter(is_key_frame),
        sample_interval_seconds=normalized_interval,
        buffer_size=normalized_buffer,
        max_input_bytes=normalized_max_input,
        max_frames=normalized_max_frames,
        max_pixels=normalized_max_pixels,
        target_frame_index=(None if target_frame_index is None else _nonnegative_frame_index(target_frame_index)),
    )


def _limit_expression(value: int | vane.Expression, *, name: str, maximum: int) -> vane.Expression:
    if isinstance(value, vane.Expression):
        return value
    return as_expression(_positive_limit(value, name=name, maximum=maximum))


def _canonical_mime_type(value: str) -> str:
    normalized = value.partition(";")[0].strip().lower()
    return _MIME_ALIASES.get(normalized, normalized)


def _container_format_names(container: Any) -> frozenset[str]:
    try:
        container_format = container.format
        format_name = None if container_format is None else container_format.name
        if not isinstance(format_name, str) or not format_name:
            raise VideoFileFormatError("video parser did not report a container format")
        names = frozenset(part.strip().lower() for part in format_name.split(",") if part.strip())
        if not names:
            raise VideoFileFormatError("video parser reported an invalid container format")
        return names
    finally:
        container = None


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
    try:
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
    finally:
        video = None


def _optional_duration(container: Any, video: Any, time_base: Fraction, av_module: Any) -> float | None:
    try:
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
    finally:
        container = None
        video = None


def _optional_frame_count(video: Any) -> int | None:
    try:
        if video.frames is None:
            return None
        frame_count = int(video.frames)
        if frame_count < 0 or frame_count > _MAX_BIGINT:
            raise VideoFileFormatError("video parser reported an out-of-range frame count")
        # FFmpeg uses zero when the container does not carry an exact frame count.
        return frame_count or None
    finally:
        video = None


def _select_video_stream(container: Any, av_module: Any) -> Any:
    video: Any = None
    try:
        attached_pic = av_module.stream.Disposition.attached_pic
        video = next(
            (
                stream
                for stream in container.streams
                if stream.type == "video" and not (stream.disposition & attached_pic)
            ),
            None,
        )
        if video is None:
            raise VideoFileFormatError("logical FILE view does not contain a video stream")
        return video
    finally:
        container = None
        video = None


def _metadata_from_container(
    container: Any,
    content_type: str | None,
    av_module: Any,
    *,
    max_pixels: int = _MAX_VIDEO_FRAME_PIXELS,
) -> VideoMetadata:
    video: Any = None
    try:
        video = _select_video_stream(container, av_module)

        format_names = _container_format_names(container)
        _validate_content_type(content_type, format_names)
        if getattr(video, "codec_context", None) is None:
            raise VideoFileFormatError("video stream does not have an available decoder")

        width = int(video.width)
        height = int(video.height)
        if width <= 0 or height <= 0:
            raise VideoFileFormatError(f"video dimensions must be positive, found {width}x{height}")
        if width > _MAX_UINTEGER or height > _MAX_UINTEGER:
            raise VideoFileFormatError(f"video dimensions must fit positive UINTEGER values, found {width}x{height}")
        if width * height > max_pixels:
            if max_pixels == _MAX_VIDEO_FRAME_PIXELS:
                limit_description = f"the metadata pixel limit of {max_pixels}"
            else:
                limit_description = f"max_pixels={max_pixels}"
            raise VideoFileLimitError(f"video dimensions {width}x{height} exceed {limit_description}")
        time_base = _positive_fraction(video.time_base, name="time base")
        return VideoMetadata(
            width=width,
            height=height,
            fps=_optional_rate(video),
            duration=_optional_duration(container, video, time_base, av_module),
            frame_count=_optional_frame_count(video),
            time_base=time_base,
        )
    finally:
        video = None
        container = None


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
        "max_pixels": str(_MAX_VIDEO_CODED_FRAME_PIXELS),
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
        if isinstance(error, (MemoryError, OSError)):
            raise
        if isinstance(error, ValueError) and str(error).startswith(_PYAV_UNSAFE_STREAM_OPTIONS_ERROR):
            raise VideoFileFormatError(
                "video container cannot be inspected safely within metadata resource limits"
            ) from error
        if _is_pyav_media_or_codec_error(error, av_module):
            if stream.budget_exhausted:
                raise VideoFileLimitError(f"video metadata requires more than max_bytes={max_bytes}") from error
            if stream.fetch_limit_exhausted:
                raise VideoFileLimitError(
                    f"video metadata requires more than {_MAX_VIDEO_METADATA_FETCHES} source ranges"
                ) from error
            raise VideoFileFormatError("logical FILE view is not a supported encoded video") from error
        raise
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


def _optional_frame_integer(value: Any, *, name: str, nonnegative: bool = False) -> int | None:
    if value is None:
        return None
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise VideoFileFormatError(f"video decoder reported an invalid {name}") from error
    if result < _MIN_BIGINT or result > _MAX_BIGINT or (nonnegative and result < 0):
        raise VideoFileFormatError(f"video decoder reported an out-of-range {name}")
    return result


def _frame_time_base(frame: Any, video: Any) -> Fraction | None:
    try:
        value = frame.time_base if frame.time_base is not None else video.time_base
        if value is None:
            return None
        return _positive_fraction(value, name="frame time base")
    finally:
        frame = None
        video = None


def _stream_time_origin(video: Any, time_base: Fraction) -> Fraction:
    try:
        stream_start = _optional_frame_integer(getattr(video, "start_time", None), name="stream start time")
        if stream_start is None:
            return Fraction(0)
        return stream_start * time_base
    finally:
        video = None


def _frame_time(
    frame_pts: int | None,
    time_base: Fraction | None,
    stream_time_origin: Fraction,
) -> tuple[Fraction | None, float | None]:
    if frame_pts is None or time_base is None:
        return None, None
    exact_time = frame_pts * time_base - stream_time_origin
    result = float(exact_time)
    if not math.isfinite(result):
        raise VideoFileFormatError("video decoder reported a non-finite frame time")
    return exact_time, result


def _decoded_frame_dimensions(frame: Any, options: _VideoFrameOptions) -> tuple[int, int]:
    try:
        try:
            width = int(frame.width)
            height = int(frame.height)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise VideoFileFormatError("video decoder reported invalid frame dimensions") from error
        if width <= 0 or height <= 0:
            raise VideoFileFormatError(f"video frame dimensions must be positive, found {width}x{height}")
        pixels = width * height
        if pixels > options.max_pixels:
            raise VideoFileLimitError(
                f"decoded video frame contains {pixels} pixels, exceeding max_pixels={options.max_pixels}"
            )
        return width, height
    finally:
        frame = None


def _close_image(image: Any) -> None:
    try:
        image.close()
    except BaseException:
        pass


@contextmanager
def _neutralized_color_conversion_metadata(frame: Any) -> Iterator[None]:
    """Apply PyAV 18's safe default behavior while Python 3.10 uses PyAV 17."""
    saved: list[tuple[str, Any]] = []
    body_failed = True
    try:
        for attribute in ("color_trc", "color_primaries"):
            try:
                value = getattr(frame, attribute)
            except AttributeError:
                continue
            saved.append((attribute, value))
            try:
                setattr(frame, attribute, 2)  # FFmpeg's UNSPECIFIED value for both fields.
            except (TypeError, ValueError, OverflowError) as error:
                raise VideoFileFormatError("video decoder reported invalid color metadata") from error
        yield
        body_failed = False
    finally:
        restore_error: BaseException | None = None
        restore_system_error: BaseException | None = None
        restore_control_flow: BaseException | None = None
        for attribute, value in reversed(saved):
            try:
                setattr(frame, attribute, value)
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
                if not isinstance(error, Exception):
                    if restore_control_flow is None:
                        restore_control_flow = error
                elif not isinstance(error, (TypeError, ValueError, OverflowError)):
                    if restore_system_error is None:
                        restore_system_error = error
        saved.clear()
        frame = None
        if not body_failed:
            if restore_control_flow is not None:
                raise restore_control_flow
            if restore_system_error is not None:
                raise restore_system_error
            if restore_error is not None:
                raise VideoFileFormatError("video decoder color metadata could not be restored") from restore_error


def _is_pyav_media_or_codec_error(error: BaseException, av_module: Any) -> bool:
    """Return whether PyAV explicitly classified a content/decoder failure."""

    # Keep this allowlist deliberately narrow. In particular, PyAV's fallback,
    # unknown, argument, buffer, range, callback, and internal errors must not
    # become suppressible media failures merely because they inherit
    # ``FFmpegError``.
    for name in _PYAV_MEDIA_ERROR_NAMES:
        error_type = getattr(av_module.error, name, None)
        if isinstance(error_type, type) and isinstance(error, error_type):
            return True
    return False


def _frame_to_image(
    frame: Any,
    info: _DecodedFrameInfo,
    options: _VideoFrameOptions,
    av_module: Any,
    image_module: Any,
    reformatter: Any,
    check_interrupted: Callable[[], None],
) -> PILImage:
    output_frame: Any = None
    image: Any = None
    try:
        output_width = options.width if options.width is not None else info.width
        output_height = options.height if options.height is not None else info.height
        try:
            # PyAV 18 neutralizes transfer/primary tags unless conversion was
            # explicitly requested. Mirror that upstream fix for Python 3.10, where
            # PyAV 17 remains the latest supported release, and restore the source
            # frame metadata before returning.
            check_interrupted()
            with _neutralized_color_conversion_metadata(frame):
                output_frame = reformatter.reformat(
                    frame,
                    width=output_width,
                    height=output_height,
                    format="rgb24",
                    threads=1,
                )
            check_interrupted()
        except av_module.error.FFmpegError as error:
            check_interrupted()
            if _is_pyav_media_or_codec_error(error, av_module):
                raise VideoFileFormatError("video frame could not be converted to RGB pixels") from error
            raise
        try:
            check_interrupted()
            image = output_frame.to_image()
            check_interrupted()
        except av_module.error.FFmpegError as error:
            check_interrupted()
            if _is_pyav_media_or_codec_error(error, av_module):
                raise VideoFileFormatError("video frame could not be converted to RGB pixels") from error
            raise
        check_interrupted()
        image.load()
        check_interrupted()
        if (
            not isinstance(image, image_module.Image)
            or image.mode != "RGB"
            or image.size != (output_width, output_height)
        ):
            raise RuntimeError("PyAV returned an invalid RGB Pillow frame")
        result = image
        image = None
        return result
    finally:
        if image is not None:
            _close_image(image)
            image = None
        # Propagated exception tracebacks must not keep native decoder frames
        # or an undelivered Pillow image alive after cleanup.
        output_frame = None
        frame = None
        del reformatter
        del check_interrupted


def _configure_video_decoder(video: Any) -> None:
    codec_context: Any = None
    try:
        codec_context = getattr(video, "codec_context", None)
        if codec_context is None:
            raise VideoFileFormatError("video stream does not have an available decoder")
        if codec_context.is_open:
            raise VideoFileLimitError("video decoder opened before resource limits could be applied")

        decoder_options = dict(codec_context.options or {})
        decoder_options.pop("skip_frame", None)
        decoder_options.update(
            {
                # ``AVCodecContext.max_pixels`` applies to coded/aligned buffer
                # dimensions rather than the visible frame. Keep this internal
                # safety ceiling separate from the public visible-pixel limit,
                # which Vane enforces from metadata and every decoded frame.
                "max_pixels": str(_MAX_VIDEO_CODED_FRAME_PIXELS),
                "threads": "1",
            }
        )
        codec_context.options = decoder_options
        codec_context.thread_count = 1
    finally:
        codec_context = None
        video = None


def _advance_sample_target(current: Fraction, interval: Fraction, frame_time: Fraction) -> Fraction:
    skipped_intervals = (frame_time - current) // interval
    return current + (skipped_intervals + 1) * interval


def _require_frame_time_for_selection(info: _DecodedFrameInfo, options: _VideoFrameOptions) -> None:
    requires_time = (
        options.start_time > 0 or options.end_time is not None or options.sample_interval_seconds is not None
    )
    if requires_time and info.exact_time is None:
        raise VideoFileFormatError("video time selection requires a presentation timestamp for every decoded frame")


def _decoded_frame_info(
    frame: Any,
    video: Any,
    options: _VideoFrameOptions,
    *,
    frame_index: int,
    stream_time_origin: Fraction,
) -> _DecodedFrameInfo:
    try:
        width, height = _decoded_frame_dimensions(frame, options)
        frame_pts = _optional_frame_integer(frame.pts, name="frame PTS")
        frame_dts = _optional_frame_integer(frame.dts, name="frame DTS")
        frame_duration = _optional_frame_integer(
            frame.duration,
            name="frame duration",
            nonnegative=True,
        )
        time_base = _frame_time_base(frame, video)
        exact_time, frame_time = _frame_time(frame_pts, time_base, stream_time_origin)
        return _DecodedFrameInfo(
            width=width,
            height=height,
            frame_index=frame_index,
            frame_pts=frame_pts,
            frame_dts=frame_dts,
            frame_duration=frame_duration,
            time_base=time_base,
            exact_time=exact_time,
            frame_time=frame_time,
            is_key_frame=bool(frame.key_frame),
        )
    finally:
        frame = None
        video = None


def _check_video_io(reader: _VideoReaderProxy, nested_io: _NestedIOBlocker) -> None:
    """Give query cancellation precedence over stored I/O and media errors."""

    reader.check_interrupted()
    reader.raise_if_error()
    nested_io.raise_if_error()


def _decode_packet_frames(
    packet: Any,
    reader: _VideoReaderProxy,
    nested_io: _NestedIOBlocker,
) -> list[Any]:
    """Decode one atomic PyAV packet with checks on every return path."""
    frames: list[Any] | None = None
    try:
        _check_video_io(reader, nested_io)
        try:
            decoded = packet.decode()
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            # An interrupt that raced with a dependency failure takes
            # precedence, matching DuckDB's query-cancellation semantics.
            _check_video_io(reader, nested_io)
            raise
        if not isinstance(decoded, list):
            raise RuntimeError("PyAV packet.decode() returned a non-list frame batch")
        frames = decoded
        _check_video_io(reader, nested_io)
        return frames
    except BaseException:
        if frames is not None:
            frames.clear()
        raise
    finally:
        packet = None


@contextmanager
def _close_demux_iterator(packets: Any) -> Iterator[Any]:
    """Close one PyAV demux iterator without replacing its body exception."""

    close: Any = None
    try:
        try:
            yield packets
        except BaseException:
            try:
                close = getattr(packets, "close", None)
                if close is not None:
                    close()
            except BaseException:
                pass
            raise
        else:
            close = getattr(packets, "close", None)
            if close is not None:
                close()
    finally:
        # A retained close traceback must not keep the demux generator and its
        # current native packet alive.
        close = None
        packets = None


def _is_flush_packet(packet: Any) -> bool:
    """Recognize PyAV's synthetic decoder-drain packet."""

    has_sidedata: Any = None
    try:
        try:
            size = operator.index(packet.size)
            buffer_ptr = operator.index(packet.buffer_ptr)
        except AttributeError:
            # Small test doubles need not reproduce PyAV's native buffer facade.
            return False
        except (TypeError, ValueError, OverflowError) as error:
            raise VideoFileFormatError("video demuxer reported invalid packet buffer metadata") from error
        if size < 0 or buffer_ptr < 0:
            raise VideoFileFormatError("video demuxer reported invalid packet buffer metadata")
        if size != 0 or buffer_ptr != 0:
            return False

        # FFmpeg treats a zero-payload packet with side data as real decoder
        # input. PyAV's synthetic EOF packet has neither payload nor side data,
        # so both properties are required before ending demux and draining once.
        has_sidedata = packet.has_sidedata
        return not any(has_sidedata(data_type) for data_type in _PYAV_PACKET_SIDE_DATA_TYPES)
    finally:
        # An exceptional classification traceback must not retain the packet's
        # native stream, container, codec context, or FILE reader.
        has_sidedata = None
        packet = None


def _next_demuxed_packet(
    container: Any,
    video: Any,
    reader: _VideoReaderProxy,
    nested_io: _NestedIOBlocker,
) -> tuple[Any, bool] | None:
    """Read one packet and close PyAV's owning demux generator before returning."""

    packets: Any = None
    packet: Any = None
    received_packet = False
    try:
        _check_video_io(reader, nested_io)
        try:
            packets = container.demux(video)
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            _check_video_io(reader, nested_io)
            raise

        try:
            with _close_demux_iterator(packets):
                _check_video_io(reader, nested_io)
                packet = next(packets)
                received_packet = True
        except StopIteration:
            _check_video_io(reader, nested_io)
            if received_packet:
                # A dependency close unexpectedly raised StopIteration after a
                # packet had already been read; this is not successful EOF.
                raise
            return None
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            _check_video_io(reader, nested_io)
            raise

        # PyAV 17.1 retains its local Packet at the demux generator's yield.
        # The iterator has been closed above, so this check and the caller can no
        # longer leave that second native-buffer owner suspended indefinitely.
        _check_video_io(reader, nested_io)
        is_flush = _is_flush_packet(packet)
        result = (packet, is_flush)
        packet = None
        return result
    finally:
        packet = None
        packets = None
        video = None
        container = None


def _iter_decoded_packet_batches(
    container: Any,
    video: Any,
    options: _VideoFrameOptions,
    reader: _VideoReaderProxy,
    nested_io: _NestedIOBlocker,
    *,
    stream_time_origin: Fraction = Fraction(0),
) -> Generator[_DecodedPacketBatch, None, None]:
    """Validate each PyAV packet batch before exposing any frame from it.

    PyAV exposes one packet decode as an atomic dependency call, so cooperative
    interruption happens immediately before and after that call. The complete
    returned batch is then subject to frame-count, provenance, and visible
    frame-dimension validation. ``max_frames`` is a packet-boundary count
    guard, not a native-memory limit.
    """
    decoded_frames = 0
    packet_and_flush: tuple[Any, bool] | None = None
    packet: Any = None
    frames: list[Any] | None = None
    try:
        while True:
            packet_and_flush = _next_demuxed_packet(container, video, reader, nested_io)
            if packet_and_flush is None:
                break
            packet, is_flush = packet_and_flush
            packet_and_flush = None
            try:
                frames = _decode_packet_frames(packet, reader, nested_io)
            finally:
                packet = None
            try:
                batch_size = len(frames)
                if batch_size > options.max_frames - decoded_frames:
                    raise VideoFileLimitError(f"video decode exceeded max_frames={options.max_frames} decoded frames")

                infos: list[_DecodedFrameInfo] = []
                for index in range(batch_size):
                    infos.append(
                        _decoded_frame_info(
                            frames[index],
                            video,
                            options,
                            frame_index=decoded_frames + index,
                            stream_time_origin=stream_time_origin,
                        )
                    )
                for info in infos:
                    _require_frame_time_for_selection(info, options)
                decoded_frames += batch_size

                if batch_size:
                    batch = _DecodedPacketBatch(frames=frames, infos=tuple(infos))
                    yield batch
            finally:
                # This guard starts immediately after the atomic decode returns,
                # so rejected batches cannot remain pinned by an error traceback.
                frames.clear()
                frames = None
            if is_flush:
                break
    finally:
        # Exception tracebacks from this generator must not own PyAV's stream,
        # packet, container, or native codec context.
        if frames is not None:
            frames.clear()
        frames = None
        packet = None
        packet_and_flush = None
        video = None
        container = None


def _prepare_video_packet_batch(
    batch: _DecodedPacketBatch,
    options: _VideoFrameOptions,
    av_module: Any,
    image_module: Any,
    reformatter: Any,
    reader: _VideoReaderProxy,
    nested_io: _NestedIOBlocker,
    *,
    next_sample_time: Fraction | None,
    last_frame_time: Fraction | None,
) -> _PreparedFrameBatch:
    """Detach every selected image before exposing any result from a packet."""

    results: list[VideoFrameData | None] = []
    frame: Any = None
    image: Any = None
    result: VideoFrameData | None = None
    try:
        for index, info in enumerate(batch.infos):
            _check_video_io(reader, nested_io)
            frame = batch.take_frame(index)
            try:
                if options.target_frame_index is not None and info.frame_index != options.target_frame_index:
                    continue
                if info.exact_time is not None:
                    if last_frame_time is not None and info.exact_time < last_frame_time:
                        # MPEG-TS and other streaming containers can reset their
                        # presentation timeline at a discontinuity. Sampling
                        # targets belong to each monotonic segment rather than a
                        # previous segment's now-unreachable timestamp range.
                        next_sample_time = options.start_time if options.sample_interval_seconds is not None else None
                    last_frame_time = info.exact_time
                    if info.exact_time < options.start_time:
                        continue
                    if options.end_time is not None and info.exact_time > options.end_time:
                        # Do not stop globally: a later discontinuity can move
                        # presentation timestamps back into the requested window.
                        continue

                if options.is_key_frame is not None and info.is_key_frame is not options.is_key_frame:
                    continue

                if options.sample_interval_seconds is not None:
                    assert info.exact_time is not None
                    assert next_sample_time is not None
                    if info.exact_time < next_sample_time:
                        continue
                    next_sample_time = _advance_sample_target(
                        next_sample_time,
                        options.sample_interval_seconds,
                        info.exact_time,
                    )

                image = _frame_to_image(
                    frame,
                    info,
                    options,
                    av_module,
                    image_module,
                    reformatter,
                    lambda: _check_video_io(reader, nested_io),
                )
                try:
                    _check_video_io(reader, nested_io)
                    result = VideoFrameData(
                        frame_index=info.frame_index,
                        frame_time=info.frame_time,
                        frame_time_base=info.time_base,
                        frame_pts=info.frame_pts,
                        frame_dts=info.frame_dts,
                        frame_duration=info.frame_duration,
                        is_key_frame=info.is_key_frame,
                        data=image,
                    )
                    results.append(result)
                except BaseException:
                    _close_image(image)
                    image = None
                    result = None
                    raise
                else:
                    # The prepared batch now owns the detached image.
                    image = None
                    result = None
            finally:
                # Filtering and conversion must release each source frame before
                # the detached packet batch can be returned to the caller.
                frame = None

        _check_video_io(reader, nested_io)
        prepared = _PreparedFrameBatch(
            results=results,
            next_sample_time=next_sample_time,
            last_frame_time=last_frame_time,
        )
        results = []
        return prepared
    except BaseException:
        for pending in results:
            if pending is not None:
                _close_image(pending.data)
        results.clear()
        # The re-raised exception retains this frame. Do not let the loop target
        # keep the final undelivered image alive through its traceback.
        pending = None
        raise
    finally:
        if image is not None:
            _close_image(image)
        result = None
        frame = None
        batch.release()
        del batch
        reformatter = None
        av_module = None
        image_module = None


def _classify_video_decode_error(
    error: Exception,
    *,
    av_module: Any,
    reader: _VideoReaderProxy,
    nested_io: _NestedIOBlocker,
) -> None:
    _check_video_io(reader, nested_io)
    if isinstance(error, VideoFileError):
        raise error
    if isinstance(error, av_module.error.ExitError):
        raise VideoFileLimitError(
            f"a video frame decode phase exceeded its {_VIDEO_METADATA_TIMEOUT_SECONDS:g}-second I/O timeout"
        ) from error
    if isinstance(error, ValueError) and str(error).startswith(_PYAV_UNSAFE_STREAM_OPTIONS_ERROR):
        raise VideoFileFormatError("video container cannot be decoded safely within resource limits") from error
    if _is_pyav_media_or_codec_error(error, av_module):
        raise VideoFileFormatError("logical FILE view is not a supported decodable video") from error
    raise error


def _iter_video_frames(
    value: vane.VideoFile,
    options: _VideoFrameOptions,
    av_module: Any,
    image_module: Any,
    connection: vane.DuckDBPyConnection | None,
) -> Generator[VideoFrameData, None, None]:
    file_reader = value.open(buffer_size=options.buffer_size, connection=connection)
    with _close_video_reader(file_reader):
        input_size = file_reader.size()
        file_reader._check_interrupted()
        if input_size > options.max_input_bytes:
            raise VideoFileLimitError(
                f"encoded video contains {input_size} bytes, exceeding max_input_bytes={options.max_input_bytes}"
            )

        reader = _VideoReaderProxy(file_reader)
        nested_io = _NestedIOBlocker()
        probe_decoder_options = {
            # FFmpeg checks coded/aligned allocation dimensions here. The
            # caller-facing visible-frame contract is checked separately.
            "max_pixels": str(_MAX_VIDEO_CODED_FRAME_PIXELS),
            "max_samples": str(_MAX_VIDEO_AUDIO_SAMPLES),
            "skip_frame": "all",
            "threads": "1",
        }
        probe_bytes = min(input_size, DEFAULT_VIDEO_METADATA_BYTES)
        probe_options = {
            "analyzeduration": str(_MAX_VIDEO_ANALYZE_DURATION_US),
            "formatprobesize": str(max(2048, probe_bytes)),
            "fpsprobesize": str(_MAX_VIDEO_FPS_PROBE_FRAMES),
            "indexmem": str(_MAX_VIDEO_INDEX_BYTES),
            "max_probe_packets": str(_MAX_VIDEO_PROBE_PACKETS),
            "max_streams": str(_MAX_VIDEO_STREAMS),
            "probesize": str(max(32, probe_bytes)),
            "skip_estimate_duration_from_pts": "1",
        }
        container: Any = None
        video: Any = None
        reformatter: Any = None
        decoded_batches: Generator[_DecodedPacketBatch, None, None] | None = None
        prepared_batch: _PreparedFrameBatch | None = None
        error_from_consumer = False
        target_delivered = False
        try:
            _check_video_io(reader, nested_io)
            container = av_module.open(
                reader,
                mode="r",
                options=probe_decoder_options,
                container_options=probe_options,
                stream_options=[probe_decoder_options.copy()],
                metadata_encoding="utf-8",
                metadata_errors="replace",
                buffer_size=options.buffer_size,
                timeout=(_VIDEO_METADATA_TIMEOUT_SECONDS, _VIDEO_METADATA_TIMEOUT_SECONDS),
                io_open=nested_io,
            )
            with _close_container(container):
                _check_video_io(reader, nested_io)
                metadata = _metadata_from_container(
                    container,
                    value.content_type,
                    av_module,
                    max_pixels=options.max_pixels,
                )
                video = _select_video_stream(container, av_module)
                stream_time_origin = _stream_time_origin(video, metadata.time_base)
                _configure_video_decoder(video)
                reformatter = av_module.video.reformatter.VideoReformatter()

                next_sample_time = options.start_time if options.sample_interval_seconds is not None else None
                decoded_batches = _iter_decoded_packet_batches(
                    container,
                    video,
                    options,
                    reader,
                    nested_io,
                    stream_time_origin=stream_time_origin,
                )
                last_frame_time: Fraction | None = None
                try:
                    while True:
                        try:
                            batch = next(decoded_batches)
                        except StopIteration:
                            break
                        prepared_batch = _prepare_video_packet_batch(
                            batch,
                            options,
                            av_module,
                            image_module,
                            reformatter,
                            reader,
                            nested_io,
                            next_sample_time=next_sample_time,
                            last_frame_time=last_frame_time,
                        )
                        del batch
                        next_sample_time = prepared_batch.next_sample_time
                        last_frame_time = prepared_batch.last_frame_time
                        try:
                            for index in range(len(prepared_batch.results)):
                                _check_video_io(reader, nested_io)
                                result = prepared_batch.take_result(index)
                                try:
                                    yield result
                                except BaseException:
                                    # ``Generator.throw()`` injects a caller-owned
                                    # exception at this boundary. It must survive
                                    # operation cleanup without being classified as
                                    # a PyAV/FFmpeg dependency failure below.
                                    error_from_consumer = True
                                    raise
                                finally:
                                    # ``throw()`` and ``close()`` resume at the
                                    # yield itself. The caller owns this result;
                                    # only later undelivered batch entries are closed.
                                    del result
                                _check_video_io(reader, nested_io)
                                if options.target_frame_index is not None:
                                    target_delivered = True
                                    break
                        finally:
                            prepared_batch.release()
                            prepared_batch = None
                        if target_delivered:
                            break
                finally:
                    decoded_batches.close()
                _check_video_io(reader, nested_io)
            # ``container.close()`` releases the GIL, so cancellation may race
            # with an otherwise successful context-manager exit.
            _check_video_io(reader, nested_io)
        except BaseException as error:
            if error_from_consumer or not isinstance(error, Exception):
                raise
            _classify_video_decode_error(
                error,
                av_module=av_module,
                reader=reader,
                nested_io=nested_io,
            )
        finally:
            if prepared_batch is not None:
                prepared_batch.release()
            # Clear every PyAV owner before the outer reader context closes. A
            # reader-close failure traceback must not retain native codec state.
            prepared_batch = None
            decoded_batches = None
            reformatter = None
            video = None
            container = None


def _video_file_frames_value(
    value: vane.VideoFile,
    start_time: int | float = 0,
    end_time: int | float | None = None,
    width: int | None = None,
    height: int | None = None,
    is_key_frame: bool | None = None,
    sample_interval_seconds: int | float | None = None,
    buffer_size: int = DEFAULT_VIDEO_BUFFER_SIZE,
    *,
    max_input_bytes: int = DEFAULT_VIDEO_MAX_INPUT_BYTES,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    connection: vane.DuckDBPyConnection | None = None,
) -> Generator[VideoFrameData, None, None]:
    options = _normalize_frame_options(
        start_time=start_time,
        end_time=end_time,
        width=width,
        height=height,
        is_key_frame=is_key_frame,
        sample_interval_seconds=sample_interval_seconds,
        buffer_size=buffer_size,
        max_input_bytes=max_input_bytes,
        max_frames=max_frames,
        max_pixels=max_pixels,
    )
    av_module = _load_av()
    image_module = _load_pillow()
    return _iter_video_frames(value, options, av_module, image_module, connection)


def _video_file_frame_by_idx_value(
    value: vane.VideoFile,
    idx: int,
    buffer_size: int = DEFAULT_VIDEO_BUFFER_SIZE,
    *,
    max_input_bytes: int = DEFAULT_VIDEO_MAX_INPUT_BYTES,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    connection: vane.DuckDBPyConnection | None = None,
) -> PILImage:
    target_index = _nonnegative_frame_index(idx)
    options = _normalize_frame_options(
        start_time=0,
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=buffer_size,
        max_input_bytes=max_input_bytes,
        max_frames=max_frames,
        max_pixels=max_pixels,
        target_frame_index=target_index,
    )
    if target_index >= options.max_frames:
        raise VideoFileLimitError(
            f"video frame index {target_index} requires decoding at least {target_index + 1} frames, "
            f"exceeding max_frames={options.max_frames}"
        )

    av_module = _load_av()
    image_module = _load_pillow()
    frames = _iter_video_frames(value, options, av_module, image_module, connection)
    frame: VideoFrameData | None = None
    image: PILImage | None = None
    try:
        try:
            frame = next(frames)
        except StopIteration:
            raise IndexError(f"video frame index {target_index} is out of range") from None
        if frame.frame_index != target_index:
            raise RuntimeError(
                f"video decoder returned frame index {frame.frame_index!r} while selecting {target_index}"
            )
        image = frame.data
        frame = None
    except BaseException:
        if frame is not None:
            _close_image(frame.data)
            frame = None
        try:
            frames.close()
        except BaseException:
            pass
        raise
    assert image is not None
    try:
        unexpected_frame = next(frames)
    except StopIteration:
        # Advancing the dedicated single-frame generator to normal exhaustion
        # runs container and reader teardown on their success paths. In
        # particular, close-time connector failures and cancellation remain
        # observable instead of being suppressed as effects of GeneratorExit.
        return image
    except BaseException:
        _close_image(image)
        try:
            frames.close()
        except BaseException:
            pass
        raise

    _close_image(unexpected_frame.data)
    _close_image(image)
    try:
        frames.close()
    except BaseException:
        pass
    raise RuntimeError("single-frame video selection yielded more than one frame")


def _iter_keyframe_images(frames: Generator[VideoFrameData, None, None]) -> Generator[PILImage, None, None]:
    try:
        for frame in frames:
            image = frame.data
            del frame
            try:
                yield image
            finally:
                # Do not retain the previously returned image while asking the
                # source iterator to decode the next keyframe.
                del image
    except BaseException:
        close = getattr(frames, "close", None)
        if close is not None:
            try:
                close()
            except BaseException:
                pass
        raise
    else:
        close = getattr(frames, "close", None)
        if close is not None:
            close()


def _video_file_keyframes_value(
    value: vane.VideoFile,
    start_time: int | float = 0,
    end_time: int | float | None = None,
    width: int | None = None,
    height: int | None = None,
    sample_interval_seconds: int | float | None = None,
    buffer_size: int = DEFAULT_VIDEO_BUFFER_SIZE,
    *,
    max_input_bytes: int = DEFAULT_VIDEO_MAX_INPUT_BYTES,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    connection: vane.DuckDBPyConnection | None = None,
) -> Generator[PILImage, None, None]:
    frames = _video_file_frames_value(
        value,
        start_time,
        end_time,
        width,
        height,
        True,
        sample_interval_seconds,
        buffer_size,
        max_input_bytes=max_input_bytes,
        max_frames=max_frames,
        max_pixels=max_pixels,
        connection=connection,
    )
    return _iter_keyframe_images(frames)


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
    "VideoFrameData",
    "VideoFileLimitError",
    "VideoMetadata",
    "video_metadata",
]
