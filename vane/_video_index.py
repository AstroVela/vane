# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent Python video index codec, verified reader and presentation cursor.

The version 2 wire contract is documented in VIDEO_FRAME_API.md. Only governed FILE
I/O and FILE identity serialization cross the base engine boundary. This module
does not load or call the native video extension.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import math
import struct
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Callable, Generator, Iterator

import vane
from vane import _video_file as video
from vane._file import _file_open_in_datasource_context

_BLOCK_BYTES = 64 * 1024
_MAX_BYTES = 64 * 1024**2
_HEADER_BYTES = 160
_FRAME_BYTES = 88
_MAGIC = b"VVIDX002"
_NOPTS = -(1 << 63)
_FRAME = struct.Struct("<qqqQQQQ32s")
_LIMIT_MESSAGE = "video index exceeds max_index_bytes or the 256 MiB batch budget"


def _digest(data: Any, check: Callable[[], None]) -> bytes:
    digest = hashlib.sha256()
    with memoryview(data) as view:
        for offset in range(0, len(view), 1024**2):
            check()
            digest.update(view[offset : offset + 1024**2])
    check()
    return digest.digest()


def _versions(av: Any) -> tuple[int, ...]:
    return tuple(
        (major << 16) | (minor << 8) | patch
        for major, minor, patch in (
            av.library_versions[name] for name in ("libavcodec", "libavformat", "libavutil", "libswscale")
        )
    )


@dataclass(frozen=True, slots=True)
class _Frame:
    pts: int
    dts: int
    duration: int
    width: int
    height: int
    format: int
    key: bool
    digest: bytes
    anchor: int


@dataclass(slots=True)
class _Index:
    source_size: int
    binding: bytes
    base: Fraction
    origin: int
    build_bytes: int
    blocks: list[bytes]
    frames: list[_Frame]

    @property
    def size(self) -> int:
        return _HEADER_BYTES + 32 + len(self.blocks) * 32 + len(self.frames) * _FRAME_BYTES


def _encode(index: _Index, av: Any, check: Callable[[], None]) -> bytes:
    output = bytearray(_MAGIC)
    output.extend(struct.pack("<5Q", *_versions(av), index.source_size))
    output.extend(_digest(vane.__git_revision__.encode(), check))
    output.extend(index.binding)
    output.extend(
        struct.pack(
            "<QQqQQQ",
            index.base.numerator,
            index.base.denominator,
            index.origin,
            index.build_bytes,
            len(index.blocks),
            len(index.frames),
        )
    )
    for block in index.blocks:
        check()
        output.extend(block)
    for frame in index.frames:
        check()
        output.extend(
            _FRAME.pack(
                frame.pts, frame.dts, frame.duration, frame.width, frame.height, frame.format, frame.key, frame.digest
            )
        )
    output.extend(_digest(output, check))
    return bytes(output)


def _decode(data: bytes, av: Any, check: Callable[[], None]) -> _Index:
    check()
    if len(data) > _MAX_BYTES:
        raise vane.OutOfRangeException("video index exceeds 64 MiB")
    if len(data) < _HEADER_BYTES + 32 or data[:8] != _MAGIC:
        raise vane.InvalidInputException("invalid video index format")
    with memoryview(data) as view:
        if _digest(view[:-32], check) != data[-32:]:
            raise vane.InvalidInputException("video index integrity check failed")
    if struct.unpack_from("<4Q", data, 8) != _versions(av):
        raise vane.InvalidInputException("video index requires the codec build that created it")
    source_size = struct.unpack_from("<Q", data, 40)[0]
    if data[48:80] != _digest(vane.__git_revision__.encode(), check):
        raise vane.InvalidInputException("video index requires the engine SourceID that created it")
    numerator, denominator, origin, build_bytes, blocks, frames = struct.unpack_from("<QQqQQQ", data, 112)
    if (
        not 0 < source_size <= 16 * 1024**3
        or not 0 < numerator <= (1 << 31) - 1
        or not 0 < denominator <= (1 << 31) - 1
        or math.gcd(numerator, denominator) != 1
        or blocks != (source_size - 1) // _BLOCK_BYTES + 1
        or not 0 < frames <= (_MAX_BYTES - _HEADER_BYTES - 32) // _FRAME_BYTES
        or _HEADER_BYTES + 32 + blocks * 32 + frames * _FRAME_BYTES != len(data)
    ):
        raise vane.InvalidInputException("invalid video index dimensions or counts")
    result = _Index(source_size, data[80:112], Fraction(numerator, denominator), origin, build_bytes, [], [])
    offset = _HEADER_BYTES
    for _ in range(blocks):
        check()
        result.blocks.append(data[offset : offset + 32])
        offset += 32
    valid_formats = {int(av.VideoFormat(name)) for name in av.video.format.names}
    anchor = 0
    for ordinal in range(frames):
        check()
        pts, dts, duration, width, height, pixel_format, key, digest = _FRAME.unpack_from(data, offset)
        offset += _FRAME_BYTES
        if (
            pts == _NOPTS
            or duration < 0
            or (ordinal and pts <= result.frames[-1].pts)
            or width <= 0
            or height <= 0
            or width * height > video.DEFAULT_VIDEO_MAX_PIXELS
            or pixel_format not in valid_formats
            or key not in (0, 1)
            or (ordinal == 0 and not key)
        ):
            raise vane.InvalidInputException("invalid video index frame record")
        if key:
            anchor = ordinal
        result.frames.append(_Frame(pts, dts, duration, width, height, pixel_format, bool(key), digest, anchor))
    return result


def _source_binding(value: vane.VideoFile, reader: vane.VaneFileReader) -> bytes:
    if sum(len((item or "").encode()) for item in (value.url, value.content_type, value.checksum)) > 1024**2:
        raise vane.OutOfRangeException("video index FILE metadata exceeds 1 MiB")
    return _digest(reader._source_identity(), reader._check_interrupted)


class _Reader:
    """One bounded FILE cursor, with at most one authenticated block cached."""

    def __init__(self, raw: vane.VaneFileReader, limit: int, index: _Index | None):
        self.raw, self.limit, self.index = raw, limit, index
        self.size = raw.size()
        self.position = self.logical_bytes = self.bytes_read = 0
        self.cached = -1
        self.buffer = b""

    def _check_interrupted(self) -> None:
        self.raw._check_interrupted()

    def _read_at(self, count: int, offset: int) -> bytes:
        self.raw.seek(offset)
        data = self.raw._read_and_check_interrupted(count)
        if len(data) != count:
            raise vane.IOException("video FILE view ended during a read")
        return data

    def _read_and_check_interrupted(self, count: int) -> bytes:
        self._check_interrupted()
        if self.position == self.size:
            return b""
        if self.logical_bytes >= self.limit:
            raise vane.OutOfRangeException("native media exceeded its read/probe byte budget")
        count = min(count if count >= 0 else self.size, self.size - self.position, self.limit - self.logical_bytes)
        data: bytes | bytearray
        if self.index is None:
            data = self._read_at(count, self.position)
            self.bytes_read += count
        else:
            data = bytearray()
            offset = self.position
            while len(data) < count:
                self._check_interrupted()
                block, within = divmod(offset, _BLOCK_BYTES)
                if block != self.cached:
                    start = block * _BLOCK_BYTES
                    size = min(_BLOCK_BYTES, self.size - start)
                    if size > self.limit - self.bytes_read:
                        raise vane.OutOfRangeException("indexed video exceeds its physical read byte budget")
                    self.buffer = self._read_at(size, start)
                    self.bytes_read += size
                    if _digest(self.buffer, self._check_interrupted) != self.index.blocks[block]:
                        raise vane.InvalidInputException("video index source bytes have changed")
                    self.cached = block
                size = min(count - len(data), len(self.buffer) - within)
                data.extend(self.buffer[within : within + size])
                offset += size
        self.position += count
        self.logical_bytes += count
        self._check_interrupted()
        return bytes(data)

    def _readinto_and_check_interrupted(self, buffer: Any) -> int:
        data = self._read_and_check_interrupted(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._check_interrupted()
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            return -1
        position = (0, self.position, self.size)[whence] + offset
        if not 0 <= position <= self.size:
            return -1
        self.position = position
        return position

    def tell(self) -> int:
        return self.position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return self.raw.closed


@contextmanager
def _open_file(value: vane.VideoFile, execution_context: Any, connection: Any = None) -> Iterator[Any]:
    raw = (
        _file_open_in_datasource_context(value, 1, execution_context=execution_context)
        if execution_context is not None
        else value.open(buffer_size=1, connection=connection)
    )
    with video._close_video_reader(raw):
        yield raw


class _Decoder:
    def __init__(self, raw: Any, options: Any, index: _Index | None, av: Any):
        self.options, self.av = options, av
        self.io = _Reader(raw, options.max_input_bytes * 4, index)
        self.reader = video._VideoReaderProxy(self.io)
        self.nested = video._NestedIOBlocker()
        self.container: Any = None
        self.stream: Any = None
        self.frames: list[Any] = []
        self.at = 0
        self.eof = False
        self.consumer_error = False

    def check(self) -> None:
        video._check_video_io(self.reader, self.nested)

    def next(self) -> Any:
        while self.at == len(self.frames):
            self.frames.clear()
            self.at = 0
            if self.eof:
                return None
            pair = video._next_demuxed_packet(self.container, self.stream, self.reader, self.nested)
            if pair is None:
                self.eof = True
                return None
            packet, self.eof = pair
            try:
                self.frames = video._decode_packet_frames(packet, self.reader, self.nested)
            finally:
                packet = pair = None
        frame = self.frames[self.at]
        self.frames[self.at] = None
        self.at += 1
        try:
            video._decoded_frame_dimensions(frame, self.options)
            video._optional_frame_integer(frame.duration, name="frame duration", nonnegative=True)
            return frame
        finally:
            frame = None

    def seek(self, pts: int) -> None:
        self.frames.clear()
        self.at = 0
        self.check()
        self.container.seek(pts, stream=self.stream, backward=True, any_frame=False)
        self.check()
        self.eof = False


@contextmanager
def _open_decoder(raw: Any, value: Any, options: Any, index: _Index | None, av: Any) -> Iterator[_Decoder]:
    if raw.size() > options.max_input_bytes:
        raise vane.OutOfRangeException("native media input exceeds max_input_bytes")
    decoder = _Decoder(raw, options, index, av)
    try:
        codec_options, probe_options = video._video_probe_options(min(raw.size(), video.DEFAULT_VIDEO_METADATA_BYTES))
        decoder.container = av.open(
            decoder.reader,
            mode="r",
            options=codec_options,
            container_options=probe_options,
            stream_options=[codec_options.copy()],
            metadata_encoding="utf-8",
            metadata_errors="replace",
            buffer_size=_BLOCK_BYTES,
            timeout=(video._VIDEO_METADATA_TIMEOUT_SECONDS,) * 2,
            io_open=decoder.nested,
        )
        with video._close_container(decoder.container):
            decoder.check()
            video._metadata_from_container(decoder.container, value.content_type, av, max_pixels=options.max_pixels)
            decoder.stream = video._select_video_stream(decoder.container, av)
            video._configure_video_decoder(decoder.stream)
            yield decoder
            decoder.check()
        decoder.check()
    except BaseException as error:
        if decoder.consumer_error or not isinstance(error, Exception):
            raise
        video._classify_video_decode_error(error, av_module=av, reader=decoder.reader, nested_io=decoder.nested)
    finally:
        decoder.frames.clear()
        decoder.stream = decoder.container = None


def _frame_digest(frame: Any, decoder: _Decoder, reformatter: Any, image_module: Any) -> bytes:
    image = None
    try:
        info = video._decoded_frame_info(
            frame, decoder.stream, decoder.options, frame_index=0, stream_time_origin=Fraction(0)
        )
        image = video._frame_to_image(
            frame,
            info,
            replace(decoder.options, width=None, height=None),
            decoder.av,
            image_module,
            reformatter,
            decoder.check,
        )
        digest = hashlib.sha256(struct.pack("<3Q", frame.width, frame.height, int(frame.format)))
        pixels = image.tobytes()
        with memoryview(pixels) as view:
            for offset in range(0, len(view), 1024**2):
                decoder.check()
                digest.update(view[offset : offset + 1024**2])
        return digest.digest()
    finally:
        if image is not None:
            video._close_image(image)
        frame = None


def _build_video_index(value: Any, options: dict[str, Any], max_index_bytes: int, execution_context: Any) -> bytes:
    normalized = video._normalize_frame_options(buffer_size=_BLOCK_BYTES, **options)
    av, images = video._load_av(), video._load_pillow()
    with _open_file(value, execution_context) as raw:
        size = raw.size()
        if not 0 < size <= normalized.max_input_bytes:
            raise vane.OutOfRangeException("video index input exceeds max_input_bytes or is empty")
        index = _Index(size, _source_binding(value, raw), Fraction(1), 0, 0, [], [])
        if _HEADER_BYTES + 32 + ((size - 1) // _BLOCK_BYTES + 1) * 32 > max_index_bytes:
            raise vane.OutOfRangeException(_LIMIT_MESSAGE)
        for offset in range(0, size, _BLOCK_BYTES):
            raw._check_interrupted()
            data = raw._read_and_check_interrupted(min(_BLOCK_BYTES, size - offset))
            if len(data) != min(_BLOCK_BYTES, size - offset):
                raise vane.IOException("video FILE view ended during a read")
            index.blocks.append(_digest(data, raw._check_interrupted))
        raw.seek(0)
        with _open_decoder(raw, value, normalized, index, av) as decoder:
            index.base = Fraction(decoder.stream.time_base)
            index.origin = decoder.stream.start_time or 0
            reformatter = av.video.reformatter.VideoReformatter()
            frame = None
            anchor = 0
            try:
                while (frame := decoder.next()) is not None:
                    if len(index.frames) >= normalized.max_frames:
                        raise vane.OutOfRangeException("video indexing exceeds max_decoded_frames")
                    if _FRAME_BYTES > max_index_bytes - index.size:
                        raise vane.OutOfRangeException(_LIMIT_MESSAGE)
                    pts = frame.pts
                    if (
                        pts is None
                        or (index.frames and pts <= index.frames[-1].pts)
                        or (not index.frames and not frame.key_frame)
                    ):
                        raise vane.NotImplementedException(
                            "video indexing requires unique increasing presentation timestamps and an initial keyframe"
                        )
                    if frame.key_frame:
                        anchor = len(index.frames)
                    index.frames.append(
                        _Frame(
                            pts,
                            frame.dts if frame.dts is not None else _NOPTS,
                            frame.duration,
                            frame.width,
                            frame.height,
                            int(frame.format),
                            bool(frame.key_frame),
                            _frame_digest(frame, decoder, reformatter, images),
                            anchor,
                        )
                    )
                    frame = None
                if not index.frames:
                    raise video.VideoFileFormatError("video index has no decoded frames")
                if _source_binding(value, raw) != index.binding:
                    raise vane.InvalidInputException("video index source metadata changed during indexing")
                index.build_bytes = size + decoder.io.bytes_read
                return _encode(index, av, decoder.check)
            finally:
                frame = reformatter = None


def _video_index_info(data: bytes, execution_context: Any) -> tuple[Any, ...]:
    av = video._load_av()
    index = _decode(data, av, execution_context._check_interrupted)
    return (
        len(index.frames),
        sum(frame.key for frame in index.frames),
        index.source_size,
        index.size,
        index.build_bytes,
        av.ffmpeg_version_info,
    )


class _Selection:
    def __init__(self, options: Any):
        self.options = options
        self.next_sample = options.start_time
        self.last_time: Fraction | None = None

    def select(self, ordinal: int, pts: int | None, base: Fraction, origin: int, key: bool) -> bool:
        options = self.options
        if options.target_frame_index is not None:
            return ordinal == options.target_frame_index
        time = None if pts is None else (pts - origin) * base
        if time is None and (
            options.start_time > 0 or options.end_time is not None or options.sample_interval_seconds is not None
        ):
            raise video.VideoFileFormatError("video time selection requires presentation timestamps")
        if time is not None:
            if self.last_time is not None and time < self.last_time:
                self.next_sample = options.start_time
            self.last_time = time
            if time < options.start_time or (options.end_time is not None and time > options.end_time):
                return False
        if options.is_key_frame is not None and options.is_key_frame != key:
            return False
        if options.sample_interval_seconds is not None:
            assert time is not None
            if time < self.next_sample:
                return False
            self.next_sample = video._advance_sample_target(self.next_sample, options.sample_interval_seconds, time)
        return True


def _selected_frames(
    decoder: _Decoder, index: _Index | None, images: Any, stats: dict[str, int]
) -> Generator[tuple[Any, video._DecodedFrameInfo], None, None]:
    options = decoder.options
    base = Fraction(decoder.stream.time_base)
    origin = decoder.stream.start_time or 0
    if index is not None and (base != index.base or origin != index.origin):
        raise vane.InvalidInputException("video index stream metadata does not match")
    selection = _Selection(options)
    reformatter = decoder.av.video.reformatter.VideoReformatter() if index is not None else None
    current = 0
    positioned = False
    frame = None

    def decode() -> Any:
        frame = decoder.next()
        try:
            if frame is not None:
                if stats["decoded_frames"] >= options.max_frames:
                    raise vane.OutOfRangeException("video exceeds max_decoded_frames")
                stats["decoded_frames"] += 1
            return frame
        finally:
            frame = None

    try:
        if index is None:
            while (frame := decode()) is not None:
                ordinal = stats["decoded_frames"] - 1
                if selection.select(ordinal, frame.pts, base, origin, bool(frame.key_frame)):
                    stats["selected_frames"] += 1
                    info = video._decoded_frame_info(
                        frame, decoder.stream, options, frame_index=ordinal, stream_time_origin=origin * base
                    )
                    yield frame, info
                    if options.target_frame_index is not None:
                        return
                frame = None
            return
        ordinals = range(len(index.frames)) if options.target_frame_index is None else (options.target_frame_index,)
        timestamps = [record.pts for record in index.frames]
        for target in ordinals:
            decoder.check()
            if target >= len(index.frames):
                return
            expected = index.frames[target]
            if (
                options.target_frame_index is None
                and options.end_time is not None
                and (expected.pts - origin) * base > options.end_time
            ):
                return
            if not selection.select(target, expected.pts, base, origin, expected.key):
                continue
            if not positioned or expected.anchor > current + 1:
                try:
                    decoder.seek(index.frames[expected.anchor].pts)
                except decoder.av.error.FFmpegError as error:
                    decoder.check()
                    if not video._is_pyav_media_or_codec_error(error, decoder.av):
                        raise
                    raise vane.NotImplementedException("indexed video keyframe seek is unavailable") from error
                stats["seeks"] += 1
                positioned = False
            while True:
                try:
                    frame = decode()
                except decoder.av.error.FFmpegError as error:
                    decoder.check()
                    if not video._is_pyav_media_or_codec_error(error, decoder.av):
                        raise
                    raise vane.NotImplementedException("video keyframe seek cannot decode the indexed frame") from error
                if frame is None:
                    raise vane.InvalidInputException("indexed video ended before the requested frame")
                current = (
                    current + 1
                    if positioned
                    else bisect.bisect_left(timestamps, frame.pts if frame.pts is not None else _NOPTS)
                )
                positioned = True
                if current > target or current >= len(index.frames) or frame.pts != index.frames[current].pts:
                    raise vane.NotImplementedException(
                        "video keyframe seek cannot reproduce the indexed presentation order"
                    )
                entry = index.frames[current]
                if (
                    bool(frame.key_frame) != entry.key
                    or _frame_digest(frame, decoder, reformatter, images) != entry.digest
                ):
                    raise vane.NotImplementedException("video keyframe seek cannot reproduce the indexed frame")
                if current == target:
                    info = video._decoded_frame_info(
                        frame, decoder.stream, options, frame_index=current, stream_time_origin=origin * base
                    )
                    info = replace(
                        info, frame_dts=None if entry.dts == _NOPTS else entry.dts, frame_duration=entry.duration
                    )
                    stats["selected_frames"] += 1
                    yield frame, info
                    break
                frame = None
    finally:
        frame = reformatter = None


def _iter_frames(
    value: Any,
    options: Any,
    av: Any,
    images: Any,
    execution_context: Any,
    data: bytes | None,
    stats: dict[str, int],
    *,
    pixels: bool = True,
    connection: Any = None,
) -> Generator[video.VideoFrameData, None, None]:
    index = None if data is None else _decode(data, av, execution_context._check_interrupted)
    with _open_file(value, execution_context, connection) as raw:
        if index is not None and (raw.size() != index.source_size or _source_binding(value, raw) != index.binding):
            raise vane.InvalidInputException("video index does not match the FILE view or source metadata")
        with _open_decoder(raw, value, options, index, av) as decoder:
            reformatter = av.video.reformatter.VideoReformatter() if pixels else None
            selected = _selected_frames(decoder, index, images, stats)
            frame = None
            image = None
            try:
                for frame, info in selected:
                    if pixels:
                        image = video._frame_to_image(frame, info, options, av, images, reformatter, decoder.check)
                    frame = None
                    result = video.VideoFrameData(
                        info.frame_index,
                        info.frame_time,
                        info.time_base,
                        info.frame_pts,
                        info.frame_dts,
                        info.frame_duration,
                        info.is_key_frame,
                        image,
                    )
                    image = None
                    try:
                        yield result
                    except BaseException:
                        decoder.consumer_error = True
                        raise
                    finally:
                        del result
                decoder.check()
                stats["bytes_read"] = decoder.io.bytes_read
            finally:
                selected.close()
                if image is not None:
                    video._close_image(image)
                frame = image = reformatter = None


def _scan_stats(value: Any, options: dict[str, Any], data: bytes | None, execution_context: Any) -> tuple[int, ...]:
    normalized = video._normalize_frame_options(buffer_size=_BLOCK_BYTES, **options)
    av = video._load_av()
    images = video._load_pillow() if data is not None else None
    stats = dict(bytes_read=0, decoded_frames=0, seeks=0, selected_frames=0)
    frames = _iter_frames(value, normalized, av, images, execution_context, data, stats, pixels=False)
    try:
        for _ in frames:
            pass
    except BaseException:
        try:
            frames.close()
        except BaseException:
            pass
        raise
    else:
        frames.close()
    return tuple(stats.values())


def _video_frames(
    value: Any,
    options: Any,
    av: Any,
    images: Any,
    connection: Any,
    execution_context: Any = None,
    index: bytes | None = None,
) -> Generator[video.VideoFrameData, None, None]:
    stats = dict(bytes_read=0, decoded_frames=0, seeks=0, selected_frames=0)
    return _iter_frames(value, options, av, images, execution_context, index, stats, connection=connection)
