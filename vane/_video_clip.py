# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded MP4 clips on the source video's presentation timeline."""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import vane
from vane._expressions import as_expression
from vane._file import _file_open_in_datasource_context
from vane._video_expressions import _argument
from vane._video_file import (
    DEFAULT_VIDEO_BUFFER_SIZE,
    DEFAULT_VIDEO_MAX_PIXELS,
    DEFAULT_VIDEO_METADATA_BYTES,
    VideoFileFormatError,
    VideoFileLimitError,
    _check_video_io,
    _classify_video_decode_error,
    _close_container,
    _close_demux_iterator,
    _close_video_reader,
    _configure_video_decoder,
    _load_av,
    _metadata_from_container,
    _NestedIOBlocker,
    _neutralized_color_conversion_metadata,
    _nonnegative_time,
    _positive_limit,
    _select_video_stream,
    _stream_time_origin,
    _video_probe_options,
    _VideoReaderProxy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vane._native import _DataSourceExecutionContext

_TIME_BASE = Fraction(1, 60_000)
_LIMITS = {
    "max_input_bytes": (1024**3, 16 * 1024**3),
    "max_decoded_frames": (100_000, 1_000_000),
    "max_decoded_samples": (100_000_000, 1_000_000_000),
    "max_pixels": (DEFAULT_VIDEO_MAX_PIXELS, DEFAULT_VIDEO_MAX_PIXELS),
    "max_output_bytes": (64 * 1024**2, 256 * 1024**2),
}
_AUDIO_RATES = {7350, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000}


@dataclass(frozen=True, slots=True)
class VideoClip:
    """An encoded MP4 and its mapping to the source video timeline.

    Add ``start_time`` to a clip presentation timestamp to recover the source
    time. ``end_time`` is the actual exclusive endpoint; ``duration`` is the
    encoded video duration. Timestamps are quantized to 1/60000 second, and
    audio to its original sample grid. ``data`` owns the detached bytes.
    """

    data: bytes
    content_type: str
    start_time: float
    end_time: float
    duration: float
    frame_count: int
    has_audio: bool


def video_clip(
    value: vane.VideoFile | vane.Expression,
    start_time: float | vane.Expression,
    end_time: float | vane.Expression,
    *,
    include_audio: bool | vane.Expression = True,
    max_duration: float | vane.Expression = 300,
    max_input_bytes: int | vane.Expression = 1024**3,
    max_decoded_frames: int | vane.Expression = 100_000,
    max_decoded_samples: int | vane.Expression = 100_000_000,
    max_pixels: int | vane.Expression = DEFAULT_VIDEO_MAX_PIXELS,
    max_output_bytes: int | vane.Expression = 64 * 1024**2,
    timeout_seconds: float | vane.Expression = 60,
) -> vane.Expression:
    """Build a lazy ``video_clip`` expression returning a STRUCT like VideoClip.

    Select ``[start_time, end_time)`` relative to the first video stream's
    declared origin. The picture already displayed at the start is retained
    for the intersecting part of its presentation interval. Transcode to
    MP4 (MPEG-4 Part 2 / yuv420p and, when present, the first audio track as
    AAC). No keyframe expansion, codec fallback, resizing or audio downmixing
    is performed. Requires ``vane-ai[video]`` and the Python video backend.

    NULL inputs return NULL. Invalid media, missing codecs, exhausted limits
    and intervals outside the video raise errors. The timeout is checked at
    I/O, codec and cleanup boundaries; an atomic codec call is not preempted.
    """
    options = dict(
        start_time=start_time,
        end_time=end_time,
        include_audio=include_audio,
        max_duration=max_duration,
        max_input_bytes=max_input_bytes,
        max_decoded_frames=max_decoded_frames,
        max_decoded_samples=max_decoded_samples,
        max_pixels=max_pixels,
        max_output_bytes=max_output_bytes,
        timeout_seconds=timeout_seconds,
    )
    arguments = [as_expression(value)]
    for name, option in options.items():
        kind = bool if name == "include_audio" else int if name in _LIMITS else float
        arguments.append(_argument(option, name, kind, _LIMITS[name][1] if name in _LIMITS else None))
    if not any(isinstance(option, vane.Expression) for option in options.values()):
        _normalize(options)
    return vane.FunctionExpression("_vane_video_clip", *arguments)


def _normalize(options: dict[str, Any]) -> dict[str, Any]:
    result = dict(options)
    for name in ("start_time", "end_time", "max_duration", "timeout_seconds"):
        result[name] = _nonnegative_time(result[name], name=name)
    if not 0 <= result["start_time"] < result["end_time"] <= 1_000_000_000:
        raise ValueError("video_clip requires 0 <= start_time < end_time <= 1000000000")
    for name in ("max_duration", "timeout_seconds"):
        if not 0 < result[name] <= 3600:
            raise ValueError(f"{name} must be positive and at most 3600 seconds")
    duration = result["end_time"] - result["start_time"]
    if duration > result["max_duration"]:
        raise VideoFileLimitError("video_clip interval exceeds max_duration")
    if duration < _TIME_BASE:
        raise ValueError("video_clip interval must be at least 1/60000 second")
    if not isinstance(result["include_audio"], bool):
        raise TypeError("include_audio must be bool")
    for name, (_, maximum) in _LIMITS.items():
        result[name] = _positive_limit(result[name], name=name, maximum=maximum)
    return result


class _ClipOutput(io.BytesIO):
    """Bound the muxer's allocation before every write, including its trailer."""

    def __init__(self, maximum: int, check: Callable[[], None]) -> None:
        super().__init__()
        self.maximum = maximum
        self.check = check

    def write(self, data: Any) -> int:
        try:
            self.check()
            if self.tell() + len(data) > self.maximum:
                raise VideoFileLimitError("video_clip exceeds max_output_bytes")
            return super().write(data)
        finally:
            data = None

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.check()
        position = super().seek(offset, whence)
        if position > self.maximum:
            raise VideoFileLimitError("video_clip exceeds max_output_bytes")
        return position


class _ClipEncoder:
    def __init__(
        self, av: Any, output: Any, video: Any, audio: Any, options: dict[str, Any], check: Callable[[], None]
    ):
        try:
            self.av, self.output, self.options, self.check = av, output, options, check
            self.origin = _stream_time_origin(video, Fraction(video.time_base))
            self.start, self.end = options["start_time"], options["end_time"]
            self.duration_ticks = math.floor(math.nextafter(float((self.end - self.start) / _TIME_BASE), math.inf))
            self.end = self.start + self.duration_ticks * _TIME_BASE
            if video.average_rate is None or not 0 < video.average_rate <= 60_000:
                raise VideoFileFormatError("video_clip requires a declared video frame rate between 0 and 60000")
            self.video = output.add_stream("mpeg4", rate=video.average_rate)
            self.video.width, self.video.height = video.width, video.height
            self.video.pix_fmt = "yuv420p"
            # Reformat pixels to this matrix/range before encoding. Initialize
            # primaries/transfer from a decoded picture, since bounded probing
            # may not discover color metadata carried only in the bitstream.
            self.video.codec_context.colorspace = 1  # BT.709
            self.video.codec_context.color_range = 1  # MPEG / limited
            self.color: tuple[int, int] | None = None
            self.pending_audio_packets: list[Any] = []
            self.pending_audio_bytes = 0
            self.video.time_base = self.video.codec_context.time_base = _TIME_BASE
            self.video.codec_context.max_b_frames = 0
            self.video.codec_context.thread_count = 1
            self.video.codec_context.bit_rate = min(20_000_000, max(400_000, video.width * video.height * 8))
            self.previous: Any = None
            self.previous_time: Fraction | None = None
            self.previous_sar: Fraction | None = None
            self.video_done = False
            self.frame_count = self.decoded_frames = self.decoded_samples = 0
            self.video_end_ticks = 0
            self.durations: dict[int, int] = {}
            self.audio: Any = None
            self.resampler: Any = None
            self.audio_end: Fraction | None = None
            self.audio_done = audio is None
            self.audio_samples = 0
            self.sample_rate = 0
            self.audio_limit = 0
            self.layout = ""
            if audio is not None:
                self.sample_rate = audio.codec_context.sample_rate
                channels = audio.codec_context.layout.nb_channels
                if self.sample_rate not in _AUDIO_RATES or audio.codec_context.layout.name not in ("mono", "stereo"):
                    raise VideoFileFormatError("video_clip audio requires an AAC-supported sample rate and mono/stereo")
                self.layout = "mono" if channels == 1 else "stereo"
                self.audio_limit = math.floor(self.duration_ticks * _TIME_BASE * self.sample_rate)
                if self.audio_limit == 0:
                    raise VideoFileFormatError("video_clip interval is shorter than one audio sample")
                self.audio = output.add_stream("aac", rate=self.sample_rate)
                self.audio.layout = self.layout
                self.audio.codec_context.thread_count = 1
                self.audio.codec_context.bit_rate = 96_000 * channels
                self.resampler = av.AudioResampler(format="fltp", layout=self.layout, rate=self.sample_rate)
        except BaseException:
            self.close()
            raise
        finally:
            output = video = audio = None

    def close(self) -> None:
        # Container.close() does not release stream/codec references retained
        # by Python. Drop every native owner even if a caller keeps traceback(s).
        self.previous = self.video = self.audio = self.resampler = self.output = None
        self.pending_audio_packets = []

    def mux(self, stream: Any, frame: Any = None) -> None:
        packet = packets = None
        try:
            self.check()
            packets = stream.encode(frame)
            try:
                self.check()
                for packet in packets:
                    self.check()
                    if stream is self.video:
                        # MPEG-4 emits one packet per frame, without reordering.
                        duration = self.durations.pop(packet.pts, None)
                        if duration is None:
                            raise VideoFileFormatError("video_clip encoder changed the video timeline")
                        packet.duration = duration
                    if self.color is None:
                        # Muxing even an audio packet opens the video codec and
                        # writes its header. Keep compressed audio bounded until
                        # the first selected picture supplies its color metadata.
                        self.pending_audio_bytes += packet.size
                        if self.pending_audio_bytes > self.options["max_output_bytes"]:
                            raise VideoFileLimitError("video_clip exceeds max_output_bytes")
                        self.pending_audio_packets.append(packet)
                    else:
                        self.output.mux(packet)
                    self.check()
            finally:
                packets.clear()
        finally:
            stream = frame = packet = packets = None

    def emit_video(self, end: Fraction) -> None:
        frame = packet = None
        try:
            assert self.previous_time is not None
            assert self.previous_sar is not None
            start, stop = max(self.previous_time, self.start), min(end, self.end)
            if stop <= start:
                return
            start_ticks = round((start - self.start) / _TIME_BASE)
            end_ticks = min(round((stop - self.start) / _TIME_BASE), self.duration_ticks)
            if self.frame_count == 0 and start_ticks > 0:
                raise VideoFileFormatError("video_clip start precedes the first video picture")
            # A partial boundary picture may disappear at output clock precision.
            # Interior intervals must still be representable without dropping frames.
            if end_ticks <= start_ticks and (start_ticks == 0 or end_ticks == self.duration_ticks):
                return
            if end_ticks <= start_ticks or start_ticks != self.video_end_ticks:
                raise VideoFileFormatError("video_clip timestamps cannot be represented on its 1/60000-second grid")
            self.check()
            primaries, trc = self.previous.color_primaries, self.previous.color_trc
            if self.color is None:
                self.video.codec_context.color_primaries = primaries
                self.video.codec_context.color_trc = trc
                self.video.codec_context.sample_aspect_ratio = self.previous_sar
                self.color = primaries, trc
                for packet in self.pending_audio_packets:
                    self.check()
                    self.output.mux(packet)
                    self.check()
                self.pending_audio_packets.clear()
                self.pending_audio_bytes = 0
            elif (primaries, trc) != self.color:
                raise VideoFileFormatError(
                    "video_clip does not support changing color primaries or transfer characteristics"
                )
            if self.previous_sar != self.video.codec_context.sample_aspect_ratio:
                raise VideoFileFormatError("video_clip does not support changing sample aspect ratios")
            # Only convert the matrix/range, including on PyAV 17 where the
            # reformatter otherwise attempts transfer/primaries conversion.
            with _neutralized_color_conversion_metadata(self.previous):
                frame = self.previous.reformat(
                    format="yuv420p", dst_colorspace="ITU709", dst_color_range="MPEG", threads=1
                )
            frame.color_primaries, frame.color_trc = primaries, trc
            frame.pts, frame.time_base = start_ticks, _TIME_BASE
            frame.duration = end_ticks - start_ticks
            # The output encoder owns picture ordering; do not force input B-frame
            # picture types onto the no-reordering output stream.
            frame.pict_type = 0
            self.check()
            self.durations[start_ticks] = frame.duration
            self.mux(self.video, frame)
            self.video_end_ticks = end_ticks
            self.frame_count += 1
        finally:
            frame = packet = None

    def video_frame(self, frame: Any, sample_aspect_ratio: Fraction | None) -> None:
        side = None
        try:
            self.check()
            if frame.width * frame.height > self.options["max_pixels"]:
                raise VideoFileLimitError("video_clip exceeds max_pixels")
            if (frame.width, frame.height) != (self.video.width, self.video.height):
                raise VideoFileFormatError("video_clip does not support changing video dimensions")
            if frame.pts is None or frame.time_base is None:
                raise VideoFileFormatError("video_clip requires video presentation timestamps")
            for side in frame.side_data:
                if side.type.name == "DISPLAYMATRIX":
                    raise VideoFileFormatError("video_clip does not support video display matrices")
            timestamp = frame.pts * Fraction(frame.time_base) - self.origin
            if self.previous_time is not None:
                if timestamp <= self.previous_time:
                    raise VideoFileFormatError("video_clip requires strictly increasing video timestamps")
                self.emit_video(timestamp)
            if float(timestamp) >= float(self.end):
                self.previous = None
                self.video_done = True
            else:
                self.previous, self.previous_time = frame, timestamp
                # PyAV exposes SAR on the stream/decoder, not the frame. Retain
                # the decoded state with this picture until its end is known.
                self.previous_sar = (
                    sample_aspect_ratio if sample_aspect_ratio is not None and sample_aspect_ratio > 0 else Fraction(1)
                )
        finally:
            frame = side = None

    def audio_frame(self, frame: Any) -> None:
        converted = planar = clipped = source = target = None
        try:
            self.check()
            if frame.sample_rate != self.sample_rate or frame.layout.name != self.layout:
                raise VideoFileFormatError("video_clip does not support changing audio rate or channel count")
            if frame.pts is None or frame.time_base is None:
                raise VideoFileFormatError("video_clip requires audio presentation timestamps")
            timestamp = frame.pts * Fraction(frame.time_base) - self.origin
            if self.audio_end is not None:
                # Only absorb sub-tick timestamp quantization. A whole tick
                # (including one sample on a sample-rate clock) is a real gap
                # or overlap, and must not shift subsequent audio.
                if abs(timestamp - self.audio_end) < Fraction(frame.time_base):
                    timestamp = self.audio_end
                elif timestamp < self.audio_end:
                    raise VideoFileFormatError("video_clip does not support overlapping audio timestamps")
            self.audio_end = timestamp + Fraction(frame.samples, self.sample_rate)
            if timestamp >= self.end:
                self.audio_done = True
                return
            if self.audio_end <= self.start:
                return
            first = max(0, math.ceil((self.start - timestamp) * self.sample_rate))
            # Round ties toward +infinity, so adding whole samples commutes with
            # rounding. Round-to-even alternates across odd-sized blocks at
            # half-sample offsets, inventing gaps/overlaps in continuous audio.
            output_pts = math.floor((timestamp - self.start) * self.sample_rate + Fraction(1, 2)) + first
            last = min(frame.samples, first + self.audio_limit - output_pts)
            if last <= first:
                # Rounding may leave no output samples even when this frame
                # reaches the endpoint. Do not decode subsequent audio packets.
                if self.audio_end >= self.end:
                    self.audio_done = True
                return
            converted = self.resampler.resample(frame)
            self.check()
            if len(converted) != 1 or converted[0].samples != frame.samples:
                raise VideoFileFormatError("video_clip audio conversion changed the sample timeline")
            planar = converted[0]
            clipped = self.av.AudioFrame(format="fltp", layout=self.layout, samples=last - first)
            for source, target in zip(planar.planes, clipped.planes, strict=True):
                target.update(memoryview(source)[first * 4 : last * 4])
            clipped.sample_rate = self.sample_rate
            clipped.time_base = Fraction(1, self.sample_rate)
            clipped.pts = output_pts
            self.pad_audio(output_pts)
            self.mux(self.audio, clipped)
            self.audio_samples += clipped.samples
            if self.audio_end >= self.end:
                self.audio_done = True
        finally:
            frame = converted = planar = clipped = source = target = None

    def pad_audio(self, until: int) -> None:
        silent = plane = None
        try:
            if until < self.audio_samples:
                raise VideoFileFormatError("video_clip audio timestamps overlap on the output sample grid")
            while self.audio_samples < until:
                self.check()
                count = min(4096, until - self.audio_samples)
                silent = self.av.AudioFrame(format="fltp", layout=self.layout, samples=count)
                for plane in silent.planes:
                    plane.update(bytes(plane.buffer_size))
                silent.sample_rate = self.sample_rate
                silent.time_base = Fraction(1, self.sample_rate)
                silent.pts = self.audio_samples
                self.mux(self.audio, silent)
                self.audio_samples += count
        finally:
            silent = plane = None

    def finish(self) -> None:
        if not self.video_done and self.previous is not None:
            assert self.previous_time is not None
            duration = self.previous.duration
            if duration is None or duration <= 0:
                raise VideoFileFormatError("video_clip requires the final picture's presentation duration")
            final_end = self.previous_time + duration * Fraction(self.previous.time_base)
            # Input float endpoints may round one ULP past the exact stream end.
            if float(final_end) < float(self.end):
                raise VideoFileFormatError("video_clip end exceeds the video timeline")
            self.emit_video(final_end)
        if not self.frame_count:
            raise VideoFileFormatError("video_clip interval contains no video pictures")
        self.previous = None
        self.mux(self.video)
        if self.durations:
            raise VideoFileFormatError("video_clip encoder did not emit all video pictures")
        if self.audio is not None:
            self.pad_audio(self.audio_limit)
            self.mux(self.audio)


def _clip(
    value: vane.VideoFile,
    options: dict[str, Any],
    connection: vane.DuckDBPyConnection | None = None,
    execution_context: _DataSourceExecutionContext | None = None,
) -> VideoClip:
    normalized = _normalize(options)
    deadline = time.monotonic() + float(normalized["timeout_seconds"])

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise VideoFileLimitError("video_clip exceeded timeout_seconds")

    av = _load_av()
    check_deadline()
    file_reader = (
        value.open(buffer_size=DEFAULT_VIDEO_BUFFER_SIZE, connection=connection)
        if execution_context is None
        else _file_open_in_datasource_context(value, DEFAULT_VIDEO_BUFFER_SIZE, execution_context=execution_context)
    )
    encoder = source = output = video = audio = streams = packets = packet = frames = frame = None
    data = None
    completed = False
    try:
        with _close_video_reader(file_reader):
            check_deadline()
            input_size = file_reader.size()
            if input_size > normalized["max_input_bytes"]:
                raise VideoFileLimitError("video_clip exceeds max_input_bytes")
            reader, nested = _VideoReaderProxy(file_reader), _NestedIOBlocker()

            def check() -> None:
                _check_video_io(reader, nested)
                check_deadline()

            with _ClipOutput(normalized["max_output_bytes"], check) as buffer:
                decoder_options, probe_options = _video_probe_options(min(input_size, DEFAULT_VIDEO_METADATA_BYTES))
                try:
                    with _close_container(
                        av.open(
                            reader,
                            mode="r",
                            options=decoder_options,
                            container_options=probe_options,
                            stream_options=[decoder_options.copy()],
                            metadata_encoding="utf-8",
                            metadata_errors="replace",
                            buffer_size=DEFAULT_VIDEO_BUFFER_SIZE,
                            timeout=min(5.0, float(normalized["timeout_seconds"])),
                            io_open=nested,
                        )
                    ) as source:
                        check()
                        metadata = _metadata_from_container(
                            source, value.content_type, av, max_pixels=normalized["max_pixels"]
                        )
                        if metadata.width % 2 or metadata.height % 2:
                            raise VideoFileFormatError("video_clip requires even video dimensions for yuv420p")
                        video = _select_video_stream(source, av)
                        if video.metadata.get("rotate", "0") not in ("0", "0.0"):
                            raise VideoFileFormatError("video_clip does not support rotated video")
                        _configure_video_decoder(video)
                        audio = next(iter(source.streams.audio), None) if normalized["include_audio"] else None
                        if audio is not None:
                            _configure_video_decoder(audio)
                        # MP4 edit lists use the movie clock, including AAC delay
                        # removal and its final partial packet. Preserve both grids
                        # exactly instead of rounding endpoints to milliseconds.
                        movie_timescale = math.lcm(
                            _TIME_BASE.denominator, audio.codec_context.sample_rate if audio is not None else 1
                        )
                        with _close_container(
                            av.open(
                                buffer,
                                mode="w",
                                format="mp4",
                                options={"max_interleave_delta": "100000", "movie_timescale": str(movie_timescale)},
                            )
                        ) as output:
                            encoder = _ClipEncoder(av, output, video, audio, normalized, check)
                            streams = [video] if audio is None else [video, audio]
                            decoded_sar: Fraction | None = None
                            with _close_demux_iterator(source.demux(streams)) as packets:
                                for packet in packets:
                                    check()
                                    is_video = packet.stream.index == video.index
                                    if encoder.video_done if is_video else encoder.audio_done:
                                        continue
                                    frames = packet.decode()
                                    try:
                                        check()
                                        if is_video:
                                            encoder.decoded_frames += len(frames)
                                            if encoder.decoded_frames > normalized["max_decoded_frames"]:
                                                raise VideoFileLimitError("video_clip exceeds max_decoded_frames")
                                            # Decoder metadata may advance before reordered pictures
                                            # are returned. PyAV has no per-frame SAR, so reject changes
                                            # before assigning a newer picture's ratio to older frames.
                                            sar = video.codec_context.sample_aspect_ratio
                                            if sar or frames:
                                                sar = sar or Fraction(1)
                                                if decoded_sar is not None and sar != decoded_sar:
                                                    raise VideoFileFormatError(
                                                        "video_clip does not support changing decoder sample aspect ratios"
                                                    )
                                                decoded_sar = sar
                                        else:
                                            for frame in frames:
                                                encoder.decoded_samples += frame.samples
                                            if encoder.decoded_samples > normalized["max_decoded_samples"]:
                                                raise VideoFileLimitError("video_clip exceeds max_decoded_samples")
                                        for frame in frames:
                                            if is_video:
                                                if not encoder.video_done:
                                                    # A container declaration takes precedence. Without
                                                    # one, only the decoder has the bitstream's SAR.
                                                    encoder.video_frame(
                                                        frame,
                                                        video.sample_aspect_ratio or decoded_sar,
                                                    )
                                            elif not encoder.audio_done:
                                                encoder.audio_frame(frame)
                                    finally:
                                        frames.clear()
                                    if encoder.video_done and encoder.audio_done:
                                        break
                            encoder.finish()
                            check()
                        check()
                    check()
                    data = buffer.getvalue()
                    check()
                except Exception as error:
                    check()
                    _classify_video_decode_error(error, av_module=av, reader=reader, nested_io=nested)
        assert encoder is not None
        duration = float(encoder.video_end_ticks * _TIME_BASE)
        frame_count, has_audio = encoder.frame_count, bool(encoder.audio_samples)
        completed = True
    finally:
        if encoder is not None:
            encoder.close()
        if frames is not None:
            frames.clear()
        encoder = source = output = video = audio = streams = packets = packet = frames = frame = None
        if not completed:
            data = None
    try:
        check_deadline()
        assert data is not None
        return VideoClip(
            data,
            "video/mp4",
            float(normalized["start_time"]),
            float(normalized["start_time"]) + duration,
            duration,
            frame_count,
            has_audio,
        )
    finally:
        data = None


def _video_file_clip_value(
    value: vane.VideoFile,
    start_time: float,
    end_time: float,
    *,
    include_audio: bool = True,
    max_duration: float = 300,
    max_input_bytes: int = 1024**3,
    max_decoded_frames: int = 100_000,
    max_decoded_samples: int = 100_000_000,
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    max_output_bytes: int = 64 * 1024**2,
    timeout_seconds: float = 60,
    connection: vane.DuckDBPyConnection | None = None,
) -> VideoClip:
    options = dict(locals())
    options.pop("value")
    options.pop("connection")
    return _clip(value, options, connection)


def _scalar_video_clip(
    value: vane.VideoFile, options: dict[str, Any], execution_context: _DataSourceExecutionContext
) -> tuple[bytes, str, float, float, float, int, bool]:
    clip = _clip(value, options, execution_context=execution_context)
    return (
        clip.data,
        clip.content_type,
        clip.start_time,
        clip.end_time,
        clip.duration,
        clip.frame_count,
        clip.has_audio,
    )
