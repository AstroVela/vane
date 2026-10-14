# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded container extraction for the Python video embedding pipeline."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from vane.ai._audio_embedding import AudioClip
from vane.ai._video_embedding import VideoClip
from vane.ai._video_pipeline_types import MODALITIES, ClipWindow, TimedContent, VideoMediaInfo, positive_number


class VaneVideoSource:
    """Read Vane FILE URIs using PyAV; ASR and OCR are explicit timed extractors.

    The first non-attached video, first audio, and first subtitle stream are used.
    Override extraction to select languages or external subtitles. Callbacks use
    ``(uri, clip, media)`` and return a sequence of absolute ``TimedContent``.
    They may call this source's ``extract`` for another channel (e.g. ASR reads
    audio). No model is selected or downloaded by the source.
    """

    def __init__(
        self,
        *,
        extractors: Mapping[str, Callable[[str, ClipWindow, VideoMediaInfo], Sequence[TimedContent]]] | None = None,
        connection: Any = None,
        frames_per_clip: int = 8,
        sample_rate: int = 48000,
        max_input_bytes: int = 8 * 1024**3,
        max_decoded_frames: int = 100000,
        max_clip_bytes: int = 64 * 1024**2,
        max_pixels: int = 32 * 1024**2,
        timeout_seconds: float = 30,
    ) -> None:
        for name, value in (
            ("frames_per_clip", frames_per_clip),
            ("sample_rate", sample_rate),
            ("max_input_bytes", max_input_bytes),
            ("max_decoded_frames", max_decoded_frames),
            ("max_clip_bytes", max_clip_bytes),
            ("max_pixels", max_pixels),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if frames_per_clip > 256:
            raise ValueError("frames_per_clip cannot exceed 256")
        self._extractors = dict(extractors or {})
        if any(key not in MODALITIES or not callable(fn) for key, fn in self._extractors.items()):
            raise ValueError("extractors must map known modalities to callables")
        self.supported_modalities = frozenset({"visual", "audio", "subtitles"} | self._extractors.keys())
        self.connection = connection
        self.frames_per_clip = frames_per_clip
        self.sample_rate = sample_rate
        self.max_input_bytes = max_input_bytes
        self.max_decoded_frames = max_decoded_frames
        self.max_clip_bytes = max_clip_bytes
        self.max_pixels = max_pixels
        self.timeout_seconds = positive_number(timeout_seconds, "timeout_seconds")

    @contextmanager
    def _open(self, uri: str) -> Iterator[tuple[Any, Any]]:
        import vane
        from vane._video_file import (
            DEFAULT_VIDEO_METADATA_BYTES,
            _close_container,
            _container_format_names,
            _load_av,
            _NestedIOBlocker,
            _validate_content_type,
            _video_probe_options,
            _VideoReaderProxy,
        )

        av = _load_av()
        with vane.File(uri).open(connection=self.connection) as reader:
            if reader.size() > self.max_input_bytes:
                raise ValueError("Video source exceeds max_input_bytes")
            adapter, nested = _VideoReaderProxy(reader), _NestedIOBlocker()
            decoder, probe = _video_probe_options(min(reader.size(), DEFAULT_VIDEO_METADATA_BYTES))
            try:
                container = av.open(
                    adapter,
                    mode="r",
                    options=decoder,
                    container_options=probe,
                    stream_options=[decoder.copy()],
                    io_open=nested,
                    timeout=(self.timeout_seconds, self.timeout_seconds),
                )
                with _close_container(container):
                    _validate_content_type(None, _container_format_names(container))
                    yield container, av
            finally:
                adapter.raise_if_error()
                nested.raise_if_error()

    @staticmethod
    def _streams(container: Any, av: Any) -> dict[str, Any]:
        streams: dict[str, Any] = {}
        for stream in container.streams:
            if stream.type == "video" and not (stream.disposition & av.stream.Disposition.attached_pic):
                streams.setdefault("visual", stream)
            elif stream.type in {"audio", "subtitle"}:
                streams.setdefault("subtitles" if stream.type == "subtitle" else "audio", stream)
        return streams

    def probe(self, uri: str) -> VideoMediaInfo:
        from vane._video_file import _container_format_names

        with self._open(uri) as (container, av):
            streams = self._streams(container, av)
            starts = [
                float(s.start_time * s.time_base)
                for s in streams.values()
                if s.start_time is not None and s.time_base is not None
            ]
            origin = (
                float(container.start_time / av.time_base)
                if container.start_time is not None
                else min(starts, default=0.0)
            )
            durations = []
            if container.duration is not None:
                duration = float(container.duration / av.time_base)
                # Matroska/WebM's segment duration is an end timestamp from
                # segment zero; rebasing its first packet also rebases that end.
                if {"matroska", "webm"}.intersection(_container_format_names(container)):
                    duration -= origin
                durations.append(duration)
            durations.extend(
                float((s.start_time or 0) * s.time_base + s.duration * s.time_base) - origin
                for s in streams.values()
                if s.duration is not None and s.time_base is not None
            )
            duration = max(durations, default=0)
            available = set(streams)
            if "audio" in available:
                available.add("transcription")
            if "visual" in available:
                available.add("ocr")
            # An explicitly configured text extractor can supply external captions.
            if "subtitles" in self._extractors:
                available.add("subtitles")
            return VideoMediaInfo(duration, frozenset(available), origin)

    def extract(self, uri: str, modality: str, clip: ClipWindow, media: VideoMediaInfo) -> Sequence[TimedContent]:
        if modality in self._extractors:
            return self._extractors[modality](uri, clip, media)
        if modality not in {"visual", "audio", "subtitles"}:
            raise ValueError(f"Configure an extractor for {modality}")
        with self._open(uri) as (container, av):
            stream = self._streams(container, av).get(modality)
            if stream is None:
                # Probe already established presence; disappearing tracks are errors.
                raise ValueError("Video tracks changed after container probing")
            if modality == "subtitles":
                return self._subtitles(container, stream, clip, media)
            if clip.start + media.origin > 0:
                if stream.time_base is None:
                    raise ValueError("Media stream has no time base")
                # Seek on the selected track: a video-keyframe seek can discard
                # an audio packet that starts before and overlaps this clip.
                container.seek(math.floor((clip.start + media.origin) / stream.time_base), stream=stream, backward=True)
            if modality == "visual":
                return self._visual(container, stream, clip, media)
            return self._audio(container, stream, clip, media, av)

    @staticmethod
    def _time(frame: Any, media: VideoMediaInfo) -> float:
        if frame.pts is None or frame.time_base is None:
            raise ValueError("Decoded content has no presentation timestamp")
        value = float(frame.pts * frame.time_base) - media.origin
        if not math.isfinite(value):
            raise ValueError("Decoded content has an invalid presentation timestamp")
        return value

    def _visual(self, container: Any, stream: Any, clip: ClipWindow, media: VideoMediaInfo) -> list[TimedContent]:
        from vane._video_file import _configure_video_decoder, _neutralized_color_conversion_metadata

        _configure_video_decoder(stream)
        frames: list[np.ndarray[Any, Any]] = []
        times: list[float] = []
        indices: list[int] = []
        size = 0
        interval = (clip.end - clip.start) / self.frames_per_clip
        previous = -math.inf
        for index, frame in enumerate(container.decode(stream)):
            if index >= self.max_decoded_frames:
                raise ValueError("Video decoding exceeds max_decoded_frames")
            timestamp = self._time(frame, media)
            if timestamp < previous:
                raise ValueError("Video frames are not in presentation order")
            previous = timestamp
            if timestamp >= clip.end:
                break
            if timestamp < clip.start + len(frames) * interval:
                continue
            if frame.width * frame.height > self.max_pixels:
                raise ValueError("Video frame exceeds max_pixels")
            size += frame.width * frame.height * 3
            if size > self.max_clip_bytes:
                raise ValueError("Visual clip exceeds max_clip_bytes")
            with _neutralized_color_conversion_metadata(frame):
                pixels = frame.to_ndarray(format="rgb24")
            frames.append(pixels)
            times.append(timestamp)
            indices.append(index)
            if len(frames) == self.frames_per_clip:
                break
        return (
            [TimedContent(clip.start, clip.end, VideoClip(tuple(frames), tuple(times), tuple(indices)))]
            if frames
            else []
        )

    def _audio(
        self, container: Any, stream: Any, clip: ClipWindow, media: VideoMediaInfo, av: Any
    ) -> list[TimedContent]:
        count = math.ceil((clip.end - clip.start) * self.sample_rate)
        if count * 4 > self.max_clip_bytes:
            raise ValueError("Audio clip exceeds max_clip_bytes")
        # A fixed window retains gaps and the offset of delayed audio tracks.
        samples: np.ndarray[Any, np.dtype[np.float32]] = np.zeros((count, 1), dtype=np.float32)
        present = False
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=self.sample_rate)
        stream.codec_context.thread_count = 1

        def collect(frame: Any) -> None:
            nonlocal present
            timestamp = self._time(frame, media)
            offset = round((timestamp - clip.start) * self.sample_rate)
            first, last = max(0, offset), min(count, offset + frame.samples)
            if first < last:
                data = frame.to_ndarray().T
                samples[first:last] = data[first - offset : last - offset]
                present = True

        for index, frame in enumerate(container.decode(stream)):
            if index >= self.max_decoded_frames:
                raise ValueError("Audio decoding exceeds max_decoded_frames")
            timestamp = self._time(frame, media)
            if timestamp >= clip.end:
                break
            if timestamp + frame.samples / frame.sample_rate <= clip.start:
                continue
            for converted in resampler.resample(frame):
                collect(converted)
        for converted in resampler.resample(None):
            collect(converted)
        return [TimedContent(clip.start, clip.end, AudioClip(samples, self.sample_rate))] if present else []

    def _subtitles(self, container: Any, stream: Any, clip: ClipWindow, media: VideoMediaInfo) -> list[TimedContent]:
        # Scan from the start: seeking at a clip boundary can lose an overlapping
        # cue. Packet and output byte bounds keep this scan explicit and bounded.
        result = []
        size = 0
        for index, packet in enumerate(container.demux(stream)):
            if index >= self.max_decoded_frames:
                raise ValueError("Subtitle scan exceeds max_decoded_frames")
            if packet.pts is None:
                if packet.size:
                    raise ValueError("Subtitle packet has no timestamp")
                continue
            start = float(packet.pts * packet.time_base) - media.origin
            if start >= clip.end:
                break
            if not packet.duration:
                raise ValueError("Subtitle packet has no duration")
            end = start + float(packet.duration * packet.time_base)
            if end <= clip.start:
                continue
            for subtitle in packet.decode():
                text = getattr(subtitle, "dialogue", None)
                if text is None:
                    raise ValueError("Bitmap subtitles require a configured subtitle OCR extractor")
                if isinstance(text, bytes):
                    text = text.decode("utf-8", errors="strict")
                text = re.sub(r"\{[^}]*\}", "", text).replace(r"\N", "\n").replace(r"\n", "\n")
                size += len(text.encode("utf-8"))
                if size > self.max_clip_bytes:
                    raise ValueError("Subtitle clip exceeds max_clip_bytes")
                result.append(TimedContent(max(clip.start, start), min(clip.end, end), text))
        return result
