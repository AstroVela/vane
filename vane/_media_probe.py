# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded container facts, independent of model or ingest policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from vane._file import _positive_buffer_size
from vane._video_file import (
    DEFAULT_VIDEO_METADATA_BUFFER_SIZE,
    DEFAULT_VIDEO_METADATA_BYTES,
    MAX_VIDEO_METADATA_BYTES,
    VideoFileFormatError,
    _container_format_names,
    _load_av,
    _positive_limit,
    _probe_container,
)

if TYPE_CHECKING:
    import vane


@dataclass(frozen=True, slots=True)
class MediaStream:
    """Declared stream facts; times are in seconds on the container timeline.

    Language and dispositions come from the container, not an AI classifier.
    A present stream does not establish speech, readable text or decoder support.
    """

    index: int
    kind: str
    codec: str | None
    language: str | None
    default: bool
    forced: bool
    attached_picture: bool
    time_base: Fraction | None
    start_time: float | None
    duration: float | None
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    channels: int | None = None


@dataclass(frozen=True, slots=True)
class MediaProbe:
    """A successful bounded container inspection, without semantic inference."""

    formats: tuple[str, ...]
    duration: float | None
    streams: tuple[MediaStream, ...]


def _seconds(value, time_base, *, duration=False):
    if value is None or time_base is None:
        return None
    result = float(value * time_base)
    if not math.isfinite(result) or (duration and result < 0):
        raise VideoFileFormatError("media container reported invalid timing metadata")
    if duration and result == 0:
        return None
    return result


def _known_positive(value):
    return value if value is not None and value > 0 else None


def _inspect_media(container, av) -> MediaProbe:
    formats = tuple(sorted(_container_format_names(container)))
    streams = []
    for stream in container.streams:
        codec = stream.codec_context
        base = stream.time_base
        if base is not None and base <= 0:
            raise VideoFileFormatError("media stream reported an invalid time base")
        disposition = stream.disposition
        flags = av.stream.Disposition
        language = stream.metadata.get("language")
        streams.append(
            MediaStream(
                index=stream.index,
                kind=stream.type,
                codec=codec.name if codec is not None else None,
                language=language if language and language != "und" else None,
                default=bool(disposition & flags.default),
                forced=bool(disposition & flags.forced),
                attached_picture=bool(disposition & flags.attached_pic),
                time_base=base,
                start_time=_seconds(stream.start_time, base),
                duration=_seconds(stream.duration, base, duration=True),
                width=_known_positive(codec.width) if stream.type == "video" and codec is not None else None,
                height=_known_positive(codec.height) if stream.type == "video" and codec is not None else None,
                sample_rate=_known_positive(codec.sample_rate)
                if stream.type == "audio" and codec is not None
                else None,
                channels=_known_positive(codec.channels) if stream.type == "audio" and codec is not None else None,
            )
        )
    if not streams:
        raise VideoFileFormatError("media container has no streams")
    return MediaProbe(formats, _seconds(container.duration, Fraction(1, av.time_base), duration=True), tuple(streams))


def probe_media(
    value: vane.File,
    *,
    max_bytes: int = DEFAULT_VIDEO_METADATA_BYTES,
    buffer_size: int = DEFAULT_VIDEO_METADATA_BUFFER_SIZE,
    connection: vane.DuckDBPyConnection | None = None,
) -> MediaProbe:
    """Inspect the streams of an encoded audio/video FILE using bounded reads.

    This is an eager Python value operation, also usable inside a public Vane
    UDF. It accepts File, AudioFile and VideoFile values, observes their logical
    byte ranges and does not perform model inference. Detection uses the bytes,
    not the filename or declared MIME type. Requires the ``video`` extra.

    Metadata probing may inspect codec headers under the same limits as video
    metadata. A malformed container, exhausted budget or I/O error is raised;
    incomplete results are never returned as an absence observation. Exceptions
    use the existing VideoFileFormatError/VideoFileLimitError hierarchy.
    """
    import vane

    if not isinstance(value, vane.File):
        raise TypeError("probe_media requires a File value")
    limit = _positive_limit(max_bytes, name="max_bytes", maximum=MAX_VIDEO_METADATA_BYTES)
    buffer = _positive_buffer_size(buffer_size, none_default=None, name="buffer_size")
    if buffer > 2**31 - 1:
        raise OverflowError("buffer_size must fit in the C int accepted by PyAV")
    _load_av()
    with value.open(buffer_size=1, connection=connection) as reader:

        def read_at(offset, size):
            reader.seek(offset)
            return reader.read(size)

        return _probe_container(read_at, reader.size(), limit, buffer, _inspect_media)


__all__ = ["MediaProbe", "MediaStream", "probe_media"]
