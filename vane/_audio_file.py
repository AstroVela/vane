# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Encoded AUDIOFILE metadata and Python waveform decoding helpers."""

from __future__ import annotations

import importlib
import io
import math
import tempfile
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from email.message import EmailMessage
from email.policy import default as default_email_policy
from typing import TYPE_CHECKING, Any

import vane
from vane._expressions import as_expression
from vane._file import _positive_buffer_size

if TYPE_CHECKING:
    import numpy as np


DEFAULT_AUDIO_METADATA_BYTES = 8 * 1024 * 1024
DEFAULT_AUDIO_BUFFER_SIZE = 1024 * 1024
DEFAULT_AUDIO_MAX_INPUT_BYTES = 512 * 1024 * 1024
DEFAULT_AUDIO_MAX_FRAMES = 100_000_000
DEFAULT_AUDIO_MAX_DECODED_BYTES = 512 * 1024 * 1024
MAX_AUDIO_METADATA_BYTES = 64 * 1024 * 1024
_MAX_AUDIO_METADATA_FETCH_BYTES = 64 * 1024
_AUDIO_METADATA_FETCHES_PER_BUDGET = 8
_MAX_AUDIO_METADATA_FETCHES = 1024
_MAX_BIGINT = (1 << 63) - 1
_MAX_UBIGINT = (1 << 64) - 1

_MIME_ALIASES = {
    "application/ogg": "audio/ogg",
    "audio/aif": "audio/aiff",
    "audio/au": "audio/x-au",
    "audio/mp3": "audio/mpeg",
    "audio/vnd.sun.audio": "audio/x-au",
    "audio/vnd.wave": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/x-aiff": "audio/aiff",
    "audio/x-flac": "audio/flac",
    "audio/x-mp3": "audio/mpeg",
    "audio/x-wav": "audio/wav",
}
_GENERIC_MIME_TYPES = frozenset({"application/octet-stream", "audio/*", "binary/octet-stream"})
_FORMAT_MIME_TYPES = {
    "AIFF": "audio/aiff",
    "AU": "audio/x-au",
    "CAF": "audio/x-caf",
    "FLAC": "audio/flac",
    "MP3": "audio/mpeg",
    "NIST": "audio/x-nist",
    "OGG": "audio/ogg",
    "RF64": "audio/wav",
    "SVX": "audio/x-iff",
    "VOC": "audio/x-voc",
    "W64": "audio/x-w64",
    "WAV": "audio/wav",
    "WAVEX": "audio/wav",
}
_FORMAT_SUBTYPE_CODECS: dict[tuple[str, str | None], frozenset[str]] = {
    ("OGG", "OPUS"): frozenset({"opus"}),
    ("OGG", "VORBIS"): frozenset({"vorbis"}),
}
_WAVE_SUBTYPE_CODECS = {
    "ALAW": "6",
    "DOUBLE": "3",
    "FLOAT": "3",
    "GSM610": "31",
    "G721_32": "40",
    "IMA_ADPCM": "11",
    "MPEG_LAYER_III": "55",
    "MS_ADPCM": "2",
    "NMS_ADPCM_16": "38",
    "NMS_ADPCM_24": "38",
    "NMS_ADPCM_32": "38",
    "PCM_16": "1",
    "PCM_24": "1",
    "PCM_32": "1",
    "PCM_U8": "1",
    "ULAW": "7",
}


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Decoder-reported metadata for an encoded :class:`vane.AudioFile`."""

    sample_rate: int
    channels: int
    frames: int | None
    duration: float | None
    format: str
    subtype: str | None


class AudioFileError(RuntimeError):
    """Base class for failures caused by encoded audio content."""


class AudioFileFormatError(AudioFileError):
    """The logical FILE view could not be opened or decoded as supported audio."""


class AudioFileLimitError(AudioFileError):
    """An AudioFile operation exceeded an explicit resource limit."""


@contextmanager
def _close_sound_file(audio: Any) -> Iterator[Any]:
    """Close a decoder without replacing an exception from its operation."""
    try:
        yield audio
    except BaseException:
        try:
            audio.close()
        except BaseException:
            pass
        raise
    else:
        audio.close()


class _AudioMetadataView:
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
            _MAX_AUDIO_METADATA_FETCH_BYTES,
            max(1, max_bytes // _AUDIO_METADATA_FETCHES_PER_BUDGET),
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
                raise RuntimeError("audio metadata cache received overlapping ranges")
        if insert_at < len(self._cache) and start + len(data) > self._cache[insert_at][0]:
            raise RuntimeError("audio metadata cache received overlapping ranges")
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
            if self._fetches >= _MAX_AUDIO_METADATA_FETCHES:
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
                    f"audio metadata source returned {len(fetched) if isinstance(fetched, bytes) else 0} bytes "
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
        if self._error is not None:
            return -1
        try:
            return self._position
        except BaseException as error:
            self._remember(error)
            return -1

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error.with_traceback(self._error.__traceback__)


class _AudioReaderProxy:
    """Keep reader failures from being swallowed by CFFI virtual-I/O callbacks."""

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
            return self._reader.read(size)
        except BaseException as error:
            self._remember(error)
            return b""

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
        if self._error is not None:
            return -1
        try:
            return self._reader.seek(offset, whence)
        except BaseException as error:
            self._remember(error)
            return -1

    def tell(self) -> int:
        if self._error is not None:
            return -1
        try:
            return self._reader.tell()
        except BaseException as error:
            self._remember(error)
            return -1

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error.with_traceback(self._error.__traceback__)


def _load_soundfile() -> Any:
    try:
        soundfile = importlib.import_module("soundfile")
        soundfile.SoundFile
        soundfile.SoundFileError
    except (AttributeError, ImportError, OSError) as error:
        raise ImportError(
            "AudioFile metadata and decoding require the 'soundfile' package and a usable libsndfile library. "
            "Please `pip install 'vane-ai[audio]'`."
        ) from error
    return soundfile


def _positive_limit(value: object, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, not {type(value).__name__!r}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _limit_expression(value: int | vane.Expression, *, name: str, maximum: int | None = None) -> vane.Expression:
    if isinstance(value, vane.Expression):
        return value
    return as_expression(_positive_limit(value, name=name, maximum=maximum))


def _canonical_mime_type(value: str) -> str:
    normalized = value.strip().lower()
    return _MIME_ALIASES.get(normalized, normalized)


def _parse_content_type(value: str) -> tuple[str, str | None, frozenset[str] | None]:
    message = EmailMessage(policy=default_email_policy)
    try:
        message["Content-Type"] = value
        header = message["Content-Type"]
        if header is None or getattr(header, "defects", False):
            raise ValueError("invalid Content-Type syntax")
        parameters = message.get_params(header="content-type", unquote=True)
    except (TypeError, ValueError) as error:
        raise AudioFileFormatError(f"AUDIOFILE content_type {value!r} is invalid") from error
    if not parameters:
        raise AudioFileFormatError(f"AUDIOFILE content_type {value!r} is invalid")

    declared = _canonical_mime_type(message.get_content_type())
    codec_values = [parameter for name, parameter in parameters[1:] if name.lower() == "codec"]
    if len(codec_values) > 1 or (codec_values and not isinstance(codec_values[0], str)):
        raise AudioFileFormatError(f"AUDIOFILE content_type {value!r} has an invalid codec parameter")
    declared_codec = codec_values[0].strip().lower() if codec_values else None
    if declared_codec == "" or (declared_codec is not None and "," in declared_codec):
        raise AudioFileFormatError(f"AUDIOFILE content_type {value!r} has an invalid codec parameter")

    codec_values = [parameter for name, parameter in parameters[1:] if name.lower() == "codecs"]
    if not codec_values:
        return declared, declared_codec, None

    codecs: set[str] = set()
    for codec_value in codec_values:
        if not isinstance(codec_value, str):
            raise AudioFileFormatError(f"AUDIOFILE content_type {value!r} has an invalid codecs parameter")
        parsed = [codec.strip().lower() for codec in codec_value.split(",")]
        if not parsed or any(not codec for codec in parsed):
            raise AudioFileFormatError(f"AUDIOFILE content_type {value!r} has an invalid codecs parameter")
        codecs.update(parsed)
    if declared_codec is not None:
        raise AudioFileFormatError(f"AUDIOFILE content_type {value!r} cannot declare both codec and codecs")
    return declared, None, frozenset(codecs)


def _detected_audio_mime_type(audio_format: str) -> str | None:
    return _FORMAT_MIME_TYPES.get(audio_format)


def _validate_content_type(
    content_type: str | None,
    audio_format: str,
    subtype: str | None,
    detected_mime_type: str | None,
) -> None:
    if content_type is None:
        return
    declared, declared_codec, declared_codecs = _parse_content_type(content_type)
    is_generic = declared in _GENERIC_MIME_TYPES
    if not is_generic and not declared.startswith("audio/"):
        raise AudioFileFormatError(f"AUDIOFILE content_type {content_type!r} contradicts the decoded audio content")
    if not is_generic:
        if detected_mime_type is None:
            raise AudioFileFormatError(
                f"AUDIOFILE content_type {content_type!r} cannot be validated against detected audio format "
                f"{audio_format!r}"
            )
        detected = _canonical_mime_type(detected_mime_type)
        if declared != detected:
            raise AudioFileFormatError(
                f"AUDIOFILE content_type {content_type!r} contradicts detected MIME type {detected_mime_type!r}"
            )
    if declared_codec is not None:
        detected_codec = (
            _WAVE_SUBTYPE_CODECS.get(subtype)
            if audio_format in {"RF64", "WAV", "WAVEX"} and subtype is not None
            else None
        )
        if declared != "audio/wav" or detected_codec is None:
            raise AudioFileFormatError(
                f"AUDIOFILE content_type {content_type!r} codec parameter cannot be validated against "
                f"detected audio format {audio_format!r}"
            )
        if declared_codec != detected_codec:
            raise AudioFileFormatError(
                f"AUDIOFILE content_type {content_type!r} contradicts detected audio codec {subtype!r}"
            )
    if declared_codecs is not None:
        detected_codecs = _FORMAT_SUBTYPE_CODECS.get((audio_format, subtype))
        if declared_codecs != detected_codecs:
            raise AudioFileFormatError(
                f"AUDIOFILE content_type {content_type!r} contradicts detected audio codec {subtype!r}"
            )


def _metadata_from_sound_file(audio: Any, *, content_type: str | None) -> AudioMetadata:
    sample_rate = int(audio.samplerate)
    channels = int(audio.channels)
    reported_frames = int(audio.frames)
    if sample_rate <= 0 or sample_rate > _MAX_BIGINT:
        raise AudioFileFormatError(f"audio sample rate must be between 1 and {_MAX_BIGINT}, found {sample_rate}")
    if channels <= 0 or channels > _MAX_BIGINT:
        raise AudioFileFormatError(f"audio channel count must be between 1 and {_MAX_BIGINT}, found {channels}")
    if reported_frames < 0 or reported_frames > _MAX_BIGINT:
        raise AudioFileFormatError(f"audio frame count must be between 0 and {_MAX_BIGINT}, found {reported_frames}")
    frames = None if reported_frames == _MAX_BIGINT else reported_frames

    audio_format = audio.format
    if not isinstance(audio_format, str) or not audio_format:
        raise AudioFileFormatError("audio decoder did not report an encoded format")
    audio_format = audio_format.upper()
    subtype = audio.subtype
    if subtype is not None:
        if not isinstance(subtype, str):
            raise AudioFileFormatError("audio decoder reported an invalid subtype")
        subtype = subtype.upper() or None

    detected_mime_type = _detected_audio_mime_type(audio_format)
    _validate_content_type(
        content_type,
        audio_format,
        subtype,
        detected_mime_type,
    )
    duration = None if frames is None else frames / sample_rate
    if duration is not None and (not math.isfinite(duration) or duration < 0):
        raise AudioFileFormatError("audio decoder reported an invalid duration")
    return AudioMetadata(sample_rate, channels, frames, duration, audio_format, subtype)


def _probe_audio_metadata(
    read_at: Callable[[int, int], bytes],
    logical_size: int,
    content_type: str | None,
    max_bytes: int,
) -> tuple[int, int, int | None, float | None, str, str | None]:
    """Bounded helper called by the native SQL scalar function."""
    soundfile = _load_soundfile()
    stream = _AudioMetadataView(read_at, logical_size=logical_size, max_bytes=max_bytes)
    try:
        # libsndfile owns container parsing. Vane only supplies a bounded,
        # range-aware logical FILE view and classifies the library's result.
        with _close_sound_file(soundfile.SoundFile(stream, mode="r", closefd=False)) as audio:
            stream.raise_if_error()
            metadata = _metadata_from_sound_file(audio, content_type=content_type)
            stream.raise_if_error()
    except BaseException as error:
        # Cancellation and interpreter-control exceptions must never be
        # rewritten as a media or resource-limit failure.
        if not isinstance(error, Exception):
            raise
        stream.raise_if_error()
        if stream.budget_exhausted:
            raise AudioFileLimitError(f"audio metadata requires more than max_bytes={max_bytes}") from error
        if stream.fetch_limit_exhausted:
            raise AudioFileLimitError(
                f"audio metadata requires more than {_MAX_AUDIO_METADATA_FETCHES} source ranges"
            ) from error
        if isinstance(error, AudioFileError):
            raise
        if isinstance(error, soundfile.SoundFileError):
            raise AudioFileFormatError("logical FILE view is not a supported encoded audio file") from error
        raise
    if stream.budget_exhausted:
        raise AudioFileLimitError(f"audio metadata requires more than max_bytes={max_bytes}")
    if stream.fetch_limit_exhausted:
        raise AudioFileLimitError(f"audio metadata requires more than {_MAX_AUDIO_METADATA_FETCHES} source ranges")
    return (
        metadata.sample_rate,
        metadata.channels,
        metadata.frames,
        metadata.duration,
        metadata.format,
        metadata.subtype,
    )


def _audio_file_metadata_value(
    value: vane.AudioFile,
    *,
    max_bytes: int = DEFAULT_AUDIO_METADATA_BYTES,
    connection: vane.DuckDBPyConnection | None = None,
) -> AudioMetadata:
    normalized_max_bytes = _positive_limit(max_bytes, name="max_bytes", maximum=MAX_AUDIO_METADATA_BYTES)
    _load_soundfile()
    with value.open(buffer_size=1, connection=connection) as reader:
        logical_size = reader.size()

        def read_at(offset: int, size: int) -> bytes:
            reader.seek(offset)
            return reader.read(size)

        fields = _probe_audio_metadata(read_at, logical_size, value.content_type, normalized_max_bytes)
    return AudioMetadata(*fields)


def _decode_audio_file(
    value: vane.AudioFile,
    buffer_size: int = DEFAULT_AUDIO_BUFFER_SIZE,
    *,
    max_input_bytes: int = DEFAULT_AUDIO_MAX_INPUT_BYTES,
    max_frames: int = DEFAULT_AUDIO_MAX_FRAMES,
    max_decoded_bytes: int = DEFAULT_AUDIO_MAX_DECODED_BYTES,
    connection: vane.DuckDBPyConnection | None = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    normalized_buffer_size = _positive_buffer_size(buffer_size, none_default=None, name="buffer_size")
    normalized_max_input = _positive_limit(max_input_bytes, name="max_input_bytes", maximum=_MAX_UBIGINT)
    normalized_max_frames = _positive_limit(max_frames, name="max_frames", maximum=_MAX_BIGINT)
    normalized_max_decoded = _positive_limit(max_decoded_bytes, name="max_decoded_bytes", maximum=_MAX_UBIGINT)
    soundfile = _load_soundfile()
    numpy = importlib.import_module("numpy")

    class _AudioDecoder(soundfile.SoundFile):  # type: ignore[name-defined]
        def seekable(self) -> bool:
            # libsndfile reports SF_COUNT_MAX when a container omits its
            # total frame count. The encoded stream can still seek while
            # probing, but frame-based seeks are unavailable. SoundFile's
            # read wrapper otherwise performs such a seek after every read.
            return int(self.frames) != _MAX_BIGINT and super().seekable()

    with value.open(buffer_size=normalized_buffer_size, connection=connection) as reader:
        input_size = reader.size()
        if input_size > normalized_max_input:
            raise AudioFileLimitError(
                f"encoded audio contains {input_size} bytes, exceeding max_input_bytes={normalized_max_input}"
            )

        proxy = _AudioReaderProxy(reader)
        try:
            with _close_sound_file(_AudioDecoder(proxy, mode="r", closefd=False)) as audio:
                proxy.raise_if_error()
                metadata = _metadata_from_sound_file(audio, content_type=value.content_type)
                if metadata.frames is not None and metadata.frames > normalized_max_frames:
                    raise AudioFileLimitError(
                        f"audio contains {metadata.frames} frames, exceeding max_frames={normalized_max_frames}"
                    )
                if metadata.frames is None:
                    frame_bytes = metadata.channels * 8
                    decoded_frame_limit = normalized_max_decoded // frame_bytes
                    frame_limit = min(normalized_max_frames, decoded_frame_limit)
                    if frame_limit == 0:
                        raise AudioFileLimitError(
                            f"one decoded audio frame requires {frame_bytes} bytes, "
                            f"exceeding max_decoded_bytes={normalized_max_decoded}"
                        )

                    # Unknown-length streams cannot accumulate both a decoded
                    # output and a same-sized scratch buffer without violating
                    # max_decoded_bytes. Spool bounded chunks, release the
                    # scratch allocation, then materialize the exact ndarray
                    # directly from the temporary file.
                    with tempfile.TemporaryFile(mode="w+b", buffering=0, prefix="vane_audio_") as decoded_file:
                        decoded_frames = 0
                        while decoded_frames < frame_limit:
                            requested_frames = min(64 * 1024, frame_limit - decoded_frames)
                            chunk = bytearray(requested_frames * frame_bytes)
                            chunk_frames = audio.buffer_read_into(chunk, dtype="float64")
                            proxy.raise_if_error()
                            if chunk_frames < 0 or chunk_frames > requested_frames:
                                raise AudioFileFormatError("audio decoder returned an invalid frame count")
                            chunk_bytes = chunk_frames * frame_bytes
                            chunk_view = memoryview(chunk)[:chunk_bytes]
                            while chunk_view:
                                written = decoded_file.write(chunk_view)
                                if written is None or written <= 0:
                                    raise OSError("temporary audio spool made no write progress")
                                chunk_view = chunk_view[written:]
                            decoded_frames += chunk_frames
                            reached_end = chunk_frames < requested_frames
                            del chunk_view
                            del chunk
                            if reached_end:
                                break

                        if decoded_frames == frame_limit:
                            extra = bytearray(frame_bytes)
                            extra_frames = audio.buffer_read_into(extra, dtype="float64")
                            proxy.raise_if_error()
                            del extra
                            if extra_frames < 0 or extra_frames > 1:
                                raise AudioFileFormatError("audio decoder returned an invalid frame count")
                            if extra_frames:
                                if normalized_max_frames <= decoded_frame_limit:
                                    raise AudioFileLimitError(
                                        f"audio contains more than max_frames={normalized_max_frames} frames"
                                    )
                                raise AudioFileLimitError(
                                    f"audio decode requires more than max_decoded_bytes={normalized_max_decoded}"
                                )

                        samples = numpy.empty((decoded_frames, metadata.channels), dtype=numpy.float64)
                        output_view = memoryview(samples).cast("B")
                        decoded_file.seek(0)
                        output_offset = 0
                        while output_offset < len(output_view):
                            read_count = decoded_file.readinto(output_view[output_offset:])
                            if read_count is None or read_count <= 0:
                                raise OSError("temporary audio spool ended before the decoded output")
                            output_offset += read_count
                        del output_view
                else:
                    frame_bytes = metadata.channels * 8
                    if frame_bytes > normalized_max_decoded:
                        raise AudioFileLimitError(
                            f"one decoded audio frame requires {frame_bytes} bytes, "
                            f"exceeding max_decoded_bytes={normalized_max_decoded}"
                        )
                    decoded_bytes = metadata.frames * metadata.channels * 8
                    if decoded_bytes > normalized_max_decoded:
                        raise AudioFileLimitError(
                            f"audio decode requires {decoded_bytes} bytes, "
                            f"exceeding max_decoded_bytes={normalized_max_decoded}"
                        )
                    if metadata.frames == 0:
                        samples = numpy.empty((0, metadata.channels), dtype=numpy.float64)
                    else:
                        decoded = bytearray(decoded_bytes)
                        decoded_frames = audio.buffer_read_into(decoded, dtype="float64")
                        proxy.raise_if_error()
                        if decoded_frames != metadata.frames:
                            raise AudioFileFormatError(
                                f"audio decoder returned {decoded_frames} frames after reporting {metadata.frames}"
                            )
                        samples = numpy.frombuffer(decoded, dtype=numpy.float64).reshape(
                            metadata.frames,
                            metadata.channels,
                        )
                    extra = bytearray(frame_bytes)
                    extra_frames = audio.buffer_read_into(extra, dtype="float64")
                    proxy.raise_if_error()
                    del extra
                    if extra_frames < 0 or extra_frames > 1:
                        raise AudioFileFormatError("audio decoder returned an invalid frame count")
                    if extra_frames:
                        raise AudioFileFormatError(
                            f"audio decoder produced more frames after reporting {metadata.frames}"
                        )
                proxy.raise_if_error()
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            proxy.raise_if_error()
            if isinstance(error, AudioFileError):
                raise
            if isinstance(error, soundfile.SoundFileError):
                raise AudioFileFormatError("logical FILE view is not a supported encoded audio file") from error
            raise

    # The backing bytearray (when present) is owned by the returned array and
    # is independent of the reader, which has already been closed.
    return samples


def audio_metadata(
    value: vane.AudioFile | vane.Expression,
    *,
    max_bytes: int | vane.Expression = DEFAULT_AUDIO_METADATA_BYTES,
) -> vane.Expression:
    """Inspect bounded encoded AUDIOFILE metadata without decoding samples."""
    return vane.FunctionExpression(
        "audio_metadata",
        as_expression(value),
        _limit_expression(max_bytes, name="max_bytes", maximum=MAX_AUDIO_METADATA_BYTES),
    )


__all__ = [
    "AudioFileError",
    "AudioFileFormatError",
    "AudioFileLimitError",
    "AudioMetadata",
    "audio_metadata",
]
