# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Encoded AUDIOFILE metadata, decoding, and resampling helpers."""

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
DEFAULT_AUDIO_MAX_OUTPUT_FRAMES = 100_000_000
DEFAULT_AUDIO_MAX_OUTPUT_BYTES = 512 * 1024 * 1024
MAX_AUDIO_METADATA_BYTES = 64 * 1024 * 1024
_AUDIO_RESAMPLE_CHUNK_BYTES = 1024 * 1024
# python-soxr owns an internal output buffer and returns a same-sized NumPy
# copy for every streaming call. Keep each allocation at or below 32 MiB; together
# with the ratio and channel caps this bounds native work between interruption
# checks without relying on libsoxr to reject hostile but technically valid
# rates itself.
_MAX_AUDIO_RESAMPLE_NATIVE_BUFFER_BYTES = 32 * 1024 * 1024
_MAX_AUDIO_RESAMPLE_RATIO = 64
_MAX_AUDIO_RESAMPLE_CHANNELS = 1024
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


@dataclass(frozen=True, slots=True)
class _AudioResampleSpool:
    """Own one bounded resample spool until its consumer closes it."""

    _stream: Any
    frames: int
    channels: int

    def readinto(self, target: Any) -> int:
        return self._stream.readinto(target)

    def close(self) -> None:
        self._stream.close()

    @property
    def closed(self) -> bool:
        return bool(self._stream.closed)


@contextmanager
def _close_preserving_primary(resource: Any) -> Iterator[Any]:
    """Close a resource without replacing an exception from its operation."""
    try:
        yield resource
    except BaseException:
        try:
            resource.close()
        except BaseException:
            pass
        raise
    else:
        resource.close()


@contextmanager
def _close_on_error(resource: Any) -> Iterator[Any]:
    """Transfer resource ownership on success and close it on failure."""
    try:
        yield resource
    except BaseException:
        try:
            resource.close()
        except BaseException:
            pass
        raise


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


class _AudioRandomAccessView:
    """Expose exact random-access callbacks as a seekable binary stream."""

    def __init__(self, read_at: Callable[[int, int], bytes], logical_size: int) -> None:
        self._read_at = read_at
        self._logical_size = logical_size
        self._position = 0

    def read(self, size: int | None = -1, /) -> bytes:
        if self._position >= self._logical_size or size == 0:
            return b""
        if size is None or size < 0:
            read_size = self._logical_size - self._position
        else:
            read_size = min(size, self._logical_size - self._position)
        data = self._read_at(self._position, read_size)
        if not isinstance(data, bytes) or len(data) != read_size:
            raise OSError(
                f"audio source returned {len(data) if isinstance(data, bytes) else 0} bytes "
                f"after requesting {read_size}"
            )
        self._position += read_size
        return data

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
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

    def tell(self) -> int:
        return self._position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


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


def _load_soxr() -> Any:
    try:
        soxr = importlib.import_module("soxr")
        soxr.ResampleStream
    except (AttributeError, ImportError, OSError) as error:
        raise ImportError(
            "AudioFile resampling requires the 'soxr' package and a usable libsoxr library. "
            "Please `pip install 'vane-ai[audio]'`."
        ) from error
    return soxr


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
        with _close_preserving_primary(soundfile.SoundFile(stream, mode="r", closefd=False)) as audio:
            stream.raise_if_error()
            metadata = _metadata_from_sound_file(audio, content_type=content_type)
            stream.raise_if_error()
    except BaseException as error:
        # Cancellation and interpreter-control exceptions must never be
        # rewritten as a media or resource-limit failure.
        if not isinstance(error, Exception):
            raise
        stream.raise_if_error()
        if isinstance(error, AudioFileError):
            raise
        if not isinstance(error, soundfile.SoundFileError):
            raise
        if stream.budget_exhausted:
            raise AudioFileLimitError(f"audio metadata requires more than max_bytes={max_bytes}") from error
        if stream.fetch_limit_exhausted:
            raise AudioFileLimitError(
                f"audio metadata requires more than {_MAX_AUDIO_METADATA_FETCHES} source ranges"
            ) from error
        raise AudioFileFormatError("logical FILE view is not a supported encoded audio file") from error
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
            with _close_preserving_primary(_AudioDecoder(proxy, mode="r", closefd=False)) as audio:
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
                    else:
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
                            if samples.size:
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


def _resample_audio_reader(
    reader: Any,
    *,
    logical_size: int,
    content_type: str | None,
    sample_rate: int,
    max_input_bytes: int,
    max_frames: int,
    max_decoded_bytes: int,
    max_output_frames: int,
    max_output_bytes: int,
    max_batch_output_bytes: int | None,
    check_interrupted: Callable[[], None] | None,
    soundfile: Any,
    soxr: Any,
    numpy: Any,
) -> _AudioResampleSpool:
    """Decode and stream-resample one logical FILE view into a bounded spool."""
    if check_interrupted is not None:
        check_interrupted()
    if logical_size > max_input_bytes:
        raise AudioFileLimitError(
            f"encoded audio contains {logical_size} bytes, exceeding max_input_bytes={max_input_bytes}"
        )

    class _AudioDecoder(soundfile.SoundFile):  # type: ignore[name-defined]
        def seekable(self) -> bool:
            return int(self.frames) != _MAX_BIGINT and super().seekable()

    proxy = _AudioReaderProxy(reader)
    output_file: Any | None = None
    spool: _AudioResampleSpool | None = None
    try:
        with _close_preserving_primary(_AudioDecoder(proxy, mode="r", closefd=False)) as audio:
            proxy.raise_if_error()
            metadata = _metadata_from_sound_file(audio, content_type=content_type)
            frame_bytes = metadata.channels * 8
            source_frame_limit: int | None = None
            if metadata.frames is not None:
                if metadata.frames > max_frames:
                    raise AudioFileLimitError(
                        f"audio contains {metadata.frames} frames, exceeding max_frames={max_frames}"
                    )
                decoded_bytes = metadata.frames * frame_bytes
                if decoded_bytes > max_decoded_bytes:
                    raise AudioFileLimitError(
                        f"audio decode requires {decoded_bytes} bytes, exceeding max_decoded_bytes={max_decoded_bytes}"
                    )
            else:
                if frame_bytes > max_decoded_bytes:
                    raise AudioFileLimitError(
                        f"one decoded audio frame requires {frame_bytes} bytes, "
                        f"exceeding max_decoded_bytes={max_decoded_bytes}"
                    )
                source_frame_limit = min(max_frames, max_decoded_bytes // frame_bytes)

            if sample_rate != metadata.sample_rate:
                if metadata.channels > _MAX_AUDIO_RESAMPLE_CHANNELS:
                    raise AudioFileLimitError(
                        f"audio resampling supports at most {_MAX_AUDIO_RESAMPLE_CHANNELS} channels, "
                        f"found {metadata.channels}"
                    )
                lower_rate = min(metadata.sample_rate, sample_rate)
                upper_rate = max(metadata.sample_rate, sample_rate)
                if upper_rate > lower_rate * _MAX_AUDIO_RESAMPLE_RATIO:
                    raise AudioFileLimitError(
                        f"audio resample ratio from {metadata.sample_rate} Hz to {sample_rate} Hz "
                        f"exceeds the safe {_MAX_AUDIO_RESAMPLE_RATIO}:1 limit"
                    )

            source_chunk_by_bytes = max(1, _AUDIO_RESAMPLE_CHUNK_BYTES // frame_bytes)
            target_chunk_by_bytes = max(1, _AUDIO_RESAMPLE_CHUNK_BYTES // frame_bytes)
            source_chunk_for_target = target_chunk_by_bytes * metadata.sample_rate // sample_rate
            if source_chunk_for_target == 0:
                raise AudioFileLimitError("one source frame exceeds the bounded audio resampler output chunk")
            chunk_frames = min(64 * 1024, source_chunk_by_bytes, source_chunk_for_target)
            decoded_frames = 0
            output_frames = 0
            resampler: Any | None = None

            output_file = tempfile.TemporaryFile(mode="w+b", buffering=0, prefix="vane_audio_resample_")
            with _close_on_error(output_file):

                def enforce_output_limits(frames: int) -> None:
                    if frames > max_output_frames:
                        raise AudioFileLimitError(
                            f"resampled audio contains more than max_output_frames={max_output_frames} frames"
                        )
                    output_bytes = frames * frame_bytes
                    if output_bytes > max_output_bytes:
                        raise AudioFileLimitError(
                            f"audio resample requires more than max_output_bytes={max_output_bytes}"
                        )
                    if max_batch_output_bytes is not None and output_bytes > max_batch_output_bytes:
                        raise AudioFileLimitError(
                            "audio_resample() exceeds the remaining "
                            f"per-batch output budget of {max_batch_output_bytes} bytes"
                        )

                def append_output(output: Any) -> None:
                    nonlocal output_frames
                    if (
                        not isinstance(output, numpy.ndarray)
                        or output.dtype != numpy.dtype(numpy.float64)
                        or output.ndim != 2
                        or output.shape[1] != metadata.channels
                        or not output.flags.c_contiguous
                    ):
                        raise RuntimeError("audio resampler returned a non-contiguous float64 (frames, channels) array")
                    next_output_frames = output_frames + int(output.shape[0])
                    enforce_output_limits(next_output_frames)

                    if output.size:
                        output_view = memoryview(output).cast("B")
                        while output_view:
                            written = output_file.write(output_view)
                            if written is None or written <= 0:
                                raise OSError("temporary audio resample spool made no write progress")
                            output_view = output_view[written:]
                    output_frames = next_output_frames
                    if check_interrupted is not None:
                        check_interrupted()

                def resampler_delay_frames() -> int:
                    assert resampler is not None
                    if check_interrupted is not None:
                        check_interrupted()
                    delay = float(resampler.delay())
                    if not math.isfinite(delay) or delay < 0:
                        raise RuntimeError(f"audio resampler returned an invalid delay: {delay!r}")
                    return math.ceil(delay)

                def resample_chunk_bounded(decoded_array: Any, *, last: bool) -> None:
                    assert resampler is not None
                    delay_frames = resampler_delay_frames()
                    input_frames = int(decoded_array.shape[0])
                    produced_frame_bound = (
                        input_frames * sample_rate + metadata.sample_rate - 1
                    ) // metadata.sample_rate
                    required_frames = delay_frames + produced_frame_bound + 1
                    if required_frames * frame_bytes > _MAX_AUDIO_RESAMPLE_NATIVE_BUFFER_BYTES:
                        raise AudioFileLimitError(
                            "audio resampler requires more than the fixed "
                            f"{_MAX_AUDIO_RESAMPLE_NATIVE_BUFFER_BYTES}-byte native output-buffer limit"
                        )
                    if check_interrupted is not None:
                        check_interrupted()
                    append_output(resampler.resample_chunk(decoded_array, last=last))

                def process_input(decoded: bytearray, frames: int) -> None:
                    nonlocal resampler
                    if frames == 0:
                        return
                    minimum_output_frames = decoded_frames * sample_rate // metadata.sample_rate
                    enforce_output_limits(minimum_output_frames)

                    decoded_array = numpy.frombuffer(
                        decoded,
                        dtype=numpy.float64,
                        count=frames * metadata.channels,
                    ).reshape(frames, metadata.channels)
                    if sample_rate == metadata.sample_rate:
                        append_output(decoded_array)
                        return
                    if resampler is None:
                        if check_interrupted is not None:
                            check_interrupted()
                        try:
                            resampler = soxr.ResampleStream(
                                metadata.sample_rate,
                                sample_rate,
                                metadata.channels,
                                dtype="float64",
                                quality="HQ",
                            )
                        except ValueError as error:
                            raise AudioFileFormatError(
                                f"audio cannot be resampled from {metadata.sample_rate} Hz to {sample_rate} Hz "
                                f"with {metadata.channels} channels"
                            ) from error

                    processed_frames = 0
                    native_frame_capacity = _MAX_AUDIO_RESAMPLE_NATIVE_BUFFER_BYTES // frame_bytes
                    while processed_frames < frames:
                        delay_frames = resampler_delay_frames()
                        available_output_frames = native_frame_capacity - delay_frames - 1
                        native_input_capacity = available_output_frames * metadata.sample_rate // sample_rate
                        if native_input_capacity <= 0:
                            raise AudioFileLimitError(
                                "audio resampler requires more than the fixed "
                                f"{_MAX_AUDIO_RESAMPLE_NATIVE_BUFFER_BYTES}-byte native output-buffer limit"
                            )
                        call_frames = min(frames - processed_frames, native_input_capacity)
                        resample_chunk_bounded(
                            decoded_array[processed_frames : processed_frames + call_frames],
                            last=False,
                        )
                        processed_frames += call_frames

                if metadata.frames is not None:
                    while decoded_frames < metadata.frames:
                        requested_frames = min(chunk_frames, metadata.frames - decoded_frames)
                        decoded = bytearray(requested_frames * frame_bytes)
                        returned_frames = audio.buffer_read_into(decoded, dtype="float64")
                        proxy.raise_if_error()
                        if returned_frames < 0 or returned_frames > requested_frames:
                            raise AudioFileFormatError("audio decoder returned an invalid frame count")
                        decoded_frames += returned_frames
                        process_input(decoded, returned_frames)
                        del decoded
                        if returned_frames != requested_frames:
                            raise AudioFileFormatError(
                                f"audio decoder returned {decoded_frames} frames after reporting {metadata.frames}"
                            )
                else:
                    assert source_frame_limit is not None
                    while decoded_frames < source_frame_limit:
                        requested_frames = min(chunk_frames, source_frame_limit - decoded_frames)
                        decoded = bytearray(requested_frames * frame_bytes)
                        returned_frames = audio.buffer_read_into(decoded, dtype="float64")
                        proxy.raise_if_error()
                        if returned_frames < 0 or returned_frames > requested_frames:
                            raise AudioFileFormatError("audio decoder returned an invalid frame count")
                        decoded_frames += returned_frames
                        process_input(decoded, returned_frames)
                        del decoded
                        if returned_frames < requested_frames:
                            break

                    if decoded_frames == source_frame_limit:
                        extra = bytearray(frame_bytes)
                        extra_frames = audio.buffer_read_into(extra, dtype="float64")
                        proxy.raise_if_error()
                        if extra_frames < 0 or extra_frames > 1:
                            raise AudioFileFormatError("audio decoder returned an invalid frame count")
                        if extra_frames:
                            if max_frames <= max_decoded_bytes // frame_bytes:
                                raise AudioFileLimitError(f"audio contains more than max_frames={max_frames} frames")
                            raise AudioFileLimitError(
                                f"audio decode requires more than max_decoded_bytes={max_decoded_bytes}"
                            )

                if resampler is not None:
                    empty = numpy.empty((0, metadata.channels), dtype=numpy.float64)
                    resample_chunk_bounded(empty, last=True)
                proxy.raise_if_error()
                if check_interrupted is not None:
                    check_interrupted()

                output_file.seek(0)
                spool = _AudioResampleSpool(output_file, output_frames, metadata.channels)
    except BaseException as error:
        if output_file is not None and not output_file.closed:
            try:
                output_file.close()
            except BaseException:
                pass
        if not isinstance(error, Exception):
            raise
        proxy.raise_if_error()
        if isinstance(error, AudioFileError):
            raise
        if isinstance(error, soundfile.SoundFileError):
            raise AudioFileFormatError("logical FILE view is not a supported encoded audio file") from error
        raise

    if spool is None:
        raise RuntimeError("audio resample did not produce a spool")
    return spool


def _materialize_audio_spool(
    spool: _AudioResampleSpool,
    *,
    numpy: Any,
    check_interrupted: Callable[[], None] | None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Materialize an owned spool once for the Python value API."""
    with _close_preserving_primary(spool):
        samples = numpy.empty((spool.frames, spool.channels), dtype=numpy.float64)
        if samples.size:
            output_view = memoryview(samples).cast("B")
            output_offset = 0
            while output_offset < len(output_view):
                read_end = min(len(output_view), output_offset + _AUDIO_RESAMPLE_CHUNK_BYTES)
                read_count = spool.readinto(output_view[output_offset:read_end])
                if read_count is None or read_count <= 0:
                    raise OSError("temporary audio resample spool ended before the decoded output")
                output_offset += read_count
                if check_interrupted is not None:
                    check_interrupted()
            del output_view
        return samples


def _resample_audio_file(
    value: vane.AudioFile,
    sample_rate: int,
    buffer_size: int = DEFAULT_AUDIO_BUFFER_SIZE,
    *,
    max_input_bytes: int = DEFAULT_AUDIO_MAX_INPUT_BYTES,
    max_frames: int = DEFAULT_AUDIO_MAX_FRAMES,
    max_decoded_bytes: int = DEFAULT_AUDIO_MAX_DECODED_BYTES,
    max_output_frames: int = DEFAULT_AUDIO_MAX_OUTPUT_FRAMES,
    max_output_bytes: int = DEFAULT_AUDIO_MAX_OUTPUT_BYTES,
    connection: vane.DuckDBPyConnection | None = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    normalized_sample_rate = _positive_limit(sample_rate, name="sample_rate", maximum=_MAX_BIGINT)
    normalized_buffer_size = _positive_buffer_size(buffer_size, none_default=None, name="buffer_size")
    normalized_max_input = _positive_limit(max_input_bytes, name="max_input_bytes", maximum=_MAX_UBIGINT)
    normalized_max_frames = _positive_limit(max_frames, name="max_frames", maximum=_MAX_BIGINT)
    normalized_max_decoded = _positive_limit(max_decoded_bytes, name="max_decoded_bytes", maximum=_MAX_UBIGINT)
    normalized_max_output_frames = _positive_limit(
        max_output_frames,
        name="max_output_frames",
        maximum=_MAX_BIGINT,
    )
    normalized_max_output = _positive_limit(max_output_bytes, name="max_output_bytes", maximum=_MAX_UBIGINT)
    soundfile = _load_soundfile()
    soxr = _load_soxr()
    numpy = importlib.import_module("numpy")

    with value.open(buffer_size=normalized_buffer_size, connection=connection) as reader:
        spool = _resample_audio_reader(
            reader,
            logical_size=reader.size(),
            content_type=value.content_type,
            sample_rate=normalized_sample_rate,
            max_input_bytes=normalized_max_input,
            max_frames=normalized_max_frames,
            max_decoded_bytes=normalized_max_decoded,
            max_output_frames=normalized_max_output_frames,
            max_output_bytes=normalized_max_output,
            max_batch_output_bytes=None,
            check_interrupted=reader._check_interrupted,
            soundfile=soundfile,
            soxr=soxr,
            numpy=numpy,
        )
        return _materialize_audio_spool(
            spool,
            numpy=numpy,
            check_interrupted=reader._check_interrupted,
        )


def _resample_audio_stream(
    read_at: Callable[[int, int], bytes],
    logical_size: int,
    content_type: str | None,
    sample_rate: int,
    max_input_bytes: int,
    max_frames: int,
    max_decoded_bytes: int,
    max_output_frames: int,
    max_output_bytes: int,
    max_batch_output_bytes: int,
    check_interrupted: Callable[[], None],
) -> _AudioResampleSpool:
    """Native SQL callback entry point over the executing ClientContext."""
    return _resample_audio_reader(
        _AudioRandomAccessView(read_at, logical_size),
        logical_size=logical_size,
        content_type=content_type,
        sample_rate=sample_rate,
        max_input_bytes=max_input_bytes,
        max_frames=max_frames,
        max_decoded_bytes=max_decoded_bytes,
        max_output_frames=max_output_frames,
        max_output_bytes=max_output_bytes,
        max_batch_output_bytes=max_batch_output_bytes,
        check_interrupted=check_interrupted,
        soundfile=_load_soundfile(),
        soxr=_load_soxr(),
        numpy=importlib.import_module("numpy"),
    )


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


def audio_resample(
    value: vane.AudioFile | vane.Expression,
    sample_rate: int | vane.Expression,
    *,
    max_input_bytes: int | vane.Expression = DEFAULT_AUDIO_MAX_INPUT_BYTES,
    max_frames: int | vane.Expression = DEFAULT_AUDIO_MAX_FRAMES,
    max_decoded_bytes: int | vane.Expression = DEFAULT_AUDIO_MAX_DECODED_BYTES,
    max_output_frames: int | vane.Expression = DEFAULT_AUDIO_MAX_OUTPUT_FRAMES,
    max_output_bytes: int | vane.Expression = DEFAULT_AUDIO_MAX_OUTPUT_BYTES,
) -> vane.Expression:
    """Build a bounded SoXR HQ AUDIOFILE resampling expression.

    The result STRUCT stores float64 ``samples`` in frame-major order together
    with its ``sample_rate``, ``frames``, and ``channels`` dimensions.
    SQL execution also caps flattened sample storage at 512 MiB per vector batch.
    Ratios above 64:1, non-identity inputs above 1024 channels, and native calls
    that would exceed the fixed SoXR output-buffer bound are rejected before
    native work.
    """
    return vane.FunctionExpression(
        "audio_resample",
        as_expression(value),
        _limit_expression(sample_rate, name="sample_rate", maximum=_MAX_BIGINT),
        _limit_expression(max_input_bytes, name="max_input_bytes", maximum=_MAX_UBIGINT),
        _limit_expression(max_frames, name="max_frames", maximum=_MAX_BIGINT),
        _limit_expression(max_decoded_bytes, name="max_decoded_bytes", maximum=_MAX_UBIGINT),
        _limit_expression(max_output_frames, name="max_output_frames", maximum=_MAX_BIGINT),
        _limit_expression(max_output_bytes, name="max_output_bytes", maximum=_MAX_UBIGINT),
    )


__all__ = [
    "AudioFileError",
    "AudioFileFormatError",
    "AudioFileLimitError",
    "AudioMetadata",
    "audio_metadata",
    "audio_resample",
]
