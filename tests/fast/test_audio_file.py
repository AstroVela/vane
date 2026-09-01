# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import io

import numpy as np
import pytest
import soundfile

import vane
from vane import _audio_file


def _encoded_audio(
    audio_format: str = "WAV",
    subtype: str = "PCM_16",
    *,
    sample_rate: int = 8000,
    frames: int = 32,
    channels: int = 2,
) -> tuple[bytes, np.ndarray]:
    samples = np.linspace(-0.75, 0.75, frames * channels, dtype=np.float64).reshape(frames, channels)
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, sample_rate, format=audio_format, subtype=subtype)
    return buffer.getvalue(), samples


def _flac_with_total_samples(payload: bytes, total_samples: int) -> bytes:
    assert payload[:4] == b"fLaC"
    assert payload[4] & 0x7F == 0
    assert int.from_bytes(payload[5:8], "big") == 34
    assert 0 <= total_samples < 1 << 36
    result = bytearray(payload)
    stream_info = int.from_bytes(result[18:26], "big")
    result[18:26] = ((stream_info & ~((1 << 36) - 1)) | total_samples).to_bytes(8, "big")
    return bytes(result)


def _flac_with_unknown_total_samples(payload: bytes) -> bytes:
    return _flac_with_total_samples(payload, 0)


@pytest.mark.parametrize(
    ("audio_format", "subtype", "content_type"),
    [
        ("WAV", "PCM_16", "audio/wav"),
        ("FLAC", "PCM_16", "audio/flac"),
        ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
        ("OGG", "VORBIS", "audio/ogg"),
        ("OGG", "OPUS", "audio/ogg; codecs=opus"),
    ],
)
def test_audio_metadata_sql_and_python_value(duckdb_cursor, tmp_path, audio_format, subtype, content_type):
    payload, _ = _encoded_audio(audio_format, subtype, sample_rate=16000, frames=40, channels=2)
    path = tmp_path / f"audio.{audio_format.lower()}"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), content_type)

    result_type, metadata, null_metadata = duckdb_cursor.execute(
        """
        SELECT
            typeof(audio_metadata($1)),
            audio_metadata($1),
            audio_metadata(NULL::AUDIOFILE)
        """,
        [value],
    ).fetchone()

    assert result_type == (
        "STRUCT(sample_rate BIGINT, channels BIGINT, frames BIGINT, duration DOUBLE, format VARCHAR, subtype VARCHAR)"
    )
    assert metadata == {
        "sample_rate": 16000,
        "channels": 2,
        "frames": 40,
        "duration": pytest.approx(40 / 16000),
        "format": audio_format,
        "subtype": subtype,
    }
    assert null_metadata is None
    assert value.metadata(connection=duckdb_cursor) == vane.AudioMetadata(
        16000,
        2,
        40,
        40 / 16000,
        audio_format,
        subtype,
    )


def test_audio_metadata_facades(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio(frames=16, channels=1)
    path = tmp_path / "audio.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/wav")

    function_result = duckdb_cursor.sql("SELECT 1").select(vane.audio_metadata(value, max_bytes=4096)).fetchone()[0]
    method_result = duckdb_cursor.sql("SELECT 1").select(vane.audio_file(value).audio_metadata()).fetchone()[0]

    assert function_result == method_result
    assert function_result["sample_rate"] == 8000
    assert function_result["channels"] == 1
    assert function_result["frames"] == 16


def test_audio_metadata_and_decode_honor_logical_range(duckdb_cursor, tmp_path):
    payload, expected = _encoded_audio("WAV", "FLOAT", frames=24, channels=2)
    prefix = b"not-an-audio-prefix"
    suffix = b"not-an-audio-suffix"
    path = tmp_path / "ranged.bin"
    path.write_bytes(prefix + payload + suffix)
    value = vane.AudioFile(str(path), "audio/wav", len(prefix), len(payload))

    assert duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]["frames"] == 24
    assert value.metadata(connection=duckdb_cursor).format == "WAV"
    decoded = value.to_numpy(buffer_size=64, connection=duckdb_cursor)

    assert decoded.dtype == np.float64
    assert decoded.shape == (24, 2)
    np.testing.assert_allclose(decoded, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    ("audio_format", "subtype", "content_type"),
    [
        ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
        ("OGG", "OPUS", "audio/ogg; codecs=opus"),
    ],
)
def test_compressed_audio_metadata_and_decode_honor_logical_range(
    duckdb_cursor,
    tmp_path,
    audio_format,
    subtype,
    content_type,
):
    payload, _ = _encoded_audio(audio_format, subtype, sample_rate=16000, frames=240, channels=2)
    prefix = b"not-an-audio-prefix"
    suffix = b"not-an-audio-suffix"
    path = tmp_path / "compressed-range.bin"
    path.write_bytes(prefix + payload + suffix)
    value = vane.AudioFile(str(path), content_type, len(prefix), len(payload))

    sql_metadata = duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]
    python_metadata = value.metadata(connection=duckdb_cursor)
    decoded = value.to_numpy(buffer_size=64, connection=duckdb_cursor)

    assert sql_metadata["format"] == audio_format
    assert sql_metadata["subtype"] == subtype
    assert python_metadata.frames == 240
    assert decoded.dtype == np.float64
    assert decoded.shape == (240, 2)
    assert decoded.flags.c_contiguous
    assert np.isfinite(decoded).all()
    assert np.any(decoded != 0)


def test_mp3_metadata_uses_bounded_random_access(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio("MP3", "MPEG_LAYER_III", sample_rate=16000, frames=64_000, channels=2)
    assert len(payload) > 4096
    prefix = b"not-an-audio-prefix"
    suffix = b"not-an-audio-suffix"
    path = tmp_path / "large-ranged-mp3.bin"
    path.write_bytes(prefix + payload + suffix)
    value = vane.AudioFile(str(path), "audio/mpeg", len(prefix), len(payload))

    python_metadata = value.metadata(max_bytes=4096, connection=duckdb_cursor)
    sql_metadata = duckdb_cursor.execute("SELECT audio_metadata($1, 4096)", [value]).fetchone()[0]

    assert python_metadata.frames == 64_000
    assert python_metadata.format == "MP3"
    assert sql_metadata["frames"] == 64_000
    assert sql_metadata["format"] == "MP3"


@pytest.mark.parametrize("channels", [1, 2, 4])
def test_audio_to_numpy_returns_detached_frame_major_float64(duckdb_cursor, tmp_path, channels):
    payload, expected = _encoded_audio("WAV", "FLOAT", frames=20, channels=channels)
    path = tmp_path / f"audio-{channels}.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/x-wav")

    decoded = value.to_numpy(connection=duckdb_cursor)
    path.unlink()

    assert decoded.dtype == np.float64
    assert decoded.shape == (20, channels)
    assert decoded.flags.c_contiguous
    np.testing.assert_allclose(decoded, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("channels", [1, 2])
def test_empty_audio_decodes_under_sub_frame_byte_limit(duckdb_cursor, tmp_path, channels):
    payload, _ = _encoded_audio("WAV", "PCM_16", frames=0, channels=channels)
    path = tmp_path / f"empty-{channels}.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/wav")

    metadata = value.metadata(connection=duckdb_cursor)
    decoded = value.to_numpy(max_decoded_bytes=1, connection=duckdb_cursor)

    assert metadata.frames == 0
    assert metadata.duration == 0
    assert decoded.dtype == np.float64
    assert decoded.shape == (0, channels)
    assert decoded.nbytes == 0


@pytest.mark.parametrize(
    ("channels", "decoded_frames"),
    [(1, 0), (2, 0), (1, 1)],
    ids=["empty-mono", "empty-stereo", "nonempty"],
)
def test_unknown_length_audio_probes_eof_under_sub_frame_byte_limit(
    duckdb_cursor,
    tmp_path,
    monkeypatch,
    channels,
    decoded_frames,
):
    probes = []

    class FakeSoundFileError(Exception):
        pass

    class UnknownLengthSoundFile:
        samplerate = 8000
        frames = _audio_file._MAX_BIGINT
        format = "FLAC"
        subtype = "PCM_16"

        def __init__(self, stream, *, mode, closefd):
            assert mode == "r"
            assert closefd is False
            self.channels = channels

        def buffer_read_into(self, buffer, *, dtype):
            assert dtype == "float64"
            probes.append(len(buffer))
            return decoded_frames

        def close(self):
            pass

    class FakeSoundFileModule:
        SoundFile = UnknownLengthSoundFile
        SoundFileError = FakeSoundFileError

    path = tmp_path / "unknown-total.flac"
    path.write_bytes(b"encoded-audio")
    value = vane.AudioFile(str(path), "audio/flac")
    monkeypatch.setattr(_audio_file, "_load_soundfile", lambda: FakeSoundFileModule)

    if decoded_frames:
        with pytest.raises(vane.AudioFileLimitError, match="one decoded audio frame requires"):
            value.to_numpy(max_decoded_bytes=1, connection=duckdb_cursor)
    else:
        decoded = value.to_numpy(max_decoded_bytes=1, connection=duckdb_cursor)
        assert decoded.dtype == np.float64
        assert decoded.shape == (0, channels)
        assert decoded.nbytes == 0

    assert probes == [channels * 8]


def test_audio_metadata_can_use_header_without_reading_complete_waveform(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio("WAV", "PCM_16", frames=100_000, channels=2)
    path = tmp_path / "large.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/wav")

    metadata = value.metadata(max_bytes=64, connection=duckdb_cursor)
    sql_metadata = duckdb_cursor.execute("SELECT audio_metadata($1, 64)", [value]).fetchone()[0]

    assert metadata.frames == 100_000
    assert sql_metadata["frames"] == 100_000


def test_audio_metadata_limit_is_reported_when_parser_reads_past_budget(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio("OGG", "VORBIS", frames=64, channels=2)
    path = tmp_path / "audio.ogg"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/ogg")

    with pytest.raises(vane.AudioFileLimitError, match="max_bytes=16"):
        value.metadata(max_bytes=16, connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="max_bytes=16"):
        duckdb_cursor.execute("SELECT audio_metadata($1, 16)", [value]).fetchone()


def test_audio_metadata_probe_preserves_cancellation_after_budget_exhaustion(monkeypatch):
    class FakeSoundFileError(Exception):
        pass

    class InterruptingSoundFile:
        def __init__(self, stream, *, mode, closefd):
            assert mode == "r"
            assert closefd is False
            assert stream.read(1) == b"x"
            assert stream.read(1) == b""
            raise KeyboardInterrupt

    class FakeSoundFileModule:
        SoundFile = InterruptingSoundFile
        SoundFileError = FakeSoundFileError

    monkeypatch.setattr(_audio_file, "_load_soundfile", lambda: FakeSoundFileModule)

    with pytest.raises(KeyboardInterrupt):
        _audio_file._probe_audio_metadata(
            lambda offset, size: b"x" * size,
            logical_size=2,
            content_type=None,
            max_bytes=1,
        )


@pytest.mark.parametrize(
    "failure",
    [
        MemoryError("allocation failed"),
        RuntimeError("decoder invariant failed"),
        _audio_file.AudioFileFormatError("classified media failure"),
    ],
    ids=["memory", "internal", "classified-media"],
)
def test_audio_metadata_probe_preserves_non_decoder_errors_after_budget_exhaustion(monkeypatch, failure):
    class FakeSoundFileError(Exception):
        pass

    class FailingSoundFile:
        def __init__(self, stream, *, mode, closefd):
            assert mode == "r"
            assert closefd is False
            assert stream.read(1) == b"x"
            assert stream.read(1) == b""
            raise failure

    class FakeSoundFileModule:
        SoundFile = FailingSoundFile
        SoundFileError = FakeSoundFileError

    monkeypatch.setattr(_audio_file, "_load_soundfile", lambda: FakeSoundFileModule)

    with pytest.raises(type(failure)) as raised:
        _audio_file._probe_audio_metadata(
            lambda offset, size: b"x" * size,
            logical_size=2,
            content_type=None,
            max_bytes=1,
        )
    assert raised.value is failure


def test_audio_operations_preserve_current_cancellation_over_stored_reader_error(
    duckdb_cursor,
    tmp_path,
    monkeypatch,
):
    class FakeSoundFileError(Exception):
        pass

    class InterruptingSoundFile:
        def __init__(self, stream, *, mode, closefd):
            assert mode == "r"
            assert closefd is False
            assert stream.read(1) == b""
            raise KeyboardInterrupt

    class FakeSoundFileModule:
        SoundFile = InterruptingSoundFile
        SoundFileError = FakeSoundFileError

    path = tmp_path / "audio.bin"
    path.write_bytes(b"audio")
    value = vane.AudioFile(str(path))

    def fail_read(self, size=-1):
        raise OSError("earlier connector read failed")

    monkeypatch.setattr(_audio_file, "_load_soundfile", lambda: FakeSoundFileModule)
    monkeypatch.setattr(vane.VaneFileReader, "read", fail_read)

    with pytest.raises(KeyboardInterrupt):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(KeyboardInterrupt):
        value.to_numpy(connection=duckdb_cursor)


def test_audio_metadata_sql_maps_non_exception_control_flow_to_interrupt(duckdb_cursor, tmp_path, monkeypatch):
    class StopAudioMetadata(BaseException):
        pass

    path = tmp_path / "audio.bin"
    path.write_bytes(b"audio")
    value = vane.AudioFile(str(path))

    def stop_metadata(*args, **kwargs):
        raise StopAudioMetadata

    monkeypatch.setattr(_audio_file, "_probe_audio_metadata", stop_metadata)

    with pytest.raises(vane.InterruptException):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()


def test_audio_metadata_sql_prioritizes_connection_interrupt_over_probe_error(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "audio.bin"
    path.write_bytes(b"audio")
    value = vane.AudioFile(str(path))

    def interrupt_then_fail(*args, **kwargs):
        duckdb_cursor.interrupt()
        raise _audio_file.AudioFileFormatError("competing format failure")

    monkeypatch.setattr(_audio_file, "_probe_audio_metadata", interrupt_then_fail)

    with pytest.raises(vane.InterruptException):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()


def test_audio_metadata_sql_maps_python_memory_error_to_out_of_memory(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "audio.bin"
    path.write_bytes(b"audio")
    value = vane.AudioFile(str(path))

    def exhaust_memory(*args, **kwargs):
        raise MemoryError("allocation failed")

    monkeypatch.setattr(_audio_file, "_probe_audio_metadata", exhaust_memory)

    with pytest.raises(vane.OutOfMemoryException, match="ran out of memory"):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()


def test_audio_metadata_view_serves_one_large_request_with_one_source_read():
    payload = b"x" * (7 * 1024 * 1024)
    requests = []

    def read_at(offset, size):
        requests.append((offset, size))
        return payload[offset : offset + size]

    stream = _audio_file._AudioMetadataView(read_at, logical_size=len(payload), max_bytes=len(payload))

    assert stream.read(len(payload)) == payload
    assert requests == [(0, len(payload))]


def test_audio_metadata_view_preserves_cache_failures(monkeypatch):
    stream = _audio_file._AudioMetadataView(lambda offset, size: b"x" * size, logical_size=4, max_bytes=4)

    def fail_cache(offset, data):
        raise RuntimeError("metadata cache insertion failed")

    monkeypatch.setattr(stream, "_cache_bytes", fail_cache)

    assert stream.read(1) == b""
    with pytest.raises(RuntimeError, match="metadata cache insertion failed"):
        stream.raise_if_error()


def test_audio_metadata_view_bounds_reverse_adjacent_fetches():
    payload = bytes(range(256)) * 256
    requests = []

    def read_at(offset, size):
        requests.append((offset, size))
        return payload[offset : offset + size]

    stream = _audio_file._AudioMetadataView(read_at, logical_size=len(payload), max_bytes=len(payload))
    for offset in range(len(payload) - 1, -1, -1):
        stream.seek(offset)
        assert stream.read(1) == payload[offset : offset + 1]

    assert len(requests) <= len(payload) // stream._fetch_size + 1
    assert not stream.fetch_limit_exhausted


def test_audio_decode_limits_are_enforced(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio("WAV", "FLOAT", frames=16, channels=2)
    path = tmp_path / "audio.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/wav")

    with pytest.raises(vane.AudioFileLimitError, match="max_input_bytes"):
        value.to_numpy(max_input_bytes=len(payload) - 1, connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileLimitError, match="max_frames=15"):
        value.to_numpy(max_frames=15, connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileLimitError, match="requires 256 bytes"):
        value.to_numpy(max_decoded_bytes=255, connection=duckdb_cursor)
    assert value.to_numpy(max_frames=16, max_decoded_bytes=256, connection=duckdb_cursor).shape == (16, 2)


@pytest.mark.parametrize(
    ("content_type", "message"),
    [("video/mp4", "contradicts"), ("audio/flac", "detected MIME type")],
)
def test_audio_file_rejects_contradictory_content_type(duckdb_cursor, tmp_path, content_type, message):
    payload, _ = _encoded_audio("WAV", "PCM_16")
    path = tmp_path / "audio.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), content_type)

    with pytest.raises(vane.AudioFileFormatError, match=message):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileFormatError, match=message):
        value.to_numpy(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match=message):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()


@pytest.mark.parametrize("content_type", ["audio/wave", "audio/vnd.wave", "application/octet-stream", "audio/*"])
def test_audio_file_accepts_mime_aliases_and_generic_types(duckdb_cursor, tmp_path, content_type):
    payload, _ = _encoded_audio("WAV", "PCM_16")
    path = tmp_path / "audio.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), content_type)

    assert value.metadata(connection=duckdb_cursor).format == "WAV"
    assert value.to_numpy(connection=duckdb_cursor).shape == (32, 2)


def test_audio_file_accepts_ogg_container_and_codec_mimes(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio("OGG", "VORBIS")
    path = tmp_path / "audio.ogg"
    path.write_bytes(payload)

    for content_type in ("application/ogg", "audio/ogg; codecs=vorbis"):
        assert vane.AudioFile(str(path), content_type).metadata(connection=duckdb_cursor).subtype == "VORBIS"


@pytest.mark.parametrize(
    ("subtype", "content_type"),
    [("OPUS", "audio/opus"), ("VORBIS", "audio/vorbis")],
)
def test_audio_file_rejects_rtp_mime_for_ogg_container(duckdb_cursor, tmp_path, subtype, content_type):
    payload, _ = _encoded_audio("OGG", subtype)
    path = tmp_path / "audio.ogg"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), content_type)

    with pytest.raises(vane.AudioFileFormatError, match="detected MIME type"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileFormatError, match="detected MIME type"):
        value.to_numpy(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="detected MIME type"):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()


def test_audio_file_validates_wave_codec_parameter(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio("WAV", "PCM_16")
    path = tmp_path / "audio.wav"
    path.write_bytes(payload)

    valid = vane.AudioFile(str(path), "audio/vnd.wave; codec=1")
    assert valid.metadata(connection=duckdb_cursor).subtype == "PCM_16"
    assert valid.to_numpy(connection=duckdb_cursor).shape == (32, 2)
    assert duckdb_cursor.execute("SELECT audio_metadata($1)", [valid]).fetchone()[0]["subtype"] == "PCM_16"

    contradictory = vane.AudioFile(str(path), "audio/vnd.wave; codec=55")
    with pytest.raises(vane.AudioFileFormatError, match="detected audio codec"):
        contradictory.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileFormatError, match="detected audio codec"):
        contradictory.to_numpy(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="detected audio codec"):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [contradictory]).fetchone()


@pytest.mark.parametrize(
    ("subtype", "codec"),
    [
        ("G721_32", "40"),
        ("NMS_ADPCM_16", "38"),
        ("NMS_ADPCM_24", "38"),
        ("NMS_ADPCM_32", "38"),
    ],
)
def test_audio_file_validates_additional_wave_codec_parameters(duckdb_cursor, tmp_path, subtype, codec):
    payload, _ = _encoded_audio("WAV", subtype, frames=1024, channels=1)
    path = tmp_path / f"audio-{subtype}.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), f"audio/vnd.wave; codec={codec}")

    assert value.metadata(connection=duckdb_cursor).subtype == subtype
    assert duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]["subtype"] == subtype


def test_audio_file_validates_waveformatextensible_codec_parameter(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio("WAVEX", "PCM_16", frames=32, channels=2)
    path = tmp_path / "audio-wavex.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/vnd.wave; codec=1")

    assert value.metadata(connection=duckdb_cursor).format == "WAVEX"
    assert value.to_numpy(connection=duckdb_cursor).shape == (32, 2)
    assert duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]["format"] == "WAVEX"


@pytest.mark.parametrize("content_type", ["audio/x-au", "audio/au", "audio/vnd.sun.audio"])
def test_audio_file_accepts_au_container_mime_aliases(duckdb_cursor, tmp_path, content_type):
    payload, _ = _encoded_audio("AU", "ULAW", sample_rate=8000, frames=32, channels=1)
    path = tmp_path / "audio.au"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), content_type)

    assert value.metadata(connection=duckdb_cursor) == vane.AudioMetadata(8000, 1, 32, 0.004, "AU", "ULAW")
    assert value.to_numpy(connection=duckdb_cursor).shape == (32, 1)
    assert duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]["subtype"] == "ULAW"


@pytest.mark.parametrize(
    ("subtype", "sample_rate", "channels"),
    [
        ("ULAW", 8000, 1),
        ("PCM_16", 8000, 1),
        ("ULAW", 16000, 1),
        ("ULAW", 8000, 2),
    ],
)
def test_audio_file_rejects_audio_basic_for_au_container(
    duckdb_cursor,
    tmp_path,
    subtype,
    sample_rate,
    channels,
):
    payload, _ = _encoded_audio("AU", subtype, sample_rate=sample_rate, frames=32, channels=channels)
    path = tmp_path / "audio.au"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/basic")

    with pytest.raises(vane.AudioFileFormatError, match="detected MIME type"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileFormatError, match="detected MIME type"):
        value.to_numpy(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="detected MIME type"):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()


@pytest.mark.parametrize(
    ("subtype", "content_type"),
    [
        ("VORBIS", "audio/ogg; codecs=opus"),
        ("OPUS", "audio/ogg; codecs=vorbis"),
    ],
)
def test_audio_file_rejects_contradictory_ogg_codec_parameter(duckdb_cursor, tmp_path, subtype, content_type):
    payload, _ = _encoded_audio("OGG", subtype)
    path = tmp_path / "audio.ogg"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), content_type)

    with pytest.raises(vane.AudioFileFormatError, match="detected audio codec"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileFormatError, match="detected audio codec"):
        value.to_numpy(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="detected audio codec"):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()


def test_audio_file_rejects_specific_mime_for_unmapped_decoder_format(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio("MAT5", "DOUBLE")
    path = tmp_path / "audio.mat"
    path.write_bytes(payload)

    declared = vane.AudioFile(str(path), "audio/wav")
    with pytest.raises(vane.AudioFileFormatError, match="cannot be validated.*MAT5"):
        declared.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileFormatError, match="cannot be validated.*MAT5"):
        declared.to_numpy(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="cannot be validated.*MAT5"):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [declared]).fetchone()

    assert vane.AudioFile(str(path)).metadata(connection=duckdb_cursor).format == "MAT5"


def test_audio_file_preserves_unknown_flac_frame_count_and_decodes_incrementally(duckdb_cursor, tmp_path):
    payload, _ = _encoded_audio("FLAC", "PCM_16", frames=64, channels=2)
    path = tmp_path / "unknown-total.flac"
    path.write_bytes(_flac_with_unknown_total_samples(payload))
    value = vane.AudioFile(str(path), "audio/flac")

    metadata = value.metadata(connection=duckdb_cursor)
    sql_metadata = duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]
    decoded = value.to_numpy(max_frames=64, max_decoded_bytes=64 * 2 * 8, connection=duckdb_cursor)

    assert metadata.frames is None
    assert metadata.duration is None
    assert sql_metadata["frames"] is None
    assert sql_metadata["duration"] is None
    assert decoded.shape == (64, 2)

    with pytest.raises(vane.AudioFileLimitError, match="max_frames=63"):
        value.to_numpy(max_frames=63, connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileLimitError, match="max_decoded_bytes"):
        value.to_numpy(max_decoded_bytes=63 * 2 * 8, connection=duckdb_cursor)


def test_soundfile_cleanup_does_not_replace_primary_audio_errors(duckdb_cursor, tmp_path, monkeypatch):
    payload, _ = _encoded_audio("WAV", "PCM_16", frames=16, channels=1)
    path = tmp_path / "audio.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/wav")
    original_close = soundfile.SoundFile.close
    original_metadata = _audio_file._metadata_from_sound_file

    def fail_first_close(self):
        was_open = not self.closed
        original_close(self)
        if was_open:
            raise RuntimeError("competing close failure")

    def interrupt_metadata(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(soundfile.SoundFile, "close", fail_first_close)
    monkeypatch.setattr(_audio_file, "_metadata_from_sound_file", interrupt_metadata)
    with pytest.raises(KeyboardInterrupt):
        value.metadata(connection=duckdb_cursor)

    monkeypatch.setattr(_audio_file, "_metadata_from_sound_file", original_metadata)
    with pytest.raises(vane.AudioFileLimitError, match="max_frames=15"):
        value.to_numpy(max_frames=15, connection=duckdb_cursor)


def test_audio_file_classifies_invalid_media_but_propagates_io(duckdb_cursor, tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not an audio file")
    corrupt_value = vane.AudioFile(str(corrupt), "audio/wav")

    with pytest.raises(vane.AudioFileFormatError, match="supported encoded audio"):
        corrupt_value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.AudioFileFormatError, match="supported encoded audio"):
        corrupt_value.to_numpy(connection=duckdb_cursor)

    missing = vane.AudioFile(str(tmp_path / "missing.wav"), "audio/wav")
    with pytest.raises(vane.IOException):
        missing.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.IOException):
        missing.to_numpy(connection=duckdb_cursor)
    with pytest.raises(vane.IOException):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [missing]).fetchone()


def test_audio_operations_propagate_reader_failures_from_virtual_io(duckdb_cursor, tmp_path, monkeypatch):
    payload, _ = _encoded_audio("WAV", "FLOAT")
    path = tmp_path / "audio.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/wav")

    def fail_read(self, size=-1):
        raise OSError("connector read failed")

    monkeypatch.setattr(vane.VaneFileReader, "read", fail_read)
    with pytest.raises(OSError, match="connector read failed"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(OSError, match="connector read failed"):
        value.to_numpy(connection=duckdb_cursor)


def test_audio_metadata_requires_audiofile(duckdb_cursor):
    with pytest.raises(vane.BinderException, match="requires AUDIOFILE, not FILE"):
        duckdb_cursor.sql("SELECT audio_metadata(file('memory://generic', NULL, NULL, NULL, NULL))")
    with pytest.raises(vane.BinderException, match="requires AUDIOFILE, not IMAGEFILE"):
        duckdb_cursor.sql("SELECT audio_metadata(image_file('memory://image'))")


@pytest.mark.parametrize(
    ("method", "kwargs", "error_type", "message"),
    [
        ("metadata", {"max_bytes": True}, TypeError, "max_bytes must be int"),
        ("metadata", {"max_bytes": 0}, ValueError, "greater than zero"),
        ("metadata", {"max_bytes": 64 * 1024 * 1024 + 1}, ValueError, "at most"),
        ("to_numpy", {"buffer_size": True}, TypeError, "buffer_size must be int"),
        ("to_numpy", {"buffer_size": 0}, ValueError, "greater than zero"),
        ("to_numpy", {"max_input_bytes": 0}, ValueError, "greater than zero"),
        ("to_numpy", {"max_frames": 0}, ValueError, "greater than zero"),
        ("to_numpy", {"max_decoded_bytes": 0}, ValueError, "greater than zero"),
    ],
)
def test_audio_file_python_argument_validation(method, kwargs, error_type, message):
    value = vane.AudioFile("memory://not-opened")

    with pytest.raises(error_type, match=message):
        getattr(value, method)(**kwargs)


def test_audio_file_optional_dependency_is_lazy(monkeypatch):
    original_import = importlib.import_module

    def fail_soundfile(name, package=None):
        if name == "soundfile":
            raise ImportError("missing soundfile")
        return original_import(name, package)

    monkeypatch.setattr(_audio_file.importlib, "import_module", fail_soundfile)

    value = vane.AudioFile("memory://not-opened")
    with pytest.raises(ImportError, match=r"vane-ai\[audio\]"):
        value.metadata()
    with pytest.raises(ImportError, match=r"vane-ai\[audio\]"):
        value.to_numpy()


def test_audio_file_unusable_native_dependency_is_actionable(duckdb_cursor, tmp_path, monkeypatch):
    payload, _ = _encoded_audio()
    path = tmp_path / "audio.wav"
    path.write_bytes(payload)
    value = vane.AudioFile(str(path), "audio/wav")
    original_import = importlib.import_module

    def fail_libsndfile(name, package=None):
        if name == "soundfile":
            raise OSError("libsndfile cannot be loaded")
        return original_import(name, package)

    monkeypatch.setattr(_audio_file.importlib, "import_module", fail_libsndfile)

    with pytest.raises(ImportError, match=r"usable libsndfile.*vane-ai\[audio\]"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(ImportError, match=r"usable libsndfile.*vane-ai\[audio\]"):
        value.to_numpy(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match=r"usable libsndfile.*vane-ai\[audio\]"):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [value]).fetchone()


def test_audio_metadata_sql_preflights_dependency_before_opening_file(duckdb_cursor, tmp_path, monkeypatch):
    missing = vane.AudioFile(str(tmp_path / "missing.wav"), "audio/wav")

    def fail_soundfile():
        raise ImportError("install vane-ai[audio]")

    monkeypatch.setattr(_audio_file, "_load_soundfile", fail_soundfile)

    with pytest.raises(vane.InvalidInputException, match=r"install vane-ai\[audio\]"):
        duckdb_cursor.execute("SELECT audio_metadata($1)", [missing]).fetchone()
