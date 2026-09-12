# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import shutil
import subprocess
import time

import pytest

import vane
from tests.fast.test_audio_file import _flac_with_unknown_total_samples
from tests.fast.test_file_reader import _start_object_server
from tests.fast.test_native_media_extensions import _connect, _wav


def _audio_value(tmp_path, *, frames, channels=2, source_rate=48000, format="WAV", subtype="FLOAT", unknown=False):
    np = pytest.importorskip("numpy")
    soundfile = pytest.importorskip("soundfile")
    times = np.arange(frames) / source_rate
    samples = np.stack(
        [0.2 * np.sin(2 * np.pi * (331 + 117 * channel) * times) + 0.03 * (channel + 1) for channel in range(channels)],
        axis=1,
    )
    if frames:
        samples[0] = 0.4
        samples[-1] = -0.35
    encoded = io.BytesIO()
    soundfile.write(encoded, samples, source_rate, format=format, subtype=subtype)
    payload = encoded.getvalue()
    if unknown:
        payload = _flac_with_unknown_total_samples(payload)
    prefix = b"outside the audio FILE view"
    path = tmp_path / "audio.bundle"
    path.write_bytes(prefix + payload + b"outside suffix")
    return vane.AudioFile(str(path), None, len(prefix), len(payload)), len(payload)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize(
    "format,subtype,source_rate,frames,channels,unknown",
    [
        ("WAV", "PCM_16", 48000, 37, 1, False),
        ("WAV", "PCM_16", 48000, 0, 2, False),
        *[
            ("WAV", subtype, 44100, 37, 2, False)
            for subtype in ("PCM_U8", "PCM_24", "PCM_32", "FLOAT", "DOUBLE", "ULAW", "ALAW")
        ],
        ("WAVEX", "PCM_16", 44100, 37, 4, False),
        ("RF64", "PCM_24", 44100, 37, 2, False),
        *[("AIFF", subtype, 44100, 37, 2, False) for subtype in ("PCM_S8", "PCM_16", "PCM_24", "FLOAT", "DOUBLE")],
        *[("FLAC", subtype, 48000, 4801, 2, False) for subtype in ("PCM_S8", "PCM_16", "PCM_24")],
        ("FLAC", "PCM_24", 48000, 4801, 2, True),
        ("MP3", "MPEG_LAYER_III", 48000, 4801, 2, False),
        ("OGG", "VORBIS", 48000, 4801, 2, False),
        *[("OGG", "OPUS", rate, 1001, 2, False) for rate in (8000, 24000, 48000)],
    ],
)
def test_native_audio_metadata_matches_python(tmp_path, format, subtype, source_rate, frames, channels, unknown):
    value, _ = _audio_value(
        tmp_path,
        frames=frames,
        channels=channels,
        source_rate=source_rate,
        format=format,
        subtype=subtype,
        unknown=unknown,
    )
    with vane.connect() as python, _connect("audio") as native:
        expected = python.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]
        result = native.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]
        expression = native.sql("SELECT 1").select(vane.audio_metadata(value)).fetchone()[0]
        assert result == expected == expression
        assert result["format"] == format and result["subtype"] == subtype
        assert (result["sample_rate"], result["channels"]) == (source_rate, channels)
        assert result["frames"] == (None if unknown else frames)
        assert result["duration"] == (None if unknown else frames / source_rate)
        assert value.metadata(connection=python) == vane.AudioMetadata(**result)
        assert native.execute("SELECT audio_metadata(NULL::AUDIOFILE)").fetchone()[0] is None


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize(
    "format,subtype,content_type,valid",
    [
        ("OGG", "VORBIS", "audio/ogg; codecs=vorbis", True),
        ("OGG", "OPUS", "audio/ogg; codecs=opus", True),
        ("OGG", "VORBIS", "application/ogg; codecs=vorbis", True),
        ("OGG", "OPUS", 'Audio/OGG; CODECS=" OPUS "', True),
        ("OGG", "OPUS", 'audio/ogg; codecs="opus, opus"', True),
        ("OGG", "OPUS", 'audio/ogg; note="x; codecs=vorbis"; codecs=opus', True),
        ("OGG", "OPUS", 'audio/ogg; note=""; codecs=opus', True),
        ("OGG", "OPUS", r'audio/ogg; codecs="op\us"', True),
        ("OGG", "OPUS", "audio/ogg; codecs=(encoder)opus", True),
        ("OGG", "OPUS", "audio/ogg; codecs*=utf-8''%6f%70%75%73", True),
        ("OGG", "OPUS", "audio/ogg; codecs*=''opus", True),
        ("OGG", "OPUS", "audio/ogg; codecs*1=us; codecs*0=op", True),
        ("OGG", "OPUS", 'audio/ogg; codecs*0="op"; codecs*1="us"', True),
        ("OGG", "OPUS", 'audio/ogg; codecs*0=""; codecs*1=opus', True),
        ("OGG", "OPUS", "audio/ogg; codecs*0*=us-ascii''op; codecs*1*=us", True),
        *[
            ("OGG", "OPUS", f"{mime}; codecs=opus", True)
            for mime in ("audio/*", "application/octet-stream", "binary/octet-stream")
        ],
        ("WAV", "PCM_16", "audio/vnd.wave; codec=1", True),
        ("WAVEX", "PCM_16", "audio/wave; codec=1", True),
        ("RF64", "PCM_24", 'audio/x-wav; CODEC="1"', True),
        ("WAV", "FLOAT", "audio/wav; codec=3", True),
        ("WAV", "DOUBLE", "audio/wav; codec=3", True),
        ("WAV", "ALAW", "audio/wav; codec=6", True),
        ("WAV", "ULAW", "audio/wav; codec=7", True),
        ("WAV", "PCM_16", "audio/wav; codec*=utf-8''1", True),
        ("OGG", "VORBIS", "audio/ogg; codecs=opus", False),
        ("OGG", "OPUS", "audio/ogg; codecs=vorbis", False),
        ("OGG", "OPUS", "application/ogg; codecs=vorbis", False),
        ("OGG", "OPUS", "audio/*; codecs=vorbis", False),
        ("OGG", "OPUS", "audio/ogg; codec=opus", False),
        ("OGG", "OPUS", 'audio/ogg; codecs="opus, vorbis"', False),
        ("OGG", "OPUS", 'audio/ogg; codecs="opus,"', False),
        ("OGG", "OPUS", 'audio/ogg; codecs=""', False),
        ("OGG", "OPUS", "audio/ogg; codecs=", False),
        ("OGG", "OPUS", "audio/ogg; codecs*0=; codecs*1=opus", False),
        ("OGG", "OPUS", "audio/ogg; note=; codecs=opus", False),
        ("OGG", "OPUS", "audio/ogg; codecs=opus; note=", False),
        ("OGG", "OPUS", 'audio/ogg; codecs="opus', False),
        ("OGG", "OPUS", "audio/ogg; codecs=opus; codecs=vorbis", False),
        ("OGG", "OPUS", "audio/ogg; codecs=opus; codec=1", False),
        ("OGG", "OPUS", "audio/ogg; codecs*0=op; codecs*2=us", False),
        ("OGG", "OPUS", "audio/ogg; codecs*00=opus", False),
        ("OGG", "OPUS", "audio/ogg; codecs*0=op; codecs*01=us", False),
        ("OGG", "OPUS", "audio/ogg; codecs**=opus", False),
        ("OGG", "OPUS", 'audio/ogg; codecs="op\x00us"', False),
        ("OGG", "OPUS", "audio/ogg; codecs*=utf-8''vorbis", False),
        ("OGG", "OPUS", "audio/ogg; codecs*=utf-16''opus", False),
        ("OGG", "OPUS", "audio/ogg; codecs*=\"utf-8''opus\"", False),
        ("OGG", "OPUS", "audio/ogg; codecs*0*=\"utf-8''op\"; codecs*1*=us", False),
        ("OGG", "OPUS", "audio/ogg; codecs*0*=utf-8''op; codecs*1*=\"us\"", False),
        ("OGG", "OPUS", "audio/ogg; note*=\"utf-8''abc\"; codecs=opus", False),
        ("WAV", "PCM_16", "audio/wav; codec*=\"utf-8''1\"", False),
        ("WAV", "PCM_16", "audio/wav; codec=55", False),
        ("WAV", "FLOAT", "audio/wav; codec=1", False),
        ("WAV", "PCM_16", "audio/*; codec=1", False),
        ("WAV", "PCM_16", "audio/wav; codecs=pcm", False),
        ("WAV", "PCM_16", 'audio/wav; codec="1,3"', False),
        ("WAV", "PCM_16", "audio/wav; codec=1; codec=1", False),
        ("AIFF", "PCM_16", "audio/aiff; codec=1", False),
        ("FLAC", "PCM_16", "audio/flac; codecs=flac", False),
    ],
)
def test_native_audio_codec_declarations_match_python(tmp_path, format, subtype, content_type, valid):
    np = pytest.importorskip("numpy")
    source, _ = _audio_value(tmp_path, frames=128, format=format, subtype=subtype)
    value = vane.AudioFile(source.url, content_type, source.position, source.size)
    with vane.connect() as python, _connect("audio") as native:
        for sql, expression in (
            ("audio_metadata($1)", vane.audio_metadata(value)),
            ("resample($1, 16000)", vane.resample(value, 16000)),
        ):
            if not valid:
                for connection in (python, native):
                    with pytest.raises(vane.InvalidInputException):
                        connection.execute(f"SELECT {sql}", [value]).fetchone()
                with pytest.raises(vane.InvalidInputException):
                    native.sql("SELECT 1").select(expression).fetchone()
                continue
            expected = python.execute(f"SELECT {sql}", [value]).fetchone()[0]
            result = native.execute(f"SELECT {sql}", [value]).fetchone()[0]
            via_expression = native.sql("SELECT 1").select(expression).fetchone()[0]
            if isinstance(expected, dict):
                assert result == via_expression == expected
            else:
                np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
                np.testing.assert_array_equal(result, via_expression)
        assert native.execute("SELECT 1").fetchone() == (1,)


@pytest.mark.usefixtures("ray_query")
def test_native_audio_metadata_retains_probe_budget(tmp_path):
    value, size = _audio_value(tmp_path, frames=100_000)
    budget = 128 * 1024
    assert size > budget
    with vane.connect() as python, _connect("audio") as native:
        query = "SELECT audio_metadata($1, $2::UBIGINT)"
        expected = python.execute(query, [value, budget]).fetchone()[0]
        assert native.execute(query, [value, budget]).fetchone()[0] == expected
        with pytest.raises(vane.OutOfRangeException, match="read/probe byte budget"):
            native.execute("SELECT audio_metadata($1, 8)", [value]).fetchone()


@pytest.mark.usefixtures("ray_query")
def test_native_audio_metadata_partial_header_budget_is_not_format_error(tmp_path):
    value, size = _audio_value(tmp_path, frames=37, subtype="PCM_16")
    with _connect("audio") as native:
        # FFmpeg probes this small FILE view in one read. Leave only four
        # bytes for libsndfile's header: the budget must retain precedence
        # even if the parser returns "format not recognised" immediately.
        with pytest.raises(vane.OutOfRangeException, match="read/probe byte budget"):
            native.execute("SELECT audio_metadata($1, $2::UBIGINT)", [value, size + 4]).fetchone()


@pytest.mark.usefixtures("ray_query")
def test_native_audio_metadata_shares_deadline_between_parsers():
    payload = _wav(frames=37)
    with _connect("audio") as native:
        server, thread, handler = _start_object_server(payload)
        original = handler._send_object
        delayed = []

        def slow_read(self, include_body):
            if include_body:
                bounds = self.headers.get("Range")
                if bounds == f"bytes=0-{len(payload) - 1}" and not delayed:
                    # FFmpeg probes the complete small WAV in one AVIO read.
                    delayed.append("ffmpeg")
                    time.sleep(15)
                elif bounds == "bytes=0-11" and delayed == ["ffmpeg"]:
                    # libsndfile identifies RIFF from its own 12-byte read.
                    # Each parser spends less than 30 seconds, but together
                    # they exceed the operation's single probe deadline.
                    delayed.append("soundfile")
                    time.sleep(16)
            original(self, include_body)

        handler._send_object = slow_read
        try:
            value = vane.AudioFile(f"http://127.0.0.1:{server.server_port}/bucket/object.bin")
            with pytest.raises(vane.OutOfRangeException, match="metadata probe exceeded its time budget"):
                native.execute("SELECT audio_metadata($1)", [value]).fetchone()
            assert delayed == ["ffmpeg", "soundfile"]
            assert native.execute("SELECT 1").fetchone() == (1,)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("frames", [0, 1, 2, 7, 31, 32, 33, 65])
@pytest.mark.parametrize("target_rate", [4000, 16000])
def test_native_audio_short_waveform_matches_python(tmp_path, channels, frames, target_rate):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soxr")
    value, _ = _audio_value(tmp_path, frames=frames, channels=channels, source_rate=8000)
    with vane.connect() as python, _connect("audio") as native:
        expected = value.resample(target_rate, connection=python)
        result = native.execute("SELECT resample($1, $2)", [value, target_rate]).fetchone()[0]
        expression = native.sql("SELECT 1").select(vane.resample(value, target_rate)).fetchone()[0]
        assert result.dtype == np.float64 and result.flags.c_contiguous
        assert result.shape == ((frames * target_rate + 7999) // 8000, channels)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result, expression)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize(
    "frames,channels,source_rate,target_rate,format,subtype,unknown",
    [
        (1001, 1, 44100, 16000, "WAV", "PCM_16", False),
        (1001, 2, 44100, 16000, "WAV", "FLOAT", False),
        (997, 4, 44100, 48000, "WAV", "DOUBLE", False),
        (70001, 2, 48000, 44100, "WAV", "PCM_24", False),
        (1001, 2, 44100, 16000, "FLAC", "PCM_24", True),
    ],
)
def test_native_audio_fractional_and_unknown_lengths_match_python(
    tmp_path, frames, channels, source_rate, target_rate, format, subtype, unknown
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soxr")
    value, _ = _audio_value(
        tmp_path,
        frames=frames,
        channels=channels,
        source_rate=source_rate,
        format=format,
        subtype=subtype,
        unknown=unknown,
    )
    with vane.connect() as python, _connect("audio") as native:
        expected = value.resample(target_rate, connection=python)
        result = native.execute("SELECT resample($1, $2)", [value, target_rate]).fetchone()[0]
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result[-1], 0)
        profile = native.execute("SELECT native_audio_resample_profile($1, $2)", [value, target_rate]).fetchone()[0]
        assert profile["decoded_frames"] == frames
        assert profile["output_frames"] == len(expected)
        assert profile["decoder_library"] == "libsndfile"
        assert profile["resampler_library"] == "soxr_hq"


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("format,subtype", [("OGG", "VORBIS"), ("OGG", "OPUS"), ("MP3", "MPEG_LAYER_III")])
@pytest.mark.parametrize("target_rate", [16000, 48000])
def test_native_audio_lossy_decode_and_tail_match_python(tmp_path, format, subtype, target_rate):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soxr")
    value, _ = _audio_value(tmp_path, frames=4801, format=format, subtype=subtype)
    with vane.connect() as python, _connect("audio") as native:
        expected = value.resample(target_rate, connection=python)
        result = native.execute("SELECT resample($1, $2)", [value, target_rate]).fetchone()[0]
        assert result.shape == expected.shape == (1601 if target_rate == 16000 else 4801, 2)
        # Different builds of the same lossy codec may vary in the last bits.
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("source_rate", [8000, 24000])
def test_native_audio_ogg_opus_matches_soundfile_sample_rate(tmp_path, source_rate):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soxr")
    value, _ = _audio_value(tmp_path, frames=1001, source_rate=source_rate, format="OGG", subtype="OPUS")
    with vane.connect() as python, _connect("audio") as native:
        expected = value.resample(16000, connection=python)
        result = native.execute("SELECT resample($1, 16000)", [value]).fetchone()[0]
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("function", ["resample", "native_audio_resample_profile"])
def test_native_audio_ceil_padding_counts_toward_limits(tmp_path, function):
    np = pytest.importorskip("numpy")
    value, size = _audio_value(tmp_path, frames=1001, source_rate=44100)
    with _connect("audio") as native:
        query = f"SELECT {function}($1, 16000, $2::UBIGINT, 1001, 16016, $3::UBIGINT, $4::UBIGINT)"
        for frames, byte_limit in [(363, 5824), (364, 5823)]:
            with pytest.raises(vane.OutOfRangeException):
                native.execute(query, [value, size, frames, byte_limit]).fetchone()
        result = native.execute(query, [value, size, 364, 5824]).fetchone()[0]
        if function == "resample":
            assert result.shape == (364, 2)
            np.testing.assert_array_equal(result[-1], 0)
        else:
            assert (result["output_frames"], result["output_bytes"]) == (364, 5824)


@pytest.mark.usefixtures("ray_query")
def test_native_audio_batch_output_budget(tmp_path):
    value, _ = _audio_value(tmp_path, frames=156250, source_rate=6000, subtype="PCM_16")
    with _connect("audio") as native:
        # Each waveform is 160 MB; two rows must exceed the shared 256 MiB
        # vector budget. The profile executes the same output allocation path.
        with pytest.raises(vane.OutOfRangeException, match="batch byte limit"):
            native.execute(
                "SELECT native_audio_resample_profile(f, 384000) FROM (VALUES ($1), ($1)) input(f)", [value]
            ).fetchall()


@pytest.mark.usefixtures("ray_query")
def test_native_audio_unknown_length_enforces_actual_input_limit(tmp_path):
    value, size = _audio_value(tmp_path, frames=1001, source_rate=44100, format="FLAC", subtype="PCM_16", unknown=True)
    with _connect("audio") as native:
        query = "SELECT resample($1, 16000, $2::UBIGINT, $3::UBIGINT, $4::UBIGINT, 364, 5824)"
        for frames, byte_limit, message in [(1000, 16016, "max_frames"), (1001, 16015, "decoded audio bytes")]:
            with pytest.raises(vane.OutOfRangeException, match=message):
                native.execute(query, [value, size, frames, byte_limit]).fetchone()


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("target_rate", [8000, 16000])
def test_native_audio_webm_uses_actual_decoder_sample_rate(tmp_path, target_rate):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    path = tmp_path / "input-rate-8000.webm"
    samples = (0.3 * np.sin(2 * np.pi * 331 * np.arange(800) / 8000)).astype("float32")
    with av.open(str(path), "w", format="webm") as output:
        stream = output.add_stream("libopus", rate=8000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(samples[None, :], format="fltp", layout="mono")
        frame.sample_rate = 8000
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as source:
        decoded = list(source.decode(audio=0))
        assert decoded[0].sample_rate == 48000
        decoded_frames = sum(frame.samples for frame in decoded)
    with _connect("audio") as native:
        value = vane.AudioFile(str(path))
        metadata = native.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]
        assert metadata["frames"] is None and metadata["duration"] is None
        result = native.execute("SELECT resample($1, $2)", [value, target_rate]).fetchone()[0]
        assert result.shape == ((decoded_frames * target_rate + 47999) // 48000, 1)
        assert np.isfinite(result).all() and np.max(np.abs(result)) > 0.1
        profile = native.execute("SELECT native_audio_resample_profile($1, $2)", [value, target_rate]).fetchone()[0]
        assert profile["decoded_frames"] == decoded_frames
        assert profile["output_frames"] == len(result)
        assert profile["decoder_library"] == "ffmpeg"
        assert profile["source_sample_rate"] == 48000


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("container", ["ogg", "flac"])
@pytest.mark.parametrize("target_rate", [8000, 16000])
def test_native_audio_retains_flac_formats_unsupported_by_soundfile(tmp_path, container, target_rate):
    np = pytest.importorskip("numpy")
    soundfile = pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg CLI is required to encode Ogg FLAC and 32-bit FLAC fixtures")
    times = np.arange(800) / 8000
    samples = np.stack([0.3 * np.sin(2 * np.pi * frequency * times) for frequency in (331, 448)], axis=1)
    source = tmp_path / "source.wav"
    soundfile.write(source, samples, 8000, subtype="PCM_16" if container == "ogg" else "PCM_32")
    encoded = tmp_path / f"encoded.{container}"
    codec_options = ["-sample_fmt", "s16"]
    if container == "flac":
        codec_options = ["-sample_fmt", "s32", "-bits_per_raw_sample", "32", "-strict", "experimental"]
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source), "-c:a", "flac"]
        + codec_options
        + ["-f", container, str(encoded)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    payload = encoded.read_bytes()
    if container == "flac":
        # Confirm that the encoder really wrote 32-bit FLAC rather than
        # silently reducing precision to libsndfile's supported 24 bits.
        assert payload[:4] == b"fLaC"
        assert ((int.from_bytes(payload[18:26], "big") >> 36) & 31) + 1 == 32
    prefix = b"outside the audio FILE view"
    bundle = tmp_path / "audio.bundle"
    bundle.write_bytes(prefix + payload + b"outside suffix")
    value = vane.AudioFile(str(bundle), None, len(prefix), len(payload))
    with vane.connect() as python, _connect("audio") as native:
        # FLAC is lossless: the decoded source WAV is an independent reference
        # without requiring Python SoundFile to support the FLAC container.
        expected = vane.AudioFile(str(source)).resample(target_rate, connection=python)
        metadata = native.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]
        assert metadata == {
            "sample_rate": 8000,
            "channels": 2,
            "frames": None,
            "duration": None,
            "format": container,
            "subtype": "flac",
        }
        assert native.sql("SELECT 1").select(vane.audio_metadata(value)).fetchone()[0] == metadata
        result = native.execute("SELECT resample($1, $2)", [value, target_rate]).fetchone()[0]
        expression = native.sql("SELECT 1").select(vane.resample(value, target_rate)).fetchone()[0]
        assert result.dtype == np.float64 and result.flags.c_contiguous
        assert result.shape == expected.shape == (target_rate // 10, 2)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result, expression)
        profile = native.execute("SELECT native_audio_resample_profile($1, $2)", [value, target_rate]).fetchone()[0]
        assert profile["decoder_library"] == "ffmpeg"
        assert profile["decoded_frames"] == 800
        assert profile["source_sample_rate"] == 8000
