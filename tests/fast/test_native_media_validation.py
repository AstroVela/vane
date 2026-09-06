# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import math
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import vane
from tests.fast.test_file_reader import _start_object_server
from tests.fast.test_native_media_extensions import _artifact, _connect, _png, _wav


def _waveform(rate, channels, frames):
    np = pytest.importorskip("numpy")
    times = np.arange(frames) / rate
    return (
        np.stack([np.sin(2 * np.pi * (200 + 100 * channel) * times) for channel in range(channels)], axis=1) * 10000
    ).astype("int16")


@pytest.mark.parametrize(
    "format,channels,source_rate,target_rate",
    [
        ("WAV", 1, 8000, 16000),
        ("WAV", 2, 48000, 16000),
        ("WAV", 6, 44100, 48000),
        ("WAV", 64, 48000, 48000),
        ("AIFF", 2, 44100, 44100),
        ("FLAC", 1, 96000, 16000),
        ("FLAC", 8, 48000, 48000),
    ],
)
def test_native_audio_lossless_matrix_preserves_layout_and_file_window(
    tmp_path, format, channels, source_rate, target_rate
):
    np = pytest.importorskip("numpy")
    soundfile = pytest.importorskip("soundfile")
    frames = source_rate // 10
    source_samples = _waveform(source_rate, channels, frames)
    encoded = io.BytesIO()
    soundfile.write(encoded, source_samples, source_rate, format=format, subtype="PCM_16")
    payload = encoded.getvalue()
    prefix = b"outside audio view" * 17
    path = tmp_path / "audio.bundle"
    path.write_bytes(prefix + payload + b"outside suffix")
    mime = {"WAV": "audio/wav", "AIFF": "audio/aiff", "FLAC": "audio/flac"}[format]
    value = vane.AudioFile(str(path), mime, len(prefix), len(payload))
    with _connect("audio") as con:
        rows = con.execute(
            "SELECT audio_resample(f, $2) FROM (VALUES ($1), (NULL::AUDIOFILE), ($1)) input(f)",
            [value, target_rate],
        ).fetchall()
        assert rows[1] == (None,)
        assert rows[0] == rows[2]
        result = rows[0][0]
        assert (result["sample_rate"], result["channels"], result["frames"]) == (
            target_rate,
            channels,
            target_rate // 10,
        )
        samples = np.array(result["samples"]).reshape(result["frames"], channels)
        assert np.isfinite(samples).all()
        if source_rate == target_rate:
            np.testing.assert_array_equal(samples, source_samples.astype("float64") / 32768)
        else:
            times = np.arange(len(samples)) / target_rate
            expected = np.stack(
                [np.sin(2 * np.pi * (200 + 100 * channel) * times) for channel in range(channels)], axis=1
            ) * (10000 / 32768)
            np.testing.assert_allclose(samples[64:-64], expected[64:-64], atol=0.005, rtol=0)
        metadata = con.execute("SELECT audio_metadata($1)", [value]).fetchone()[0]
        assert (metadata["sample_rate"], metadata["channels"]) == (source_rate, channels)


@pytest.mark.parametrize(
    "container,codec,mime",
    [("adts", "aac", "audio/aac"), ("mp4", "aac", "audio/mp4"), ("webm", "libopus", "audio/webm")],
)
def test_native_audio_encoded_container_matrix(tmp_path, container, codec, mime):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    path = tmp_path / f"audio.{container}"
    with av.open(str(path), "w", format=container) as output:
        stream = output.add_stream(codec, rate=48000)
        stream.layout = "stereo"
        stream.bit_rate = 96000
        samples = _waveform(48000, 2, 4800).astype("float32") / 32768
        frame = av.AudioFrame.from_ndarray(samples.T.copy(), format="fltp", layout="stereo")
        frame.sample_rate = 48000
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with _connect("audio") as con:
        file = vane.AudioFile(str(path), mime)
        metadata = con.execute("SELECT audio_metadata($1)", [file]).fetchone()[0]
        assert metadata["sample_rate"] == 48000 and metadata["channels"] == 2
        assert metadata["frames"] is None
        result = con.execute("SELECT audio_resample($1, 16000)", [file]).fetchone()[0]
        samples = np.array(result["samples"]).reshape(result["frames"], 2)
        assert result["sample_rate"] == 16000 and result["channels"] == 2
        assert 1400 <= len(samples) <= 2100
        assert np.isfinite(samples).all() and np.max(np.abs(samples)) < 1
        assert np.mean(samples[:, 0] ** 2) > 0.01


def test_native_audio_profile_executes_real_output_and_preserves_nulls(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(_wav(4800, 48000))
    value = vane.AudioFile(str(path))
    with _connect("audio") as con:
        profiles = con.execute(
            "SELECT native_audio_resample_profile(f, 16000) FROM (VALUES ($1), (NULL::AUDIOFILE), ($1)) input(f)",
            [value],
        ).fetchall()
        assert profiles[1] == (None,)
        for ordinal, (profile,) in enumerate((profiles[0], profiles[2]), start=1):
            assert profile["decoded_frames"] == 4800
            assert profile["output_frames"] == 1600
            assert profile["output_bytes"] == 1600 * 2 * 8
            assert profile["buffer_capacity_bytes"] >= ordinal * profile["output_bytes"]
            assert profile["codec_version"] > 0 and profile["resampler_version"] > 0
            assert profile["file_bytes_read"] > 0 and profile["file_read_calls"] > 0
            for name, seconds in profile.items():
                if name.endswith("_seconds"):
                    assert math.isfinite(seconds) and seconds >= 0
        normal = con.execute("SELECT audio_resample($1, 16000)", [value]).fetchone()[0]
        assert len(normal["samples"]) * 8 == profiles[0][0]["output_bytes"]
        assert con.execute("SELECT native_audio_resample_profile($1, NULL)", [value]).fetchone() == (None,)


@pytest.mark.parametrize("function", ["audio_resample", "native_audio_resample_profile"])
@pytest.mark.parametrize("limit", range(5))
def test_native_audio_output_and_profile_share_limits(tmp_path, function, limit):
    path = tmp_path / "audio.wav"
    path.write_bytes(_wav())
    limits = [len(path.read_bytes()), 800, 800 * 2 * 8, 1600, 1600 * 2 * 8]
    limits[limit] -= 1
    with _connect("audio") as con:
        with pytest.raises(vane.OutOfRangeException):
            con.execute(f"SELECT {function}(audio_file(?), 16000, ?, ?, ?, ?, ?)", [str(path), *limits]).fetchone()


def test_native_audio_profile_counts_http_bytes_inside_the_view():
    payload = _wav(48000, 48000)
    prefix = b"outside FILE window" * 11
    server, thread, handler = _start_object_server(prefix + payload + b"outside suffix")
    served = []
    original = handler._send_object

    def record(self, include_body):
        if include_body:
            bounds = self.headers.get("Range")
            assert bounds is not None
            start, end = map(int, bounds.removeprefix("bytes=").split("-"))
            assert len(prefix) <= start <= end < len(prefix) + len(payload)
            served.append(end - start + 1)
        original(self, include_body)

    handler._send_object = record
    try:
        file = vane.AudioFile(
            f"http://127.0.0.1:{server.server_port}/bucket/object.bin", "audio/wav", len(prefix), len(payload)
        )
        with _connect("audio") as con:
            result = con.execute("SELECT native_audio_resample_profile($1, 16000)", [file]).fetchone()[0]
            assert result["file_bytes_read"] == sum(served)
            assert result["file_read_calls"] == len(served)
            assert result["output_frames"] == 16000
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "domain,payload,function",
    [
        ("image", _png, "decode_image_file($1, NULL, 'null')"),
        ("audio", _wav, "audio_resample($1, 16000)"),
        ("audio", _wav, "native_audio_resample_profile($1, 16000)"),
    ],
)
def test_native_media_cancellation_during_io_is_not_suppressed(domain, payload, function):
    con = _connect(domain)
    server, server_thread, handler = _start_object_server(payload())
    handler.block_reads = True
    errors = []
    file_type = vane.ImageFile if domain == "image" else vane.AudioFile
    file = file_type(f"http://127.0.0.1:{server.server_port}/bucket/object.bin")

    def query():
        try:
            con.execute(f"SELECT {function}", [file]).fetchone()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=query, daemon=True)
    try:
        worker.start()
        assert handler.read_started.wait(timeout=5)
        con.interrupt()
        handler.release_read.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], vane.InterruptException)
        assert con.execute("SELECT 1").fetchone() == (1,)
    finally:
        handler.release_read.set()
        worker.join(timeout=5)
        con.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


@pytest.mark.parametrize(
    "domain,operation,payload", [("image", "image_decode", _png), ("audio", "audio_resample", _wav)]
)
def test_media_benchmark_observes_http_and_spool_costs(tmp_path, domain, operation, payload):
    source = tmp_path / "input.bin"
    source.write_bytes(payload())
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_native_media.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            operation,
            str(source),
            "--extension",
            str(_artifact(domain)),
            "--rows",
            "2",
            "--concurrency",
            "2",
            "--repetitions",
            "1",
            "--transport",
            "http",
            "--diagnostics",
            "--allow-unsigned-development-artifact",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["aggregate_results_match"]
    assert result["concurrency"] == 2
    for backend in ("python", "native"):
        traffic = result["samples"][backend][0]["http"]
        assert traffic["body_bytes"] > 0 and traffic["get_requests"] > 0
        assert traffic["errors"] == 0 and traffic["active_requests"] == 0
    assert result["diagnostics"]["python"]["python_temporary_files"]["written_bytes"] > 0
    assert result["diagnostics"]["native"]["python_temporary_files"]["files"] == 0
    if domain == "audio":
        profiles = result["diagnostics"]["native"]["audio_profiles"]
        assert len(profiles) == 4
        assert (
            sum(profile["file_bytes_read"] for profile in profiles)
            == result["samples"]["native"][0]["http"]["body_bytes"]
        )
