# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import io
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

import vane
from vane import _video_file


def _encoded_video(
    container_format: str = "mp4",
    *,
    width: int = 16,
    height: int = 12,
    frame_count: int = 4,
    frame_rate: int = 24,
) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format) as container:
        stream = container.add_stream("mpeg4", rate=frame_rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            pixels = np.full((height, width, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def test_video_metadata_sql_and_python_value(duckdb_cursor, tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")

    result_type, metadata, null_metadata = duckdb_cursor.execute(
        """
        SELECT
            typeof(video_metadata($1)),
            video_metadata($1),
            video_metadata(NULL::VIDEOFILE)
        """,
        [value],
    ).fetchone()

    assert result_type == (
        "STRUCT(width UINTEGER, height UINTEGER, fps DOUBLE, duration DOUBLE, frame_count BIGINT, "
        "time_base STRUCT(numerator BIGINT, denominator BIGINT))"
    )
    assert metadata["width"] == 16
    assert metadata["height"] == 12
    assert metadata["fps"] == pytest.approx(24)
    assert metadata["duration"] == pytest.approx(4 / 24)
    assert metadata["frame_count"] == 4
    assert metadata["time_base"]["numerator"] > 0
    assert metadata["time_base"]["denominator"] > 0
    assert null_metadata is None

    value_metadata = value.metadata(connection=duckdb_cursor)
    assert value_metadata.width == 16
    assert value_metadata.height == 12
    assert value_metadata.fps == pytest.approx(24)
    assert value_metadata.duration == pytest.approx(4 / 24)
    assert value_metadata.frame_count == 4
    assert value_metadata.time_base == Fraction(
        metadata["time_base"]["numerator"],
        metadata["time_base"]["denominator"],
    )


def test_video_metadata_facades(duckdb_cursor, tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video(width=20, height=14, frame_count=3, frame_rate=30))
    value = vane.VideoFile(str(path), "video/mp4")

    function_result = duckdb_cursor.sql("SELECT 1").select(vane.video_metadata(value, max_bytes=4096)).fetchone()[0]
    method_result = duckdb_cursor.sql("SELECT 1").select(vane.video_file(value).video_metadata()).fetchone()[0]

    assert function_result == method_result
    assert function_result["width"] == 20
    assert function_result["height"] == 14
    assert function_result["fps"] == pytest.approx(30)


def test_video_metadata_honors_logical_range(duckdb_cursor, tmp_path):
    payload = _encoded_video(width=18, height=10)
    prefix = b"not-a-video-prefix"
    suffix = b"not-a-video-suffix"
    path = tmp_path / "ranged.bin"
    path.write_bytes(prefix + payload + suffix)
    value = vane.VideoFile(str(path), "video/mp4", len(prefix), len(payload))

    sql_metadata = duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()[0]
    value_metadata = value.metadata(connection=duckdb_cursor)

    assert (sql_metadata["width"], sql_metadata["height"]) == (18, 10)
    assert (value_metadata.width, value_metadata.height) == (18, 10)


def test_video_metadata_preserves_unknown_frame_count(duckdb_cursor, tmp_path):
    path = tmp_path / "video.mkv"
    path.write_bytes(_encoded_video("matroska", frame_count=5))
    value = vane.VideoFile(str(path), "video/x-matroska")

    metadata = value.metadata(connection=duckdb_cursor)
    sql_metadata = duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()[0]

    # FFmpeg reports zero when Matroska does not carry an exact count. Vane
    # keeps that unknown instead of estimating duration * fps.
    assert metadata.frame_count is None
    assert sql_metadata["frame_count"] is None
    assert metadata.duration == pytest.approx(5 / 24, rel=0.01)


def test_video_metadata_treats_zero_parser_sentinels_as_unknown():
    video = SimpleNamespace(
        type="video",
        width=16,
        height=12,
        time_base=Fraction(1, 1000),
        average_rate=Fraction(0, 1),
        guessed_rate=Fraction(30, 1),
        duration=0,
        frames=0,
    )
    container = SimpleNamespace(
        streams=[video],
        format=SimpleNamespace(name="matroska,webm"),
        duration=2_000_000,
    )
    av_module = SimpleNamespace(time_base=1_000_000)

    metadata = _video_file._metadata_from_container(container, "video/webm", av_module)

    assert metadata.fps == 30
    assert metadata.duration == 2
    assert metadata.frame_count is None

    video.guessed_rate = Fraction(0, 1)
    container.duration = 0
    unknown_metadata = _video_file._metadata_from_container(container, "video/webm", av_module)
    assert unknown_metadata.fps is None
    assert unknown_metadata.duration is None


def test_video_metadata_rejects_missing_container_format():
    video = SimpleNamespace(
        type="video",
        width=16,
        height=12,
        time_base=Fraction(1, 1000),
        average_rate=None,
        guessed_rate=None,
        duration=None,
        frames=None,
    )
    container = SimpleNamespace(streams=[video], format=None, duration=None)

    with pytest.raises(vane.VideoFileFormatError, match="did not report a container format"):
        _video_file._metadata_from_container(container, None, SimpleNamespace(time_base=1_000_000))


def test_video_metadata_budget_is_enforced(duckdb_cursor, tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")

    with pytest.raises(vane.VideoFileLimitError, match="max_bytes=16"):
        value.metadata(max_bytes=16, connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="max_bytes=16"):
        duckdb_cursor.execute("SELECT video_metadata($1, 16)", [value]).fetchone()


@pytest.mark.parametrize(
    "failure",
    [
        MemoryError("allocation failed"),
        RuntimeError("parser invariant failed"),
        _video_file.VideoFileFormatError("classified media failure"),
    ],
    ids=["memory", "internal", "classified-media"],
)
def test_video_metadata_probe_preserves_non_parser_errors_after_budget_exhaustion(monkeypatch, failure):
    class FakeFFmpegError(Exception):
        pass

    def failing_open(stream, **kwargs):
        del kwargs
        assert stream.read(1) == b"x"
        assert stream.read(1) == b""
        raise failure

    fake_av = SimpleNamespace(
        open=failing_open,
        error=SimpleNamespace(FFmpegError=FakeFFmpegError),
        time_base=1_000_000,
    )
    monkeypatch.setattr(_video_file, "_load_av", lambda: fake_av)

    with pytest.raises(type(failure)) as raised:
        _video_file._probe_video_metadata(
            lambda offset, size: b"x" * size,
            logical_size=2,
            content_type=None,
            max_bytes=1,
        )
    assert raised.value is failure


@pytest.mark.parametrize("content_type", ["audio/mp4", "image/png", "video/x-msvideo"])
def test_video_metadata_rejects_contradictory_mime(duckdb_cursor, tmp_path, content_type):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), content_type)

    with pytest.raises(vane.VideoFileFormatError, match="content_type"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="content_type"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


@pytest.mark.parametrize(
    ("container_format", "content_type"),
    [
        ("mp4", "video/quicktime"),
        ("mp4", "video/x-m4v"),
        ("mp4", 'video/mp4; codecs="mp4v.20.9"'),
        ("mp4", "application/octet-stream"),
        ("mp4", "video/*"),
        ("matroska", "video/mkv"),
    ],
)
def test_video_metadata_accepts_compatible_and_generic_mimes(
    duckdb_cursor,
    tmp_path,
    container_format,
    content_type,
):
    path = tmp_path / f"video.{container_format}"
    path.write_bytes(_encoded_video(container_format))
    value = vane.VideoFile(str(path), content_type)

    assert value.metadata(connection=duckdb_cursor).width == 16
    assert duckdb_cursor.execute("SELECT (video_metadata($1)).width", [value]).fetchone()[0] == 16


def test_video_metadata_requires_a_video_stream(duckdb_cursor, tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00"
        b"\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    value = vane.VideoFile(str(path), "video/*")

    with pytest.raises(vane.VideoFileFormatError, match="does not contain a video stream"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="does not contain a video stream"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_file_classifies_invalid_media_but_propagates_io(duckdb_cursor, tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video file")
    corrupt_value = vane.VideoFile(str(corrupt), "video/mp4")

    with pytest.raises(vane.VideoFileFormatError, match="supported encoded video"):
        corrupt_value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="supported encoded video"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [corrupt_value]).fetchone()

    missing = vane.VideoFile(str(tmp_path / "missing.mp4"), "video/mp4")
    with pytest.raises(vane.IOException):
        missing.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.IOException):
        duckdb_cursor.execute("SELECT video_metadata($1)", [missing]).fetchone()


def test_video_metadata_propagates_reader_failures_from_virtual_io(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")

    def fail_read(self, size=-1):
        raise OSError("connector read failed")

    monkeypatch.setattr(vane.VaneFileReader, "read", fail_read)
    with pytest.raises(OSError, match="connector read failed"):
        value.metadata(connection=duckdb_cursor)


def test_video_metadata_requires_videofile(duckdb_cursor):
    with pytest.raises(vane.BinderException, match="requires VIDEOFILE, not FILE"):
        duckdb_cursor.sql("SELECT video_metadata(file('memory://generic', NULL, NULL, NULL, NULL))")
    with pytest.raises(vane.BinderException, match="requires VIDEOFILE, not AUDIOFILE"):
        duckdb_cursor.sql("SELECT video_metadata(audio_file('memory://audio'))")


@pytest.mark.parametrize(
    ("max_bytes", "error_type", "message"),
    [
        (True, TypeError, "max_bytes must be int"),
        (0, ValueError, "greater than zero"),
        (64 * 1024 * 1024 + 1, ValueError, "at most"),
    ],
)
def test_video_file_python_argument_validation(max_bytes, error_type, message):
    value = vane.VideoFile("memory://not-opened")

    with pytest.raises(error_type, match=message):
        value.metadata(max_bytes=max_bytes)


def test_video_file_optional_dependency_is_lazy(monkeypatch):
    original_import = importlib.import_module

    def fail_av(name, package=None):
        if name == "av":
            raise ImportError("missing av")
        return original_import(name, package)

    monkeypatch.setattr(_video_file.importlib, "import_module", fail_av)

    value = vane.VideoFile("memory://not-opened")
    with pytest.raises(ImportError, match=r"vane-ai\[video\]"):
        value.metadata()


def test_video_file_unusable_native_dependency_is_actionable(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")
    original_import = importlib.import_module

    def fail_ffmpeg(name, package=None):
        if name == "av":
            raise OSError("FFmpeg libraries cannot be loaded")
        return original_import(name, package)

    monkeypatch.setattr(_video_file.importlib, "import_module", fail_ffmpeg)

    with pytest.raises(ImportError, match=r"bundled FFmpeg libraries.*vane-ai\[video\]"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match=r"bundled FFmpeg libraries.*vane-ai\[video\]"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_sql_preflights_dependency_before_opening_file(duckdb_cursor, tmp_path, monkeypatch):
    missing = vane.VideoFile(str(tmp_path / "missing.mp4"), "video/mp4")

    def fail_av():
        raise ImportError("install vane-ai[video]")

    monkeypatch.setattr(_video_file, "_load_av", fail_av)

    with pytest.raises(vane.InvalidInputException, match=r"install vane-ai\[video\]"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [missing]).fetchone()


def test_video_metadata_sql_maps_non_exception_control_flow_to_interrupt(duckdb_cursor, tmp_path, monkeypatch):
    class StopVideoMetadata(BaseException):
        pass

    path = tmp_path / "video.bin"
    path.write_bytes(b"video")
    value = vane.VideoFile(str(path))

    def stop_metadata(*args, **kwargs):
        raise StopVideoMetadata

    monkeypatch.setattr(_video_file, "_probe_video_metadata", stop_metadata)

    with pytest.raises(vane.InterruptException):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_sql_prioritizes_connection_interrupt_over_probe_error(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "video.bin"
    path.write_bytes(b"video")
    value = vane.VideoFile(str(path))

    def interrupt_then_fail(*args, **kwargs):
        duckdb_cursor.interrupt()
        raise _video_file.VideoFileFormatError("competing format failure")

    monkeypatch.setattr(_video_file, "_probe_video_metadata", interrupt_then_fail)

    with pytest.raises(vane.InterruptException):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_sql_maps_python_memory_error_to_out_of_memory(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "video.bin"
    path.write_bytes(b"video")
    value = vane.VideoFile(str(path))

    def exhaust_memory(*args, **kwargs):
        raise MemoryError("allocation failed")

    monkeypatch.setattr(_video_file, "_probe_video_metadata", exhaust_memory)

    with pytest.raises(vane.OutOfMemoryException, match="ran out of memory"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_view_reuses_cached_ranges():
    payload = bytes(range(64))
    requests = []

    def read_at(offset, size):
        requests.append((offset, size))
        return payload[offset : offset + size]

    stream = _video_file._VideoMetadataView(read_at, logical_size=len(payload), max_bytes=64)
    assert stream.read(8) == payload[:8]
    stream.seek(2)
    assert stream.read(4) == payload[2:6]
    stream.seek(-4, io.SEEK_END)
    assert stream.read() == payload[-4:]
    assert sum(size for _, size in requests) <= 64
    assert len(requests) == 2


def test_video_metadata_view_serves_one_large_request_with_one_source_read():
    payload = b"x" * (7 * 1024 * 1024)
    requests = []

    def read_at(offset, size):
        requests.append((offset, size))
        return payload[offset : offset + size]

    stream = _video_file._VideoMetadataView(read_at, logical_size=len(payload), max_bytes=len(payload))

    assert stream.read(len(payload)) == payload
    assert requests == [(0, len(payload))]


def test_video_metadata_view_preserves_cache_failures(monkeypatch):
    stream = _video_file._VideoMetadataView(lambda offset, size: b"x" * size, logical_size=4, max_bytes=4)

    def fail_cache(offset, data):
        raise RuntimeError("metadata cache insertion failed")

    monkeypatch.setattr(stream, "_cache_bytes", fail_cache)

    assert stream.read(1) == b""
    with pytest.raises(RuntimeError, match="metadata cache insertion failed"):
        stream.raise_if_error()


def test_video_metadata_view_bounds_reverse_adjacent_fetches():
    payload = bytes(range(256)) * 256
    requests = []

    def read_at(offset, size):
        requests.append((offset, size))
        return payload[offset : offset + size]

    stream = _video_file._VideoMetadataView(read_at, logical_size=len(payload), max_bytes=len(payload))
    for offset in range(len(payload) - 1, -1, -1):
        stream.seek(offset)
        assert stream.read(1) == payload[offset : offset + 1]

    assert len(requests) <= len(payload) // stream._fetch_size + 1
    assert not stream.fetch_limit_exhausted


def test_video_metadata_probe_preserves_control_flow_over_stored_reader_error(monkeypatch):
    class FakeFFmpegError(Exception):
        pass

    def interrupting_open(stream, **kwargs):
        del kwargs
        assert stream.read(1) == b""
        raise KeyboardInterrupt

    fake_av = SimpleNamespace(
        open=interrupting_open,
        error=SimpleNamespace(FFmpegError=FakeFFmpegError),
        time_base=1_000_000,
    )
    monkeypatch.setattr(_video_file, "_load_av", lambda: fake_av)

    def fail_read(offset, size):
        del offset, size
        raise OSError("connector read failed")

    with pytest.raises(KeyboardInterrupt):
        _video_file._probe_video_metadata(fail_read, logical_size=2, content_type=None, max_bytes=2)


def test_video_container_cleanup_does_not_replace_primary_error(monkeypatch):
    class FakeContainer:
        def close(self):
            raise RuntimeError("competing close failure")

    def interrupt_metadata(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(_video_file, "_metadata_from_container", interrupt_metadata)

    with pytest.raises(KeyboardInterrupt):
        with _video_file._close_container(FakeContainer()) as container:
            _video_file._metadata_from_container(container, None, None)


def test_video_nested_io_is_rejected():
    blocker = _video_file._NestedIOBlocker()

    with pytest.raises(vane.VideoFileFormatError, match="external resource"):
        blocker("file:///tmp/segment.ts", 0, {})
    with pytest.raises(vane.VideoFileFormatError, match="external resource"):
        blocker.raise_if_error()
