# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import math
import os
import struct
import subprocess
import sys
import wave
import zlib
from pathlib import Path

import pytest

import vane
from vane.datasource.video_reader import VideoFrameSource


def _artifact(domain: str) -> Path:
    variable = f"VANE_TEST_NATIVE_{domain.upper()}_EXTENSION"
    path = os.environ.get(variable)
    if not path:
        pytest.skip(f"set {variable} to test the optional {domain} artifact")
    result = Path(path).resolve()
    assert result.is_file(), result
    return result


def _connect(domain: str):
    artifact = _artifact(domain)
    connection = vane.connect(config={"allow_unsigned_extensions": "true"})
    connection.load_extension(str(artifact))
    connection.execute(f"SET {domain}_backend='native'")
    return connection


def _png(width: int = 5, height: int = 3) -> bytes:
    def chunk(name: bytes, content: bytes) -> bytes:
        return struct.pack(">I", len(content)) + name + content + struct.pack(">I", zlib.crc32(name + content))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = (b"\0" + bytes((20, 80, 160)) * width) * height
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


def _wav(frames: int = 800, sample_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((2, 2, sample_rate, frames, "NONE", "not compressed"))
        output.writeframes(
            b"".join(
                struct.pack("<hh", int(12000 * math.sin(i / 8)), -int(12000 * math.sin(i / 8))) for i in range(frames)
            )
        )
    return buffer.getvalue()


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(_png())
    return path


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(_wav())
    return path


@pytest.fixture
def video_path(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "yuv420p"
        stream.codec_context.gop_size = 3
        stream.codec_context.max_b_frames = 2
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((12, 16, 3), index * 15, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


@pytest.mark.parametrize(
    "domain,function", [("image", "image_file_metadata"), ("audio", "audio_metadata"), ("video", "video_metadata")]
)
def test_backend_is_explicit_and_requires_matching_extension(domain, function):
    with vane.connect() as con:
        assert con.execute(f"SELECT current_setting('{domain}_backend')").fetchone() == ("python",)
        with pytest.raises(vane.InvalidInputException, match="python.*native"):
            con.execute(f"SET {domain}_backend='automatic'")
        con.execute(f"SET {domain}_backend='native'")
        with pytest.raises(vane.BinderException, match=f"requires the {domain} extension"):
            con.sql(f"SELECT {function}({domain}_file('unopened://file'))")


@pytest.mark.parametrize("domain", ["image", "audio", "video"])
def test_backend_connection_configuration_is_validated(domain):
    option = f"{domain}_backend"
    for backend in ("python", "native"):
        with vane.connect(config={option: backend}) as con:
            assert con.execute(f"SELECT current_setting('{option}')").fetchone() == (backend,)
    for invalid in ("automatic", None):
        with pytest.raises(vane.InvalidInputException, match="python.*native"):
            vane.connect(config={option: invalid})


def test_image_native_decode_metadata_nulls_and_backend_switch(image_path, monkeypatch):
    import vane._image_file as helper

    with _connect("image") as con:
        monkeypatch.setattr(
            helper, "_probe_image_metadata", lambda *a, **kw: pytest.fail("Python metadata helper called")
        )
        monkeypatch.setattr(helper, "_decode_image_stream", lambda *a, **kw: pytest.fail("Python decode helper called"))
        assert con.execute("SELECT image_file_metadata(image_file(?))", [str(image_path)]).fetchone()[0] == {
            "width": 5,
            "height": 3,
            "format": "PNG",
            "mode": "RGB",
        }
        rows = con.execute(
            "SELECT (decode_image_file(image_file(url), 'RGB')).data FROM (VALUES (?), (NULL), (?)) t(url)",
            [str(image_path), str(image_path)],
        ).fetchall()
        assert rows == [(bytes((20, 80, 160)) * 15,), (None,), (bytes((20, 80, 160)) * 15,)]
        quoted_path = str(image_path).replace("'", "''")
        query = f"SELECT image_file_metadata(image_file(url)) FROM (SELECT '{quoted_path}' AS url FROM range(1))"
        native_plan = con.sql(query)
        assert "native_image_file_metadata" in native_plan.explain()
        con.execute(f"PREPARE native_metadata AS SELECT image_file_metadata(image_file('{quoted_path}'))")
        con.execute("SET image_backend='python'")
        assert con.execute("EXECUTE native_metadata").fetchone()[0]["width"] == 5
        assert "native_image_file_metadata" not in con.sql(query).explain()
        assert con.execute("SELECT current_setting('audio_backend'), current_setting('video_backend')").fetchone() == (
            "python",
            "python",
        )


@pytest.mark.parametrize("mode,channels", [("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4)])
def test_native_image_modes(image_path, mode, channels):
    with _connect("image") as con:
        result = con.execute("SELECT (decode_image_file(image_file(?), ?)).data", [str(image_path), mode]).fetchone()[0]
        assert len(result) == 15 * channels


@pytest.mark.parametrize(
    "domain,fixture,function,field,expected",
    [
        ("image", "image_path", "image_file_metadata", "width", 5),
        ("audio", "audio_path", "audio_metadata", "sample_rate", 8000),
        ("video", "video_path", "video_metadata", "width", 16),
    ],
)
def test_native_generic_mime_declarations(domain, fixture, function, field, expected, request):
    path = request.getfixturevalue(fixture)
    query = f"SELECT {function}({domain}_file(file(?, ?, NULL, NULL, NULL)))"
    with _connect(domain) as con:
        for declared in (
            "application/octet-stream",
            "binary/octet-stream",
            f"{domain}/*",
            f" {domain.upper()}/*; hint=1 ",
        ):
            assert con.execute(query, [str(path), declared]).fetchone()[0][field] == expected
        with pytest.raises(vane.InvalidInputException, match="content_type"):
            con.execute(query, [str(path), "text/*"])


def test_native_image_mime_alias(image_path):
    with _connect("image") as con:
        query = "image_file(file(?, 'image/x-png', NULL, NULL, NULL))"
        assert con.execute(f"SELECT image_file_metadata({query})", [str(image_path)]).fetchone()[0]["width"] == 5
        assert con.execute(f"SELECT (decode_image_file({query})).data", [str(image_path)]).fetchone() == (
            bytes((20, 80, 160)) * 15,
        )


@pytest.mark.parametrize(
    "format,subtype,declared",
    [
        ("MP3", "MPEG_LAYER_III", "audio/mp3"),
        ("MP3", "MPEG_LAYER_III", "audio/x-mp3"),
        ("OGG", "VORBIS", "application/ogg"),
        ("AIFF", "PCM_16", "audio/aif"),
    ],
)
def test_native_audio_mime_aliases(tmp_path, format, subtype, declared):
    np = pytest.importorskip("numpy")
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / ("audio." + format.lower())
    soundfile.write(path, np.zeros((3200, 2)), 8000, format=format, subtype=subtype)
    with _connect("audio") as con:
        value = "audio_file(file(?, ?, NULL, NULL, NULL))"
        metadata = con.execute(f"SELECT audio_metadata({value})", [str(path), declared]).fetchone()[0]
        assert (metadata["sample_rate"], metadata["channels"]) == (8000, 2)
        result = con.execute(f"SELECT audio_resample({value}, 16000)", [str(path), declared]).fetchone()[0]
        assert result["sample_rate"] == 16000 and result["frames"] > 0


@pytest.mark.parametrize("container,declared", [("mp4", "video/x-m4v"), ("matroska", "video/mkv")])
def test_native_video_mime_aliases(tmp_path, video_path, container, declared):
    av = pytest.importorskip("av")
    path = tmp_path / "remuxed.bin"
    with av.open(str(video_path)) as source, av.open(str(path), "w", format=container) as output:
        stream = output.add_stream_from_template(source.streams.video[0])
        for packet in source.demux(video=0):
            if packet.dts is not None:
                packet.stream = stream
                output.mux(packet)
    with _connect("video") as con:
        metadata = con.execute(
            "SELECT video_metadata(video_file(file(?, ?, NULL, NULL, NULL)))", [str(path), declared]
        ).fetchone()[0]
        assert (metadata["width"], metadata["height"]) == (16, 12)
        source = VideoFrameSource([vane.VideoFile(str(path), declared)], width=8, height=6)
        assert con.from_datasource(source).count("*").fetchone() == (12,)


@pytest.mark.parametrize("domain,fixture", [("audio", "audio_path"), ("video", "video_path")])
def test_native_metadata_read_budgets(domain, fixture, request):
    path = request.getfixturevalue(fixture)
    query = f"SELECT {domain}_metadata({domain}_file(?), ?)"
    with _connect(domain) as con:
        with pytest.raises(vane.OutOfRangeException, match="read/probe byte budget"):
            con.execute(query, [str(path), 1])
        assert con.execute(query, [str(path), 64 * 1024 * 1024]).fetchone()[0] is not None
        with pytest.raises(vane.InvalidInputException, match="max_bytes"):
            con.execute(query, [str(path), 64 * 1024 * 1024 + 1])


def test_native_image_errors_and_limits(tmp_path, image_path):
    broken = tmp_path / "bad.png"
    broken.write_bytes(b"not an image")
    with _connect("image") as con:
        assert con.execute("SELECT decode_image_file(image_file(?), NULL, 'null')", [str(broken)]).fetchone() == (None,)
        with pytest.raises(vane.InvalidInputException, match="native"):
            con.execute("SELECT decode_image_file(image_file(?))", [str(broken)])
        with pytest.raises(Exception, match="does not exist|No such file|Cannot open"):
            con.execute("SELECT decode_image_file(image_file(?), NULL, 'null')", [str(tmp_path / "absent.png")])
        with pytest.raises(vane.OutOfRangeException, match="pixels"):
            con.execute("SELECT decode_image_file(image_file(?), NULL, 'null', 1024, 1, 1024)", [str(image_path)])
        with pytest.raises(vane.OutOfRangeException, match="decoded frame bytes"):
            con.execute("SELECT decode_image_file(image_file(?), NULL, 'null', 1024, 100, 16)", [str(image_path)])
        with pytest.raises(vane.OutOfRangeException, match="max_bytes"):
            con.execute("SELECT image_file_metadata(image_file(?), 8, 100)", [str(image_path)])
        with pytest.raises(vane.InvalidInputException, match="content_type"):
            con.execute(
                "SELECT decode_image_file(image_file(file(?, 'image/jpeg', NULL, NULL, NULL)))", [str(image_path)]
            )


@pytest.mark.parametrize(
    "domain,mime,function,payload",
    [
        ("image", "image/png", "image_file_metadata", _png),
        ("audio", "audio/wav", "audio_metadata", _wav),
    ],
)
def test_native_uses_exact_file_window(tmp_path, domain, mime, function, payload):
    encoded = payload()
    prefix = b"unrelated prefix" * 20
    path = tmp_path / "bundle.bin"
    path.write_bytes(prefix + encoded + b"unrelated suffix" * 20)
    with _connect(domain) as con:
        value = con.execute(
            f"SELECT {function}({domain}_file(file(?, ?, ?, ?, NULL)))", [str(path), mime, len(prefix), len(encoded)]
        ).fetchone()[0]
        assert value is not None
        if domain == "image":
            assert con.execute(
                "SELECT (decode_image_file(image_file(file(?, ?, ?, ?, NULL)))).data",
                [str(path), mime, len(prefix), len(encoded)],
            ).fetchone() == (bytes((20, 80, 160)) * 15,)
        with pytest.raises(vane.InvalidInputException):
            con.execute(f"SELECT {function}({domain}_file(file(?, ?, ?, ?, NULL)))", [str(path), mime, len(prefix), 8])


def test_native_audio_resamples_without_python_helpers(audio_path, monkeypatch):
    import vane._audio_file as helper

    with _connect("audio") as con:
        monkeypatch.setattr(helper, "_probe_audio_metadata", lambda *a, **kw: pytest.fail("Python probe called"))
        monkeypatch.setattr(helper, "_resample_audio_stream", lambda *a, **kw: pytest.fail("Python resampler called"))
        metadata = con.execute("SELECT audio_metadata(audio_file(?))", [str(audio_path)]).fetchone()[0]
        assert (metadata["sample_rate"], metadata["channels"], metadata["frames"]) == (8000, 2, 800)
        result = con.execute("SELECT audio_resample(audio_file(?), 16000)", [str(audio_path)]).fetchone()[0]
        assert (result["sample_rate"], result["frames"], result["channels"]) == (16000, 1600, 2)
        assert len(result["samples"]) == 3200
        assert max(abs(a + b) for a, b in zip(result["samples"][::2], result["samples"][1::2])) < 1e-10
        assert con.execute("SELECT audio_resample(NULL::AUDIOFILE, 16000)").fetchone() == (None,)
        with pytest.raises(vane.OutOfRangeException, match="max_frames"):
            con.execute(
                "SELECT audio_resample(audio_file(?), 16000, 100000, 10, 100000, 100000, 100000)", [str(audio_path)]
            )


def test_native_video_metadata_and_streamed_frames(video_path, monkeypatch):
    import vane._video_file as helper
    from vane.datasource import read_datasource

    with _connect("video") as con:
        monkeypatch.setattr(helper, "_probe_video_metadata", lambda *a, **kw: pytest.fail("Python video probe called"))
        monkeypatch.setattr(vane.VideoFile, "frames", lambda *a, **kw: pytest.fail("Python frame decoder called"))
        metadata = con.execute("SELECT video_metadata(video_file(?))", [str(video_path)]).fetchone()[0]
        assert (metadata["width"], metadata["height"], metadata["frame_count"]) == (16, 12, 12)
        source = VideoFrameSource(
            [str(video_path)], width=8, height=6, start_time=0.5, end_time=2, sample_interval_seconds=0.5
        )
        for relation in (read_datasource(source, con=con), con.from_datasource(source)):
            assert "NATIVE_VIDEO_TENSOR_FRAMES" in relation.explain().upper()
            rows = relation.project("file.url, frame_index, frame_time, frame_time_base_denominator, frame").fetchall()
            assert [(r[1], r[2]) for r in rows] == [(2, 0.5), (4, 1.0), (6, 1.5), (8, 2.0)]
            assert all(r[0] == str(video_path) and r[3] > 0 and len(r[4]) == 144 for r in rows)
        limited = VideoFrameSource([str(video_path)] * 2, width=8, height=6, frame_limit=3)
        assert con.from_datasource(limited).project("frame_index").fetchall() == [(0,), (1,), (2,)]
        image_relation = con.table_function("native_video_frames", source._native_parameters())
        assert image_relation.project("frame.width, frame.height, frame.channels").fetchall() == [(8, 6, 3)] * 4


def test_native_video_empty_and_format_error_policy(tmp_path, video_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"invalid")
    with _connect("video") as con:
        assert con.from_datasource(VideoFrameSource([], width=8, height=6)).fetchall() == []
        source = VideoFrameSource([str(broken), str(video_path)], width=8, height=6, on_error="skip", frame_limit=2)
        assert con.from_datasource(source).project("frame_index").fetchall() == [(0,), (1,)]
        limited = VideoFrameSource([str(video_path)], width=8, height=6, max_decoded_frames=1, on_error="skip")
        with pytest.raises(vane.OutOfRangeException, match="max_decoded_frames"):
            con.from_datasource(limited).fetchall()


@pytest.mark.parametrize("on_error", ["raise", "skip"])
@pytest.mark.parametrize("budget,message", [(1, "video output frame bytes"), (144, "row exceeds max_partition_bytes")])
def test_native_video_partition_limit_is_hard(video_path, on_error, budget, message):
    with _connect("video") as con:
        source = VideoFrameSource([str(video_path)], width=8, height=6, max_partition_bytes=budget, on_error=on_error)
        with pytest.raises(vane.OutOfRangeException, match=message):
            con.from_datasource(source)


def test_native_video_rejects_subclasses_without_bypassing_custom_tasks(video_path):
    pa = pytest.importorskip("pyarrow")
    from vane.datasource import DataSourceTask

    class CustomTask(DataSourceTask):
        def execute(self):
            yield pa.record_batch([pa.array([42], type=pa.int64())], names=["custom_value"])

    class CustomVideoSource(VideoFrameSource):
        @property
        def schema(self):
            return {"custom_value": "BIGINT"}

        def get_tasks(self):
            yield CustomTask()

    source = CustomVideoSource([str(video_path)], width=8, height=6)
    with _connect("video") as con:
        with pytest.raises(vane.InvalidInputException, match="built-in VideoFrameSource, not subclasses"):
            con.from_datasource(source)
        con.execute("SET video_backend='python'")
        relation = con.from_datasource(source)
        assert relation.columns == ["custom_value"]
        assert relation.fetchall() == [(42,)]


@pytest.mark.parametrize("task_count", [None, 1, 2, 20])
def test_native_video_grouped_local_scan(video_path, task_count):
    with _connect("video") as con:
        con.execute("SET threads=4")
        source = VideoFrameSource([str(video_path)] * 5, width=8, height=6, read_task_count=task_count)
        rows = con.from_datasource(source).project("file.url, frame_index").fetchall()
        assert len(rows) == 60
        assert sorted(index for _, index in rows) == sorted(list(range(12)) * 5)
        assert all(url == str(video_path) for url, _ in rows)


@pytest.mark.parametrize(
    "domain,function,fixture",
    [
        ("image", "decode_image_file", "image_path"),
        ("audio", "audio_resample", "audio_path"),
        ("video", "video_metadata", "video_path"),
    ],
)
def test_native_does_not_import_python_codec_packages(domain, function, fixture, request):
    path = request.getfixturevalue(fixture)
    suffix = ", 16000" if domain == "audio" else ""
    script = """
import importlib.abc, sys
class BlockCodecs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PIL', 'av', 'soundfile', 'soxr'}:
            raise AssertionError('native execution imported ' + fullname)
sys.meta_path.insert(0, BlockCodecs())
import vane
with vane.connect(config={'allow_unsigned_extensions': 'true'}) as con:
    con.load_extension(sys.argv[1])
    con.execute('SET ' + sys.argv[2] + "_backend='native'")
    con.execute(sys.argv[3], [sys.argv[4]]).fetchall()
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(_artifact(domain)),
            domain,
            f"SELECT {function}({domain}_file(?){suffix})",
            str(path),
        ],
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize("format,mode", [("JPEG", "RGB"), ("JPEG", "L"), ("PNG", "RGBA")])
def test_native_encoded_modes_and_jpeg_headers(tmp_path, format, mode):
    image = pytest.importorskip("PIL.Image")
    color = 80 if mode == "L" else (20, 80, 160, 78) if mode == "RGBA" else (20, 80, 160)
    path = tmp_path / ("picture." + format.lower())
    image.new(mode, (5, 3), color).save(path, format=format)
    with _connect("image") as con:
        result = con.execute("SELECT image_file_metadata(image_file(?))", [str(path)]).fetchone()[0]
        assert (result["width"], result["height"], result["format"], result["mode"]) == (5, 3, format, mode)
        pixels, channels = con.execute(
            "SELECT decoded.data, decoded.channels FROM (SELECT decode_image_file(image_file(?)) AS decoded)",
            [str(path)],
        ).fetchone()
        assert len(pixels) == 15 * channels
        if mode == "RGBA":
            assert pixels == bytes(color) * 15
        else:
            expected = [color] if mode == "L" else color
            assert max(abs(actual - target) for actual, target in zip(pixels[:channels], expected)) <= 3


def test_native_file_adapter_preserves_registered_filesystem_rejection():
    fsspec = pytest.importorskip("fsspec")

    class GuardedFS(fsspec.AbstractFileSystem):
        protocol = "nativeguard"
        cachable = False

        def __init__(self):
            super().__init__()
            self.opens = 0

        def _open(self, path, mode="rb", **kwargs):
            self.opens += 1
            pytest.fail("Native FILE resolver attempted unsupported Python filesystem I/O")

    fs = GuardedFS()
    with _connect("image") as con:
        con.register_filesystem(fs)
        with pytest.raises(vane.NotImplementedException, match="Nonblocking opens are not supported"):
            con.execute("SELECT decode_image_file(image_file('nativeguard://image'), NULL, 'null')")
        assert fs.opens == 0
