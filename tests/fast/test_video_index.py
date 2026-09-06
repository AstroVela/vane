# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import pytest

import vane
from tests.fast import test_native_media_extensions as media
from tests.fast.test_file_reader import _start_object_server


@pytest.fixture
def native_video():
    with media._connect("video") as con:
        yield con


@pytest.fixture(params=[("mp4", False, 0), ("mp4", True, 120), ("matroska", True, 120)])
def indexed_clip(request, tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    container, variable, origin = request.param
    path = tmp_path / ("clip.mp4" if container == "mp4" else "clip.mkv")
    rng = np.random.default_rng(714)
    with av.open(str(path), "w", format=container) as output:
        stream = output.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        # Keep the fixture larger than container-probe and verified-block overhead.
        stream.bit_rate = 4_000_000
        stream.codec_context.gop_size = 12
        stream.codec_context.max_b_frames = 2
        stream.codec_context.time_base = Fraction(1, 24)
        for index in range(240):
            pixels = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = origin + index + (index // 7 if variable else 0)
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return vane.VideoFile(str(path), "video/mp4" if container == "mp4" else "video/webm")


def _build(con, file):
    return con.execute("SELECT build_video_index($1)", [file]).fetchone()[0]


def test_video_index_construction_is_lazy_and_requires_its_extension():
    expression = vane.build_video_index(vane.VideoFile("unopened://missing"))
    with vane.connect(config={"video_backend": "native"}) as con:
        with pytest.raises(vane.BinderException, match="requires the video extension"):
            con.sql("SELECT 1").select(expression)


def test_video_index_does_not_invoke_python_codecs(native_video, indexed_clip, monkeypatch):
    import vane._read_video_frames as source
    import vane._video_expressions as expressions
    import vane._video_file as helpers

    monkeypatch.setattr(helpers, "_load_av", lambda: pytest.fail("native indexing invoked a Python codec"))
    monkeypatch.setattr(
        expressions, "_scalar_video_frames", lambda *a, **kw: pytest.fail("native selection invoked Python")
    )
    monkeypatch.setattr(source, "_image_video_source", lambda *a, **kw: pytest.fail("native scan invoked Python"))
    index = _build(native_video, indexed_clip)
    native_video.execute("SELECT get_video_frame_by_idx($1, 200, index => $2)", [indexed_clip, index]).fetchone()
    assert (
        len(
            vane.read_video_frames(
                indexed_clip, 6, 8, frame_limit=1, indexes=[index], connection=native_video
            ).fetchall()
        )
        == 1
    )
    native_video.execute("SET enable_external_access=false")
    assert native_video.sql("SELECT 1").select(vane.video_index_info(index)).fetchone()[0]["frame_count"] == 240


def test_video_index_reproduces_exact_native_frames(native_video, indexed_clip):
    con, file = native_video, indexed_clip
    baseline = con.execute("SELECT video_frames($1)", [file]).fetchone()[0]
    index = _build(con, file)
    info = con.execute("SELECT video_index_info($1)", [index]).fetchone()[0]
    assert info["frame_count"] == len(baseline) == 240
    assert baseline[0]["frame_index"] == 0
    with pytest.importorskip("av").open(file.url) as encoded:
        assert info["keyframe_count"] == sum(frame.key_frame for frame in encoded.decode(video=0))
    assert info["index_bytes"] == len(index)
    assert info["build_bytes_read"] >= info["source_bytes"]
    # Check exact ordinals independently as well as complete list metadata.
    assert con.execute("SELECT video_frames($1, index => $2)", [file, index]).fetchone()[0] == baseline
    for target in (0, 1, 11, 12, 13, 199, 200, 239):
        result = con.execute("SELECT get_video_frame_by_idx($1, $2, index => $3)", [file, target, index]).fetchone()[0]
        expected = con.execute("SELECT get_video_frame_by_idx($1, $2)", [file, target]).fetchone()[0]
        assert result == expected
    start, end = baseline[199]["frame_time"], baseline[220]["frame_time"]
    query = "SELECT video_frames($1, start_time => $2, end_time => $3, index => $4)"
    result = con.execute(query, [file, start, end, index]).fetchone()[0]
    assert result == baseline[199:221]
    assert con.execute("SELECT video_keyframes($1, index => $2)", [file, index]).fetchone()[0] == [
        frame["data"] for frame in baseline if frame["is_key_frame"]
    ]


def test_video_index_streaming_and_expression_options_agree(native_video, indexed_clip):
    con, file = native_video, indexed_clip
    index = _build(con, file)
    options = dict(start_time=7, end_time=9, width=48, height=32, sample_interval_seconds=0.4)
    source = con.sql("SELECT $1 AS file, $2 AS frame_index", params=[file, index])
    direct = source.select(vane.video_frames(vane.col("file"), index=vane.col("frame_index"), **options)).fetchone()[0]
    method = source.select(vane.col("file").video_frames(index=vane.col("frame_index"), **options)).fetchone()[0]
    sequential = source.select(vane.video_frames(vane.col("file"), **options)).fetchone()[0]
    assert direct == method == sequential
    rows = (
        vane.read_video_frames(
            file, 32, 48, start_time=7, end_time=9, sample_interval_seconds=0.4, indexes=[index], connection=con
        )
        .order("frame_index")
        .fetchall()
    )
    assert [row[2] for row in rows] == [frame["frame_index"] for frame in direct]
    assert [row[-1] for row in rows] == [frame["data"] for frame in direct]
    assert all(row[1] == file for row in rows)
    assert (
        con.execute(
            "SELECT * FROM read_video_frames($1, 32, 48, start_time => 7, end_time => 9, "
            "sample_interval_seconds => 0.4, indexes => $2) ORDER BY frame_index",
            [file, [index]],
        ).fetchall()
        == rows
    )


def test_video_index_reduces_actual_decode_work(native_video, indexed_clip):
    con, file = native_video, indexed_clip
    index = _build(con, file)
    source = con.sql("SELECT $1 AS file", params=[file])
    sequential = source.select(vane.video_scan_stats(vane.col("file"), idx=200)).fetchone()[0]
    indexed = source.select(vane.video_scan_stats(vane.col("file"), idx=200, index=index)).fetchone()[0]
    assert sequential["decoded_frames"] == 201
    assert 0 < indexed["decoded_frames"] < 40
    assert indexed["seeks"] == 1 and indexed["selected_frames"] == 1
    assert sequential["seeks"] == 0
    assert indexed["bytes_read"] < sequential["bytes_read"]
    con.execute("SELECT get_video_frame_by_idx($1, 200, index => $2, max_decoded_frames => 40)", [file, index])
    with pytest.raises(vane.OutOfRangeException, match="max_decoded_frames"):
        con.execute("SELECT get_video_frame_by_idx($1, 200, max_decoded_frames => 40)", [file])
    with pytest.raises(vane.OutOfRangeException, match="max_decoded_frames"):
        con.execute("SELECT video_frames($1, index => $2, max_decoded_frames => 1, on_error => 'null')", [file, index])


def test_video_index_is_a_persistent_value(native_video, indexed_clip, tmp_path):
    con, file = native_video, indexed_clip
    index = _build(con, file)
    path = tmp_path / "index.parquet"
    con.sql("SELECT $1 AS seek_index", params=[index]).write_parquet(str(path))
    result = (
        con.read_parquet(str(path))
        .select(vane.get_video_frame_by_idx(file, 200, index=vane.col("seek_index")))
        .fetchone()[0]
    )
    assert isinstance(result, vane.Image)
    assert result == con.execute("SELECT get_video_frame_by_idx($1, 200)", [file]).fetchone()[0]


def test_video_index_requires_native_and_retains_python_path(native_video, indexed_clip):
    con, file = native_video, indexed_clip
    index = _build(con, file)
    con.execute("SET video_backend='python'")
    for query in ("SELECT build_video_index($1)", "SELECT video_scan_stats($1)"):
        with pytest.raises(vane.BinderException, match="video_backend='native'"):
            con.execute(query, [file])
    with pytest.raises(vane.InvalidInputException, match="video_backend='native'"):
        con.execute("SELECT get_video_frame_by_idx($1, 0, index => $2)", [file, index])
    with pytest.raises(vane.BinderException, match="video_backend='native'"):
        vane.read_video_frames(file, 6, 8, indexes=[index], connection=con)
    assert isinstance(con.execute("SELECT get_video_frame_by_idx($1, 0)", [file]).fetchone()[0], vane.Image)


def test_video_index_null_and_empty_selections(native_video, indexed_clip):
    con, file = native_video, indexed_clip
    index = _build(con, file)
    assert con.execute(
        "SELECT build_video_index(NULL), video_index_info(NULL), video_frames(NULL, index => $1)", [index]
    ).fetchone() == (None, None, None)
    assert con.execute("SELECT video_frames($1, start_time => 999, index => $2)", [file, index]).fetchone()[0] == []
    assert (
        con.execute(
            "SELECT get_video_frame_by_idx($1, 999, index => $2, on_error => 'null')", [file, index]
        ).fetchone()[0]
        is None
    )
    with pytest.raises(vane.InvalidInputException, match="out of range"):
        con.execute("SELECT get_video_frame_by_idx($1, 999, index => $2)", [file, index])


def test_video_index_binding_and_content_errors_are_not_suppressed(native_video, indexed_clip):
    con, file = native_video, indexed_clip
    index = _build(con, file)
    other = vane.VideoFile(file.url, file.content_type, checksum="sha256:different")
    with pytest.raises(vane.InvalidInputException, match="does not match"):
        con.execute("SELECT get_video_frame_by_idx($1, 0, index => $2, on_error => 'null')", [other, index])
    before = os.stat(file.url)
    with open(file.url, "r+b") as output:
        original = output.read(1)
        output.seek(0)
        output.write(bytes([original[0] ^ 1]))
    os.utime(file.url, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(vane.InvalidInputException, match="source bytes have changed"):
        con.execute("SELECT get_video_frame_by_idx($1, 0, index => $2, on_error => 'null')", [file, index])


def test_video_index_resource_and_malformed_input_limits(native_video, indexed_clip):
    con, file = native_video, indexed_clip
    index = _build(con, file)
    for option, value in (("max_index_bytes", 128), ("max_input_bytes", 128), ("max_decoded_frames", 2)):
        with pytest.raises(vane.OutOfRangeException, match=option):
            con.execute(f"SELECT build_video_index($1, {option} => $2)", [file, value])
    for broken in (b"", index[:20], index[:-1], index[:180] + bytes([index[180] ^ 1]) + index[181:]):
        with pytest.raises(vane.InvalidInputException, match="video index"):
            con.execute("SELECT get_video_frame_by_idx($1, 0, index => $2, on_error => 'null')", [file, broken])
    with pytest.raises(vane.BinderException, match="correspond"):
        con.sql("SELECT * FROM read_video_frames($1, 6, 8, indexes => []::BLOB[])", params=[file])
    with pytest.raises(vane.BinderException, match="NULL"):
        con.sql("SELECT * FROM read_video_frames($1, 6, 8, indexes => [NULL]::BLOB[])", params=[file])


def test_video_index_reproduction_failure_is_not_a_nullable_media_error(native_video, indexed_clip):
    index = bytearray(_build(native_video, indexed_clip))
    # Corrupt the last indexed frame's digest, then repair the outer checksum:
    # a structurally valid index must still verify decoded content after seek.
    index[-33] ^= 1
    index[-32:] = hashlib.sha256(index[:-32]).digest()
    with pytest.raises(vane.NotImplementedException, match="cannot reproduce"):
        native_video.execute(
            "SELECT get_video_frame_by_idx($1, 239, index => $2, on_error => 'null')", [indexed_clip, bytes(index)]
        )


def test_video_index_size_is_checked_before_parsing(native_video):
    with pytest.raises(vane.OutOfRangeException, match="64 MiB"):
        native_video.execute("SELECT video_index_info(repeat('x', 67108865)::BLOB)")


def test_video_index_range_reads_and_missing_source(native_video, indexed_clip):
    from pathlib import Path

    con = native_video
    payload = Path(indexed_clip.url).read_bytes()
    prefix = b"private surrounding bytes\0" * 17
    server, thread, handler = _start_object_server(prefix + payload + b"outside suffix")
    try:
        url = f"http://127.0.0.1:{server.server_port}/bucket/object.bin"
        file = vane.VideoFile(url, indexed_clip.content_type, len(prefix), len(payload))
        index = _build(con, file)
        handler.requests.clear()
        con.execute("SELECT get_video_frame_by_idx($1, 200, index => $2)", [file, index]).fetchone()
        ranges = [request["range"] for request in handler.requests if request["range"]]
        assert ranges
        total = 0
        for request in ranges:
            start, end = map(int, request.removeprefix("bytes=").split("-"))
            assert len(prefix) <= start <= end < len(prefix) + len(payload)
            total += end - start + 1
        assert total < len(payload)
        con.execute("SET enable_external_access=false")
        with pytest.raises(vane.PermissionException):
            con.execute("SELECT get_video_frame_by_idx($1, 200, index => $2, on_error => 'null')", [file, index])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_video_indexing_can_be_cancelled(native_video, indexed_clip):
    from pathlib import Path

    con = native_video
    server, thread, handler = _start_object_server(Path(indexed_clip.url).read_bytes())
    handler.block_reads = True
    try:
        file = vane.VideoFile(f"http://127.0.0.1:{server.server_port}/bucket/object.bin", indexed_clip.content_type)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_build, con, file)
            assert handler.read_started.wait(timeout=10)
            con.interrupt()
            handler.release_read.set()
            with pytest.raises(vane.InterruptException):
                future.result(timeout=20)
    finally:
        handler.release_read.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
