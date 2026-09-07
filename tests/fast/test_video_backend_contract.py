# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare actual public outputs, including every timestamp and RGB byte."""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import pytest

import vane
from tests.fast import test_native_media_extensions as media


@pytest.fixture
def backends():
    pytest.importorskip("av")
    pytest.importorskip("PIL.Image")
    with vane.connect(config={"video_backend": "python"}) as python, media._connect("video") as native:
        yield python, native


@pytest.fixture(params=["mp4", "mkv", "vfr", "offset", "ntsc", "audio_tail", "avi"])
def contract_clip(request, tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    kind = request.param
    path = tmp_path / ("clip.mkv" if kind == "mkv" else "clip.avi" if kind == "avi" else "clip.mp4")
    rate = Fraction(30000, 1001) if kind == "ntsc" else Fraction(8)
    ticks = [i + (i // 3 if kind == "vfr" else 0) + (40 if kind == "offset" else 0) for i in range(24)]
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4" if kind == "avi" else "libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.gop_size = 8
        stream.codec_context.max_b_frames = 2
        stream.codec_context.time_base = 1 / rate
        stream.options = {"threads": "1", "sc_threshold": "0"}
        audio = output.add_stream("aac", rate=8000) if kind == "audio_tail" else None
        if audio is not None:
            audio.layout = "mono"
        y, x = np.indices((48, 64))
        for i, pts in enumerate(ticks):
            pixels = np.stack(
                ((x * 7 + i * 19) % 256, (y * 11 + i * 31) % 256, ((x ^ y) * 13 + i * 47) % 256), axis=-1
            ).astype("uint8")
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts, 1 / rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        if audio is not None:
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 40000), dtype="float32"), format="flt", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = 8000, 0, Fraction(1, 8000)
            for packet in audio.encode(frame):
                output.mux(packet)
            for packet in audio.encode():
                output.mux(packet)
    return vane.VideoFile(str(path)), kind


def _query(con, file, function="video_frames", options="", extra=()):
    return con.execute(f"SELECT {function}($1{options})", [file, *extra]).fetchone()[0]


def test_complete_metadata_and_frame_records(backends, contract_clip):
    python, native = backends
    file, kind = contract_clip
    metadata = [_query(con, file, "video_metadata") for con in backends]
    assert metadata[0] == metadata[1]
    if kind == "audio_tail":
        assert metadata[0]["duration"] == 3
        assert metadata[0]["container_duration"] == 5
    if kind == "mkv":
        # FFmpeg reports a stream duration for this clip after bounded probing.
        # Unknown stream duration is covered separately in test_video_file.
        assert metadata[0]["duration"] == 3
        assert metadata[0]["container_duration"] == 3
        assert metadata[0]["frame_count"] is None
    options = [
        "",
        ", width => 23, height => 17",
        ", width => 1, height => 1",
        ", start_time => 0.5, end_time => 1",
        ", start_time => 0.5, end_time => 0.5",
        ", is_key_frame => true",
        ", is_key_frame => false",
        ", sample_interval_seconds => 0.5000001",
        ", sample_interval_seconds => 0.4",
        ", start_time => 0.1, end_time => 2.1, sample_interval_seconds => 0.3",
        ", start_time => 999",
    ]
    baseline = _query(python, file)
    assert [frame["frame_index"] for frame in baseline] == list(range(24))
    value_metadata = file.metadata(connection=python)
    for name in ("width", "height", "fps", "duration", "container_duration", "frame_count"):
        assert getattr(value_metadata, name) == metadata[0][name]
    assert value_metadata.time_base == Fraction(
        metadata[0]["time_base"]["numerator"], metadata[0]["time_base"]["denominator"]
    )
    values = file.frames(connection=python)
    try:
        for ordinal, record in enumerate(values):
            expected = baseline[ordinal]
            try:
                for name in ("frame_index", "frame_time", "frame_pts", "frame_dts", "frame_duration", "is_key_frame"):
                    assert getattr(record, name) == expected[name]
                assert record.frame_time_base == Fraction(
                    expected["frame_time_base_numerator"], expected["frame_time_base_denominator"]
                )
                assert record.data.tobytes() == expected["data"].data
            finally:
                record.data.close()
        assert ordinal == 23
    finally:
        values.close()
    for option in options:
        assert _query(python, file, options=option) == _query(native, file, options=option), option
    for target in (0, 1, 22, 23):
        assert _query(python, file, "get_video_frame_by_idx", ", $2", [target]) == baseline[target]["data"]
        assert _query(native, file, "get_video_frame_by_idx", ", $2", [target]) == baseline[target]["data"]
        for con in backends:
            assert (
                _query(con, file, "get_video_frame_by_idx", ", $2, max_decoded_frames => $3", [target, target + 1])
                == baseline[target]["data"]
            )
        image = file.get_frame_by_idx(target, max_frames=target + 1, connection=python)
        try:
            assert image.tobytes() == baseline[target]["data"].data
        finally:
            image.close()
    assert _query(python, file, "video_keyframes") == _query(native, file, "video_keyframes")


def test_independent_index_builds_cross_reads_and_counters(backends, contract_clip):
    file, _ = contract_clip
    indexes = [_query(con, file, "build_video_index") for con in backends]
    # Includes the source binding, source blocks, original DTS, frame digests and
    # build I/O. Dropping fields or comparing only decoded images is insufficient.
    assert indexes[0] == indexes[1]
    baseline = _query(backends[0], file)
    info = [con.execute("SELECT video_index_info($1)", [indexes[0]]).fetchone()[0] for con in backends]
    assert info[0] == info[1]
    assert info[0]["frame_count"] == 24
    for index in indexes:
        for con in backends:
            assert _query(con, file, options=", index => $2", extra=[index]) == baseline
            assert _query(con, file, "get_video_frame_by_idx", ", 23, index => $2", [index]) == baseline[23]["data"]
        for options in (
            "",
            ", idx => 23",
            ", is_key_frame => true",
            ", start_time => 1, sample_interval_seconds => 0.4",
        ):
            counts = [_query(con, file, "video_scan_stats", ", index => $2" + options, [index]) for con in backends]
            assert counts[0] == counts[1]
    for options in ("", ", idx => 23", ", is_key_frame => false", ", start_time => 1, end_time => 2"):
        counts = [_query(con, file, "video_scan_stats", options) for con in backends]
        assert counts[0] == counts[1]


def test_streaming_indexes_preserve_every_output_field(backends, contract_clip):
    file, _ = contract_clip
    indexes = [_query(con, file, "build_video_index") for con in backends]
    for options in (
        {},
        {"is_key_frame": False},
        {"start_time": 0.5, "sample_interval_seconds": 0.4},
        {"frame_limit": 3},
    ):
        baseline = (
            vane.read_video_frames(file, 17, 23, connection=backends[0], **options).order("frame_index").fetchall()
        )
        for con in backends:
            for index in (None, *indexes):
                result = (
                    vane.read_video_frames(
                        file, 17, 23, indexes=None if index is None else [index], connection=con, **options
                    )
                    .order("frame_index")
                    .fetchall()
                )
                assert result == baseline


@pytest.mark.parametrize("interval", [math.nextafter(0.5, 0), math.nextafter(0.5, 1), 5e-324, 1e300])
def test_sampling_uses_exact_decimal_options(backends, contract_clip, interval):
    file, _ = contract_clip
    baseline = _query(backends[0], file)
    next_time = Fraction(0)
    step = Fraction(str(interval))
    expected = []
    with pytest.importorskip("av").open(file.url) as encoded:
        origin = encoded.streams.video[0].start_time or 0
    for frame in baseline:
        base = Fraction(frame["frame_time_base_numerator"], frame["frame_time_base_denominator"])
        time = (frame["frame_pts"] - origin) * base
        if time >= next_time:
            expected.append(frame)
            next_time += ((time - next_time) // step + 1) * step
    for con in backends:
        assert _query(con, file, options=", sample_interval_seconds => $2", extra=[interval]) == expected


def test_index_errors_and_nulls_match(backends, contract_clip):
    file, _ = contract_clip
    index = _query(backends[0], file, "build_video_index")
    for con in backends:
        assert con.execute("SELECT build_video_index(NULL), video_index_info(NULL), video_frames(NULL)").fetchone() == (
            None,
            None,
            None,
        )
        assert _query(con, file, options=", start_time => 999, index => $2", extra=[index]) == []
        for query in (
            "SELECT build_video_index($1, max_decoded_frames => 1)",
            "SELECT video_frames($1, max_decoded_frames => 1, on_error => 'null')",
        ):
            with pytest.raises(vane.OutOfRangeException, match="max_decoded_frames"):
                con.execute(query, [file])
        for broken in (b"", index[:20], index[:-1]):
            with pytest.raises(vane.InvalidInputException, match="video index"):
                _query(con, file, options=", index => $2, on_error => 'null'", extra=[broken])
        changed = bytearray(index)
        changed[-33] ^= 1
        changed[-32:] = hashlib.sha256(changed[:-32]).digest()
        with pytest.raises(vane.NotImplementedException, match="cannot reproduce"):
            _query(con, file, "get_video_frame_by_idx", ", 23, index => $2, on_error => 'null'", [bytes(changed)])


def test_python_indexes_work_without_loading_video_extension(contract_clip):
    file, _ = contract_clip
    with vane.connect(config={"video_backend": "python"}) as con:
        index = _query(con, file, "build_video_index")
        assert _query(con, file, "video_scan_stats", ", idx => 23, index => $2", [index])["selected_frames"] == 1
        assert con.execute("SELECT video_index_info($1)", [index]).fetchone()[0]["frame_count"] == 24


def test_public_failure_categories_and_policies_match(backends, contract_clip, tmp_path):
    file, _ = contract_clip
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not encoded video")
    for con in backends:
        for function in ("video_frames", "video_keyframes"):
            with pytest.raises(vane.InvalidInputException):
                _query(con, vane.VideoFile(str(broken)), function)
            assert _query(con, vane.VideoFile(str(broken)), function, ", on_error => 'null'") is None
            with pytest.raises(vane.OutOfRangeException):
                _query(con, file, function, ", max_pixels => 100, on_error => 'null'")
        with pytest.raises(vane.OutOfRangeException):
            _query(con, file, "video_metadata", ", 16")
        with pytest.raises(vane.IOException):
            _query(con, vane.VideoFile(str(tmp_path / "missing.mp4")), options=", on_error => 'null'")
        con.execute("SET enable_external_access=false")
        with pytest.raises(vane.PermissionException):
            _query(con, file, options=", on_error => 'null'")


def test_streaming_failure_categories_survive_arrow(backends, contract_clip, tmp_path):
    file, _ = contract_clip
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not encoded video")
    for con in backends:

        def read(value=file, **options):
            return vane.read_video_frames(value, 1, 1, connection=con, **options).fetchall()

        with pytest.raises(vane.InvalidInputException):
            read(vane.VideoFile(str(broken)))
        assert read(vane.VideoFile(str(broken)), on_error="skip") == []
        for options in ({"max_decoded_frames": 1}, {"max_pixels": 100}):
            with pytest.raises(vane.OutOfRangeException):
                read(on_error="skip", **options)
        with pytest.raises(vane.InvalidInputException, match="video index"):
            read(indexes=[b"invalid index"], on_error="skip")
        with pytest.raises(vane.IOException):
            read(vane.VideoFile(str(tmp_path / "missing.mp4")), on_error="skip")
        assert con.execute("SELECT 42").fetchone() == (42,)


def test_python_streaming_skip_keeps_partial_batch_before_content_failure(monkeypatch):
    pytest.importorskip("av")
    import vane._video_index as cursor
    from vane._video_file import VideoFrameData

    image = pytest.importorskip("PIL.Image")

    def damaged_decode(*args):
        for ordinal in range(2):
            yield VideoFrameData(
                ordinal,
                float(ordinal),
                Fraction(1),
                ordinal,
                ordinal,
                1,
                True,
                image.new("RGB", (1, 1), (ordinal, 2, 3)),
            )
        raise vane.VideoFileFormatError("damaged packet after valid prefix")

    monkeypatch.setattr(cursor, "_video_frames", damaged_decode)
    with vane.connect(config={"video_backend": "python"}) as con:
        result = (
            vane.read_video_frames(
                [vane.VideoFile("unopened://clip")] * 2,
                1,
                1,
                on_error="skip",
                frame_limit=3,
                connection=con,
            )
            .project("frame_index, data")
            .fetchall()
        )
        assert [row[0] for row in result] == [0, 1, 0]
        assert [row[1].data for row in result] == [b"\x00\x02\x03", b"\x01\x02\x03", b"\x00\x02\x03"]


def test_streaming_metadata_budget_includes_files_and_indexes(backends):
    index = b"x" * (64 * 1024**2 - 512)
    for con in backends:
        with pytest.raises(vane.BinderException, match="source metadata exceeds 64 MiB"):
            vane.read_video_frames(vane.VideoFile("unopened://clip"), 1, 1, indexes=[index], connection=con)


def test_video_frame_source_bound_relations_match(backends, contract_clip):
    from vane.datasource.video_reader import VideoFrameSource

    file, _ = contract_clip
    for options in (
        {},
        {"frame_limit": 3},
        {"is_key_frame": False},
        {"start_time": 0.5, "sample_interval_seconds": 0.4},
    ):
        source = VideoFrameSource([file, file], width=3, height=2, read_task_count=2, **options)
        relations = [con.from_datasource(source).order("frame_index") for con in backends]
        assert relations[0].types == relations[1].types
        assert relations[0].types[-1].is_image()
        assert relations[0].fetchall() == relations[1].fetchall()
    for con in backends:
        with pytest.raises(vane.OutOfRangeException):
            con.from_datasource(
                VideoFrameSource([file], width=1, height=1, max_decoded_frames=1, on_error="skip")
            ).fetchall()


@pytest.mark.parametrize(
    "options",
    [
        {"width": 100001},
        {"height": 100001},
        {"max_input_bytes": 16 * 1024**3 + 1},
        {"max_decoded_frames": 100000001},
        {"max_partition_bytes": 256 * 1024**2 + 1},
        {"frame_limit": 1 << 63},
        {"read_task_count": 1 << 63},
        {"start_time": 10**400},
        {"max_partition_bytes": 3},
    ],
)
def test_video_frame_source_connection_limits_match_without_io(backends, options):
    from vane.datasource.video_reader import VideoFrameSource

    arguments = {"width": 1, "height": 1, **options}
    source = VideoFrameSource(["unopened://clip"], **arguments)
    for con in backends:
        with pytest.raises(vane.OutOfRangeException):
            con.from_datasource(source)
