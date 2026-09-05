# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys

import pytest

import vane
from tests.fast import test_native_media_extensions as media_tests

video_path = media_tests.video_path

COLUMNS = [
    "path",
    "file",
    "frame_index",
    "frame_time",
    "frame_time_base_numerator",
    "frame_time_base_denominator",
    "frame_pts",
    "frame_dts",
    "frame_duration",
    "is_key_frame",
    "data",
]


def test_streaming_video_default_connection_does_not_reenter_type_binding(video_path):
    pytest.importorskip("psutil")
    # Isolate a lock regression so it cannot stall the complete pytest shard.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys, vane; "
            "rows = vane.read_video_frames(sys.argv[1], 6, 8, frame_limit=1).fetchall(); "
            "assert len(rows) == 1 and rows[0][-1].mode == 'RGB'",
            str(video_path),
        ],
        check=True,
        timeout=30,
    )


@pytest.fixture(params=["python", "native"])
def video_connection(request):
    if request.param == "python":
        pytest.importorskip("psutil")
    con = vane.connect() if request.param == "python" else media_tests._connect("video")
    try:
        yield con
    finally:
        con.close()


def test_streaming_video_python_and_sql_api(video_connection, video_path):
    con = video_connection
    relation = vane.read_video_frames(
        video_path, 6, 8, start_time=0.5, end_time=2, sample_interval_seconds=0.5, connection=con
    )
    assert relation.columns == COLUMNS
    assert relation.types[1] == vane.file_type(vane.MediaType.video())
    assert relation.types[-1] == vane.image_type("RGB", 6, 8)
    rows = relation.order("frame_index").fetchall()
    assert [row[2] for row in rows] == [2, 4, 6, 8]
    assert [row[3] for row in rows] == [0.5, 1.0, 1.5, 2.0]
    assert all(row[0] == str(video_path) and row[1] == vane.VideoFile(str(video_path)) for row in rows)
    assert all(row[4] == 1 and row[5] > 0 and row[6] is not None for row in rows)
    assert all(isinstance(row[9], bool) for row in rows)
    assert all(
        isinstance(row[10], vane.Image)
        and row[10].width == 8
        and row[10].height == 6
        and row[10].mode == "RGB"
        and len(row[10].data) == 144
        for row in rows
    )
    sql = con.sql(
        "SELECT * FROM read_video_frames(?, 6, 8, start_time => 0.5, end_time => 2, "
        "sample_interval_seconds => 0.5) ORDER BY frame_index",
        params=[str(video_path)],
    )
    assert sql.types == relation.types
    assert sql.fetchall() == rows


def test_streaming_video_preserves_file_window(video_connection, video_path, tmp_path):
    payload = video_path.read_bytes()
    prefix = b"outside the file view\0" * 37
    path = tmp_path / "bundle'with-quote.bin"
    path.write_bytes(prefix + payload + b"outside suffix")
    file = vane.File(str(path), "video/mp4", len(prefix), len(payload), "sha256:opaque")
    relation = vane.read_video_frames(file, 6, 8, frame_limit=2, connection=video_connection)
    rows = relation.order("frame_index").fetchall()
    expected = vane.VideoFile(file.url, file.content_type, file.position, file.size, file.checksum)
    assert [row[1] for row in rows] == [expected] * 2
    assert [row[2] for row in rows] == [0, 1]
    assert all(row[0] == file.url for row in rows)
    sql = video_connection.sql(
        "SELECT * FROM read_video_frames(file(?, ?, ?, ?, ?), 6, 8, frame_limit => 2) ORDER BY frame_index",
        params=[file.url, file.content_type, file.position, file.size, file.checksum],
    )
    assert sql.fetchall() == rows


def test_streaming_video_keyframe_positional_and_named_arguments(video_connection, video_path):
    con = video_connection
    query = "SELECT frame_index, is_key_frame FROM read_video_frames(?, 6, 8, %s) ORDER BY frame_index"
    positional = con.execute(query % "TRUE", [str(video_path)]).fetchall()
    named = con.execute(query % "is_key_frame => TRUE", [str(video_path)]).fetchall()
    python = (
        vane.read_video_frames(video_path, 6, 8, True, connection=con)
        .project("frame_index, is_key_frame")
        .order("frame_index")
        .fetchall()
    )
    assert positional == named == python
    assert positional and all(key for _, key in positional)
    assert len(positional) < 12


@pytest.mark.parametrize("input", [None, [], "unopened://missing"])
def test_streaming_video_empty_and_zero_limit_do_not_open_files(video_connection, input):
    relation = vane.read_video_frames(
        input, 5000, 5000, frame_limit=0, max_partition_bytes=256 * 1024**2, connection=video_connection
    )
    assert relation.types[-1] == vane.image_type("RGB", 5000, 5000)
    assert relation.fetchall() == []
    if input != "unopened://missing":
        assert video_connection.execute("SELECT * FROM read_video_frames(?, 6, 8)", [input]).fetchall() == []


def test_streaming_video_native_needs_loaded_extension():
    with vane.connect(config={"video_backend": "native"}) as con:
        with pytest.raises(vane.BinderException, match="requires the video extension"):
            vane.read_video_frames("unopened://missing", 6, 8, connection=con)


def test_streaming_video_backend_plan_and_helper_dispatch(video_connection, video_path, monkeypatch):
    import vane._video_file as helper

    con = video_connection
    backend = con.execute("SELECT current_setting('video_backend')").fetchone()[0]
    calls = []
    original = helper._video_file_frames_value

    def observe(*args, **kwargs):
        assert backend == "python", "native scan invoked the Python video helper"
        assert kwargs.get("_execution_context") is not None
        calls.append(args[0].url)
        return original(*args, **kwargs)

    monkeypatch.setattr(helper, "_video_file_frames_value", observe)
    relation = vane.read_video_frames(video_path, 6, 8, frame_limit=1, connection=con)
    plan = relation.explain().upper()
    assert ("NATIVE_READ_VIDEO_FRAMES" if backend == "native" else "DATASOURCE_SCAN") in plan
    assert calls == []
    assert len(relation.fetchall()) == 1
    assert bool(calls) == (backend == "python")


@pytest.mark.parametrize(
    "option,value",
    [
        ("image_height", 0),
        ("image_width", -1),
        ("start_time", None),
        ("start_time", float("nan")),
        ("end_time", -1),
        ("sample_interval_seconds", 0),
        ("max_input_bytes", 0),
        ("max_decoded_frames", 0),
        ("max_pixels", 32 * 1024**2 + 1),
        ("max_partition_bytes", 1),
        ("frame_limit", -1),
        ("read_task_count", 0),
        ("on_error", "null"),
    ],
)
def test_streaming_video_validates_options_before_io(video_connection, option, value):
    options = {"image_height": 6, "image_width": 8, option: value}
    with pytest.raises(vane.BinderException):
        vane.read_video_frames("unopened://missing", connection=video_connection, **options)


def test_streaming_video_null_elements_and_other_media_are_invalid(video_connection):
    for expression in ("['unopened://video', NULL]", "image_file('unopened://image')"):
        with pytest.raises(vane.BinderException):
            video_connection.sql(f"SELECT * FROM read_video_frames({expression}, 6, 8)")
    for input in ([None], vane.ImageFile("unopened://image")):
        with pytest.raises(TypeError):
            vane.read_video_frames(input, 6, 8, connection=video_connection)


def test_streaming_video_budget_accounts_for_path_and_file(video_connection):
    url = "unopened://" + "x" * 2000
    with pytest.raises(vane.BinderException, match="row exceeds max_partition_bytes"):
        vane.read_video_frames(url, 6, 8, max_partition_bytes=3000, connection=video_connection)


@pytest.mark.parametrize("task_count", [None, 1, 2, 20])
def test_streaming_video_task_groups_and_global_frame_limit(video_connection, video_path, task_count):
    files = [video_path] * 3
    rows = (
        vane.read_video_frames(
            files,
            6,
            8,
            start_time=0.5,
            end_time=1,
            frame_limit=4,
            read_task_count=task_count,
            connection=video_connection,
        )
        .project("frame_index")
        .fetchall()
    )
    assert rows == [(2,), (3,), (4,), (2,)]
    rows = (
        vane.read_video_frames(
            files, 6, 8, start_time=0.5, end_time=1, read_task_count=task_count, connection=video_connection
        )
        .project("frame_index")
        .fetchall()
    )
    assert sorted(rows) == [(2,)] * 3 + [(3,)] * 3 + [(4,)] * 3


def test_streaming_video_skip_propagates_io_and_resource_errors(video_connection, video_path, tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    rows = (
        vane.read_video_frames([corrupt, video_path], 6, 8, frame_limit=2, on_error="skip", connection=video_connection)
        .project("path")
        .fetchall()
    )
    assert rows == [(str(video_path),)] * 2
    with pytest.raises(vane.Error):
        vane.read_video_frames(tmp_path / "missing.mp4", 6, 8, on_error="skip", connection=video_connection).fetchall()
    with pytest.raises(vane.Error, match="max_input_bytes"):
        vane.read_video_frames(
            video_path, 6, 8, max_input_bytes=1, on_error="skip", connection=video_connection
        ).fetchall()


def test_streaming_video_uses_query_connection(video_connection, video_path):
    con = video_connection
    con.execute("SET home_directory = ?", [str(video_path.parent)])
    rows = (
        vane.read_video_frames(f"~/{video_path.name}", 6, 8, frame_limit=1, connection=con)
        .project("frame_index")
        .fetchall()
    )
    assert rows == [(0,)]


def test_python_image_batches_bound_payload_and_survive_reuse(video_path):
    from vane.datasource.video_reader import _decode_video_batches, _ImageVideoFrameSource

    source = _ImageVideoFrameSource([str(video_path)], height=6, width=8, max_partition_bytes=2048)
    with vane.connect() as con:
        batches = list(
            _decode_video_batches(source.files[0], options=source.options, max_output_frames=None, connection=con)
        )
    assert sum(batch.num_rows for batch in batches) == 12
    assert all(batch.nbytes <= 2048 for batch in batches)
    assert [index for batch in batches for index in batch.column("frame_index").to_pylist()] == list(range(12))
    assert len({batch.column("frame")[0].as_py()["data"] for batch in batches}) == len(batches)
