# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import pickle
import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.datasource import _schema_to_arrow, read_datasource, video_reader
from vane.datasource.video_reader import (
    LimitedVideoFrameTask,
    VideoFrameSource,
    VideoFrameTask,
    VideoReadError,
    _decode_video_batches,
    _decode_video_with_policy,
    _file_storage_bounds,
    _flush_frame_batch,
    _split_video_file_groups,
    _video_output_batch_size,
)


def _encoded_video(
    container_format: str = "mp4",
    *,
    width: int = 16,
    height: int = 12,
    frame_count: int = 8,
    frame_rate: int = 4,
    gop_size: int = 3,
    max_b_frames: int = 0,
) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format) as container:
        stream = container.add_stream("mpeg4", rate=frame_rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = gop_size
        stream.codec_context.max_b_frames = max_b_frames
        for index in range(frame_count):
            horizontal = np.broadcast_to(np.arange(width, dtype=np.uint8), (height, width))
            pixels = np.stack(
                (
                    horizontal + index,
                    np.full((height, width), index * 5, dtype=np.uint8),
                    np.flip(horizontal, axis=1) + index,
                ),
                axis=2,
            )
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _ranged_video(tmp_path: Path, *, frame_count: int = 8) -> vane.VideoFile:
    payload = _encoded_video(frame_count=frame_count)
    prefix = b"not-a-video-prefix"
    suffix = b"not-a-video-suffix"
    path = tmp_path / "ranged-video.bin"
    path.write_bytes(prefix + payload + suffix)
    return vane.VideoFile(str(path), "video/mp4", len(prefix), len(payload), "sha256:" + "1" * 64)


def _fake_batch(
    value: vane.VideoFile,
    indices: list[int],
    *,
    height: int = 2,
    width: int = 3,
) -> pa.RecordBatch:
    count = len(indices)
    pixels = np.stack(
        [np.full((height, width, 3), index, dtype=np.uint8) for index in indices],
        axis=0,
    )
    return _flush_frame_batch(
        value,
        pixels,
        count,
        frame_indices=indices,
        frame_times=[index / 4 for index in indices],
        time_base_numerators=[1] * count,
        time_base_denominators=[16_384] * count,
        frame_pts=[index * 4_096 for index in indices],
        frame_dts=[index * 4_096 for index in indices],
        frame_durations=[4_096] * count,
        key_frame_flags=[index % 3 == 0 for index in indices],
    )


class _TrackingImage:
    def __init__(self, value: int, shape: tuple[int, int, int]):
        self.pixels = np.full(shape, value, dtype=np.uint8)
        self.closed = False

    def __array__(self, dtype=None, copy=None):
        del copy
        return np.asarray(self.pixels, dtype=dtype)

    def close(self):
        self.closed = True


class _FakeVideoFile:
    def __init__(self, records=None, error: Exception | None = None):
        self.url = "memory://fake.mp4"
        self.content_type = "video/mp4"
        self.position = None
        self.size = None
        self.checksum = None
        self.records = list(records or [])
        self.error = error
        self.closed = False
        self.frames_kwargs = None
        self.connection = None

    def frames(self, **kwargs):
        self.connection = kwargs.pop("connection", None)
        if self.connection is None:
            raise AssertionError("test decode must provide an explicit connection")
        self.frames_kwargs = kwargs

        def generate():
            try:
                yield from self.records
                if self.error is not None:
                    raise self.error
            finally:
                self.closed = True

        return generate()


def _fake_record(index: int, *, height: int = 2, width: int = 3):
    return SimpleNamespace(
        frame_index=index,
        frame_time=index / 4,
        frame_time_base=Fraction(1, 16_384),
        frame_pts=index * 4_096,
        frame_dts=index * 4_096,
        frame_duration=4_096,
        is_key_frame=index % 3 == 0,
        data=_TrackingImage(index, (height, width, 3)),
    )


def test_video_frame_source_normalizes_paths_and_preserves_all_file_fields(tmp_path):
    ranged = _ranged_video(tmp_path)
    generic = vane.File(
        ranged.url,
        ranged.content_type,
        ranged.position,
        ranged.size,
        ranged.checksum,
    )

    source = VideoFrameSource([Path("relative.mp4"), generic, ranged], height=4, width=5, max_pixels=100)

    assert all(isinstance(value, vane.VideoFile) for value in source.files)
    assert source.files[0] == vane.VideoFile("relative.mp4")
    assert source.files[1] == ranged
    assert source.files[2] is ranged


@pytest.mark.parametrize("value", [vane.ImageFile("memory://image"), vane.AudioFile("memory://audio")])
def test_video_frame_source_rejects_other_media_file_types(value):
    with pytest.raises(TypeError, match="VIDEOFILE, generic FILE, or path"):
        VideoFrameSource([value])


@pytest.mark.parametrize("value", ["video.mp4", Path("video.mp4"), vane.VideoFile("memory://video")])
def test_video_frame_source_rejects_unwrapped_single_input(value):
    with pytest.raises(TypeError, match="iterable.*not a single value"):
        VideoFrameSource(value)


def test_video_frame_source_preserves_entry_validation_errors_and_reports_non_iterables():
    class BrokenPath:
        def __fspath__(self):
            raise RuntimeError("path conversion failed")

    with pytest.raises(RuntimeError, match="path conversion failed"):
        VideoFrameSource([BrokenPath()])
    with pytest.raises(TypeError, match="must be an iterable of file values"):
        VideoFrameSource(42)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"height": 0}, ValueError, "height must be greater than zero"),
        ({"width": True}, TypeError, "width must be int"),
        ({"max_partition_bytes": 0}, ValueError, "max_partition_bytes must be greater than zero"),
        ({"frame_limit": -1}, ValueError, "frame_limit must be non-negative"),
        ({"read_task_count": 0}, ValueError, "read_task_count must be greater than zero"),
        ({"start_time": -1}, ValueError, "start_time must be non-negative"),
        ({"end_time": -1}, ValueError, "end_time must be non-negative"),
        ({"start_time": 2, "end_time": 1}, ValueError, "end_time must be greater"),
        ({"sample_interval_seconds": 0}, ValueError, "sample_interval_seconds must be greater"),
        ({"is_key_frame": 1}, TypeError, "is_key_frame must be bool or None"),
        ({"buffer_size": 0}, ValueError, "buffer_size must be greater than zero"),
        ({"max_input_bytes": 0}, ValueError, "max_input_bytes must be greater than zero"),
        ({"max_decoded_frames": 0}, ValueError, "max_decoded_frames must be greater than zero"),
        ({"max_pixels": 1, "height": 2, "width": 2}, vane.VideoFileLimitError, "exceeding max_pixels"),
        ({"on_error": "null"}, ValueError, "on_error must be one of"),
    ],
)
def test_video_frame_source_validates_options_without_io(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        VideoFrameSource(["memory://not-opened"], **kwargs)


def test_video_frame_source_schema_exposes_source_and_exact_frame_provenance():
    source = VideoFrameSource(["a.mp4"], height=4, width=5)

    assert source.schema == {
        "file": "VIDEOFILE",
        "frame_index": "BIGINT",
        "frame_time": "DOUBLE",
        "frame_time_base_numerator": "BIGINT",
        "frame_time_base_denominator": "BIGINT",
        "frame_pts": "BIGINT",
        "frame_dts": "BIGINT",
        "frame_duration": "BIGINT",
        "is_key_frame": "BOOLEAN",
        "frame": {"kind": "tensor", "dtype": "UINT8", "shape": [4, 5, 3]},
    }
    arrow_schema = _schema_to_arrow(source.schema)
    assert arrow_schema.field("file").type == pa.struct(
        [
            ("url", pa.string()),
            ("content_type", pa.string()),
            ("position", pa.int64()),
            ("size", pa.int64()),
            ("checksum", pa.string()),
        ]
    )
    assert arrow_schema.field("frame").type == pa.fixed_shape_tensor(pa.uint8(), (4, 5, 3))


def test_video_frame_source_task_preserves_exact_file_and_options(tmp_path):
    value = _ranged_video(tmp_path)
    source = VideoFrameSource(
        [value],
        height=7,
        width=9,
        start_time=0.25,
        end_time=1.5,
        is_key_frame=False,
        sample_interval_seconds=0.5,
        buffer_size=64,
        max_input_bytes=10_000,
        max_decoded_frames=50,
        max_pixels=1000,
        on_error="skip",
    )
    task = next(source.get_tasks())

    assert isinstance(task, VideoFrameTask)
    assert task.video_file is value
    assert task.options == source.options


def test_video_frame_source_groups_files_into_read_tasks():
    source = VideoFrameSource([f"memory://{index}" for index in range(5)], read_task_count=2)
    tasks = list(source.get_tasks())

    assert [[value.url for value in task.files] for task in tasks] == [
        ["memory://0", "memory://1", "memory://2"],
        ["memory://3", "memory://4"],
    ]


def test_global_frame_limit_forces_one_ordered_task():
    source = VideoFrameSource(["a.mp4", "b.mp4"], frame_limit=3, read_task_count=2)

    tasks = list(source.get_tasks())

    assert len(tasks) == 1
    assert isinstance(tasks[0], LimitedVideoFrameTask)
    assert [value.url for value in tasks[0].files] == ["a.mp4", "b.mp4"]


def test_video_frame_tasks_are_pickle_safe():
    value = vane.VideoFile(
        "memory://it's/a-video",
        "video/x-'quoted'\x00tail",
        7,
        13,
        "sha256:" + "1" * 64,
    )
    source = VideoFrameSource([value], height=2, width=3, max_pixels=100)
    task = next(source.get_tasks())

    restored = pickle.loads(pickle.dumps(task))

    assert isinstance(restored, VideoFrameTask)
    assert restored.video_file == value
    assert restored.options == source.options


def test_split_video_file_groups_is_balanced_and_ordered():
    files = tuple(vane.VideoFile(f"memory://{index}") for index in range(5))

    groups = _split_video_file_groups(files, 3)

    assert [[value.url for value in group] for group in groups] == [
        ["memory://0", "memory://1"],
        ["memory://2", "memory://3"],
        ["memory://4"],
    ]


def test_decode_video_batches_honors_range_selection_resize_and_provenance(tmp_path, duckdb_cursor):
    value = _ranged_video(tmp_path, frame_count=8)
    source = VideoFrameSource(
        [value],
        height=6,
        width=8,
        max_partition_bytes=800,
        start_time=0.25,
        end_time=1.25,
        sample_interval_seconds=0.5,
        buffer_size=64,
        max_pixels=1000,
    )

    batches = list(
        _decode_video_batches(
            value,
            options=source.options,
            max_output_frames=None,
            connection=duckdb_cursor,
        )
    )
    table = pa.Table.from_batches(batches)

    assert [batch.num_rows for batch in batches] == [2, 1]
    assert (
        table.column("file").to_pylist()
        == [
            {
                "url": value.url,
                "content_type": value.content_type,
                "position": value.position,
                "size": value.size,
                "checksum": value.checksum,
            }
        ]
        * 3
    )
    assert table.column("frame_index").to_pylist() == [1, 3, 5]
    assert table.column("frame_time").to_pylist() == pytest.approx([0.25, 0.75, 1.25])
    assert table.column("frame_time_base_numerator").to_pylist() == [1, 1, 1]
    assert all(value > 0 for value in table.column("frame_time_base_denominator").to_pylist())
    assert all(value is not None for value in table.column("frame_pts").to_pylist())
    assert all(value is not None for value in table.column("frame_dts").to_pylist())
    assert all(value is not None for value in table.column("frame_duration").to_pylist())
    frames = np.concatenate([batch.column("frame").to_numpy_ndarray() for batch in batches])
    assert frames.dtype == np.uint8
    assert frames.shape == (3, 6, 8, 3)


def test_decode_video_batches_closes_images_and_decoder_at_output_limit(duckdb_cursor):
    records = [_fake_record(index) for index in range(5)]
    value = _FakeVideoFile(records)
    source = VideoFrameSource(["memory://fake.mp4"], height=2, width=3, max_pixels=100)

    batches = list(_decode_video_batches(value, options=source.options, max_output_frames=2, connection=duckdb_cursor))

    assert sum(batch.num_rows for batch in batches) == 2
    assert all(record.data.closed for record in records[:2])
    assert all(not record.data.closed for record in records[2:])
    assert value.closed
    assert value.connection is duckdb_cursor
    assert value.frames_kwargs == {
        "start_time": 0,
        "end_time": None,
        "width": 3,
        "height": 2,
        "is_key_frame": None,
        "sample_interval_seconds": None,
        "buffer_size": 1024 * 1024,
        "max_input_bytes": 8 * 1024**3,
        "max_frames": 1_000_000,
        "max_pixels": 100,
    }


def test_decode_video_batches_closes_decoder_when_limit_ends_on_full_batch(duckdb_cursor):
    records = [_fake_record(index) for index in range(2)]
    value = _FakeVideoFile(records)
    source = VideoFrameSource(
        ["memory://fake.mp4"],
        height=2,
        width=3,
        max_partition_bytes=1,
        max_pixels=100,
    )

    batches = list(_decode_video_batches(value, options=source.options, max_output_frames=1, connection=duckdb_cursor))

    assert [batch.num_rows for batch in batches] == [1]
    assert records[0].data.closed
    assert not records[1].data.closed
    assert value.closed


def test_decode_video_batches_does_not_mutate_emitted_arrow_buffers(duckdb_cursor):
    records = [_fake_record(index) for index in range(4)]
    value = _FakeVideoFile(records)
    source = VideoFrameSource(
        ["memory://fake.mp4"],
        height=2,
        width=3,
        max_partition_bytes=200,
        max_pixels=100,
    )
    batches = _decode_video_batches(
        value,
        options=source.options,
        max_output_frames=None,
        connection=duckdb_cursor,
    )

    first = next(batches)
    snapshot = first.column("frame").to_numpy_ndarray().copy()
    remaining = list(batches)

    np.testing.assert_array_equal(first.column("frame").to_numpy_ndarray(), snapshot)
    assert sum(batch.num_rows for batch in remaining) + first.num_rows == 4


def test_decode_video_batches_requires_exactly_one_explicit_io_context(duckdb_cursor):
    value = _FakeVideoFile([_fake_record(0)])
    source = VideoFrameSource(["memory://fake.mp4"], height=2, width=3, max_pixels=100)

    with pytest.raises(ValueError, match="exactly one explicit connection"):
        list(_decode_video_batches(value, options=source.options, max_output_frames=None))
    with pytest.raises(ValueError, match="exactly one explicit connection"):
        list(
            _decode_video_batches(
                value,
                options=source.options,
                max_output_frames=None,
                connection=duckdb_cursor,
                execution_context=object(),
            )
        )

    assert value.frames_kwargs is None


def test_memory_admission_observes_execution_context_interruption(monkeypatch):
    class AdmissionCancelled(Exception):
        pass

    checks = 0
    memory_checks = 0

    def check_interrupted():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise AdmissionCancelled

    def virtual_memory():
        nonlocal memory_checks
        memory_checks += 1
        return SimpleNamespace(available=0, percent=100.0)

    monkeypatch.setattr(
        video_reader,
        "_import_video_dependency",
        lambda _module_name, _package_name: SimpleNamespace(virtual_memory=virtual_memory),
    )

    with pytest.raises(AdmissionCancelled):
        video_reader._wait_for_memory(check_interrupted)

    assert checks == 2
    assert memory_checks == 1


def test_decode_admission_observes_execution_context_interruption(monkeypatch):
    class AdmissionCancelled(Exception):
        pass

    class ExecutionContext:
        def __init__(self):
            self.checks = 0

        def _check_interrupted(self):
            self.checks += 1
            if self.checks == 3:
                raise AdmissionCancelled

    class UnavailableSemaphore:
        def __init__(self):
            self.acquire_timeouts = []

        def acquire(self, *, timeout):
            self.acquire_timeouts.append(timeout)
            return False

        def release(self):
            raise AssertionError("an unavailable decode slot must not be released")

    execution_context = ExecutionContext()
    semaphore = UnavailableSemaphore()
    source = VideoFrameSource(["memory://fake.mp4"], height=2, width=3, max_pixels=100)
    monkeypatch.setattr(video_reader, "_wait_for_memory", lambda check: check())
    monkeypatch.setattr(video_reader, "_decode_semaphore", semaphore)

    with pytest.raises(AdmissionCancelled):
        list(
            video_reader._decode_video_guarded(
                vane.VideoFile("memory://fake.mp4", "video/mp4"),
                options=source.options,
                max_output_frames=None,
                execution_context=execution_context,
            )
        )

    assert execution_context.checks == 3
    assert semaphore.acquire_timeouts == [video_reader._ADMISSION_INTERRUPT_CHECK_INTERVAL]


def test_flush_frame_batch_compacts_short_tail():
    value = vane.VideoFile("memory://video")
    backing = np.zeros((100, 2, 3, 3), dtype=np.uint8)
    batch = _flush_frame_batch(
        value,
        backing,
        1,
        frame_indices=[0],
        frame_times=[0.0],
        time_base_numerators=[1],
        time_base_denominators=[1000],
        frame_pts=[0],
        frame_dts=[0],
        frame_durations=[1],
        key_frame_flags=[True],
    )

    tensor = batch.column("frame")
    assert tensor.to_numpy_ndarray().shape == (1, 2, 3, 3)
    assert tensor.storage.values.buffers()[1].size == 2 * 3 * 3


def test_video_error_policy_wraps_or_skips_only_format_errors(caplog, duckdb_cursor):
    source_raise = VideoFrameSource(["memory://fake"], height=2, width=3, max_pixels=100)
    bad_raise = _FakeVideoFile(error=vane.VideoFileFormatError("bad codec"))

    with pytest.raises(VideoReadError, match="bad codec") as raised:
        list(
            _decode_video_with_policy(
                bad_raise,
                options=source_raise.options,
                max_output_frames=None,
                connection=duckdb_cursor,
            )
        )
    assert isinstance(raised.value.__cause__, vane.VideoFileFormatError)

    source_skip = VideoFrameSource(
        ["memory://fake"],
        height=2,
        width=3,
        max_pixels=100,
        on_error="skip",
    )
    bad_skip = _FakeVideoFile(error=vane.VideoFileFormatError("bad codec"))
    with caplog.at_level("WARNING"):
        assert (
            list(
                _decode_video_with_policy(
                    bad_skip,
                    options=source_skip.options,
                    max_output_frames=None,
                    connection=duckdb_cursor,
                )
            )
            == []
        )
    assert "Skipping unreadable VIDEOFILE" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        OSError("filesystem failed"),
        PermissionError("permission denied"),
        vane.VideoFileLimitError("resource bound"),
        RuntimeError("internal invariant"),
    ],
)
def test_video_skip_policy_propagates_system_resource_and_internal_errors(error, duckdb_cursor):
    source = VideoFrameSource(
        ["memory://fake"],
        height=2,
        width=3,
        max_pixels=100,
        on_error="skip",
    )
    value = _FakeVideoFile(error=error)

    with pytest.raises(type(error), match=str(error)):
        list(
            _decode_video_with_policy(
                value,
                options=source.options,
                max_output_frames=None,
                connection=duckdb_cursor,
            )
        )


def test_consumer_error_is_not_reclassified_as_bad_media(duckdb_cursor):
    record = _fake_record(0)
    value = _FakeVideoFile([record])
    source = VideoFrameSource(
        ["memory://fake"],
        height=2,
        width=3,
        max_pixels=100,
        on_error="skip",
    )
    batches = _decode_video_with_policy(
        value,
        options=source.options,
        max_output_frames=None,
        connection=duckdb_cursor,
    )
    next(batches)
    consumer_error = vane.VideoFileFormatError("downstream failure")

    with pytest.raises(vane.VideoFileFormatError) as raised:
        batches.throw(consumer_error)
    assert raised.value is consumer_error
    assert value.closed


def test_limited_video_task_propagates_context_and_global_limit(monkeypatch):
    source = VideoFrameSource(
        ["memory://a", "memory://b"],
        height=2,
        width=3,
        max_partition_bytes=1_000,
        frame_limit=3,
        max_pixels=100,
    )
    task = next(source.get_tasks())
    execution_context = object()
    decoded = []

    def fake_decode(value, *, options, max_output_frames, connection=None, execution_context=None):
        del options, connection
        decoded.append((value.url, max_output_frames, execution_context))
        indices = [0, 1]
        if max_output_frames is not None:
            indices = indices[:max_output_frames]
        yield _fake_batch(value, indices)

    monkeypatch.setattr(video_reader, "_decode_video_guarded", fake_decode)

    batches = list(task._execute_with_context(execution_context))

    assert decoded == [
        ("memory://a", 3, execution_context),
        ("memory://b", 1, execution_context),
    ]
    assert sum(batch.num_rows for batch in batches) == 3


def test_grouped_video_task_propagates_one_context_to_every_file(monkeypatch):
    source = VideoFrameSource(
        ["memory://a", "memory://b"],
        height=2,
        width=3,
        max_partition_bytes=10_000,
        read_task_count=1,
        max_pixels=100,
    )
    task = next(source.get_tasks())
    execution_context = object()
    decoded = []

    def fake_decode(value, *, options, max_output_frames, connection=None, execution_context=None):
        del options, max_output_frames, connection
        decoded.append((value.url, execution_context))
        yield _fake_batch(value, [0])

    monkeypatch.setattr(video_reader, "_decode_video_guarded", fake_decode)

    batches = list(task._execute_with_context(execution_context))

    assert decoded == [("memory://a", execution_context), ("memory://b", execution_context)]
    assert [batch.num_rows for batch in batches] == [1, 1]


def test_output_batch_size_accounts_for_file_and_provenance_columns():
    value = vane.VideoFile(
        "memory://" + "x" * 100,
        "video/mp4",
        checksum="sha256:" + "0" * 64,
    )
    bounds = _file_storage_bounds((value,))
    partition_bytes = 10_000

    actual = _video_output_batch_size(
        10,
        10,
        partition_bytes,
        max_file_string_bytes=bounds.max_string_bytes,
        max_file_row_bytes=bounds.max_row_bytes,
    )
    frame_only = partition_bytes // (10 * 10 * 3) + 1

    assert actual < frame_only


def test_output_batch_size_respects_arrow_string_offset_limit(monkeypatch):
    monkeypatch.setattr(video_reader, "_ARROW_STRING_DATA_MAX_BYTES", 100)

    assert (
        _video_output_batch_size(
            1,
            1,
            10_000,
            max_file_string_bytes=40,
            max_file_row_bytes=80,
        )
        == 2
    )


def test_video_frame_source_builds_typed_datasource_scan_plan(duckdb_cursor):
    source = VideoFrameSource([vane.VideoFile("memory://video")], height=8, width=9)

    relation = read_datasource(source, con=duckdb_cursor)
    plan = relation.explain()

    assert "DATASOURCE_SCAN" in plan
    assert "STREAMING_UDF" not in plan
    assert str(relation.types[0]) == "VIDEOFILE"


def test_video_frame_source_execution_preserves_range_alias_and_provenance(duckdb_cursor, tmp_path):
    value = _ranged_video(tmp_path, frame_count=8)
    source = VideoFrameSource(
        [value],
        height=6,
        width=8,
        max_partition_bytes=400,
        start_time=0.25,
        end_time=1.25,
        sample_interval_seconds=0.5,
        buffer_size=64,
        max_pixels=1000,
    )

    relation = read_datasource(source, con=duckdb_cursor).order("frame_index")
    assert str(relation.types[0]) == "VIDEOFILE"
    assert str(relation.types[-1]) == "TENSOR(UTINYINT, [6, 8, 3])"
    rows = relation.select(
        "file",
        "frame_index",
        "frame_time",
        "frame_time_base_numerator",
        "frame_time_base_denominator",
        "frame_pts",
        "frame_dts",
        "frame_duration",
        "is_key_frame",
        "frame",
    ).fetchall()

    assert [row[0] for row in rows] == [value] * 3
    assert [row[1] for row in rows] == [1, 3, 5]
    assert [row[2] for row in rows] == pytest.approx([0.25, 0.75, 1.25])
    assert all(row[3] == 1 and row[4] > 0 for row in rows)
    assert all(row[5] is not None and row[6] is not None and row[7] is not None for row in rows)
    assert all(isinstance(row[8], bool) for row in rows)
    assert all(isinstance(row[9], tuple) and len(row[9]) == 6 * 8 * 3 for row in rows)


def test_video_frame_source_uses_query_connection_context_without_default_fallback(duckdb_cursor, tmp_path):
    scoped_home = tmp_path / "query-home"
    default_home = tmp_path / "default-home"
    scoped_home.mkdir()
    default_home.mkdir()
    relative_name = "connection-scoped.mp4"
    (scoped_home / relative_name).write_bytes(_encoded_video(frame_count=1))
    duckdb_cursor.execute("SET home_directory = ?", [str(scoped_home)])

    previous_default = vane.default_connection()
    wrong_default = vane.connect()
    wrong_default.execute("SET home_directory = ?", [str(default_home)])
    vane.set_default_connection(wrong_default)
    try:
        rows = (
            read_datasource(
                VideoFrameSource(
                    [vane.VideoFile(f"~/{relative_name}", "video/mp4")],
                    height=6,
                    width=8,
                    max_pixels=1000,
                ),
                con=duckdb_cursor,
            )
            .select("frame_index")
            .fetchall()
        )
    finally:
        vane.set_default_connection(previous_default)
        wrong_default.close()

    assert rows == [(0,)]


def test_video_frame_source_interrupts_while_waiting_for_decode_slot(monkeypatch):
    entered = threading.Event()

    class UnavailableSemaphore:
        def acquire(self, *, timeout):
            entered.set()
            threading.Event().wait(timeout)
            return False

        def release(self):
            raise AssertionError("an unavailable decode slot must not be released")

    monkeypatch.setattr(video_reader, "_decode_semaphore", UnavailableSemaphore())
    connection = vane.connect()
    errors = []
    worker = None
    try:
        relation = read_datasource(
            VideoFrameSource(["memory://never-opened.mp4"], height=2, width=3, max_pixels=100),
            con=connection,
        )

        def fetch_frames():
            try:
                relation.fetchall()
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=fetch_frames)
        worker.start()
        assert entered.wait(timeout=5)
        connection.interrupt()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], vane.InvalidInputException)
        assert "INTERRUPT Error: Interrupted!" in str(errors[0])
    finally:
        connection.interrupt()
        if worker is not None:
            worker.join(timeout=5)
        connection.close()


def test_video_frame_source_global_frame_limit_is_ordered(duckdb_cursor, tmp_path):
    first_path = tmp_path / "first.mp4"
    second_path = tmp_path / "second.mp4"
    first_path.write_bytes(_encoded_video(frame_count=2))
    second_path.write_bytes(_encoded_video(frame_count=4))
    first = vane.VideoFile(str(first_path), "video/mp4")
    second = vane.VideoFile(str(second_path), "video/mp4")
    source = VideoFrameSource(
        [first, second],
        height=6,
        width=8,
        frame_limit=3,
        max_pixels=1000,
    )

    rows = read_datasource(source, con=duckdb_cursor).select("file", "frame_index").fetchall()

    assert rows == [(first, 0), (first, 1), (second, 0)]


def test_empty_video_frame_source_preserves_output_schema(duckdb_cursor):
    relation = read_datasource(VideoFrameSource([]), con=duckdb_cursor)

    assert [str(dtype) for dtype in relation.types] == [
        "VIDEOFILE",
        "BIGINT",
        "DOUBLE",
        "BIGINT",
        "BIGINT",
        "BIGINT",
        "BIGINT",
        "BIGINT",
        "BOOLEAN",
        "TENSOR(UTINYINT, [640, 480, 3])",
    ]
    assert relation.fetchall() == []


def test_video_frame_source_executes_ranged_videofile_on_real_ray(ray_local, monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    vane.teardown_runner()
    vane.set_runner_ray(noop_if_initialized=True)
    value = _ranged_video(tmp_path, frame_count=4)
    connection = vane.connect()
    try:
        relation = read_datasource(
            VideoFrameSource(
                [value],
                height=6,
                width=8,
                frame_limit=2,
                buffer_size=64,
                max_pixels=1000,
            ),
            con=connection,
        )
        assert str(relation.types[0]) == "VIDEOFILE"
        rows = relation.select("file", "frame_index", "frame_pts").fetchall()
    finally:
        connection.close()

    assert rows == [(value, 0, 0), (value, 1, 4096)]


def test_video_frame_source_skip_continues_after_corrupt_media_but_not_missing_file(
    monkeypatch,
    duckdb_cursor,
    tmp_path,
):
    monkeypatch.setattr(video_reader, "_wait_for_memory", lambda check: check())
    corrupt_path = tmp_path / "corrupt.mp4"
    valid_path = tmp_path / "valid.mp4"
    corrupt_path.write_bytes(b"not a video")
    valid_path.write_bytes(_encoded_video(frame_count=2))
    corrupt = vane.VideoFile(str(corrupt_path), "video/mp4")
    valid = vane.VideoFile(str(valid_path), "video/mp4")
    source = VideoFrameSource(
        [corrupt, valid],
        height=6,
        width=8,
        max_pixels=1000,
        on_error="skip",
    )
    rows = read_datasource(source, con=duckdb_cursor).select("file").fetchall()

    assert rows == [(valid,), (valid,)]

    missing = vane.VideoFile(str(tmp_path / "missing.mp4"), "video/mp4")
    missing_source = VideoFrameSource(
        [missing],
        height=6,
        width=8,
        max_pixels=1000,
        on_error="skip",
    )
    with pytest.raises(vane.IOException):
        list(
            _decode_video_with_policy(
                missing,
                options=missing_source.options,
                max_output_frames=None,
                connection=duckdb_cursor,
            )
        )


def test_video_frame_source_max_input_limit_propagates_in_skip_mode(tmp_path, duckdb_cursor):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")
    source = VideoFrameSource(
        [value],
        height=6,
        width=8,
        max_input_bytes=1,
        max_pixels=1000,
        on_error="skip",
    )

    with pytest.raises(vane.VideoFileLimitError, match="max_input_bytes=1"):
        list(
            _decode_video_with_policy(
                value,
                options=source.options,
                max_output_frames=None,
                connection=duckdb_cursor,
            )
        )
