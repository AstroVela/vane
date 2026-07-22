# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from vane.datasource import _schema_to_arrow
from vane.datasource.video_reader import (
    LimitedVideoFrameTask,
    VideoFrameSource,
    VideoFrameTask,
    _coalesce_video_frame_batches,
    _decode_video_batches,
    _flush_frame_batch,
    _resize_frame_batch,
    _split_video_path_groups,
    _video_frame_source_manifest_sql,
    _video_frame_source_map_batches,
    _video_source_udf_output_batch_size,
)


def test_video_frame_source_uses_one_ordered_task_for_frame_limit():
    source = VideoFrameSource(["a.avi", "b.avi"], height=8, width=8, frame_limit=3)

    tasks = list(source.get_tasks())

    assert len(tasks) == 1
    assert isinstance(tasks[0], LimitedVideoFrameTask)
    assert tasks[0].paths == ["a.avi", "b.avi"]
    assert tasks[0].max_frames == 3


def test_video_frame_source_keeps_parallel_per_file_tasks_without_frame_limit():
    source = VideoFrameSource(["a.avi", "b.avi"], height=8, width=8)

    tasks = list(source.get_tasks())

    assert len(tasks) == 2
    assert all(isinstance(task, VideoFrameTask) for task in tasks)


def test_video_frame_source_manifest_groups_paths_like_ray_read_tasks():
    source = VideoFrameSource(
        [f"clip-{index}.avi" for index in range(11)],
        height=8,
        width=8,
        read_task_count=4,
    )

    sql = _video_frame_source_manifest_sql(source)

    assert [len(group) for group in _split_video_path_groups(source.paths, 4)] == [3, 3, 3, 2]
    assert sql.count("list_value(") == 4
    assert "video_paths::VARCHAR[]" in sql


def test_video_source_uses_ray_soft_block_row_boundary():
    assert _video_source_udf_output_batch_size(640, 640, 128 * 1024**2) == 110


def test_video_source_transport_does_not_resplit_ray_soft_block(monkeypatch):
    import vane.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    target_bytes = 128 * 1024**2
    kwargs = video_reader._video_source_udf_kwargs(
        height=640,
        width=640,
        max_partition_bytes=target_bytes,
    )

    assert kwargs["output_batch_size"] == 110
    assert kwargs["output_target_max_bytes"] == 2 * target_bytes
    assert kwargs["preserve_compute_batch_boundaries"] is True


def test_video_source_coalesces_file_tails_within_one_read_task():
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)

    def batches():
        for name in ("a.avi", "b.avi"):
            yield pa.record_batch(
                {
                    "video_path": [name] * 3,
                    "frame_index": [0, 1, 2],
                    "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frames),
                }
            )

    output = list(_coalesce_video_frame_batches(batches(), target_rows=5))

    assert [table.num_rows for table in output] == [5, 1]
    assert output[0].column("video_path").to_pylist() == ["a.avi"] * 3 + ["b.avi"] * 2


def test_video_decode_batches_do_not_mutate_emitted_arrow_buffers(monkeypatch):
    import vane.datasource.video_reader as video_reader

    class FakeFrame:
        def __init__(self, value):
            self._value = value

        def asnumpy(self):
            return np.full((2, 2, 3), self._value, dtype=np.uint8)

    monkeypatch.setattr(video_reader, "_open_decord_reader", lambda _path: [FakeFrame(i) for i in range(5)])
    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 1)

    batches = list(
        _decode_video_batches(
            "clip.avi",
            height=2,
            width=2,
            # One 12-byte frame reaches the soft target, so each batch has 2 rows.
            max_partition_bytes=12,
        )
    )

    values = [batch.column("frame").to_numpy_ndarray()[:, 0, 0, 0].tolist() for batch in batches]
    assert values == [[0, 1], [2, 3], [4]]


def test_datasource_schema_supports_fixed_shape_tensor_entries():
    schema = _schema_to_arrow(
        {
            "frame_index": "BIGINT",
            "frame": {"kind": "tensor", "dtype": "UINT8", "shape": [4, 5, 3]},
        }
    )

    assert schema.field("frame_index").type == pa.int64()
    assert schema.field("frame").type == pa.fixed_shape_tensor(pa.uint8(), (4, 5, 3))


def test_video_frame_source_schema_declares_typed_frame_not_blob():
    source = VideoFrameSource(["a.avi"], height=4, width=5)

    assert source.schema == {
        "video_path": "VARCHAR",
        "frame_index": "BIGINT",
        "frame": {"kind": "tensor", "dtype": "UINT8", "shape": [4, 5, 3]},
    }


def test_read_datasource_uses_datasource_udf_relation_hook():
    import vane
    from vane.datasource import DataSource, read_datasource

    class HookSource(DataSource):
        @property
        def schema(self):
            return {"value": "INTEGER"}

        def get_tasks(self):
            raise AssertionError("native datasource scan should not run")

        def to_udf_relation(self, con):
            return con.sql("select 42::INTEGER as value")

    con = vane.connect()

    assert read_datasource(HookSource(), con=con).fetchall() == [(42,)]


def test_video_frame_source_read_datasource_builds_hidden_udf_plan(monkeypatch):
    import vane
    from vane.datasource import read_datasource

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    con = vane.connect()

    plan = read_datasource(VideoFrameSource(["a.avi"], height=8, width=9), con=con).explain()
    compact_plan = "".join(ch for ch in plan if ch.isalnum() or ch == "_")

    assert "STREAMING_UDF" in plan
    assert "_video_frame_source_map_batches" in compact_plan
    assert "execution_backend" in plan
    assert "ray_task" in plan
    assert "udf_queue_depth" not in compact_plan
    assert "udf_max_outstanding_batches" not in compact_plan
    assert "udf_max_ready_rows" not in compact_plan


def test_video_source_udf_identity_is_assigned_by_physical_graph(monkeypatch):
    import vane.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_VIDEOS_PER_TASK", "1")
    kwargs = video_reader._video_source_udf_kwargs()

    assert kwargs["execution_backend"] == "ray_task"
    assert kwargs["memory_bytes"] == 512 * 1024**2
    assert kwargs["cpus"] == 1.0
    assert "queue_depth" not in kwargs
    assert "query_id" not in kwargs
    assert "fragment_id" not in kwargs
    assert "operator_id" not in kwargs
    assert "max_outstanding_batches" not in kwargs


def test_video_source_udf_cpu_default_accounts_for_resize_pool(monkeypatch):
    import vane.datasource.video_reader as video_reader

    monkeypatch.delenv("VANE_VIDEO_SOURCE_UDF_CPUS", raising=False)
    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 3)

    assert video_reader._video_source_udf_kwargs()["cpus"] == 3.0


def test_video_source_udf_cpu_allocation_is_overridable(monkeypatch):
    import vane.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_CPUS", "2.5")
    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 4)

    assert video_reader._video_source_udf_kwargs()["cpus"] == 2.5


def test_video_source_udf_cpu_allocation_must_be_positive(monkeypatch):
    import vane.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_CPUS", "0")

    with pytest.raises(ValueError, match="VANE_VIDEO_SOURCE_UDF_CPUS must be positive"):
        video_reader._video_source_udf_kwargs()


def test_video_source_udf_memory_is_stage_specific(monkeypatch):
    import vane.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_MEMORY_BYTES", "268435456")
    monkeypatch.setenv("VANE_UDF_TASK_HEAP_BYTES", "1073741824")

    assert video_reader._video_source_udf_kwargs()["memory_bytes"] == 268435456

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "subprocess_task")
    assert "memory_bytes" not in video_reader._video_source_udf_kwargs()


def test_video_source_udf_memory_must_be_positive(monkeypatch):
    import vane.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_MEMORY_BYTES", "0")

    with pytest.raises(ValueError, match="VANE_VIDEO_SOURCE_UDF_MEMORY_BYTES must be positive"):
        video_reader._video_source_udf_kwargs()


def test_video_frame_source_map_batches_reads_manifest(monkeypatch):
    import vane.datasource.video_reader as video_reader

    calls = []
    frames = np.arange(2 * 8 * 9 * 3, dtype=np.uint8).reshape(2, 8, 9, 3)

    def fake_decode(video_path, *, height, width, max_partition_bytes, max_frames=None):
        calls.append((video_path, height, width, max_partition_bytes, max_frames))
        yield pa.record_batch(
            {
                "video_path": [video_path, video_path],
                "frame_index": [0, 1],
                "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frames),
            }
        )

    monkeypatch.setattr(video_reader, "_wait_for_memory", lambda: None)
    monkeypatch.setattr(video_reader, "_decode_video_batches", fake_decode)
    manifest = pa.table(
        {
            "video_path": ["a.avi", "b.avi"],
            "height": [8, 8],
            "width": [9, 9],
            "max_partition_bytes": [1024, 1024],
            "frame_limit": pa.array([None, None], type=pa.int64()),
        }
    )

    tables = list(_video_frame_source_map_batches(manifest))

    assert calls == [
        ("a.avi", 8, 9, 1024, None),
        ("b.avi", 8, 9, 1024, None),
    ]
    assert [table.select(["video_path", "frame_index"]).to_pydict() for table in tables] == [
        {"video_path": ["a.avi", "a.avi"], "frame_index": [0, 1]},
        {"video_path": ["b.avi", "b.avi"], "frame_index": [0, 1]},
    ]
    for table in tables:
        np.testing.assert_array_equal(table.column("frame").combine_chunks().to_numpy_ndarray(), frames)


def test_video_frame_source_map_batches_honors_global_frame_limit(monkeypatch):
    import vane.datasource.video_reader as video_reader

    calls = []

    def fake_decode(video_path, *, height, width, max_partition_bytes, max_frames=None):
        calls.append((video_path, max_frames))
        row_count = min(2, max_frames if max_frames is not None else 2)
        if row_count > 0:
            frames = np.zeros((row_count, 8, 9, 3), dtype=np.uint8)
            yield pa.record_batch(
                {
                    "video_path": [video_path] * row_count,
                    "frame_index": list(range(row_count)),
                    "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frames),
                }
            )

    monkeypatch.setattr(video_reader, "_wait_for_memory", lambda: None)
    monkeypatch.setattr(video_reader, "_decode_video_batches", fake_decode)
    manifest = pa.table(
        {
            "video_paths": pa.array([["a.avi", "b.avi"]], type=pa.list_(pa.string())),
            "height": [8],
            "width": [9],
            "max_partition_bytes": [1024],
            "frame_limit": pa.array([3], type=pa.int64()),
        }
    )

    tables = list(_video_frame_source_map_batches(manifest))

    assert calls == [("a.avi", 3), ("b.avi", 1)]
    assert sum(table.num_rows for table in tables) == 3


def test_resize_frame_batch_preserves_order_and_uses_configured_threads(monkeypatch):
    import vane.datasource.video_reader as video_reader

    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 2)
    frame_a = np.zeros((2, 3, 3), dtype=np.uint8)
    frame_b = np.full((2, 3, 3), 255, dtype=np.uint8)

    resized = _resize_frame_batch([frame_a, frame_b], width=5, height=4)

    assert len(resized) == 2
    assert resized[0].shape == (4, 5, 3)
    assert resized[1].shape == (4, 5, 3)
    assert int(resized[0].mean()) == 0
    assert int(resized[1].mean()) == 255


def test_flush_frame_batch_uses_fixed_shape_tensor_for_frames():
    resized = np.arange(2 * 2 * 3 * 3, dtype=np.uint8).reshape(2, 2, 3, 3)

    batch = _flush_frame_batch("clip.avi", resized, [5, 6], 2)
    frame = batch.column("frame")

    assert batch.column("video_path").to_pylist() == ["clip.avi", "clip.avi"]
    assert batch.column("frame_index").to_pylist() == [5, 6]
    assert frame.type == pa.fixed_shape_tensor(pa.uint8(), (2, 3, 3))
    np.testing.assert_array_equal(frame.to_numpy_ndarray(), resized)
