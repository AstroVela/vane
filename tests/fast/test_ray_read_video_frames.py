# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import vane
from tests.fast import test_native_media_extensions as media_tests
from tests.fast.test_ray_native_media_extensions import _load_provider

video_path = media_tests.video_path


@pytest.mark.real_ray
@pytest.mark.parametrize("backend", ["python", "native"])
def test_ray_streaming_video_preserves_file_and_fixed_image_through_udf_and_exchange(ray_local, video_path, backend):
    from vane.runners.ray.runner import RayRunner

    if backend == "python":
        pytest.importorskip("psutil")
    dtype = vane.image_type("RGB", 6, 8)

    @vane.func.batch(return_dtype=dtype)
    def identity(images):
        return images

    with vane.connect(config={"video_backend": backend}) as con:
        if backend == "native":
            _load_provider(con, "video")
        vane.attach_function(identity, connection=con, alias="fixed_frame_identity", parameters=[dtype])
        payload_size = video_path.stat().st_size
        source = vane.VideoFile(str(video_path), "video/mp4", 0, payload_size, "sha256:opaque")
        frames = vane.read_video_frames(
            [source] * 3, 6, 8, start_time=0.5, end_time=1, read_task_count=2, connection=con
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(frames, "public-video-groups").to_physical_plan(con)
        assert [len(batches) for batches in plan.scan_split_batch_map().values()] == [2]
        relation = frames.project("path, file, frame_index, frame_time, fixed_frame_identity(data) AS data").order(
            "frame_index DESC"
        )
        assert relation.columns == ["path", "file", "frame_index", "frame_time", "data"]
        assert relation.types[-1] == dtype
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
        finally:
            runner.close()
    assert table.num_rows == 9
    # run_iter_tables exposes physical output names; the relation owns the
    # public column names checked above.
    assert table.num_columns == 5
    assert sorted(table.column(2).to_pylist()) == [2] * 3 + [3] * 3 + [4] * 3
    assert set(table.column(3).to_pylist()) == {0.5, 0.75, 1.0}
    assert table.column(0).to_pylist() == [source.url] * 9
    assert all(
        file == {name: getattr(source, name) for name in ("url", "content_type", "position", "size", "checksum")}
        for file in table.column(1).to_pylist()
    )
    assert all(
        image["width"] == 8
        and image["height"] == 6
        and image["channels"] == 3
        and image["mode"] == "RGB"
        and len(image["data"]) == 144
        for image in table.column(4).to_pylist()
    )


@pytest.mark.real_ray
def test_ray_nested_explicit_image_cast_retains_validation_mode(ray_local):
    from vane.runners.ray.runner import RayRunner

    @vane.func(return_dtype=vane.image_type())
    def pixels(index):
        return vane.Image(bytes([index]) * 3, 1, 1, "RGB")

    with vane.connect() as con:
        vane.attach_function(pixels, connection=con, alias="cast_pixels", parameters=["BIGINT"])
        relation = con.sql(
            "SELECT CAST({'images': [cast_pixels(i), NULL]} AS STRUCT(images IMAGE('RGB', 1, 1)[])) AS value "
            "FROM range(3) t(i)"
        )
        assert relation.types == [vane.struct_type({"images": vane.list_type(vane.image_type("RGB", 1, 1))})]
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
        finally:
            runner.close()
    rows = table.column(0).to_pylist()
    assert sorted(row["images"][0]["data"] for row in rows) == [bytes([i]) * 3 for i in range(3)]
    assert all(row["images"][1] is None and row["images"][0]["mode"] == "RGB" for row in rows)
