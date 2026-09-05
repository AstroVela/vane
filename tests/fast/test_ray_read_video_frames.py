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
        assert relation.types[-1] == dtype
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
        finally:
            runner.close()
    assert table.num_rows == 9
    assert sorted(table.column("frame_index").to_pylist()) == [2] * 3 + [3] * 3 + [4] * 3
    assert set(table.column("frame_time").to_pylist()) == {0.5, 0.75, 1.0}
    assert table.column("path").to_pylist() == [source.url] * 9
    assert all(
        file == {name: getattr(source, name) for name in ("url", "content_type", "position", "size", "checksum")}
        for file in table.column("file").to_pylist()
    )
    assert all(
        image["width"] == 8
        and image["height"] == 6
        and image["channels"] == 3
        and image["mode"] == "RGB"
        and len(image["data"]) == 144
        for image in table.column("data").to_pylist()
    )
