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
def test_ray_video_scalars_preserve_nested_images_and_file_windows(ray_local, video_path, tmp_path, backend):
    from vane.runners.ray.runner import RayRunner

    payload = video_path.read_bytes()
    prefix = b"outside video window\0" * 13
    path = tmp_path / "bounded-video.bin"
    path.write_bytes(prefix + payload + b"outside suffix")
    source = vane.VideoFile(str(path), "video/mp4", len(prefix), len(payload), "sha256:opaque")
    dtype = vane.list_type(vane.image_type())

    @vane.func.batch(return_dtype=dtype)
    def identity(images):
        return images

    with vane.connect(config={"video_backend": backend}) as con:
        if backend == "native":
            _load_provider(con, "video")
        vane.attach_function(identity, connection=con, alias="nested_frame_identity", parameters=[dtype])
        file_sql = str(vane.ConstantExpression(source))
        relation = con.sql(
            "SELECT video_frames(file, end_time => 0.25, width => 8, height => 6) AS frames, "
            "nested_frame_identity(video_keyframes(file, width => 8, height => 6)) AS keys, "
            "get_video_frame_by_idx(file, 1) AS image "
            f"FROM (SELECT {file_sql} AS file FROM range(3)) videos"
        )
        assert relation.columns == ["frames", "keys", "image"]
        assert relation.types[1:] == [dtype, vane.image_type()]
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
        finally:
            runner.close()
    assert table.num_rows == 3
    for frames in table.column(0).to_pylist():
        assert [frame["frame_index"] for frame in frames] == [0, 1]
        assert [frame["frame_time"] for frame in frames] == [0, 0.25]
        assert all(
            frame["file"]
            == {key: getattr(source, key) for key in ("url", "content_type", "position", "size", "checksum")}
            for frame in frames
        )
        assert all(len(frame["data"]["data"]) == 144 and frame["data"]["mode"] == "RGB" for frame in frames)
    assert all(
        keys and all(image["mode"] == "RGB" and len(image["data"]) == 144 for image in keys)
        for keys in table.column(1).to_pylist()
    )
    assert all(
        image["mode"] == "RGB" and image["width"] > 0 and image["height"] > 0 for image in table.column(2).to_pylist()
    )
