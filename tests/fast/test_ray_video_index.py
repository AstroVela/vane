# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import vane
from tests.fast import test_native_media_extensions as media
from tests.fast.test_ray_native_media_extensions import _load_provider

video_path = media.video_path


def _collect(relation):
    from vane.runners.ray.runner import RayRunner

    runner = RayRunner(address=None, max_task_backlog=None)
    try:
        parts = list(runner.run_iter_tables(relation))
        return pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
    finally:
        runner.close()


@pytest.mark.real_ray
def test_ray_builds_and_consumes_explicit_video_indexes(ray_local, video_path):
    file = vane.VideoFile(str(video_path), "video/mp4")
    with vane.connect(config={"video_backend": "native"}) as con:
        _load_provider(con, "video")
        file_sql = str(vane.ConstantExpression(file))
        relation = con.sql(f"SELECT build_video_index({file_sql}) AS seek_index FROM range(2)")
        indexes = _collect(relation).column(0).to_pylist()
        assert len(indexes) == 2 and indexes[0] == indexes[1]
        index_sql = str(vane.ConstantExpression(indexes[0]))
        relation = con.sql(
            f"SELECT get_video_frame_by_idx({file_sql}, 8, index => {index_sql}) AS image, "
            f"video_scan_stats({file_sql}, idx => 8, index => {index_sql}) AS stats FROM range(3)"
        )
        table = _collect(relation)
        baseline = con.execute("SELECT get_video_frame_by_idx($1, 8)", [file]).fetchone()[0]
        assert table.num_rows == 3
        assert all(image["data"] == baseline.data for image in table.column(0).to_pylist())
        assert all(stats["seeks"] == 1 and stats["decoded_frames"] < 9 for stats in table.column(1).to_pylist())


@pytest.mark.real_ray
@pytest.mark.parametrize("group_count", [1, 2])
def test_ray_streaming_splits_keep_indexes_with_their_file_views(ray_local, video_path, tmp_path, group_count):
    payload = video_path.read_bytes()
    files = []
    for number in range(3):
        prefix = b"outside window" * (number + 1)
        path = tmp_path / f"video-{number}.bin"
        path.write_bytes(prefix + payload + b"outside suffix")
        files.append(vane.VideoFile(str(path), "video/mp4", len(prefix), len(payload)))
    with vane.connect(config={"video_backend": "native"}) as con:
        _load_provider(con, "video")
        indexes = [con.execute("SELECT build_video_index($1)", [file]).fetchone()[0] for file in files]
        relation = vane.read_video_frames(
            files, 6, 8, start_time=2, end_time=2.25, indexes=indexes, read_task_count=group_count, connection=con
        )
        table = _collect(relation)
        assert table.num_rows == 6
        assert sorted(table.column(2).to_pylist()) == [8] * 3 + [9] * 3
        assert sorted(table.column(0).to_pylist()) == sorted([file.url for file in files] * 2)
        assert all(image["width"] == 8 and image["height"] == 6 for image in table.column(10).to_pylist())
        for file in table.column(1).to_pylist():
            expected = next(source for source in files if source.url == file["url"])
            assert file["position"] == expected.position and file["size"] == expected.size
