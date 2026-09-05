# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from importlib.metadata import entry_points

import pytest

import vane
from tests.fast.test_native_media_extensions import audio_path, image_path, video_path  # noqa: F401

pytestmark = pytest.mark.skipif(
    os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") != "1",
    reason="requires signed native image/audio/video provider wheels",
)


def _load_provider(con, name):
    if os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") != "1":
        pytest.skip("set VANE_TEST_NATIVE_MEDIA_PROVIDERS=1 with signed image/audio/video provider wheels installed")
    assert len([ep for ep in entry_points(group="vane.dynamic_extension_providers") if ep.name == name]) == 1
    vane.load_installed_extension(name, connection=con)


@pytest.mark.real_ray
@pytest.mark.parametrize(
    "domain,fixture,expression,expected",
    [
        ("image", "image_path", "(image_file_metadata(image_file(url))).width", 5),
        ("audio", "audio_path", "(audio_resample(audio_file(url), 16000)).frames", 1600),
        ("video", "video_path", "(video_metadata(video_file(url))).frame_count", 12),
    ],
)
def test_ray_executes_native_scalar_with_exact_installed_provider(
    ray_local, request, domain, fixture, expression, expected
):
    import pyarrow as pa
    from vane.runners.ray.runner import RayRunner

    path = request.getfixturevalue(fixture)
    with vane.connect(config={f"{domain}_backend": "native"}) as con:
        _load_provider(con, domain)
        quoted_path = str(path).replace("'", "''")
        relation = con.sql(f"SELECT {expression} AS result FROM (SELECT '{quoted_path}' AS url FROM range(8)) inputs")
        assert "RANGE" in relation.explain()
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            for _ in range(2):
                parts = list(runner.run_iter_tables(relation))
                table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
                assert table.num_columns == 1
                assert table.column(0).to_pylist() == [expected] * 8
        finally:
            runner.close()


@pytest.mark.real_ray
def test_ray_native_video_splits_preserve_files_and_frames(ray_local, video_path):
    import pyarrow as pa
    from vane.datasource.video_reader import VideoFrameSource
    from vane.runners.ray.runner import RayRunner

    with vane.connect(config={"video_backend": "native"}) as con:
        _load_provider(con, "video")
        source = VideoFrameSource([str(video_path)] * 3, width=8, height=6, start_time=0.5, end_time=1.0)
        relation = con.from_datasource(source).project("file, frame_index, frame_time")
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
            assert table.num_columns == 3
            assert sorted(table.column(1).to_pylist()) == [2] * 3 + [3] * 3 + [4] * 3
            assert set(table.column(2).to_pylist()) == {0.5, 0.75, 1.0}
            assert all(value["url"] == str(video_path) for value in table.column(0).to_pylist())
        finally:
            runner.close()
