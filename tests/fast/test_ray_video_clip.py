# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

import vane
from tests.fast.test_video_clip import decode_video, make_av_video


@pytest.mark.real_ray
def test_video_clips_use_default_ray_without_runner_configuration(ray_local, monkeypatch, tmp_path):
    source = make_av_video(tmp_path / "source.mp4", origin=3)
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    vane.teardown_runner()
    with vane.connect() as connection:
        relation = connection.sql(
            f"SELECT i, video_clip({vane.ConstantExpression(source)}, i * 0.25, i * 0.25 + 0.5) AS clip "
            "FROM range(3) t(i)"
        )
        rows = relation.order("i").fetchall()
    assert len(rows) == 3
    for index, clip in rows:
        assert clip["start_time"] == index * 0.25
        assert clip["has_audio"]
        frames, duration = decode_video(clip["data"])
        assert frames[0][0] == 0
        assert duration == pytest.approx(0.5)
    vane.teardown_runner()


@pytest.mark.real_ray
def test_video_clip_limit_propagates_from_ray_worker(ray_local, monkeypatch, tmp_path):
    source = make_av_video(tmp_path / "source.mp4")
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    vane.teardown_runner()
    try:
        with vane.connect() as connection:
            with pytest.raises(Exception, match="max_output_bytes"):
                connection.sql(
                    f"SELECT video_clip({vane.ConstantExpression(source)}, 0, 1, max_output_bytes => 128) FROM range(2)"
                ).fetchall()
    finally:
        vane.teardown_runner()
