# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
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
@pytest.mark.parametrize(
    "task_count,frame_limit,expected_splits", [(None, None, 5), (1, None, 1), (2, None, 2), (20, None, 5), (2, 3, 1)]
)
def test_ray_native_video_splits_preserve_files_and_frames(
    ray_local, request, task_count, frame_limit, expected_splits
):
    import pyarrow as pa

    from vane.datasource.video_reader import VideoFrameSource
    from vane.runners.ray.runner import RayRunner

    path = request.getfixturevalue("video_path")
    with vane.connect(config={"video_backend": "native"}) as con:
        _load_provider(con, "video")
        source = VideoFrameSource(
            [str(path)] * 5,
            width=8,
            height=6,
            start_time=0.5,
            end_time=1.0,
            read_task_count=task_count,
            frame_limit=frame_limit,
        )
        relation = con.from_datasource(source).project("file, frame_index, frame_time, frame")
        assert relation.types[-1].is_image()
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "native-video-groups").to_physical_plan(con)
        assert [len(batches) for batches in plan.scan_split_batch_map().values()] == [expected_splits]
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
            assert table.num_columns == 4
            repeats = 1 if frame_limit is not None else 5
            assert sorted(table.column(1).to_pylist()) == [2] * repeats + [3] * repeats + [4] * repeats
            assert set(table.column(2).to_pylist()) == {0.5, 0.75, 1.0}
            assert all(value["url"] == str(path) for value in table.column(0).to_pylist())
            assert all(
                value["width"] == 8 and value["height"] == 6 and value["mode"] == "RGB" and len(value["data"]) == 144
                for value in table.column(3).to_pylist()
            )
        finally:
            runner.close()


@pytest.mark.real_ray
def test_ray_concurrent_connections_keep_independent_media_backends(ray_local, request, tmp_path):
    import pyarrow as pa

    from vane.runners.ray.runner import RayRunner

    files = []
    for path, kind, mime in (
        (request.getfixturevalue("image_path"), vane.ImageFile, "image/png"),
        (request.getfixturevalue("audio_path"), vane.AudioFile, "audio/wav"),
    ):
        payload = path.read_bytes()
        prefix = b"outside FILE view" * 9
        bundle = tmp_path / f"{path.name}.bundle"
        bundle.write_bytes(prefix + payload + b"outside suffix")
        files.append(kind(str(bundle), mime, len(prefix), len(payload)))
    image_sql, audio_sql = (str(vane.ConstantExpression(file)) for file in files)
    with ExitStack() as stack:
        queries = []
        for image_backend, audio_backend in (("native", "python"), ("python", "native")):
            con = stack.enter_context(
                vane.connect(config={"image_backend": image_backend, "audio_backend": audio_backend})
            )
            _load_provider(con, "image" if image_backend == "native" else "audio")
            runner = RayRunner(address=None, max_task_backlog=None)
            stack.callback(runner.close)
            relation = con.sql(
                f"SELECT decode_image_file({image_sql}, 'RGB') AS image, "
                f"audio_metadata({audio_sql}) AS audio, {audio_sql} AS file FROM range(3)"
            )
            plan = con.sql(
                f"SELECT decode_image_file({image_sql}), audio_metadata({audio_sql}) FROM range(1)"
            ).explain()
            assert ("native_decode_image_file" in plan) == (image_backend == "native")
            assert ("native_audio_metadata" in plan) == (audio_backend == "native")
            queries.append((runner, relation, audio_backend))
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=2))

        def collect(query):
            runner, relation, audio_backend = query
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
            assert table.num_rows == 3
            for row in table.to_pylist():
                assert row["image"]["data"] == bytes((20, 80, 160)) * 15
                assert row["audio"]["format"] == ("wav" if audio_backend == "native" else "WAV")
                assert row["file"]["position"] == files[1].position
                assert row["file"]["size"] == files[1].size

        for _ in range(2):
            list(executor.map(collect, queries))


@pytest.mark.real_ray
def test_ray_native_audio_profile_retains_runtime_and_waveform_contract(ray_local, request):
    import pyarrow as pa

    from vane.runners.ray.runner import RayRunner

    with vane.connect(config={"audio_backend": "native"}) as con:
        _load_provider(con, "audio")
        file_sql = str(vane.ConstantExpression(vane.AudioFile(str(request.getfixturevalue("audio_path")))))
        relation = con.sql(f"SELECT native_audio_resample_profile({file_sql}, 16000) AS profile FROM range(4)")
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
            profiles = table.column(0).to_pylist()
            assert len(profiles) == 4
            reference = con.execute(f"SELECT native_audio_resample_profile({file_sql}, 16000)").fetchone()[0]
            for profile in profiles:
                for field in ("codec_version", "resampler_version", "decoded_frames", "output_frames", "output_bytes"):
                    assert profile[field] == reference[field]
                assert profile["file_bytes_read"] > 0
        finally:
            runner.close()


@pytest.mark.real_ray
@pytest.mark.parametrize("domain", ["image", "audio", "video"])
def test_ray_media_identity_rejection_and_preparation_retry(ray_local, domain):
    import ray

    from tests.fast.test_ray_dynamic_extension_replay import _run_real_ray_dynamic_extension_rejection_matrix
    from vane.runners.ray.worker import RayWorkerActor

    with vane.connect(config={f"{domain}_backend": "native"}) as con:
        _load_provider(con, domain)
        function = "image_file_metadata" if domain == "image" else f"{domain}_metadata"
        query_id = f"native-{domain}-identity"
        relation = con.sql(f"SELECT {function}({domain}_file('unopened://media')) FROM range(1)")
        physical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, query_id).to_physical_plan(con)
        state = list(physical.__getstate__())
        snapshot = dict(state[6])
        manifest = snapshot["dynamic_extensions"]
        assert len(manifest) == 1 and manifest[0]["name"] == domain
        original = manifest[0]
        snapshot["dynamic_extensions"] = [{**original, "sha256": "0" * 64}]
        state[6] = snapshot
        altered = vane.ray_cxx.DistributedPhysicalPlan.__new__(vane.ray_cxx.DistributedPhysicalPlan)
        altered.__setstate__(tuple(state))
        actor = RayWorkerActor.options(num_cpus=0).remote(1, 0, 1 << 28, 1 << 28)
        try:
            with pytest.raises(Exception, match="VANE_DYNAMIC_EXTENSION_PROVIDER_DESCRIPTOR_MISMATCH"):
                ray.get(
                    actor.register_fragments.remote(
                        [{"fragment_id": f"{query_id}:node:1", "plan": ray.put(altered), "query_id": query_id}]
                    )
                )
            assert ray.get(actor.stats_fragments.remote())["fragments_total"] == 0
            fragment = {"fragment_id": f"{query_id}:node:1", "plan": ray.put(physical), "query_id": query_id}
            assert ray.get(actor.register_fragments.remote([fragment]))["registered"] == 1
            before = ray.get(actor.stats_snapshot_databases.remote())
            assert ray.get(actor.register_fragments.remote([fragment]))["existing"] == 1
            assert ray.get(actor.stats_snapshot_databases.remote()) == before
            assert ray.get(actor.drop_query_fragments.remote(query_id)) == 1
            ray.get(actor.fte_cleanup_query.remote(query_id))
        finally:
            ray.kill(actor, no_restart=True)
        rejection = ray.remote(_run_real_ray_dynamic_extension_rejection_matrix)
        result = ray.get(rejection.remote([{"name": domain, "manifest": [original]}]))
        assert result == {domain: "PROVIDER_NOT_FOUND"}
