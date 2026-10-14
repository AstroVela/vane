# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-query video ingestion plans with deterministic encoders and timed content."""

import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from vane.ai import VideoEmbeddingPipeline, VideoEmbeddingProfile, VideoFusionProfile
from vane.ai._audio_embedding import AudioClip
from vane.ai._video_content import decode_content, encode_content
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai._video_ingestion import _stage_options
from vane.ai.protocols import TextEmbedderDescriptor, VideoEmbedderDescriptor
from vane.ai.video import TimedContent, VideoMediaInfo


class Encoder:
    def embed_text(self, inputs):
        return [np.array([len(value), 1], dtype=np.float32) for value in inputs]

    def embed_video(self, inputs):
        return [np.array([len(value.frames), 2, 3], dtype=np.float32) for value in inputs]


class TextProfile(TextEmbedderDescriptor):
    def get_provider(self):
        return "test"

    def get_model(self):
        return "text"

    def get_dimensions(self):
        return 2

    def get_options(self):
        return {}

    def instantiate(self):
        return Encoder()


class VisualProfile(VideoEmbedderDescriptor):
    def get_provider(self):
        return "test"

    def get_model(self):
        return "video"

    def get_dimensions(self):
        return 3

    def get_options(self):
        return {}

    def get_input_spec(self):
        return VideoInputSpec(max_frames=8)

    def instantiate(self):
        return Encoder()


class WideTextProfile(TextProfile):
    def get_dimensions(self):
        return 3

    def instantiate(self):
        class WideEncoder:
            def embed_text(self, inputs):
                return [np.array([len(value), 1, 2], dtype=np.float32) for value in inputs]

        return WideEncoder()


def test_video_ingestion_rejects_missing_input_identity(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local")
    with (
        vane.connect() as connection,
        VideoEmbeddingPipeline(profiles={"visual": VideoEmbeddingProfile("video-v1", VisualProfile())}) as pipeline,
    ):
        with pytest.raises(ValueError, match="VARCHAR asset_id"):
            pipeline.batch_ingest(connection.sql("SELECT 'video' AS uri"), modalities=["visual"])


@pytest.mark.real_ray
@pytest.mark.parametrize(
    "output,backend",
    [("separate", "ray_actor"), ("separate", "ray_task"), ("fused", "ray_actor"), ("both", "ray_actor")],
)
def test_video_ingestion_preserves_priority_and_independent_visual(ray_local, monkeypatch, tmp_path, output, backend):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    directory = str(tmp_path)

    class Source:
        supported_modalities = frozenset({"visual", "subtitles", "transcription", "ocr"})

        def probe(self, uri):
            with Path(directory, "probe").open("a") as log:
                log.write(uri + "\n")
            return VideoMediaInfo(20, self.supported_modalities)

        def extract(self, uri, channel, clip, media):
            with Path(directory, channel).open("a") as log:
                log.write(str(clip.start) + "\n")
            if channel == "visual":
                return [
                    TimedContent(
                        clip.start, clip.end, VideoClip((np.zeros((2, 2, 3), dtype=np.uint8),), (clip.start + 1,), (0,))
                    )
                ]
            if channel == "subtitles":
                return [TimedContent(1, 2, "subtitle")] if clip.start == 0 else []
            if channel == "transcription":
                assert clip.start == 10
                return [TimedContent(11, 12, "speech")]
            return []

    profiles = {
        c: VideoEmbeddingProfile(c + "-v1", VisualProfile() if c == "visual" else TextProfile())
        for c in Source.supported_modalities
    }
    fusion = VideoFusionProfile(
        "joint-v1",
        "test",
        "joint",
        2,
        (frozenset({"visual", "subtitles"}), frozenset({"visual", "transcription"})),
        lambda clip, selected, options: [clip.start, len(selected)],
    )
    source = tmp_path / "assets.parquet"
    pq.write_table(pa.table({"asset_id": ["a"], "source_version": ["v1"], "uri": ["memory://video"]}), source)
    with (
        vane.connect() as connection,
        VideoEmbeddingPipeline(profiles=profiles, source=Source(), fusion=fusion) as pipeline,
    ):
        relation = pipeline.batch_ingest(
            connection.read_parquet(str(source)),
            modalities=["visual", "subtitles", "transcription", "ocr"],
            selection={
                "policy": "all",
                "groups": [
                    {
                        "modalities": ["subtitles", "transcription"],
                        "policy": "first_available",
                        "priority": ["subtitles", "transcription"],
                    }
                ],
            },
            output=output,
            resources={"embed": {"execution_backend": backend, "cpus": 0.25}},
        )
        relation.write_parquet(str(tmp_path / "vectors.parquet"))
        saved = connection.read_parquet(str(tmp_path / "vectors.parquet")).to_arrow_table().to_pylist()
    rows = sorted(saved, key=lambda row: row["start_sec"])
    assert len(rows) == 2
    assert Path(directory, "probe").read_text() == "memory://video\n"
    assert Path(directory, "transcription").read_text() == "10.0\n"
    assert len(Path(directory, "visual").read_text().splitlines()) == 2
    assert rows[0]["clip_id"] != rows[1]["clip_id"]
    assert json.loads(rows[0]["transcription_metadata"])["reason"] == "lower_priority"
    assert json.loads(rows[0]["ocr_metadata"])["reason"] == "no_content"
    if output in {"separate", "both"}:
        assert rows[0]["visual_vector"] == rows[1]["visual_vector"] == [1, 2, 3]
        assert rows[0]["subtitles_vector"] == [8, 1]
        assert rows[1]["transcription_vector"] == [6, 1]
        assert rows[0]["transcription_vector"] is None
        assert rows[1]["subtitles_vector"] is None
    if output in {"fused", "both"}:
        assert rows[0]["fused_vector"] == [0, 2]
        assert rows[1]["fused_vector"] == [10, 2]
    assert all(not column.endswith("_content") for column in rows[0])


@pytest.mark.parametrize(
    "resources",
    [
        {"visual.embed": {}},
        {"probe": {"execution_backend": "auto"}},
        {"preprocess": {"execution_backend": "ray_task", "actor_number": 2}},
        {"embed": {"fallback": True}},
    ],
)
def test_invalid_stage_resources_fail_before_execution(resources):
    with pytest.raises(ValueError):
        _stage_options(resources)


def test_tasks_actors_and_gpu_resources_are_independent():
    options = _stage_options(
        {
            "preprocess": {"execution_backend": "ray_actor", "actor_number": 2},
            "embed": {"execution_backend": "ray_task", "gpus": 1},
        }
    )
    assert options["preprocess"]["gpus"] == 0
    assert options["preprocess"]["actor_number"] == 2
    assert options["embed"]["gpus"] == 1
    assert "actor_number" not in options["embed"]


@pytest.mark.parametrize("channel", ["visual", "audio", "subtitles"])
def test_content_transport_preserves_timestamps_and_values(channel):
    value = (
        VideoClip((np.arange(12, dtype=np.uint8).reshape(2, 2, 3),), (10.5,), (7,))
        if channel == "visual"
        else AudioClip(np.array([[0.25], [-0.5]]), 2)
        if channel == "audio"
        else "caption"
    )
    result = decode_content(channel, encode_content(channel, (TimedContent(10, 11, value),)))[0]
    assert (result.start, result.end) == (10, 11)
    if channel == "visual":
        np.testing.assert_array_equal(result.value.frames[0], value.frames[0])
        assert result.value.frame_times == (10.5,)
        assert result.value.frame_indices == (7,)
    elif channel == "audio":
        np.testing.assert_array_equal(result.value.samples, value.samples)
        assert result.value.sample_rate == 2
    else:
        assert result.value == value


class ParallelSource:
    supported_modalities = frozenset({"subtitles", "ocr"})

    def __init__(self, directory, *, fail=False):
        self.directory, self.fail = directory, fail

    def probe(self, uri):
        with Path(self.directory, "probe").open("a") as log:
            log.write(uri + "\n")
        return VideoMediaInfo(20, self.supported_modalities)

    def extract(self, uri, channel, clip, media):
        if clip.start == 10:
            return []
        Path(self.directory, channel).write_text(str(os.getpid()))
        peer = "subtitles" if channel == "ocr" else "ocr"
        deadline = time.monotonic() + 30
        while not Path(self.directory, peer).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("preprocessing did not run concurrently")
            time.sleep(0.02)
        if self.fail and channel == "ocr":
            raise RuntimeError("deliberate decoder failure")
        return [TimedContent(1, 2, channel)]


@pytest.mark.real_ray
@pytest.mark.parametrize(
    "destination,backend", [("parquet", "ray_task"), ("parquet", "ray_actor"), ("lance", "ray_task")]
)
def test_parallel_preprocessing_single_plan_and_single_sink(ray_local, monkeypatch, tmp_path, destination, backend):
    if destination == "lance":
        try:
            importlib.metadata.distribution("vane-extension-lance")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("matching Lance provider required")
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    source = tmp_path / "assets.parquet"
    pq.write_table(pa.table({"asset_id": ["a"], "source_version": ["v1"], "uri": ["memory://video"]}), source)
    profiles = {
        channel: VideoEmbeddingProfile(channel + "-v1", WideTextProfile() if channel == "ocr" else TextProfile())
        for channel in ParallelSource.supported_modalities
    }
    preprocess = {"execution_backend": backend, "batch_size": 1, "cpus": 0.25}
    if backend == "ray_actor":
        # A cold pool starts after its first actor is ready. Prevent prefetch
        # from queuing both sides of the test barrier on that first actor.
        monkeypatch.setenv("VANE_RAY_ACTOR_PREFETCH_DEPTH", "1")
        preprocess["actor_number"] = 2
    with (
        vane.connect() as connection,
        VideoEmbeddingPipeline(profiles=profiles, source=ParallelSource(str(tmp_path))) as pipeline,
    ):
        relation = pipeline.batch_ingest(
            connection.read_parquet(str(source)),
            modalities=["subtitles", "ocr"],
            resources={"preprocess": preprocess, "embed": {"cpus": 0.25}},
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None).to_physical_plan(connection)
        assert len(plan.collect_udf_nodes()) == 3
        assert not Path(tmp_path, "probe").exists(), "plan inspection must not execute media probing"
        if destination == "parquet":
            relation.write_parquet(str(tmp_path / "result.parquet"))
            result = connection.read_parquet(str(tmp_path / "result.parquet")).to_arrow_table().to_pylist()
        else:
            vane.load_installed_extension("lance", connection=connection)
            namespace = str(tmp_path / "lance").replace("'", "''")
            connection.execute(f"ATTACH '{namespace}' AS lance_ns (TYPE LANCE, READ_ONLY false)")
            schema = ", ".join(
                f'"{name}" {dtype}' for name, dtype in zip(relation.columns, relation.types, strict=True)
            )
            connection.execute(f"CREATE TABLE lance_ns.main.clips ({schema})")
            before = set((tmp_path / "lance").glob("**/_versions/*.manifest"))
            assert len(before) == 1
            relation.insert_into("lance_ns.main.clips")
            after = set((tmp_path / "lance").glob("**/_versions/*.manifest"))
            assert len(after - before) == 1
            result = connection.table("lance_ns.main.clips").to_arrow_table().to_pylist()
    assert Path(tmp_path, "probe").read_text() == "memory://video\n"
    assert Path(tmp_path, "subtitles").read_text() != Path(tmp_path, "ocr").read_text()
    rows = sorted(result, key=lambda row: row["start_sec"])
    assert rows[0]["subtitles_vector"] == [9, 1]
    assert rows[0]["ocr_vector"] == [3, 1, 2]
    assert rows[1]["subtitles_vector"] is None and rows[1]["ocr_vector"] is None
    assert json.loads(rows[1]["ocr_metadata"])["reason"] == "no_content"


@pytest.mark.real_ray
@pytest.mark.parametrize("destination", ["parquet", "lance"])
def test_decoder_error_fails_ingestion(ray_local, monkeypatch, tmp_path, destination):
    if destination == "lance":
        try:
            importlib.metadata.distribution("vane-extension-lance")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("matching Lance provider required")
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    source = tmp_path / "assets.parquet"
    pq.write_table(pa.table({"asset_id": ["a"], "source_version": ["v1"], "uri": ["memory://video"]}), source)
    profiles = {
        channel: VideoEmbeddingProfile(channel + "-v1", TextProfile())
        for channel in ParallelSource.supported_modalities
    }
    with (
        vane.connect() as connection,
        VideoEmbeddingPipeline(profiles=profiles, source=ParallelSource(str(tmp_path), fail=True)) as pipeline,
    ):
        relation = pipeline.batch_ingest(
            connection.read_parquet(str(source)),
            modalities=["subtitles", "ocr"],
            resources={"preprocess": {"cpus": 0.25}, "embed": {"cpus": 0.25}},
        )
        if destination == "lance":
            vane.load_installed_extension("lance", connection=connection)
            namespace = str(tmp_path / "lance").replace("'", "''")
            connection.execute(f"ATTACH '{namespace}' AS lance_ns (TYPE LANCE, READ_ONLY false)")
            schema = ", ".join(
                f'"{name}" {dtype}' for name, dtype in zip(relation.columns, relation.types, strict=True)
            )
            connection.execute(f"CREATE TABLE lance_ns.main.clips ({schema})")
            before = set((tmp_path / "lance").glob("**/_versions/*.manifest"))
            with pytest.raises(Exception, match="deliberate decoder failure"):
                relation.insert_into("lance_ns.main.clips")
            assert set((tmp_path / "lance").glob("**/_versions/*.manifest")) == before
            assert connection.table("lance_ns.main.clips").aggregate("count(*)").fetchone() == (0,)
        else:
            with pytest.raises(Exception, match="deliberate decoder failure"):
                relation.write_parquet(str(tmp_path / "result.parquet"))


@pytest.mark.real_ray
def test_duplicate_asset_identity_fails_assembly(ray_local, monkeypatch, tmp_path):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    source = tmp_path / "assets.parquet"
    pq.write_table(
        pa.table({"asset_id": ["a", "a"], "source_version": ["v1", "v1"], "uri": ["memory://video"] * 2}), source
    )

    class EmptySource:
        supported_modalities = frozenset({"subtitles"})

        def probe(self, uri):
            return VideoMediaInfo(10, frozenset())

        def extract(self, *args):
            raise AssertionError("absent channels must not be extracted")

    with (
        vane.connect() as connection,
        VideoEmbeddingPipeline(
            profiles={"subtitles": VideoEmbeddingProfile("text-v1", TextProfile())}, source=EmptySource()
        ) as pipeline,
    ):
        relation = pipeline.batch_ingest(connection.read_parquet(str(source)), modalities=["subtitles"])
        with pytest.raises(Exception, match="duplicate asset identity"):
            relation.fetchall()


def test_task_runtime_closes_after_each_batch_and_on_failure():
    from vane.ai._video_ingestion import _apply_stage

    calls = []

    class Relation:
        def map_batches(self, function, **options):
            assert options["execution_backend"] == "ray_task"
            return function

    class Runtime:
        def __init__(self):
            calls.append("open")

        def __call__(self, table):
            if table["fail"][0].as_py():
                raise RuntimeError("provider failed")
            return table

        def close(self):
            calls.append("close")

    task = _apply_stage(Relation(), Runtime, {"fail": "BOOLEAN"}, {"execution_backend": "ray_task"}, 1024)
    task(pa.table({"fail": [False]}))
    with pytest.raises(RuntimeError, match="provider failed"):
        task(pa.table({"fail": [True]}))
    assert calls == ["open", "close", "open", "close"]


def test_stage_output_byte_budget_is_enforced():
    from vane.ai._video_ingestion import _table

    with pytest.raises(MemoryError, match="max_batch_bytes"):
        _table([{"content": b"x" * 1024}], {"content": "BLOB"}, 512)


class TimelineSource:
    supported_modalities = frozenset({"subtitles", "transcription", "ocr"})

    def probe(self, uri):
        return VideoMediaInfo(20, self.supported_modalities)

    def extract(self, uri, channel, clip, media):
        if channel == "subtitles" and clip.start == 0:
            return [TimedContent(1, 2, "captions")]
        if channel == "transcription":
            if clip.start == 0:
                raise AssertionError("selected subtitles must skip ASR")
            return [TimedContent(11, 12, "speech")]
        return []


def timeline_pipeline():
    return VideoEmbeddingPipeline(
        source=TimelineSource(),
        profiles={
            channel: VideoEmbeddingProfile(channel + "-v1", TextProfile())
            for channel in TimelineSource.supported_modalities
        },
        fusion=VideoFusionProfile(
            "joint-v1",
            "test",
            "joint",
            2,
            (frozenset({"subtitles"}), frozenset({"transcription"})),
            lambda clip, selected, options: [clip.start, len(selected)],
        ),
    )


def timeline_options(output="separate"):
    return {
        "modalities": ["subtitles", "transcription", "ocr"],
        "selection": {
            "policy": "all",
            "groups": [{"modalities": ["subtitles", "transcription"], "policy": "first_available"}],
        },
        "output": output,
        "resources": {
            name: {"execution_backend": "ray_task", "cpus": 0.25} for name in ("probe", "preprocess", "embed", "fusion")
        },
    }


@pytest.mark.real_ray
@pytest.mark.parametrize("output", ["separate", "fused", "both"])
def test_ingest_batch_and_create_have_identical_clip_results(ray_local, monkeypatch, output):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    # Quotes must survive bound parameters and arbitrary source identity text.
    asset = {"asset_id": "a'1", "source_version": "immutable-v1", "uri": "memory://video'1"}
    options = timeline_options(output)
    with vane.connect() as connection, timeline_pipeline() as pipeline:
        single = pipeline.ingest(asset, connection=connection, **options)
        assert len(
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(single, None)
            .to_physical_plan(connection)
            .collect_udf_nodes()
        ) == (3 if output == "separate" else 4)
        single_rows = single.to_arrow_table().to_pylist()
        assets = connection.sql(
            "SELECT ?::VARCHAR asset_id, ?::VARCHAR source_version, ?::VARCHAR uri UNION ALL SELECT 'b', 'v2', 'memory://b'",
            params=list(asset.values()),
        )
        batch_rows = pipeline.batch_ingest(assets, **options).to_arrow_table().to_pylist()
        assert len(batch_rows) == 4
        assert sorted(single_rows, key=lambda r: r["clip_id"]) == sorted(
            [r for r in batch_rows if r["asset_id"] == asset["asset_id"]], key=lambda r: r["clip_id"]
        )
        response = pipeline.create(
            {"input": {"type": "video", **asset}, **{k: v for k, v in options.items() if k != "resources"}},
            resources=options["resources"],
        )
    assert [clip["start_sec"] for clip in response["clips"]] == [0, 10]
    rows_by_id = {row["clip_id"]: row for row in single_rows}
    assert {clip["clip_id"] for clip in response["clips"]} == set(rows_by_id)
    for vector in response["data"]:
        row = rows_by_id[vector["clip_id"]]
        channel = vector["modality"]
        assert vector["embedding"] == row[f"{channel}_vector"]
        metadata = json.loads(row[f"{channel}_metadata"])
        assert all(vector[key] == metadata[key] for key in ("model", "profile", "provider", "start_sec", "end_sec"))
    assert len(response["data"]) == (4 if output == "both" else 2)
    assert response["clips"][0]["channels"]["transcription"]["reason"] == "lower_priority"
    assert response["clips"][1]["channels"]["subtitles"]["reason"] == "no_content"


@pytest.mark.real_ray
@pytest.mark.parametrize("asynchronous", [False, True])
def test_create_uses_ray_and_enforces_sync_duration_budget(ray_local, monkeypatch, tmp_path, asynchronous):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    directory = str(tmp_path)

    class Source(TimelineSource):
        def probe(self, uri):
            Path(directory, "worker_pid").write_text(str(os.getpid()))
            return VideoMediaInfo(61, self.supported_modalities)

        def extract(self, *args):
            Path(directory, "extracted").touch()
            return []

    with VideoEmbeddingPipeline(
        source=Source(), profiles={"subtitles": VideoEmbeddingProfile("v1", TextProfile())}
    ) as pipeline:
        request = {
            "input": {"type": "video", "uri": "memory://video", "asset_id": "a", "source_version": "v1"},
            "modalities": ["subtitles"],
            "async": asynchronous,
        }
        if asynchronous:
            task = pipeline.create(request)
            assert task.id and pipeline.get_task(task.id) is task
            assert len(task.result(timeout=60)["clips"]) == 7
            assert task.status == "ready"
            assert (tmp_path / "extracted").exists()
        else:
            with pytest.raises(Exception, match="sync_budget.max_video_seconds"):
                pipeline.create(request)
            assert not (tmp_path / "extracted").exists()
    assert int((tmp_path / "worker_pid").read_text()) != os.getpid()


@pytest.mark.real_ray
@pytest.mark.parametrize("cancellation", ["explicit", "sync_budget"])
def test_create_cancels_active_ray_query(ray_local, monkeypatch, tmp_path, cancellation):
    from concurrent.futures import CancelledError

    from vane.ai.video import VideoEmbeddingTimeout

    monkeypatch.delenv("VANE_RUNNER", raising=False)
    directory = str(tmp_path)

    class Source(TimelineSource):
        def probe(self, uri):
            Path(directory, "entered").touch()
            deadline = time.monotonic() + 45
            while not Path(directory, "release").exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("test source was not released")
                time.sleep(0.05)
            return VideoMediaInfo(10, self.supported_modalities)

        def extract(self, *args):
            Path(directory, "extracted").touch()
            return []

    pipeline = VideoEmbeddingPipeline(
        source=Source(), profiles={"subtitles": VideoEmbeddingProfile("v1", TextProfile())}
    )
    request = {
        "input": {"type": "video", "uri": "memory://video", "asset_id": "a", "source_version": "v1"},
        "modalities": ["subtitles"],
    }
    try:
        if cancellation == "explicit":
            task = pipeline.create({**request, "async": True})
            deadline = time.monotonic() + 30
            while not (tmp_path / "entered").exists():
                assert time.monotonic() < deadline
                time.sleep(0.05)
            task.cancel()
        else:
            with pytest.raises(VideoEmbeddingTimeout) as error:
                pipeline.create({**request, "sync_budget": {"timeout_seconds": 8}})
            task = pipeline.get_task(error.value.task_id)
        with pytest.raises(CancelledError):
            task.result(timeout=20)
        assert task.status == "cancelled"
        assert not (tmp_path / "extracted").exists()
    finally:
        (tmp_path / "release").touch()
        pipeline.close()


@pytest.mark.real_ray
def test_ingest_and_batch_ingest_commit_to_same_lance_table(ray_local, monkeypatch, tmp_path):
    try:
        importlib.metadata.distribution("vane-extension-lance")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("matching Lance provider required")
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    with vane.connect() as connection, timeline_pipeline() as pipeline:
        options = timeline_options("both")
        relation = pipeline.ingest(
            {"asset_id": "a", "source_version": "v1", "uri": "memory://a"}, connection=connection, **options
        )
        vane.load_installed_extension("lance", connection=connection)
        namespace = str(tmp_path / "lance").replace("'", "''")
        connection.execute(f"ATTACH '{namespace}' AS lance_ns (TYPE LANCE, READ_ONLY false)")
        schema = ", ".join(f'"{name}" {dtype}' for name, dtype in zip(relation.columns, relation.types, strict=True))
        connection.execute(f"CREATE TABLE lance_ns.main.clips ({schema})")
        before = set((tmp_path / "lance").glob("**/_versions/*.manifest"))
        relation.insert_into("lance_ns.main.clips")
        after_single = set((tmp_path / "lance").glob("**/_versions/*.manifest"))
        assert len(after_single - before) == 1
        assets = connection.sql(
            "SELECT asset_id, 'v1' AS source_version, 'memory://' || asset_id AS uri FROM (VALUES ('b'), ('c')) t(asset_id)"
        )
        batch = pipeline.batch_ingest(assets, **options)
        assert batch.columns == relation.columns and batch.types == relation.types
        batch.insert_into("lance_ns.main.clips")
        after_batch = set((tmp_path / "lance").glob("**/_versions/*.manifest"))
        assert len(after_batch - after_single) == 1
        rows = connection.table("lance_ns.main.clips").to_arrow_table().to_pylist()
    assert len(rows) == len({row["clip_id"] for row in rows}) == 6
    assert {row["asset_id"] for row in rows} == {"a", "b", "c"}
    for row in rows:
        assert row["ocr_vector"] is None
        assert row["fused_vector"] == [row["start_sec"], 1]
        assert (row["subtitles_vector"] is None) == (row["start_sec"] == 10)


@pytest.mark.real_ray
@pytest.mark.parametrize("backend", ["ray_task", "ray_actor"])
@pytest.mark.parametrize("consumer", ["read", "parquet"])
def test_preprocess_error_is_not_retried_when_a_second_call_would_succeed(
    ray_local, monkeypatch, tmp_path, backend, consumer
):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    directory = str(tmp_path)

    class Source(TimelineSource):
        def probe(self, uri):
            with Path(directory, "probe").open("a") as log:
                log.write("probe\n")
            return VideoMediaInfo(10, self.supported_modalities)

        def extract(self, *args):
            marker = Path(directory, "called")
            if not marker.exists():
                marker.touch()
                raise RuntimeError("first extraction failed")
            Path(directory, "retried").touch()
            return []

    with (
        vane.connect() as connection,
        VideoEmbeddingPipeline(
            source=Source(), profiles={"subtitles": VideoEmbeddingProfile("v1", TextProfile())}
        ) as pipeline,
    ):
        relation = pipeline.ingest(
            {"asset_id": "a", "source_version": "v1", "uri": "memory://video"},
            connection=connection,
            modalities=["subtitles"],
            resources={"preprocess": {"execution_backend": backend, "cpus": 0.25}},
        )
        with pytest.raises(Exception, match="first extraction failed"):
            if consumer == "read":
                relation.to_arrow_table()
            else:
                relation.write_parquet(str(tmp_path / "failed.parquet"))
    assert not (tmp_path / "retried").exists()
    assert (tmp_path / "probe").read_text() == "probe\n"
