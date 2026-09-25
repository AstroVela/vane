# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_module(name):
    directory = Path(__file__).resolve().parents[2] / "multimodal_inference_benchmarks/video_object_detection"
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profiler():
    return load_module("batch_profile")


def test_disabled_profile_does_not_create_files(profiler, monkeypatch, tmp_path):
    monkeypatch.delenv("VIDEO_PROFILE_DIR", raising=False)
    assert profiler.BatchProfile.from_env("vane") is None
    monkeypatch.setenv("VIDEO_PROFILE_DIR", "   ")
    assert profiler.BatchProfile.from_env("vane") is None
    assert list(tmp_path.iterdir()) == []


def test_batch_times_gaps_and_unique_actor_files(profiler, monkeypatch, tmp_path):
    wall = iter([0, 10_000_000, 13_000_000, 20_000_000, 30_000_000, 32_000_000])
    cpu = iter([0, 4_000_000, 6_000_000, 11_000_000])
    monkeypatch.setattr(profiler.time, "perf_counter_ns", lambda: next(wall))
    monkeypatch.setattr(profiler.time, "thread_time_ns", lambda: next(cpu))
    profile = profiler.BatchProfile(str(tmp_path), "vane")
    other = profiler.BatchProfile(str(tmp_path), "vane")
    assert profile.path != other.path
    results = [SimpleNamespace(speed=dict(preprocess=1, inference=2, postprocess=3))] * 2
    for _ in range(2):
        profile.begin(2)
        profile.mark("model")
        profile.finish(results)
    first, second = [json.loads(line) for line in profile.path.read_text().splitlines()]
    assert first["actor_gap_ms"] is None and first["previous_write_ms"] is None
    assert first["actor_body_ms"] == 10
    assert first["phases"]["model"] == dict(wall_ms=10, thread_cpu_ms=4)
    assert first["model_ms"] == dict(preprocess=2, inference=4, postprocess=6)
    assert second["actor_gap_ms"] == 7  # Excludes the preceding 3 ms write.
    assert second["previous_write_ms"] == 3
    assert second["actor_thread_cpu_ms"] == 5
    assert second["batch_index"] == 2


@pytest.mark.parametrize("speed", [None, {}, dict(preprocess=None), dict(preprocess=float("nan")), dict(preprocess=-1)])
def test_missing_model_times_are_not_zero(profiler, speed):
    assert profiler.model_timings([SimpleNamespace(speed=speed)], 1) is None
    assert profiler.model_timings([], 0) is None
    assert profiler.model_timings([], 1) is None


def test_summary_counts_frames_and_skips_each_actor_warmup(tmp_path, capsys):
    module = load_module("summarize_profile")
    records = [
        dict(
            schema_version=1,
            engine="vane",
            actor_id=actor,
            batch_index=index,
            rows=rows,
            actor_body_ms=10,
            actor_gap_ms=None,
            phases={},
            model_ms=None,
        )
        for actor in ("a", "b")
        for index, rows in ((1, 32), (2, 3))
    ]
    path = tmp_path / "vane.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records) + '{"partial":')
    summary = module.summarize(module.read_records(tmp_path))["vane"]
    assert summary["completed_batches"] == 2
    assert summary["inferred_frames"] == 6
    assert summary["actors"] == 2
    assert summary["model_timing_batches"] == 0
    assert summary["metrics"]["actor_body_ms"]["total_ms"] == 20
    assert "Ignoring truncated" in capsys.readouterr().err
    with pytest.raises(ValueError, match="non-negative"):
        module.summarize(records, -1)
    path.write_text('{"broken":\n')
    with pytest.raises(json.JSONDecodeError):
        list(module.read_records(tmp_path))


def test_summary_rejects_unknown_schema(tmp_path):
    module = load_module("summarize_profile")
    (tmp_path / "new.jsonl").write_text('{"schema_version": 2}\n')
    with pytest.raises(ValueError, match="Unsupported profile schema"):
        list(module.read_records(tmp_path))
