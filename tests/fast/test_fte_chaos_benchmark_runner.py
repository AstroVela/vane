# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import types
from pathlib import Path

import pytest

import vane
from multimodal_inference_benchmarks import run_fte_chaos_benchmarks as runner
from multimodal_inference_benchmarks import run_fte_full_matrix as matrix_runner


def _write_parquet(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = vane.connect()
    try:
        escaped = str(path).replace("'", "''")
        con.execute(f"COPY (SELECT 1 AS id, 'ok' AS value) TO '{escaped}' (FORMAT PARQUET)")
    finally:
        con.close()


def _write_path_parquet(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = vane.connect()
    try:
        escaped = str(path).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM (VALUES ('c', 3), ('a', 1), ('b', 2)) AS t(path, value)
            )
            TO '{escaped}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def _read_manifest(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fte_chaos_runner_resume_reuses_valid_output(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    _write_parquet(input_path)
    run_root = tmp_path / "runs"
    output_file = run_root / "unit" / "output" / "part.parquet"
    _write_parquet(output_file)
    manifest_path = run_root / "manifest.jsonl"
    spec = runner.BenchmarkSpec(
        name="unit",
        script=tmp_path / "should_not_run.py",
        local_env={"INPUT_PATH": str(input_path)},
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("subprocess.run should not be called")),
    )

    runner._run_one(
        spec,
        mode="host",
        run_root=run_root,
        smoke_limit=0,
        full=True,
        timeout_s=1,
        dry_run=False,
        compare_reference=False,
        manifest_path=manifest_path,
        resume=True,
    )

    events = _read_manifest(manifest_path)
    assert [event["event"] for event in events] == ["resume_success"]
    assert events[0]["suite"] == "unit"
    assert events[0]["label"] == "chaos"
    assert events[0]["summary"]["row_count"] == 1
    assert events[0]["summary"]["column_count"] == 2


def test_fte_chaos_runner_records_failure_manifest(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    _write_parquet(input_path)
    run_root = tmp_path / "runs"
    manifest_path = run_root / "manifest.jsonl"
    spec = runner.BenchmarkSpec(
        name="unit",
        script=tmp_path / "fails.py",
        local_env={"INPUT_PATH": str(input_path)},
    )

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=7),
    )

    with pytest.raises(RuntimeError, match="benchmark failed rc=7"):
        runner._run_one(
            spec,
            mode="worker",
            run_root=run_root,
            smoke_limit=16,
            full=False,
            timeout_s=1,
            dry_run=False,
            compare_reference=False,
            manifest_path=manifest_path,
            resume=False,
        )

    events = _read_manifest(manifest_path)
    assert [event["event"] for event in events] == ["failure"]
    assert events[0]["suite"] == "unit"
    assert events[0]["label"] == "chaos"
    assert events[0]["returncode"] == 7
    assert events[0]["log_path"].endswith("run.log")


def test_fte_chaos_runner_materializes_stable_smoke_input(tmp_path):
    input_path = tmp_path / "input" / "part.parquet"
    _write_path_parquet(input_path)
    run_root = tmp_path / "runs"
    manifest_path = run_root / "manifest.jsonl"
    spec = runner.BenchmarkSpec(
        name="unit",
        script=tmp_path / "benchmark.py",
        local_env={"INPUT_PATH": str(input_path.parent)},
        stable_smoke_input_env="INPUT_PATH",
        stable_smoke_input_order_by="path",
    )

    runner._run_one(
        spec,
        mode="host",
        run_root=run_root,
        smoke_limit=2,
        full=False,
        timeout_s=1,
        dry_run=True,
        compare_reference=False,
        manifest_path=manifest_path,
        resume=False,
    )

    events = _read_manifest(manifest_path)
    assert [event["event"] for event in events] == [
        "stable_smoke_input_prepared",
        "dry_run",
    ]
    stable_input = Path(events[0]["output_path"])
    rows = vane.sql(
        "SELECT path, value FROM read_parquet(?) ORDER BY path",
        params=[str(stable_input)],
    ).fetchall()
    assert rows == [("a", 1), ("b", 2)]
    assert events[1]["env"]["INPUT_PATH"] == str(stable_input)
    assert events[1]["env"]["INPUT_LIMIT"] == "2"


def test_fte_chaos_runner_records_timeout_manifest(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    _write_parquet(input_path)
    run_root = tmp_path / "runs"
    manifest_path = run_root / "manifest.jsonl"
    spec = runner.BenchmarkSpec(
        name="unit",
        script=tmp_path / "times_out.py",
        local_env={"INPUT_PATH": str(input_path)},
    )

    def fake_run(*_args, **kwargs):
        raise runner.subprocess.TimeoutExpired(cmd=kwargs.get("args", ["benchmark"]), timeout=1)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="benchmark timed out"):
        runner._run_one(
            spec,
            mode="worker",
            run_root=run_root,
            smoke_limit=16,
            full=False,
            timeout_s=1,
            dry_run=False,
            compare_reference=False,
            manifest_path=manifest_path,
            resume=False,
        )

    events = _read_manifest(manifest_path)
    assert [event["event"] for event in events] == ["timeout"]
    assert events[0]["suite"] == "unit"
    assert events[0]["label"] == "chaos"
    assert events[0]["timeout_s"] == 1
    assert events[0]["log_path"].endswith("run.log")


def test_fte_chaos_runner_records_interrupted_manifest(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    _write_parquet(input_path)
    run_root = tmp_path / "runs"
    manifest_path = run_root / "manifest.jsonl"
    spec = runner.BenchmarkSpec(
        name="unit",
        script=tmp_path / "interrupted.py",
        local_env={"INPUT_PATH": str(input_path)},
    )

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        runner._run_one(
            spec,
            mode="host",
            run_root=run_root,
            smoke_limit=16,
            full=False,
            timeout_s=1,
            dry_run=False,
            compare_reference=False,
            manifest_path=manifest_path,
            resume=False,
        )

    events = _read_manifest(manifest_path)
    assert [event["event"] for event in events] == ["interrupted"]
    assert events[0]["suite"] == "unit"
    assert events[0]["label"] == "chaos"
    assert events[0]["chaos"] is True
    assert events[0]["log_path"].endswith("run.log")


def test_fte_chaos_runner_label_reference_runs_only_reference(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    _write_parquet(input_path)
    run_root = tmp_path / "runs"
    manifest_path = run_root / "manifest.jsonl"
    spec = runner.BenchmarkSpec(
        name="unit",
        script=tmp_path / "benchmark.py",
        local_env={"INPUT_PATH": str(input_path)},
    )
    calls = []

    def fake_run(*_args, **kwargs):
        env = kwargs["env"]
        output_path = env["OUTPUT_PATH"]
        calls.append((output_path, "VANE_FTE_CHAOS_KILL_WORKER_ON_RUNNING" in env))
        _write_parquet(tmp_path / "runs" / "unit" / "reference_output" / "part.parquet")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._run_one(
        spec,
        mode="host",
        run_root=run_root,
        smoke_limit=16,
        full=False,
        timeout_s=1,
        dry_run=False,
        compare_reference=True,
        manifest_path=manifest_path,
        resume=False,
        label="reference",
    )

    assert calls == [(str(run_root / "unit" / "reference_output"), False)]
    assert not (run_root / "unit" / "output").exists()
    events = _read_manifest(manifest_path)
    assert [event["event"] for event in events] == ["success"]
    assert events[0]["label"] == "reference"
    assert events[0]["chaos"] is False
    assert events[0]["summary"]["row_count"] == 1


def test_fte_chaos_runner_label_chaos_loads_reference_and_compares(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    _write_parquet(input_path)
    run_root = tmp_path / "runs"
    _write_parquet(run_root / "unit" / "reference_output" / "part.parquet")
    manifest_path = run_root / "manifest.jsonl"
    spec = runner.BenchmarkSpec(
        name="unit",
        script=tmp_path / "benchmark.py",
        local_env={"INPUT_PATH": str(input_path)},
    )
    calls = []

    def fake_run(*_args, **kwargs):
        env = kwargs["env"]
        output_path = env["OUTPUT_PATH"]
        calls.append((output_path, "VANE_FTE_CHAOS_KILL_WORKER_ON_RUNNING" in env))
        _write_parquet(tmp_path / "runs" / "unit" / "output" / "part.parquet")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._run_one(
        spec,
        mode="host",
        run_root=run_root,
        smoke_limit=16,
        full=False,
        timeout_s=1,
        dry_run=False,
        compare_reference=True,
        manifest_path=manifest_path,
        resume=False,
        label="chaos",
    )

    assert calls == [(str(run_root / "unit" / "output"), True)]
    events = _read_manifest(manifest_path)
    assert [event["event"] for event in events] == [
        "reference_loaded",
        "success",
        "comparison_success",
    ]
    assert events[0]["label"] == "reference"
    assert events[0]["summary"]["row_count"] == 1
    assert events[1]["label"] == "chaos"
    assert events[1]["chaos"] is True
    assert events[2]["reference"]["hash_sum"] == events[2]["chaos"]["hash_sum"]


def test_fte_chaos_runner_label_chaos_loads_reference_summary(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    _write_parquet(input_path)
    run_root = tmp_path / "runs"
    manifest_path = run_root / "manifest.jsonl"
    spec = runner.BenchmarkSpec(
        name="unit",
        script=tmp_path / "benchmark.py",
        local_env={"INPUT_PATH": str(input_path)},
    )
    reference_seed = tmp_path / "reference_seed" / "part.parquet"
    _write_parquet(reference_seed)
    reference_summary = runner._validate_output(reference_seed.parent)
    runner._write_summary_json(
        run_root / "unit" / "reference_summary.json",
        spec=spec,
        label="reference",
        mode="host",
        full=True,
        smoke_limit=0,
        chaos=False,
        output_path=run_root / "unit" / "reference_output",
        summary=reference_summary,
    )

    def fake_run(*_args, **_kwargs):
        _write_parquet(tmp_path / "runs" / "unit" / "output" / "part.parquet")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._run_one(
        spec,
        mode="host",
        run_root=run_root,
        smoke_limit=0,
        full=True,
        timeout_s=1,
        dry_run=False,
        compare_reference=True,
        manifest_path=manifest_path,
        resume=False,
        label="chaos",
    )

    events = _read_manifest(manifest_path)
    assert [event["event"] for event in events] == [
        "reference_summary_loaded",
        "success",
        "comparison_success",
    ]
    assert events[0]["summary_path"].endswith("reference_summary.json")
    assert events[2]["reference"]["hash_sum"] == events[2]["chaos"]["hash_sum"]


def test_fte_chaos_runner_cleans_reference_before_chaos(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    _write_parquet(input_path)
    run_root = tmp_path / "runs"
    manifest_path = run_root / "manifest.jsonl"
    spec = runner.BenchmarkSpec(
        name="unit",
        script=tmp_path / "benchmark.py",
        local_env={"INPUT_PATH": str(input_path)},
    )

    def fake_run(*_args, **kwargs):
        output_path = Path(kwargs["env"]["OUTPUT_PATH"])
        _write_parquet(output_path / "part.parquet")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._run_one(
        spec,
        mode="host",
        run_root=run_root,
        smoke_limit=0,
        full=True,
        timeout_s=1,
        dry_run=False,
        compare_reference=True,
        manifest_path=manifest_path,
        resume=False,
        label="all",
        cleanup_reference_before_chaos=True,
    )

    assert not (run_root / "unit" / "reference_output").exists()
    assert (run_root / "unit" / "reference_summary.json").exists()
    assert (run_root / "unit" / "chaos_summary.json").exists()
    events = _read_manifest(manifest_path)
    assert "reference_output_cleaned" in [event["event"] for event in events]
    assert events[-1]["event"] == "comparison_success"


def test_full_matrix_wrapper_applies_suite_defaults(monkeypatch, tmp_path):
    specs = {
        suite: runner.BenchmarkSpec(
            name=suite,
            script=tmp_path / f"{suite}.py",
            local_env={"INPUT_PATH": str(tmp_path / suite)},
        )
        for suite in matrix_runner.DEFAULT_SUITES
    }
    calls = []
    monkeypatch.setattr(matrix_runner.runner, "_benchmark_specs", lambda _data_root: dict(specs))
    monkeypatch.setattr(
        matrix_runner.runner,
        "_run_one",
        lambda spec, **kwargs: calls.append((spec, kwargs)),
    )
    run_root = tmp_path / "matrix"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fte_full_matrix.py",
            "--suite",
            "audio",
            "--suite",
            "video",
            "--run-root",
            str(run_root),
            "--audio-num-gpu-nodes",
            "1",
            "--smoke-limit",
            "4",
            "--dry-run",
        ],
    )

    matrix_runner.main()

    assert [call[0].name for call in calls] == ["audio", "video"]
    assert calls[0][0].full_extra_env["AUDIO_NUM_GPU_NODES"] == "1"
    assert calls[0][0].full_extra_env["NUM_GPU_NODES"] == "1"
    assert calls[0][1]["cleanup_reference_before_chaos"] is False
    assert calls[1][1]["cleanup_reference_before_chaos"] is True
    assert calls[0][1]["full"] is False
    assert calls[1][1]["full"] is False
    assert calls[0][1]["smoke_limit"] == 4
    assert calls[1][1]["smoke_limit"] == 4
    assert calls[0][1]["timeout_s"] == matrix_runner.DEFAULT_TIMEOUTS["audio"]
    assert calls[1][1]["timeout_s"] == matrix_runner.DEFAULT_TIMEOUTS["video"]
    assert (run_root / "full_matrix_report.md").exists()
