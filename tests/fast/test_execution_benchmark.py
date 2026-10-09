# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check benchmark measurement boundaries, evidence and failure reporting."""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from scripts import benchmark_execution as benchmark


@pytest.mark.parametrize(
    "changes",
    [
        {"rows": 0},
        {"repetitions": 0},
        {"warmups": 0},
        {"batch_rows": 2049},
        {"worker_count": True},
        {"consumer_rows_per_second": float("nan")},
        {"deadline": float("inf")},
        {"seed": -1},
        {"modes": ("local",)},
        {"profiles": ("default", "default")},
        {"scenarios": ()},
        {"modes": ("unknown",)},
        {"interface": "unknown"},
    ],
)
def test_invalid_benchmark_configuration_is_rejected_before_work(tmp_path, changes):
    with pytest.raises(ValueError):
        benchmark.Configuration(tmp_path / "report", **changes)
    assert not (tmp_path / "report").exists()


def test_generated_inputs_and_native_reference_are_reproducible(tmp_path):
    import vane

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    config = benchmark.Configuration(first, rows=257)
    workloads, files = benchmark.dataset(config)
    replay, copies = benchmark.dataset(replace(config, output=second))
    assert files == copies
    with vane.connect() as connection:
        for workload, other in zip(workloads, replay):
            expected = connection.execute(workload.sql).to_arrow_table()
            actual = connection.execute(other.sql).to_arrow_table()
            benchmark.compare(workload, expected, actual)
            assert actual.num_rows > 0
    assert sum(p.stat().st_size for p in (first / "inputs").glob("*.parquet")) == sum(f["bytes"] for f in files)


def test_benchmark_validation_catches_values_multiplicity_schema_and_order():
    expected = pa.table({"id": [1, 2, 2], "v": [7, None, None]})
    workload = benchmark.Workload("probe", "select 1", ("id",))
    benchmark.compare(workload, expected, expected.take([2, 0, 1]))
    for actual in (
        expected.take([0, 0, 1]),
        pa.table({"id": [1, 2, 2], "v": [8, None, None]}),
        expected.rename_columns(["renamed", "v"]),
    ):
        with pytest.raises(AssertionError):
            benchmark.compare(workload, expected, actual)
    with pytest.raises(AssertionError):
        benchmark.compare(replace(workload, ordered=True), expected, expected.take([2, 0, 1]))


def test_summary_separates_modes_phases_and_excludes_warmup():
    def sample(mode, phase, elapsed, first=None):
        return dict(
            profile="default", mode=mode, phase=phase, workload="scan", total_seconds=elapsed, first_batch_seconds=first
        )

    groups = benchmark.summarize(
        [
            sample("fte", "warmup", 10000),
            sample("fte", "warm", 1, 0.5),
            sample("fte", "warm", 3),
            sample("pipelined", "warm", 0.1, 0.05),
            sample("fte", "recovery", 8, 6),
        ]
    )
    assert len(groups) == 3
    warm = next(g for g in groups if g["mode"] == "fte" and g["phase"] == "warm")
    assert warm["samples"] == 2
    assert warm["metrics"]["total_seconds"] == dict(n=2, min=1, median=2, p95=3, max=3)
    assert warm["metrics"]["first_batch_seconds"]["n"] == 1


def test_local_benchmark_cli_uses_installed_package_and_records_each_sample(tmp_path):
    directory = tmp_path / "benchmark"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(benchmark.__file__)),
            "--output",
            str(directory),
            "--rows",
            "257",
            "--repetitions",
            "2",
            "--modes",
            "local",
            "--scenarios",
            "cold",
            "warm",
            "slow",
            "--consumer-rows-per-second",
            "100000",
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((directory / "report.json").read_text())
    assert report["complete"]
    assert report["engine_identity"] and len(report["native_sha256"]) == 64
    assert report["cluster_startup_seconds"] is None
    assert len(report["samples"]) == 16  # 2 cold, 4 warmup, 8 warm, 2 paced scans.
    assert len((directory / "samples.jsonl").read_text().splitlines()) == 16
    assert all(g["samples"] == 2 for g in report["summary"])
    for sample in report["samples"]:
        assert (
            0
            <= sample["query_return_seconds"]
            <= sample["first_batch_seconds"]
            <= sample["drain_seconds"]
            <= sample["total_seconds"]
        )
        assert sample["rows"] > 0 and sample["batches"] > 0
        if sample["phase"] == "slow":
            assert sample["requested_consumer_pause_seconds"] == sample["rows"] / 100000
            assert sample["consumer_pause_seconds"] >= sample["requested_consumer_pause_seconds"]
    assert (directory / "resources.json").exists()
    assert "First batch median" in (directory / "report.md").read_text()


def test_failure_report_preserves_partial_samples_and_existing_output(tmp_path, monkeypatch):
    config = benchmark.Configuration(tmp_path / "failed", rows=64, modes=("local",), scenarios=("warm",))
    original = benchmark.measure

    def fail(connection, workload, *args, **kwargs):
        if workload.name == "scan":
            raise RuntimeError("controlled benchmark failure")
        return original(connection, workload, *args, **kwargs)

    monkeypatch.setattr(benchmark, "measure", fail)
    with pytest.raises(RuntimeError, match="controlled benchmark failure"):
        benchmark.run(config)
    path = config.output / "report.json"
    contents = path.read_bytes()
    report = json.loads(contents)
    assert not report["complete"] and len(report["samples"]) == 1
    assert report["failure"]["message"] == "controlled benchmark failure"
    assert json.loads((config.output / "active-local-local.json").read_text())["workload"] == "scan"
    with pytest.raises(FileExistsError):
        benchmark.run(config)
    assert path.read_bytes() == contents


def test_empty_stream_has_no_first_batch_and_releases_ownership(tmp_path):
    config = benchmark.Configuration(tmp_path, modes=("local",), scenarios=("warm",))
    workload = benchmark.Workload("empty", "select 1::bigint v where false", ("v",), streaming=True)
    with benchmark.connect(config, "local") as connection:
        expected = connection.execute(workload.sql).to_arrow_table()
        sample, result = benchmark.measure(connection, workload, config, "local", expected)
        assert sample["first_batch_seconds"] is None
        assert sample["rows"] == sample["batches"] == sample["output_rows_per_second"] == 0
        assert result.execution_state == "SUCCEEDED"
        benchmark.idle(connection)


@pytest.mark.parametrize("outcome", ["deadline", "query_cancel", "delivery_cancel", "stop"])
def test_consumer_pause_checks_deadline_and_cancellation_without_reading_ahead(monkeypatch, outcome):
    from vane.execution.request_admission import RequestCancelled
    from vane.execution.result_delivery import ResultDeliveryCancelled, ResultDeliveryTimeout

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock.now)

    def check_query():
        if outcome == "query_cancel" and clock.now >= 0.05:
            raise RequestCancelled("query canceled")

    def check_delivery():
        if outcome == "delivery_cancel" and clock.now >= 0.05:
            raise ResultDeliveryCancelled("delivery canceled")

    def wait(seconds):
        clock.now += seconds

    result = SimpleNamespace(context=SimpleNamespace(check=check_query), check_preparation=check_delivery)
    stop = SimpleNamespace(is_set=lambda: outcome == "stop" and clock.now >= 0.05, wait=wait)
    error = {
        "deadline": ResultDeliveryTimeout,
        "query_cancel": RequestCancelled,
        "delivery_cancel": ResultDeliveryCancelled,
        "stop": InterruptedError,
    }[outcome]
    with pytest.raises(error):
        benchmark.consumer_pause(result, 2048, 0.1, stop)
    assert clock.now <= 0.1


@pytest.mark.timeout(10)
@pytest.mark.parametrize("outcome", ["deadline", "interrupt", "cancel", "stop"])
def test_paced_native_query_releases_batch_promptly_on_timeout_or_cancellation(tmp_path, outcome):
    from vane.execution.request_admission import RequestCancelled, RequestExecutionTimeout
    from vane.execution.result_delivery import ResultDeliveryCancelled, ResultDeliveryTimeout

    config = benchmark.Configuration(
        tmp_path,
        modes=("local",),
        scenarios=("slow",),
        deadline=0.1 if outcome == "deadline" else 60,
        consumer_rows_per_second=1,
    )
    workload = benchmark.Workload("scan", "select range from range(20)", ("range",), streaming=True)
    ready, stop = threading.Event(), threading.Event()
    with benchmark.connect(config, "local") as connection, ThreadPoolExecutor(1) as executor:
        expected = connection.execute(workload.sql).to_arrow_table()
        started = time.monotonic()
        future = executor.submit(
            benchmark.measure, connection, workload, config, "local", expected, paced=True, ready=ready, stop=stop
        )
        try:
            assert ready.wait(2)
            if outcome == "interrupt":
                connection.interrupt()
            elif outcome == "cancel":
                # Cancel the handle independently of connection.interrupt().
                with connection.query_runtime._delivery._condition:
                    result = next(iter(connection.query_runtime._delivery._results.values()))
                result.cancel()
            elif outcome == "stop":
                stop.set()
            errors = {
                "deadline": (RequestExecutionTimeout, ResultDeliveryTimeout),
                "interrupt": RequestCancelled,
                "cancel": (RequestCancelled, ResultDeliveryCancelled),
                "stop": InterruptedError,
            }[outcome]
            with pytest.raises(errors):
                future.result(timeout=2)
            assert time.monotonic() - started < 3
            benchmark.idle(connection)
        finally:
            stop.set()
            connection.interrupt()


@pytest.mark.parametrize("case", ["overlap", "queued", "different_worker"])
def test_mixed_evidence_requires_shared_worker_reservations(monkeypatch, case):
    from vane.execution.worker_resources import WorkerResourceManager

    manager = WorkerResourceManager({"contexts": 2}, 2)
    connection = SimpleNamespace(query_runtime=SimpleNamespace(pool=SimpleNamespace(admission=manager)))
    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock.now)
    intervals = []
    demand = {0: {"contexts": 2 if case == "queued" else 1}}
    fte_demand = {1 if case == "different_worker" else 0: {"contexts": 1}}
    with benchmark.worker_occupancy(connection, intervals):
        assert manager.try_acquire("pipelined/scan", "scan", demand)
        # An idempotent acquisition must not move the start or add an interval.
        clock.now = 2.0
        assert manager.try_acquire("pipelined/scan", "scan", demand)
        if case == "queued":
            assert not manager.try_acquire("fte/aggregate/task/0", "aggregate", fte_demand)
            assert len(intervals) == 1
            clock.now = 3.0
            manager.release("pipelined/scan")
        assert manager.try_acquire("fte/aggregate/task/0", "aggregate", fte_demand)
        clock.now = 4.0
        manager.release("fte/aggregate/task/0")
        clock.now = 5.0
        manager.release("pipelined/scan")
    assert len(intervals) == 2
    assert intervals[0]["acquired_at_monotonic"] == 1
    assert benchmark.mixed_overlap(intervals, "scan", "aggregate") == (2 if case == "overlap" else 0)
    assert manager.snapshot()["reservations"] == {}
    assert manager.snapshot()["waiting"] == []


def test_mixed_evidence_rejects_missing_or_unreleased_reservations():
    with pytest.raises(AssertionError, match="evidence is incomplete"):
        benchmark.mixed_overlap([], "scan", "aggregate")
    with pytest.raises(AssertionError, match="evidence is incomplete"):
        benchmark.mixed_overlap(
            [{"query_id": query, "released_at_monotonic": None} for query in ("scan", "aggregate")], "scan", "aggregate"
        )
