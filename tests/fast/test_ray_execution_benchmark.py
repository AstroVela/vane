# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise benchmark orchestration, including a real worker loss, without timing gates."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import benchmark_execution as benchmark

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


@pytest.mark.ray_fault
@pytest.mark.timeout(300)
@pytest.mark.parametrize("interface", benchmark.INTERFACES)
def test_ray_benchmark_checks_both_profiles_and_recovery_evidence(tmp_path, monkeypatch, interface):
    from vane.execution.recovery_runtime import RecoveryScheduler

    # Hold the mixed scan's first batch until FTE is dispatched. This tests
    # real overlapping reservations without relying on actor startup latency.
    dispatched = threading.Event()
    original_measure, original_dispatch = benchmark.measure, RecoveryScheduler._dispatch

    def measure(*args, **kwargs):
        ready = kwargs.get("ready")
        if ready is not None:
            dispatched.clear()

            def first_batch():
                ready.set()
                assert dispatched.wait(30), "FTE never dispatched during the retained scan"

            kwargs["ready"] = SimpleNamespace(set=first_batch)
        return original_measure(*args, **kwargs)

    def dispatch(*args, **kwargs):
        admitted = original_dispatch(*args, **kwargs)
        if admitted:
            dispatched.set()
        return admitted

    monkeypatch.setattr(benchmark, "measure", measure)
    monkeypatch.setattr(RecoveryScheduler, "_dispatch", dispatch)
    config = benchmark.Configuration(
        tmp_path / "benchmark",
        rows=1024,
        repetitions=1,
        modes=("pipelined", "fte"),
        consumer_rows_per_second=512,
        interface=interface,
    )
    report = benchmark.run(config)
    assert report["complete"]
    for profile in benchmark.PROFILES:
        samples = [s for s in report["samples"] if s["profile"] == profile]
        assert {s["phase"] for s in samples} == {
            "cold",
            "warmup",
            "warm",
            "slow",
            "mixed",
            "recovery",
            "recovery_control",
        }
        recovery = next(s for s in samples if s["phase"] == "recovery")
        assert recovery["attempt_count"] == 2 and recovery["same_input_id"] and recovery["distinct_fences"]
        assert 0 < recovery["fault_to_completion_seconds"] <= recovery["total_seconds"]
        mixed = [s for s in samples if s["phase"] == "mixed"]
        assert len(mixed) == 2 and {s["mode"] for s in mixed} == {"fte", "pipelined"}
        controls = [s for s in samples if s["phase"] in {"recovery", "recovery_control"}]
        assert len(controls) == 2 and controls[0]["rows"] == controls[1]["rows"]
    snapshots = json.loads((config.output / "resources.json").read_text())
    assert len(report["recovery_pairs"]) == 2
    assert len(report["mixed_pairs"]) == 2
    for pair in report["mixed_pairs"]:
        assert pair["max_shared_worker_overlap_seconds"] > 0
        assert all(v["released_at_monotonic"] is not None for v in pair["worker_reservations"])
        assert {v["query_id"] for v in pair["worker_reservations"]} == {
            pair["pipelined_query_id"],
            pair["fte_query_id"],
        }
    assert len(report["validated"]) == 16
    assert len(snapshots) == 4
    if interface == "flight":
        assert all(s["client_resources"]["exported_bytes"] > 0 and s["gateway_links"] == 1 for s in snapshots)
        assert all(
            s["server_diagnostics"]["session_resources"]["result_delivery"]["usage_bytes"] == 0 for s in snapshots
        )
    else:
        assert all(s["diagnostics"]["session_resources"]["result_delivery"]["usage_bytes"] > 0 for s in snapshots)
    assert json.loads((config.output / "report.json").read_text())["complete"]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("case", ["serial", "failure"])
@pytest.mark.parametrize("interface", benchmark.INTERFACES)
def test_mixed_rejects_serial_dispatch_and_stops_a_sleeping_sibling(tmp_path, monkeypatch, case, interface):
    from vane.execution.pipelined_runtime import PipelinedContext
    from vane.execution.recovery_runtime import RecoveryScheduler

    config = benchmark.Configuration(
        tmp_path / "benchmark",
        rows=1024,
        repetitions=1,
        profiles=("default",),
        modes=("pipelined", "fte"),
        scenarios=("mixed",),
        consumer_rows_per_second=1 if case == "failure" else 50_000,
        interface=interface,
    )
    mixed_started, scan_finished = threading.Event(), threading.Event()
    fault = {}
    original_measure, original_dispatch = benchmark.measure, RecoveryScheduler._dispatch
    original_start, original_failed = threading.Thread.start, PipelinedContext.failed
    failure_reported = threading.Event()

    def start(thread):
        original_start(thread)
        if case == "failure" and mixed_started.is_set() and thread.name == "vane-recovery-scheduler":
            # A fast first-attempt failure can precede Thread.start() returning.
            # Force that ordering so preparation/cleanup races cannot hide.
            assert failure_reported.wait(10), "FTE did not publish the injected failure"

    def failed(context, message):
        try:
            return original_failed(context, message)
        finally:
            if message == "controlled mixed dispatch failure":
                failure_reported.set()

    def measure(*args, **kwargs):
        if kwargs.get("ready") is None:
            return original_measure(*args, **kwargs)
        mixed_started.set()
        try:
            return original_measure(*args, **kwargs)
        finally:
            scan_finished.set()

    def dispatch(*args, **kwargs):
        if mixed_started.is_set():
            if case == "serial":
                assert scan_finished.wait(10), "scan did not release its worker reservations"
            else:
                fault["at"] = time.monotonic()
                raise RuntimeError("controlled mixed dispatch failure")
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(benchmark, "measure", measure)
    monkeypatch.setattr(RecoveryScheduler, "_dispatch", dispatch)
    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(PipelinedContext, "failed", failed)
    message = "did not overlap in worker reservations" if case == "serial" else "controlled mixed dispatch failure"
    with pytest.raises((AssertionError, RuntimeError), match=message):
        benchmark.run(config)
    report = json.loads((config.output / "report.json").read_text())
    assert not report["complete"]
    assert scan_finished.is_set()
    if case == "serial":
        pair = report["mixed_pairs"][0]
        assert pair["max_shared_worker_overlap_seconds"] == 0
        assert {v["query_id"] for v in pair["worker_reservations"]} == {
            pair["pipelined_query_id"],
            pair["fte_query_id"],
        }
    else:
        # The held batch would otherwise sleep for hundreds of seconds.
        assert report["failure"] == {"type": "RuntimeError", "message": message}
        assert time.monotonic() - fault["at"] < 10
