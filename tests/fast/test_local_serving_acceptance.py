# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
import time
from pathlib import Path

import pytest

from vane.execution.request_deadline import MonotonicDeadline
from vane.execution.result_delivery import ManagedResult, ResultDeliveryFull, RuntimeResultDelivery


def acceptance():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "validate_local_serving.py"))


@pytest.mark.parametrize("expire_before_publication", [False, True])
@pytest.mark.timeout(120)
def test_cpu_serving_acceptance_uses_one_runtime_and_returns_to_baseline(
    monkeypatch, tmp_path, expire_before_publication
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    # Capacity diagnostics are human-readable, not a parsing contract. All
    # acceptance decisions must survive changed wording at both refusal sites.
    def opaque_message(operation):
        def invoke(*args, **kwargs):
            try:
                return operation(*args, **kwargs)
            except ResultDeliveryFull as error:
                error.args = ("localized capacity diagnostic",)
                raise

        return invoke

    monkeypatch.setattr(RuntimeResultDelivery, "begin", opaque_message(RuntimeResultDelivery.begin))
    monkeypatch.setattr(ManagedResult, "own_buffer", opaque_message(ManagedResult.own_buffer))
    if expire_before_publication:
        start = MonotonicDeadline.start

        def expired_start(deadline):
            if deadline._thread_name == "vane-result-deadline":
                deadline.expires_at = time.monotonic() - 1
            start(deadline)

        monkeypatch.setattr(MonotonicDeadline, "start", expired_start)
    report = acceptance()["run_acceptance"](tmp_path, requests=4, concurrency=4)
    assert report["status"] == "passed"
    assert report["schema_version"] == 2
    assert report["configuration"]["query_apis"] == ["sql", "relation"]
    assert all(report["load_api_counts"][api] > 0 for api in ("sql", "relation"))
    assert sum(report["load_api_counts"].values()) == 9
    assert report["model"]["cold_initializations"] == 1
    assert report["model"]["healthy_additional_initializations"] == 0
    assert report["model"]["observed_worker_exit_failures"] == 1
    assert report["measurements"]["warm"]["latency_seconds"]["count"] == 4
    assert report["measurements"]["warm"]["delivery_seconds"]["count"] == 4
    assert report["measurements"]["mixed_analysis"]["latency_seconds"]["count"] == 1
    for name, executions in (("cold", 1), ("warm", 4), ("mixed", 4)):
        metrics = report["phase_request_metrics"][name]
        assert metrics["executed_requests"] == metrics["completed_requests"] == executions
        assert metrics["execution_seconds"] > 0
        assert metrics["queue_wait_seconds"] >= 0
        assert metrics["cleanup_seconds"] >= 0
    checkpoints = report["checkpoints"]
    assert checkpoints["ingress_full"]["request_admission"]["active_requests"] == 2
    assert checkpoints["ingress_full"]["request_admission"]["queued_requests"] == 2
    assert checkpoints["ingress_full"]["result_delivery"]["active_results"] == 2
    assert checkpoints["slow_consumer_slots"]["result_delivery"]["active_results"] == 2
    assert checkpoints["slow_consumer_view"]["result_delivery"]["exported_bytes"] > 0
    assert checkpoints["slow_consumer_numpy_view"]["result_delivery"]["exported_bytes"] > 0
    assert checkpoints["closed"]["closed"]
    assert checkpoints["closed"]["request_admission"]["closed"]
    assert report["before_close"]["request_admission"]["failed_executions"] == 2
    assert report["before_close"]["request_admission"]["cancelled_requests"] >= 2
    assert report["before_close"]["result_delivery"]["timed_out_results"] == 1
    deadlines = report["deadline_sessions"]
    queue = deadlines["queue"]["closed"]["request_admission"]
    assert queue["timed_out_requests"] == 1
    assert queue["cancelled_requests"] == 0
    assert deadlines["execution"]["initializations"] == 0
    assert deadlines["execution"]["closed"]["request_admission"]["execution_timed_out_requests"] == 2
    for session in deadlines.values():
        assert session["closed"]["closed"]
        assert session["closed"]["data"]["usage_bytes"] == 0
        assert session["closed"]["result_delivery"]["usage_bytes"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(("reason", "started"), [(None, None), ("slots", None), ("slots", True), ("bytes", True)])
def test_acceptance_does_not_retry_unverified_or_post_execution_refusals(tmp_path, reason, started):
    # No real runtime is needed: this tests the driver's conservative policy,
    # independently of error wording or this fixture's call markers.
    scenario_type = acceptance()["Scenario"]
    scenario = object.__new__(scenario_type)
    scenario.directory = tmp_path
    attempts = []
    error = ResultDeliveryFull("runtime result slots are full", reason=reason)
    error._execution_started = started

    def refuse():
        attempts.append(None)
        raise error

    with pytest.raises(ResultDeliveryFull) as caught:
        scenario.execute_with_slot_retry(refuse, "request")
    assert caught.value is error
    assert len(attempts) == 1


def test_latency_report_uses_nearest_rank_and_explicit_empty_groups():
    distribution = acceptance()["distribution"]
    assert distribution([]) == dict(count=0, mean=None, p95=None, p99=None, max=None)
    assert distribution([7]) == dict(count=1, mean=7, p95=7, p99=7, max=7)
    assert distribution(range(1, 101)) == dict(count=100, mean=50.5, p95=95, p99=99, max=100)


@pytest.mark.parametrize("kwargs", [{"requests": 1}, {"requests": True}, {"concurrency": 0}, {"concurrency": 5}])
def test_invalid_load_configuration_starts_no_runtime(tmp_path, kwargs):
    with pytest.raises(ValueError):
        acceptance()["run_acceptance"](tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())
