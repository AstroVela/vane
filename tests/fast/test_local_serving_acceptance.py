# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
import time
from pathlib import Path

import pytest

from vane.execution.request_deadline import MonotonicDeadline


def acceptance():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "validate_local_serving.py"))


@pytest.mark.parametrize("expire_before_publication", [False, True])
def test_cpu_serving_acceptance_uses_one_runtime_and_returns_to_baseline(
    monkeypatch, tmp_path, expire_before_publication
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    if expire_before_publication:
        start = MonotonicDeadline.start

        def expired_start(deadline):
            if deadline._thread_name == "vane-result-deadline":
                deadline.expires_at = time.monotonic() - 1
            start(deadline)

        monkeypatch.setattr(MonotonicDeadline, "start", expired_start)
    report = acceptance()["run_acceptance"](tmp_path, requests=4, concurrency=4)
    assert report["status"] == "passed"
    assert report["model"]["cold_initializations"] == 1
    assert report["model"]["healthy_additional_initializations"] == 0
    assert report["model"]["observed_worker_exit_failures"] == 1
    assert report["measurements"]["warm"]["execution_seconds"]["count"] == 4
    assert report["measurements"]["mixed_analysis"]["latency_seconds"]["count"] == 1
    checkpoints = report["checkpoints"]
    assert checkpoints["slow_consumer_slots"]["result_delivery"]["active_results"] == 2
    assert checkpoints["slow_consumer_view"]["result_delivery"]["exported_bytes"] > 0
    assert checkpoints["closed"]["closed"]
    assert checkpoints["closed"]["request_admission"]["closed"]
    assert report["before_close"]["request_admission"]["failed_executions"] == 2
    assert report["before_close"]["request_admission"]["execution_timed_out_requests"] == 1
    assert report["before_close"]["result_delivery"]["timed_out_results"] == 1
    json.dumps(report, allow_nan=False)


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
