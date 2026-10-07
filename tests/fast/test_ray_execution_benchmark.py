# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise benchmark orchestration, including a real worker loss, without timing gates."""

import json

import pytest

from scripts import benchmark_execution as benchmark

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


@pytest.mark.ray_fault
@pytest.mark.timeout(300)
def test_ray_benchmark_checks_both_profiles_and_recovery_evidence(tmp_path):
    config = benchmark.Configuration(
        tmp_path / "benchmark",
        rows=1024,
        repetitions=1,
        modes=("pipelined", "fte"),
        consumer_rows_per_second=512,
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
    assert len(report["validated"]) == 16
    assert len(snapshots) == 4
    assert all(s["diagnostics"]["session_resources"]["result_delivery"]["usage_bytes"] > 0 for s in snapshots)
    assert json.loads((config.output / "report.json").read_text())["complete"]
