# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real CUDA lifecycle soak; uses the same supervisor and workload as CPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from local_gpu_helpers import cuda_devices as cuda_devices

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.name != "posix", reason="POSIX process-group watchdog")]


@pytest.mark.timeout(270)
def test_cuda_serving_soak_keeps_one_runtime_through_pressure_and_fault_recovery(tmp_path, cuda_devices):
    root = os.environ.get("VANE_TEST_DIAGNOSTICS_DIR")
    output = (Path(root) / f"cuda-soak-{uuid.uuid4().hex}" if root else tmp_path / "evidence").resolve()
    print(f"CUDA serving soak diagnostics: {output}", file=sys.stderr, flush=True)
    script = Path(__file__).resolve().parents[2] / "scripts" / "validate_local_serving_soak.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            "--output",
            str(output),
            "--gpu-device",
            cuda_devices[0],
            "--rounds",
            "2",
            "--requests",
            "4",
            "--timeout",
            "240",
        ],
        capture_output=True,
        text=True,
        timeout=255,
    )
    log = output / "worker.log"
    assert completed.returncode == 0, (
        f"CUDA serving soak diagnostics: {output}\n"
        + completed.stdout
        + completed.stderr
        + (log.read_text() if log.exists() else "")
    )
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "passed"
    worker = report["worker_report"]
    assert worker["runtime_sessions"] == 1
    assert worker["completed_rounds"] == 2
    assert worker["load_requests"] == 17
    assert worker["observed_worker_exit_failures"] == 2
    rounds = worker["recent_rounds"]
    assert worker["total_initializations"] == 1 + sum(
        item["cancellation_replacements"] + item["deadline_replacements"] + 2 for item in rounds
    )
    previous = worker["gpu"]["cold"]["worker"]
    assert worker["gpu"]["prewarm_seconds"] > 0
    assert worker["gpu"]["sampled_memory_peaks"]["allocated_bytes"] > 0
    for item in rounds:
        assert item["healthy_additional_initializations"] == 0
        assert item["deadline_replacements"] == 1
        assert item["fault_replacements"] == {"udf_error": 1, "worker_exit": 1}
        current = item["gpu"]["worker"]
        assert current["device"] == previous["device"] == worker["configuration"]["gpu_device"]
        assert current["generation"] == previous["generation"] + item["cancellation_replacements"] + 3
        assert current["pid"] != previous["pid"]
        assert item["gpu"]["cuda"]["pid"] == current["pid"]
        assert item["resources"]["reserved_resources"]["gpu"] == 1
        assert item["transport"]["usage_bytes"] == 0
        previous = current
    assert worker["closed"]["reserved_resources"]["gpu"] == 0
    assert worker["gpu"]["closed"]["workers"] == []
    assert worker["closed_transport"]["usage_bytes"] == 0
    for filename in ("resources.json", "idle-owners.json", "threads.log", "rounds.json"):
        assert report["diagnostics"][filename]
    assert len((output / "work" / "initializations").read_text().splitlines()) == 1
    assert not list((output / "work").glob("calls-*"))
