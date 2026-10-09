# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""The standalone benchmark must keep Ray workers on the installed wheel."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import benchmark_execution as benchmark

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner]


@pytest.mark.timeout(180)
@pytest.mark.parametrize("interface", benchmark.INTERFACES)
def test_benchmark_cli_from_checkout_keeps_workers_on_installed_wheel(tmp_path, interface):
    root = Path(benchmark.__file__).resolve().parents[1]
    output = tmp_path / "benchmark"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(root / "scripts/benchmark_execution.py"),
            "--output",
            str(output),
            "--interface",
            interface,
            "--rows",
            "257",
            "--repetitions",
            "1",
            "--warmups",
            "1",
            "--worker-count",
            "1",
            "--partitions",
            "1",
            "--profiles",
            "default",
            "compact",
            "--modes",
            "pipelined",
            "fte",
            "--scenarios",
            "cold",
            "warm",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        text=True,
        capture_output=True,
        timeout=150,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] and report["cluster_startup_seconds"] > 0
    assert report["ray_cluster_resources"]["CPU"] == 1
    assert {s["mode"] for s in report["samples"]} == {"pipelined", "fte"}
    assert len(report["validated"]) == 16
    assert len(report["samples"]) == 36
