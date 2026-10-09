# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Release execution smoke must work with only the installed distribution."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
@pytest.mark.timeout(270)
def test_standalone_execution_smoke_uses_installed_wheel(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = tmp_path / "verify_execution_install.py"
    shutil.copyfile(root / "scripts/verify_execution_install.py", script)
    # Deliberately poison caller-side source resolution. Neither the driver,
    # workers nor the separate Flight client may import it.
    source = tmp_path / "vane"
    source.mkdir()
    (source / "__init__.py").write_text('raise RuntimeError("imported a checkout instead of the wheel")\n')
    report = tmp_path / "report.json"
    completed = subprocess.run(
        [sys.executable, "-I", str(script), "--report", str(report)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=250,
    )
    if report.exists() and os.environ.get("VANE_TEST_DIAGNOSTICS_DIR"):
        evidence = Path(os.environ["VANE_TEST_DIAGNOSTICS_DIR"]) / "execution-install.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report, evidence)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(report.read_text())
    assert result["status"] == "passed" and result["stage"] == "complete"
    assert result["checks"] == {
        name: ["aggregate", "topn", "empty"]
        for name in ("local", "runtime-pipelined", "runtime-fte", "flight-pipelined", "flight-fte")
    }
    assert result["remote_authentication"] == "passed" and not result["remote_ray_initialized"]
    assert len(result["identity"]["native_sha256"]) == 64
