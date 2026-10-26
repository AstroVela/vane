# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path


def test_bounded_decoders_do_not_starve_backpressured_native_queries(tmp_path):
    """Original v0.2.0 stalls; the same workload must complete with one permit."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("decoder_admission_probe.py")), str(tmp_path)],
        env={**os.environ, "VANE_RUNNER": "local-fast", "VANE_MAX_CONCURRENT_DECODES": "1"},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
