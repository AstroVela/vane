# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("flag", "output", "window", "expected"),
    [
        ("yes", "8192", "64", "must be 0 or 1"),
        ("1", None, "64", "requires VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES"),
        ("1", "8192", None, "requires VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES"),
        ("0", None, None, None),
        ("1", "8192", "64", None),
    ],
)
def test_scalar_admission_configuration_is_validated_before_execution(flag, output, window, expected):
    env = {**os.environ, "VANE_NATIVE_MEDIA_ENABLE_SCALAR_ADMISSION": flag, "VANE_RUNNER": "local-fast"}
    for name, value in (
        ("VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES", output),
        ("VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES", window),
    ):
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    script = "import vane\nwith vane.connect() as con:\n    assert con.execute('SELECT 42').fetchone() == (42,)\n"
    result = subprocess.run([sys.executable, "-I", "-c", script], env=env, text=True, capture_output=True, timeout=30)
    if expected:
        assert result.returncode != 0
        assert expected in result.stderr, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
