# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(params=[False, True], ids=["native", "no_sigalrm"])
def disable_sigalrm(request):
    return request.param


def _run_pytest(tmp_path: Path, test_source: str, *, disable_sigalrm: bool) -> subprocess.CompletedProcess[str]:
    tests_root = Path(__file__).resolve().parents[1]
    child_conftest = tmp_path / "conftest.py"
    shutil.copyfile(tests_root / "conftest.py", child_conftest)
    if disable_sigalrm:
        # Exercise the Windows fallback on POSIX CI, only in the child process.
        with child_conftest.open("a", encoding="utf-8") as conftest:
            conftest.write('\nif hasattr(signal, "SIGALRM"):\n    del signal.SIGALRM\n')
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    (tmp_path / "test_watchdog_child.py").write_text(test_source)
    env = os.environ.copy()
    env.pop("PYTEST_PLUGINS", None)
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
        TEST_TIMEOUT="2",
        VANE_RUNNER="local-fast",
    )
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "-c",
            str(config),
            "--confcutdir",
            str(tmp_path),
            "--import-mode=importlib",
            "-o",
            f'pythonpath="{tests_root.as_posix()}"',
            "-q",
            "-s",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_collection_watchdog_does_not_fire_during_running_tests(tmp_path, disable_sigalrm):
    completed = _run_pytest(
        tmp_path,
        """
import time
import pytest

@pytest.mark.parametrize("case", range(4))
def test_progress(case):
    time.sleep(0.75)
""",
        disable_sigalrm=disable_sigalrm,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "4 passed" in completed.stdout
    assert "Timeout (" not in completed.stderr


def test_collection_watchdog_still_reports_slow_collection(tmp_path, disable_sigalrm):
    completed = _run_pytest(
        tmp_path,
        """
import time
time.sleep(3)

def test_collected():
    pass
""",
        disable_sigalrm=disable_sigalrm,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Timeout (0:00:02)!" in completed.stderr
    assert "test_watchdog_child.py" in completed.stderr


def test_per_test_timeout_remains_active_after_collection(tmp_path, disable_sigalrm):
    completed = _run_pytest(
        tmp_path,
        """
import time

def test_stalled():
    time.sleep(10)
""",
        disable_sigalrm=disable_sigalrm,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "test_watchdog_child.py" in completed.stderr
    assert "in test_stalled" in completed.stderr
    if hasattr(signal, "SIGALRM") and not disable_sigalrm:
        assert "TimeoutError: Test exceeded timeout of 2 seconds" in completed.stdout
        assert "=== TEST TIMEOUT (2s)" in completed.stderr
    else:
        # The fallback exits the process instead of raising a Python exception.
        assert "Timeout (0:00:02)!" in completed.stderr
        assert "TimeoutError" not in completed.stdout + completed.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native timeout watchdog")
def test_windows_timeout_terminates_native_call_holding_gil(tmp_path):
    completed = _run_pytest(
        tmp_path,
        """
import ctypes

def test_stalled():
    # PyDLL retains the GIL across the native call, unlike ctypes.WinDLL.
    sleep = ctypes.PyDLL("kernel32").Sleep
    sleep.argtypes = (ctypes.c_uint32,)
    sleep.restype = None
    sleep(10000)
""",
        disable_sigalrm=False,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "Timeout (0:00:02)!" in completed.stderr
    assert "test_watchdog_child.py" in completed.stderr
    assert "in test_stalled" in completed.stderr
