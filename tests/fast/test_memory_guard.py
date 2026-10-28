# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""The CI watchdog must stop only the command it owns."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

_GUARD = Path(__file__).resolve().parents[2] / "scripts" / "run_with_memory_guard.py"


def _run_guard(tmp_path, script, *options):
    report = tmp_path / "memory.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_GUARD),
            "--report",
            str(report),
            "--min-available-mib=0",
            *options,
            "--",
            sys.executable,
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result, json.loads(report.read_text())


def _active_process(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize("code", [0, 7])
def test_memory_guard_preserves_command_exit_code(tmp_path, code):
    result, report = _run_guard(tmp_path, f"import sys; sys.exit({code})")
    assert result.returncode == report["exit_code"] == code
    assert report["stopped_reason"] is None


@pytest.mark.parametrize("failure", ["memory", "timeout"])
def test_memory_guard_stops_descendants_and_preserves_unrelated_process(tmp_path, failure):
    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time; from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"Path({str(pid_file)!r}).write_text(str(child.pid)); "
        + (
            "data = bytearray(128 * 1024**2); data[::4096] = b'x' * (len(data) // 4096); "
            if failure == "memory"
            else ""
        )
        + "time.sleep(60)"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        options = ["--max-process-mib=96"] if failure == "memory" else ["--timeout=2"]
        result, report = _run_guard(tmp_path, script, *options)
        assert result.returncode == report["exit_code"] == (137 if failure == "memory" else 124)
        assert failure in report["stopped_reason"] or "deadline" in report["stopped_reason"]
        assert unrelated.poll() is None
        child_pid = int(pid_file.read_text())
        deadline = time.monotonic() + 5
        while _active_process(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _active_process(child_pid)
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_memory_guard_rejects_reused_ancestor_pid():
    spec = importlib.util.spec_from_file_location("vane_memory_guard", _GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Process:
        def __init__(self, pid, parent, created):
            self.pid, self.parent, self.created = pid, parent, created

        def is_running(self):
            return True

        def ppid(self):
            return self.parent

        def create_time(self):
            return self.created

        def children(self, *, recursive):
            assert recursive
            return children

    root = Process(10, 1, 100)
    child = Process(11, 10, 101)
    reused_pid_descendant = Process(12, 11, 90)
    unrelated_descendant = Process(13, 12, 91)
    children = [child, reused_pid_descendant, unrelated_descendant]
    assert module.owned_process_tree(root) == [root, child]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process sessions")
def test_memory_guard_ignores_inaccessible_sessions(monkeypatch):
    spec = importlib.util.spec_from_file_location("vane_memory_guard", _GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = psutil.Process()
    monkeypatch.setattr(module, "owned_process_tree", lambda _: [root])
    monkeypatch.setattr(psutil, "process_iter", lambda: iter([root]))

    def deny_session_lookup(_pid):
        raise PermissionError("session is not accessible")

    monkeypatch.setattr(os, "getsid", deny_session_lookup)
    assert module.owned_processes(root) == [root]


@pytest.mark.parametrize("failure", ["memory", "timeout"])
@pytest.mark.parametrize(
    "separate_process_group",
    [False, pytest.param(True, marks=pytest.mark.skipif(os.name == "nt", reason="POSIX process groups"))],
)
def test_memory_guard_retains_children_after_their_parent_exits(tmp_path, failure, separate_process_group):
    pid_file = tmp_path / "orphan.pid"
    worker = (
        "import os, time; from pathlib import Path; "
        + ("os.setpgid(0, 0); " if separate_process_group else "")
        + f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(0.75); " + ("data = bytearray(128 * 1024**2); " if failure == "memory" else "") + "time.sleep(60)"
    )
    middle = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {worker!r}])"
    script = (
        f"import subprocess, sys, time; subprocess.run([sys.executable, '-c', {middle!r}], check=True); time.sleep(60)"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        options = ["--max-process-mib=96"] if failure == "memory" else ["--timeout=3"]
        result, report = _run_guard(tmp_path, script, *options)
        assert result.returncode == report["exit_code"] == (137 if failure == "memory" else 124)
        assert pid_file.is_file()
        assert not _active_process(int(pid_file.read_text()))
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_memory_guard_preserves_stdin_and_virtual_environment(tmp_path):
    report = tmp_path / "memory.json"
    script = "import os, sys; print(sys.prefix); print(sys.stdin.read()); print(os.environ['VANE_GUARD_TEST'])"
    result = subprocess.run(
        [
            sys.executable,
            str(_GUARD),
            "--report",
            str(report),
            "--min-available-mib=0",
            "--",
            sys.executable,
            "-c",
            script,
        ],
        input="stdin remains available",
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "VANE_GUARD_TEST": "inherited"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [sys.prefix, "stdin remains available", "inherited"]


@pytest.mark.parametrize("code", [0, 7])
def test_memory_guard_cleans_orphans_after_the_command_exits(tmp_path, code):
    pid_file = tmp_path / "orphan.pid"
    worker = (
        "import os, time; from pathlib import Path; "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    script = (
        "import subprocess, sys, time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {worker!r}]); "
        f"path = Path({str(pid_file)!r})\n"
        "while not path.is_file(): time.sleep(0.01)\n"
        f"sys.exit({code})"
    )
    result, report = _run_guard(tmp_path, script)
    assert result.returncode == report["exit_code"] == code
    assert report["stopped_reason"] is None
    assert not _active_process(int(pid_file.read_text()))
