# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise installed-test launchers without recursively running their suites."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("launcher", ["run_installed_pytest.sh", "run_release_tests.sh", "run_fast_tests.sh"])
@pytest.mark.parametrize("pytest_status", [0, 7])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_launcher_leaves_temporary_directory_before_cleanup(tmp_path, launcher, pytest_status, cleanup_fails):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required to exercise the test launchers")
    real_rm = subprocess.check_output([bash, "-c", "command -v rm"], text=True).strip()
    executable = shlex.quote(Path(sys.executable).as_posix())
    scratch = tmp_path / "temporary tests"
    scratch.mkdir()
    record = tmp_path / "probe.json"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json\nfrom pathlib import Path\n"
        f"Path({str(record)!r}).write_text(json.dumps({{'cwd': str(Path.cwd())}}))\n"
        f"raise SystemExit({pytest_status})\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "pytest" ]; then\n'
        f"  exec {executable} -I {shlex.quote(probe.as_posix())}\n"
        "fi\n"
        f'exec {executable} "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    python.chmod(0o755)
    rm = bin_dir / "rm"
    # Model Windows' current-directory lock on every host. Use the real remover
    # only after verifying the target is one of this test's owned directories.
    guard = (
        "from pathlib import Path; import sys; "
        "target = Path(sys.argv[1]).resolve(); "
        f"owned = target.parent == Path({str(scratch)!r}).resolve(); "
        "sys.exit(2 if not owned else 0 if Path.cwd().resolve() == target else 1)"
    )
    rm.write_text(
        '#!/bin/sh\nfor target in "$@"; do :; done\n'
        "status=0\n"
        f'{executable} -I -c {shlex.quote(guard)} "$target" || status=$?\n'
        'if [ "$status" = 0 ]; then\n'
        '  echo "test directory is still the shell current directory" >&2\n  exit 16\n'
        'elif [ "$status" != 1 ]; then\n  exit "$status"\nfi\n'
        + (
            'echo "injected cleanup failure" >&2\nexit 42\n' if cleanup_fails else f'exec {shlex.quote(real_rm)} "$@"\n'
        ),
        encoding="utf-8",
        newline="\n",
    )
    rm.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        TMPDIR=scratch.as_posix(),
        VANE_TEST_DIAGNOSTICS_DIR="",
        VANE_FAST_TEST_JUNIT_DIR="",
        VANE_FAST_TEST_NON_RAY_SHARD_COUNT="1",
        VANE_FAST_TEST_NON_RAY_SHARD_INDEX="0",
        VANE_FAST_TEST_PROCESS_TIMEOUT_SECONDS="0",
    )
    args = ["probe.py"] if launcher == "run_installed_pytest.sh" else ["non-ray"]
    completed = subprocess.run(
        [
            bash,
            "-c",
            'export PATH="$(cd "$1" && pwd):$PATH"; shift; source "$@"',
            "launcher-test",
            bin_dir.as_posix(),
            (scripts / launcher).as_posix(),
            *args,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == (42 if cleanup_fails else pytest_status), completed.stdout + completed.stderr
    test_directory = Path(json.loads(record.read_text())["cwd"])
    assert test_directory.parent.resolve() == scratch.resolve()
    assert test_directory.is_dir() == cleanup_fails
    if cleanup_fails:
        assert "injected cleanup failure" in completed.stderr
