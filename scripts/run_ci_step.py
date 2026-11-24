#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Actions custom shell: retain Bash semantics and guard each owned process tree."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def bash_executable() -> str:
    if sys.platform == "win32":
        # CreateProcess can select System32/bash.exe (WSL) before PATH. Use
        # the Git installation that Actions uses for its Bash shell instead.
        git = shutil.which("git")
        if git:
            for candidate in (Path(git).parent.parent / "bin/bash.exe", Path(git).parent / "bash.exe"):
                if candidate.is_file():
                    return str(candidate)
        raise RuntimeError("Git for Windows Bash is required for native CI steps")
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("Bash is required for native CI steps")
    return bash


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    script = Path(sys.argv[1]).resolve()
    report = root / "test-results/memory" / f"{script.stem}.json"
    return subprocess.call(
        [
            sys.executable,
            str(root / "scripts/run_with_memory_guard.py"),
            "--report",
            str(report),
            "--min-available-mib",
            "4096" if sys.platform == "win32" else "1536",
            "--timeout",
            os.environ.get("VANE_CI_STEP_TIMEOUT_SECONDS", "7200"),
            "--",
            bash_executable(),
            "--noprofile",
            "--norc",
            "-e",
            "-o",
            "pipefail",
            script.as_posix(),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
