# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from contextlib import contextmanager

import pytest


@pytest.fixture
def sqlite_writer():
    @contextmanager
    def lock(database):
        # Keep the inspection SQLite library in a separate process from the
        # native library, and acknowledge lock acquisition before closing pins.
        program = (
            "import sqlite3,sys; db=sqlite3.connect(sys.argv[1]); "
            "db.execute('BEGIN IMMEDIATE'); print('locked',flush=True); "
            "sys.stdin.readline(); db.rollback()"
        )
        process = subprocess.Popen(
            [sys.executable, "-I", "-u", "-c", program, str(database)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdout.readline().strip() == "locked"
            yield
        finally:
            try:
                process.communicate("\n", timeout=5)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)
            assert process.returncode == 0

    return lock
