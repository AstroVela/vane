# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the inherited marker/matching contracts against the reduced observer."""

from pathlib import Path

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent / "sqlite_step_waits/test_observer.py"
    exec(compile(parent.read_text(), str(parent), "exec"), globals())
