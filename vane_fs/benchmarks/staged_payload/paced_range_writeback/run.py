# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute the frozen original serial FUSE runner."""

from pathlib import Path

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent / "extent_fuse/run.py"
    exec(compile(parent.read_text(), str(parent), "exec"), {"__name__": "__main__", "__file__": __file__})
