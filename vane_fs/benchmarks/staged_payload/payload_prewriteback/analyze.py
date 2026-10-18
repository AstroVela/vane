# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Apply the original complete gates and retain all application write/fsync tails."""

from pathlib import Path

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent / "extent_fuse/analyze.py"
    code = parent.read_text()
    before = 'assert raw["config"]["format_versions"] == {"baseline": 1004, "updated": 1005}'
    after = 'assert raw["config"]["format_versions"] == {"baseline": 1005, "updated": 1005}'
    assert code.count(before) == 1
    code = code.replace(before, after)
    exec(compile(code, str(parent), "exec"), {"__name__": "__main__", "__file__": __file__})
