# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the unchanged original gates for two format-1005 variants."""

from pathlib import Path

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent / "extent_fuse/analyze.py"
    source = code = parent.read_text()
    before = 'assert raw["config"]["format_versions"] == {"baseline": 1004, "updated": 1005}'
    after = 'assert raw["config"]["format_versions"] == {"baseline": 1005, "updated": 1005}'
    assert code.count(before) == 1
    code = code.replace(before, after)
    assert code.replace(after, before) == source
    exec(compile(code, str(parent), "exec"), {"__name__": "__main__", "__file__": __file__})
