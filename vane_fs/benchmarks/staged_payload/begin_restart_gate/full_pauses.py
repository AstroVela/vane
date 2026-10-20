# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Repeat real payload/WAL/CLEAN FULL pause checks after changing BEGIN locking."""

from pathlib import Path

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent / "paced_range_writeback/validate.py"
    original = code = parent.read_text()
    before = 'here / "test_full_pause.cpp"'
    after = f"Path({str(parent.parent / 'test_full_pause.cpp')!r})"
    assert code.count(before) == 2
    code = code.replace(before, after)
    assert code.replace(after, before) == original
    exec(compile(code, str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
