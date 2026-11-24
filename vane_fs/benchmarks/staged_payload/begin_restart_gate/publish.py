# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain ordinary windows, native regressions and negative controls losslessly."""

from pathlib import Path

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent / "paced_range_writeback/publish.py"
    original = code = parent.read_text()
    before = 'cleanup = build["cleanup"] + full["cleanup"] + failed_report["cleanup"] + [r for d in raw for r in d["cleanup"]]'
    after = 'cleanup = build["cleanup"] + full["cleanup"] + json.loads((out / "gate-tests.json").read_text())["cleanup"] + failed_report["cleanup"] + [r for d in raw for r in d["cleanup"]]'
    assert code.count(before) == 1
    code = code.replace(before, after)
    assert code.replace(after, before) == original
    exec(compile(code, str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
