# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute post-GC overwrites and small-file tails without changing the policy."""

import importlib.util
import json
import sys
from pathlib import Path

from prepare import digest, save

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sync_attribution", HERE.parent / "fuse_sync_waits/analyze.py")
attribution = importlib.util.module_from_spec(spec)
# The reused module's common.py uses the same typed recorder validator.
sys.path.insert(0, str(HERE.parent / "fuse_sync_waits"))
spec.loader.exec_module(attribution)
sys.path.pop(0)
belongs = attribution.belongs


def main():
    out = Path(sys.argv[1]).resolve()
    parent = HERE.parent / "range_writeback_waits/analyze.py"
    source = parent.read_text()
    changes = [
        ('len(plain["full_failed_checks"]) == 7', 'len(plain["full_failed_checks"]) == 9'),
        (
            '                    if r[8] == phase["application_pid"]\n'
            "                    and overlap(r[2:4], (begin, end))\n"
            '                    and (small or r[0] == interval["kind"])',
            '                    if belongs(r, interval, phase["application_pid"])',
        ),
    ]
    code = source
    for before, after in changes:
        assert code.count(before) == 1
        code = code.replace(before, after)
    reverse = code
    for before, after in reversed(changes):
        assert reverse.count(after) == 1
        reverse = reverse.replace(after, before)
    assert reverse == source
    save(
        out / "analysis-adapter.json",
        dict(status="PASS", parent=str(parent), sha256=digest(parent), exact_reverse=True, changes=changes),
    )
    exec(compile(code, str(parent), "exec"), dict(__name__="__main__", __file__=__file__, belongs=belongs))
    findings = json.loads((out / "findings.json").read_text())
    findings["inputs"].update({str(path): digest(path) for path in (parent, Path(attribution.__file__))})
    save(out / "findings.json", findings)


if __name__ == "__main__":
    main()
