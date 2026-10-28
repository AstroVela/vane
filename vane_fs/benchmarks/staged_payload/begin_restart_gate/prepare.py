# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the original 22 gates against the original extent control before timing."""

import json
import sys
from pathlib import Path

from generate import digest, save

if __name__ == "__main__":
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    report = json.loads((out / "gate-tests.json").read_text())
    assert report["status"] == "PASS" and report["parent_negative_control"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in (report["inputs"] | report["binaries"]).items())
    for name in ("initial-preflight.json", "full-pause-preflight.json"):
        save(out / name, dict(status="NOT_NEEDED", inputs={}))
    parent = here.parent / "paced_range_writeback/prepare.py"
    original = code = parent.read_text()
    before = 'before = f"Finite-range 64 MiB payload writeback: {window} ordinary FUSE window; both formats 1005"'
    after = 'before = f"Paced 4 MiB range claims, 16 ms completion delay and FULL pause: {window} ordinary FUSE window; both formats 1005"'
    assert code.count(before) == 1
    code = code.replace(before, after)
    before2 = 'after = f"Paced 4 MiB range claims, 16 ms completion delay and FULL pause: {window} ordinary FUSE window; both formats 1005"'
    after2 = (
        'after = f"Paced range claims plus local BEGIN/RESTART gate: {window} ordinary FUSE window; both formats 1005"'
    )
    assert code.count(before2) == 1
    code = code.replace(before2, after2)
    assert code.replace(after2, before2).replace(after, before) == original
    exec(compile(code, str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
    frozen = json.loads((out / "fuse/build.json").read_text())
    frozen["inputs"].update(report["inputs"] | report["binaries"])
    frozen["inputs"][str(out / "gate-tests.json")] = digest(out / "gate-tests.json")
    save(out / "fuse/build.json", frozen)
