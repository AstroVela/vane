# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the existing typed recorder around both frozen paced-trial sources."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "range_writeback_waits/prepare.py"
source = PARENT.read_text()
before, after = '"original_plain_failed_checks=7,"', '"original_plain_failed_checks=9,"'
assert source.count(before) == 1
code = source.replace(before, after)
assert code.replace(after, before) == source
namespace = dict(__name__="paced_wait_parent", __file__=__file__)
exec(compile(code, str(PARENT), "exec"), namespace)
TRACE, digest, save, validate = [namespace[key] for key in ("TRACE", "digest", "save", "validate")]


if __name__ == "__main__":
    import sys

    namespace["scope"]["namespace"]["main"]()
    out = Path(sys.argv[1]).resolve()
    report = json.loads((out / "build.json").read_text())
    report["inputs"].update(
        {
            str(path): digest(path)
            for path in [
                PARENT,
                HERE.parent / "range_writeback_waits/test_hold.cpp",
                HERE.parent / "prewriteback_waits/prepare.py",
                HERE / "prepare.py",
                HERE / "run.py",
                HERE / "analyze.py",
                HERE / "publish.py",
            ]
        }
    )
    report["paced_adapter"] = dict(
        exact_reverse=True,
        parent=str(PARENT),
        parent_sha256=digest(PARENT),
        changes=[[before, after]],
        native_policy_unchanged=True,
    )
    save(out / "build.json", report)
