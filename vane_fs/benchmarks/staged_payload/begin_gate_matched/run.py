# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen matched windows serially without native rebuilds."""

import json
import shutil
import sys
import time
from pathlib import Path

from checks import owned_state
from prepare import save

if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    tests = json.loads((out / "tests.json").read_text())
    assert tests["status"] == "PASS"
    state = owned_state(out, tests["cleanup"])
    assert not state["active_owned_pids"] and not state["owned_mounts"]
    save(
        out / "measurement-preflight.json",
        state
        | dict(
            status="PASS",
            recorded_epoch=time.time(),
            free_bytes=shutil.disk_usage(out).free,
            owned_build_test_analysis_stopped=True,
        ),
    )
    parent = Path(__file__).resolve().parent.parent / "extent_fuse/run.py"
    exec(compile(parent.read_text(), str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
