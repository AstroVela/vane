# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse exact application matching before classifying SQL and kernel observations."""

import subprocess
import sys
import time
from pathlib import Path

from prepare import save

if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    parent = Path(__file__).resolve().parent.parent / "paced_request_waits/analyze.py"
    command = [sys.executable, str(parent), str(out)]
    begin = time.time()
    with (out / "base-analysis.log").open("w") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    save(
        out / "base-analysis-command.json",
        dict(argv=command, started_epoch=begin, ended_epoch=time.time(), returncode=process.returncode),
    )
    assert process.returncode == 0
