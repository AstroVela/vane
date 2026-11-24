# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Serially replay the unchanged complete FUSE diagnostic with wait sampling."""

import json
import subprocess
import sys
import time
from pathlib import Path

from common import digest, save


def main():
    out = Path(sys.argv[1]).resolve()
    plan = json.loads((out / "plan.json").read_text())
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS" and plan["status"] == "FROZEN"
    assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    started = time.time()
    assert plan["frozen_epoch"] < started
    command = [sys.executable, str(out / "run_fuse.py"), str(out), "--medium", "ext4", "--diagnostic"]
    with (out / "measurement.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    save(
        out / "measurement-command.json",
        dict(argv=command, returncode=result.returncode, started_epoch=started, ended_epoch=time.time()),
    )
    assert result.returncode == 0
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    rows = raw["fuse"] + raw["aged_fuse"]
    assert len(rows) == plan["mounts"]
    for row in rows:
        assert row["cpu_sampler_stopped"]
        samples = [json.loads(line) for line in Path(row["cpu_samples_file"]).read_text().splitlines()]
        terminal = [s for s in samples if s.get("sampler_finished")]
        assert len(terminal) == 1 and terminal[0] is samples[-1]
        assert terminal[0]["stop_requested"] and terminal[0]["fatal"] is None
    assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    save(
        out / "measurement-checks.json",
        dict(
            status="PASS",
            mounts=len(rows),
            native_and_measurement_inputs_unchanged=True,
            all_samplers_stopped_without_failure=True,
            checked_epoch=time.time(),
        ),
    )
    print("PASS 16 complete mounts, stopped samplers, frozen inputs and workload validations", flush=True)


if __name__ == "__main__":
    main()
