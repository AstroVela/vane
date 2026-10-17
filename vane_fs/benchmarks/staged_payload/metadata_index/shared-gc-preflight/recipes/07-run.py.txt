# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapt the fixed-buffer acceptance workload to managed external format 1004."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--medium", choices=("tmpfs", "ext4"), required=True)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    assert json.loads((out / "tests.json").read_text())["status"] == "PASS"
    assert json.loads((out / "sanitize.json").read_text())["status"] == "PASS"
    here = Path(__file__).resolve().parent
    acceptance = here.parents[1] / "partial_write/acceptance"
    plan = json.loads((out / "plan.json").read_text())
    source = acceptance / "run_fuse.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == plan["runner_source_sha256"]
    runner = source.read_text().replace(
        'assert json.loads((out / "probe-build.json").read_text())["status"] == "PASS"',
        'assert json.loads((out / "build.json").read_text())["status"] == "PASS"',
    )
    runner = runner.replace("here = Path(__file__).resolve().parent", "here = Path(" + repr(str(acceptance)) + ")")
    runner = runner.replace('here / "workload.py"', 'out / "workload.py"')
    runner = runner.replace("external_threshold_bytes=None,", "external_threshold_bytes=262144,")
    runner = runner.replace("format_version=2,", "format_version=1004,")
    runner = runner.replace(
        "schema == [(2, 4096)] and all(n == 4096 for n, _ in lengths)",
        "schema == [(1004, 4096)] and all(n in (8, 4096) for n, _ in lengths)",
    )
    workload_source = acceptance / "workload.py"
    assert hashlib.sha256(workload_source.read_bytes()).hexdigest() == plan["workload_source_sha256"]
    workload = workload_source.read_text().replace(
        'assert not Path(str(db) + "-wal").exists() and not Path(str(db) + "-shm").exists()',
        'assert Path(str(db) + ".stage-fence").read_bytes()[40:48] == bytes(8)',
    )
    assert hashlib.sha256(runner.encode()).hexdigest() == plan["prepared_runner_sha256"]
    assert hashlib.sha256(workload.encode()).hexdigest() == plan["prepared_workload_sha256"]
    (out / "run_fuse.py").write_text(runner)
    (out / "workload.py").write_text(workload)
    subprocess.run([sys.executable, str(out / "run_fuse.py"), str(out), "--medium", args.medium], check=True)


if __name__ == "__main__":
    main()
