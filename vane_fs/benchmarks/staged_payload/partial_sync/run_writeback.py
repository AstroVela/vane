# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the writeback gates, then run the unchanged external workload."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--medium", choices=("tmpfs", "ext4"), required=True)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    assert json.loads((out / "writeback-tests.json").read_text())["status"] == "PASS"
    reference = json.loads((here / "writeback-plan.json").read_text())
    if not (out / "plan.json").exists():
        build = json.loads((out / "build.json").read_text())
        assert build["status"] == "PASS"
        parent = json.loads((Path(build["parent"]) / "plan.json").read_text())
        for key in (
            "runner_source",
            "runner_source_sha256",
            "prepared_runner_sha256",
            "workload_source",
            "workload_source_sha256",
            "prepared_workload_sha256",
        ):
            reference[key] = parent[key]
        reference["frozen_epoch"] = time.time()
        reference["inputs"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                here / "writeback.patch",
                here.parent / "partial_edges/run.py",
                here.parents[1] / "checkpoint_priority/analyze.py",
            )
        }
        (out / "plan.json").write_text(json.dumps(reference, indent=2) + "\n")
    plan = json.loads((out / "plan.json").read_text())
    for key in (
        "orders",
        "media",
        "primary_gates",
        "regression_gates",
        "writeback_range_bytes",
        "buffer_page_offset",
        "large_sequence",
        "durability",
    ):
        assert plan[key] == reference[key], key
    subprocess.run(
        [sys.executable, str(here.parent / "partial_edges/run.py"), str(out), "--medium", args.medium], check=True
    )


if __name__ == "__main__":
    main()
