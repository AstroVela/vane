# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run unchanged complete workloads in a separate short-request diagnostic window."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--medium", choices=("tmpfs", "ext4"), required=True)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    parent = Path(build["parent"])
    old_plan = json.loads((parent / "plan.json").read_text())

    def check_inputs():
        for record in build["original_sources"].values():
            assert all(digest(Path(record["path"]) / p) == h for p, h in record["sha256"].items())
        for name, paths in build["source_inputs"].items():
            assert all(digest(out / "diagnostic" / name / p) == h for p, h in paths.items())
        assert all(digest(out / p) == h for p, h in build["binaries"].items())
        assert all(digest(Path(p)) == h for p, h in build["inputs"].items())

    check_inputs()
    assert digest(parent / "run_fuse.py") == old_plan["prepared_runner_sha256"]
    assert digest(parent / "workload.py") == old_plan["prepared_workload_sha256"]
    runner = (parent / "run_fuse.py").read_text()
    needle = "(1 if args.diagnostic else 3)"
    assert runner.count(needle) == 1
    runner = runner.replace(needle, "(2 if args.diagnostic else 3)")
    runner = runner.replace(".timeline.json", ".request-costs.json").replace("-timeline.json", "-request-costs.json")
    runner = runner.replace('row["timeline_file"]', 'row["request_costs_file"]')
    runner = runner.replace(
        "instrumented=args.diagnostic,",
        'instrumented=args.diagnostic,\n        diagnostic_scope="Per-request wall/CPU time, inclusive short-stage counters and SQLite VM steps; not adoption throughput",',
    )
    plan = dict(
        status="FROZEN_BEFORE_DIAGNOSTIC_WINDOW",
        frozen_epoch=time.time(),
        parent=str(parent),
        medium=args.medium,
        states=["fresh", "post_gc"],
        orders=[["baseline", "updated"], ["updated", "baseline"]] * 2,
        mounts=16,
        native_logic_unchanged=True,
        submission_budget_bytes=4194304,
        minimum_submission_gap_ms=16,
        global_cache_eviction=False,
        owned_build_or_correctness_test_overlap=False,
        stage_times="Inclusive; nested counters must not be summed as disjoint durations",
        inputs={str(p): digest(p) for p in (here / "run.py", parent / "run_fuse.py", parent / "workload.py")},
        prepared_runner_sha256=hashlib.sha256(runner.encode()).hexdigest(),
        prepared_workload_sha256=digest(parent / "workload.py"),
    )
    (out / (args.medium + "-plan.json")).write_text(json.dumps(plan, indent=2) + "\n")
    (out / "run_fuse.py").write_text(runner)
    (out / "workload.py").write_bytes((parent / "workload.py").read_bytes())
    subprocess.run(
        [sys.executable, str(out / "run_fuse.py"), str(out), "--medium", args.medium, "--diagnostic"], check=True
    )
    check_inputs()
    (out / (args.medium + "-input-check.json")).write_text(json.dumps(dict(status="PASS"), indent=2) + "\n")


if __name__ == "__main__":
    main()
