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
    build = json.loads((out / "build.json").read_text())
    sources = {"baseline": Path(build["parent"]) / "updated", "updated": out / "updated"}
    if "unbudgeted_parent" in build:
        sources["unbudgeted"] = Path(build["unbudgeted_parent"]) / "updated"

    def check_inputs():
        for name, source in sources.items():
            assert all(
                hashlib.sha256((source / p).read_bytes()).hexdigest() == h
                for p, h in build["source_inputs"][name].items()
            )
        assert all(hashlib.sha256((out / p).read_bytes()).hexdigest() == h for p, h in build["binaries"].items())
        assert (
            hashlib.sha256((Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a").read_bytes()).hexdigest()
            == build["sdk"]["sqlite_sha256"]
        )

    check_inputs()
    if not (out / "plan.json").exists():
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
                here / "budget.patch",
                here / "test_budget.cpp",
                here / "run.py",
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
        "submission_budget_bytes",
        "minimum_submission_gap_ms",
        "pause_full_publication",
        "buffer_page_offset",
        "large_sequence",
        "durability",
    ):
        assert plan[key] == reference[key], key
    subprocess.run(
        [sys.executable, str(here.parent / "partial_edges/run.py"), str(out), "--medium", args.medium], check=True
    )
    check_inputs()
    (out / (args.medium + "-input-check.json")).write_text(
        json.dumps(dict(status="PASS", sources_and_binaries_match_before_and_after=True), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
