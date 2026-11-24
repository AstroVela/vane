# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the original adoption workload with worker scheduling batches."""

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
    assert json.loads((out / "batch-tests.json").read_text())["status"] == "PASS"
    sources = {
        "baseline": Path(build["parent"]) / "updated",
        "updated": out / "updated",
        "budget-control": Path(build["budget_parent"]) / "updated",
        "rejected-control": Path(build["rejected_parent"]) / "updated",
    }

    def check():
        assert all(digest(sources[n] / p) == h for n, paths in build["source_inputs"].items() for p, h in paths.items())
        assert all(digest(out / p) == h for p, h in build["binaries"].items())
        assert digest(Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a") == build["sdk"]["sqlite_sha256"]
        assert all(digest(Path(p)) == h for p, h in build["inputs"].items())

    check()
    reference = json.loads((here.parent / "writeback_budget/writeback-plan.json").read_text())
    if not (out / "plan.json").exists():
        parent = json.loads((Path(build["parent"]) / "plan.json").read_text())
        reference.update(
            {
                k: parent[k]
                for k in (
                    "runner_source",
                    "runner_source_sha256",
                    "prepared_runner_sha256",
                    "workload_source",
                    "workload_source_sha256",
                    "prepared_workload_sha256",
                )
            }
        )
        reference.update(
            frozen_epoch=time.time(),
            candidate="Budgeted WRITE-only hints plus a fixed 2 ms first-request worker deadline; pending notifications deduplicated, CLEAN progress/retries unchanged",
            checkpoint_batching=True,
            maximum_programmed_batch_delay_ms=2,
            causal_comparison=dict(
                medium="ext4", pairs_per_state=6, baseline="preceding budget", adoption_gates_replaced=False
            ),
            diagnostic_comparison=dict(
                medium="ext4",
                pairs_per_state=4,
                baseline="preceding budget",
                instruments="C++ request costs and read-only owned CPU sampling",
            ),
        )
        reference["inputs"] = {
            str(p): digest(p)
            for p in (
                here / "batch.patch",
                here / "test_batch.cpp",
                here / "run.py",
                here / "run_causal.py",
                here / "run_diagnostic.py",
                here.parent / "partial_edges/run.py",
                here.parent / "partial_sync/analyze_writeback.py",
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
    check()
    (out / (args.medium + "-input-check.json")).write_text(json.dumps(dict(status="PASS"), indent=2) + "\n")


if __name__ == "__main__":
    main()
