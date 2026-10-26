# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare orphan batching plus payload byte budget against the unchanged adoption control."""

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
    assert json.loads((out / "budget-tests.json").read_text())["status"] == "PASS"
    sources = {
        "baseline": Path(build["parent"]) / "updated",
        "orphan-control": Path(build["orphan_parent"]) / "updated",
        "updated": out / "updated",
    }

    def check():
        assert all(digest(sources[n] / p) == h for n, paths in build["source_inputs"].items() for p, h in paths.items())
        assert all(digest(out / p) == h for p, h in build["binaries"].items())
        assert digest(Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a") == build["sdk"]["sqlite_sha256"]
        assert all(digest(Path(p)) == h for p, h in build["inputs"].items())

    check()
    original = json.loads((here.parent / "writeback_budget/writeback-plan.json").read_text())
    keys = (
        "media",
        "orders",
        "states",
        "large_sequence",
        "buffer_page_offset",
        "durability",
        "gc_preparation",
        "primary_gates",
        "regression_gates",
        "owned_build_or_test_overlap",
        "global_cache_eviction",
        "measurement_scope",
    )
    reference = {k: original[k] for k in keys}
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
            status="FROZEN_BEFORE_PLAIN_COMPARISON",
            frozen_epoch=time.time(),
            baseline="Original no-hint partial-edge control, format 1004",
            candidate="Orphan batching plus 64 MiB shared unpublished-payload budget and before-transaction admission",
            publication_presync=False,
            payload_budget_bytes=64 * 1024**2,
            payload_origin_shared_in_fence=True,
            causal_comparison=dict(
                medium="ext4", pairs_per_state=6, baseline="Preceding orphan batching", adoption_gates_replaced=False
            ),
            writeback_hints=False,
            wal_checkpoint_thresholds_and_clean_progress_unchanged=True,
            payload_worker_wake_and_admission=True,
            final_barriers_unchanged=True,
            timing_boundaries_unchanged=True,
            diagnostic_comparison=dict(
                medium="ext4",
                pairs_per_state=4,
                baseline="Preceding orphan batching",
                instruments="C++ request costs and read-only owned CPU sampling",
            ),
            inputs={
                str(p): digest(p)
                for p in (
                    here / "budget.patch",
                    here / "test_budget.cpp",
                    here / "run_causal.py",
                    here / "run.py",
                    here / "run_diagnostic.py",
                    here.parent / "partial_edges/run.py",
                    here.parent / "partial_sync/analyze_writeback.py",
                )
            },
        )
        (out / "plan.json").write_text(json.dumps(reference, indent=2) + "\n")
    plan = json.loads((out / "plan.json").read_text())
    assert all(plan[k] == reference[k] for k in keys)
    assert all(digest(Path(p)) == h for p, h in plan["inputs"].items())
    subprocess.run(
        [sys.executable, str(here.parent / "partial_edges/run.py"), str(out), "--medium", args.medium], check=True
    )
    check()
    (out / (args.medium + "-input-check.json")).write_text(json.dumps(dict(status="PASS"), indent=2) + "\n")


if __name__ == "__main__":
    main()
