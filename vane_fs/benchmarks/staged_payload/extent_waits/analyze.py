# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute each final fsync to typed synchronization and disjoint lock waits."""

import argparse
import collections
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from prepare import TRACE, digest, save


def union(intervals):
    total, end = 0, -1
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    started = time.time()
    proof = json.loads((out / "wrapper-proof.json").read_text())
    canonical = subprocess.check_output(
        ["git", "show", "HEAD:vane_fs/benchmarks/staged_payload/fuse_waits/run.py"], cwd=TRACE
    )
    assert proof["original_sha256"] == digest(TRACE / "run.py") == hashlib.sha256(canonical).hexdigest()
    assert (out / "wrapper-proof.json").stat().st_mtime < json.loads((out / "plan.json").read_text())["frozen_epoch"]
    argv = [sys.executable, "-u", str(TRACE / "analyze.py"), str(out)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    save(
        out / "analysis-command.json",
        dict(
            argv=argv,
            started_epoch=started,
            ended_epoch=time.time(),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ),
    )
    assert result.returncode == 0, result.stderr
    analysis = json.loads((out / "analysis.json").read_text())
    assert analysis["status"] == "PASS"
    stages, records, phases, outside = analysis["stages"], [], [], []
    for mount in analysis["mounts"]:
        trace = json.loads(Path(mount["trace_file"]).read_text())
        roots = {(t["tid"], r[1]): r for t in trace["threads"] for r in t["roots"]}
        events = [(t["tid"], e) for t in trace["threads"] for e in t["events"]]
        for phase in mount["phases"]:
            app = phase["application_requests"][-1]
            assert app["kind"] == "fsync" and len(app["handler_ids"]) == 1
            tid, ident = app["handler_ids"][0]
            root = roots[tid, ident]
            assert root[0] == "fsync"
            own = [e for t, e in events if t == tid and e[2] == ident]
            syncs = [e for e in own if e[0] in ("fsync", "fdatasync")]
            typed = {
                file: sum(e[4] - e[3] for e in syncs if e[1] == file)
                for file in ("payload", "wal", "database", "fence", "other", "unknown")
            }
            exclusive = {name: root[10][i][2] for i, name in enumerate(stages)}
            assert sum(exclusive.values()) == root[3] - root[2]
            assert sum(typed.values()) <= exclusive["fsync"] + exclusive["fdatasync"]
            waits = []
            for wait in (
                e
                for e in own
                if e[0]
                in ("checkpoint_wait", "publication_wait", "lease_wait", "database_wait", "payload_guard", "flock")
            ):
                overlapping = [
                    [t, e]
                    for t, e in events
                    if t != tid and e[0] in ("fsync", "fdatasync") and max(wait[3], e[3]) < min(wait[4], e[4])
                ]
                waits.append(
                    dict(
                        event=wait,
                        concurrent_other_thread_syncs=overlapping,
                        concurrent_sync_covered_ns=union(
                            (max(wait[3], e[3]), min(wait[4], e[4])) for _, e in overlapping
                        ),
                    )
                )
            categories = dict(
                payload_sync=typed["payload"],
                wal_sync=typed["wal"],
                database_sync=typed["database"],
                checkpoint_wait=exclusive["checkpoint_wait"],
                publication_wait=exclusive["publication_wait"],
                lease_wait=exclusive["lease_wait"],
                other=root[3]
                - root[2]
                - typed["payload"]
                - typed["wal"]
                - typed["database"]
                - exclusive["checkpoint_wait"]
                - exclusive["publication_wait"]
                - exclusive["lease_wait"],
            )
            assert categories["other"] >= 0
            records.append(
                dict(
                    state=mount["state"],
                    variant=mount["variant"],
                    rep=mount["rep"],
                    workload=phase["workload"],
                    tid=tid,
                    root_id=ident,
                    wall_ns=root[3] - root[2],
                    caller_cpu_ns=root[4],
                    application_wall_ns=app["end_ns"] - app["begin_ns"],
                    exclusive_wall_ns=exclusive,
                    actual_sync_ns=typed,
                    categories_ns=categories,
                    largest_component=max(categories, key=categories.get),
                    actual_sync_events=syncs,
                    waits=waits,
                )
            )
            writing_end = app["begin_ns"]
            before = [
                (t, e) for t, e in events if e[0] in ("fsync", "fdatasync") and phase["begin_ns"] <= e[3] < writing_end
            ]
            phases.append(
                dict(
                    state=mount["state"],
                    variant=mount["variant"],
                    rep=mount["rep"],
                    workload=phase["workload"],
                    actual_syncs_during_writes=[
                        dict(tid=t, thread="foreground" if t == tid else "worker", event=e) for t, e in before
                    ],
                    worker_payload_syncs_during_writes=sum(t != tid and e[1] == "payload" for t, e in before),
                    write_checkpoint_wait_ns=sum(
                        request["stage_exclusive_wall_ns"][stages.index("checkpoint_wait")]
                        for request in phase["application_requests"]
                        if request["kind"] == "write"
                    ),
                )
            )
        selected_ids = {
            (r["tid"], r["root_id"])
            for r in records
            if (r["state"], r["variant"], r["rep"]) == (mount["state"], mount["variant"], mount["rep"])
        }
        for item in mount["slow_roots"]:
            root = item["root"]
            if root[0] not in ("fsync", "fsyncdir", "workspace_close") or root[3] - root[2] < 10**9:
                continue
            if (item["tid"], root[1]) in selected_ids:
                continue
            syncs = [e for e in item["events"] if e[0] in ("fsync", "fdatasync")]
            typed = {
                file: sum(e[4] - e[3] for e in syncs if e[1] == file)
                for file in ("payload", "wal", "database", "fence", "other", "unknown")
            }
            outside.append(
                dict(
                    state=mount["state"],
                    variant=mount["variant"],
                    rep=mount["rep"],
                    tid=item["tid"],
                    root_id=root[1],
                    operation=root[0],
                    wall_ns=root[3] - root[2],
                    caller_cpu_ns=root[4],
                    actual_sync_ns=typed,
                    exclusive_wall_ns={name: root[10][i][2] for i, name in enumerate(stages)},
                    actual_sync_events=syncs,
                    waits=item["waits"],
                )
            )
    assert len(records) == len(phases) == 32
    groups = []
    for state in ("fuse", "aged_fuse"):
        for variant in ("baseline", "updated"):
            for workload in ("stream_256", "write_64"):
                selected = [
                    r for r in records if (r["state"], r["variant"], r["workload"]) == (state, variant, workload)
                ]
                assert len(selected) == 4
                groups.append(
                    dict(
                        state=state,
                        variant=variant,
                        workload=workload,
                        fsyncs=4,
                        roots_at_least_1s=sum(r["wall_ns"] >= 10**9 for r in selected),
                        median_wall_ns=statistics.median(r["wall_ns"] for r in selected),
                        maximum_wall_ns=max(r["wall_ns"] for r in selected),
                        median_actual_sync_ns={
                            file: statistics.median(r["actual_sync_ns"][file] for r in selected)
                            for file in selected[0]["actual_sync_ns"]
                        },
                        slow_root_categories=dict(
                            collections.Counter(r["largest_component"] for r in selected if r["wall_ns"] >= 10**9)
                        ),
                    )
                )
    inputs = {
        str(p): digest(p)
        for p in [
            Path(__file__),
            TRACE / "analyze.py",
            out / "analysis.json",
            out / "analysis-command.json",
            out / "plan.json",
            out / "input-check.json",
            out / "wrapper-proof.json",
            TRACE / "run.py",
        ]
    }
    for mount in analysis["mounts"]:
        inputs[mount["trace_file"]] = mount["trace_sha256"]
    save(
        out / "findings.json",
        dict(
            status="PASS",
            started_epoch=started,
            ended_epoch=time.time(),
            inputs=inputs,
            mounts=16,
            total_roots=sum(m["validation"]["roots"] for m in analysis["mounts"]),
            total_events=sum(m["validation"]["events"] for m in analysis["mounts"]),
            groups=groups,
            final_fsyncs=records,
            slow_sync_roots_outside_large_phases=outside,
            phases=phases,
            examples={
                workload: max(
                    (
                        r
                        for r in records
                        if r["state"] == "aged_fuse" and r["variant"] == "updated" and r["workload"] == workload
                    ),
                    key=lambda r: r["wall_ns"],
                )
                for workload in ("stream_256", "write_64")
            },
            interpretation="Instrumented native final-fsync attribution only. Original 14 plain performance failures remain. Syscall durations do not identify a kernel/device cause.",
        ),
    )
    for group in groups:
        print(
            group["state"],
            group["variant"],
            group["workload"],
            "median/max seconds",
            group["median_wall_ns"] / 1e9,
            group["maximum_wall_ns"] / 1e9,
            "slow",
            group["slow_root_categories"],
        )


if __name__ == "__main__":
    main()
