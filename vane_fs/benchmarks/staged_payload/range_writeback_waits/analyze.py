# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute large-write tails and subsequent random/small-file operations."""

import collections
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from prepare import TRACE, digest, save, validate


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def union(intervals):
    total = end = 0
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def distribution(values):
    values = sorted(values)
    assert values
    pos = (len(values) - 1) * 0.99
    low = int(pos)
    p99 = values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)
    return dict(count=len(values), median=statistics.median(values), p99=p99, max=values[-1], sum=sum(values))


def main():
    out = Path(sys.argv[1]).resolve()
    started = time.time()
    argv = [sys.executable, str(TRACE / "analyze.py"), str(out)]
    process = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    save(
        out / "analysis-command.json",
        dict(
            argv=argv,
            started_epoch=started,
            ended_epoch=time.time(),
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        ),
    )
    assert process.returncode == 0, process.stderr
    analysis = json.loads((out / "analysis.json").read_text())
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    build = json.loads((out / "build.json").read_text())
    plain = json.loads((Path(build["parent"]) / "analysis.json").read_text())
    assert plain["full_checks"] == 22 and len(plain["full_failed_checks"]) == 7
    cases, phases, workers = [], [], []
    total_roots = total_events = 0
    for mount in analysis["mounts"]:
        label = {key: mount[key] for key in ("state", "variant", "rep")}
        original = next(
            row for row in raw[mount["state"]] if (row["variant"], row["rep"]) == (mount["variant"], mount["rep"])
        )
        trace = json.loads(Path(mount["trace_file"]).read_text())
        stats = validate(trace)
        total_roots += stats["roots"]
        total_events += stats["events"]
        stages = trace["stages"]
        roots = {(thread["tid"], root[1]): root for thread in trace["threads"] for root in thread["roots"]}
        events = [(thread["tid"], event) for thread in trace["threads"] for event in thread["events"]]
        own_events = collections.defaultdict(list)
        for tid, event in events:
            own_events[tid, event[2]].append(event)
        syncs = [(tid, event) for tid, event in events if event[0] in ("fsync", "fdatasync", "sync_file_range")]
        assert all(event[6:8] == [0, 0] for _, event in syncs)
        for (tid, _), root in roots.items():
            if root[0] != "prewriteback_worker":
                continue
            assert root[10][stages.index("publication_hold")][0] == 0
            assert root[10][stages.index("checkpoint_hold")][0] == 0
            workers.append(dict(**label, tid=tid, root=root, events=own_events[tid, root[1]]))

        def case(workload, interval, selected):
            begin, end = interval["begin_ns"], interval["end_ns"]
            assert selected
            covered = union((max(begin, r[2]), min(end, r[3])) for _, r in selected)
            assert 0 <= covered <= end - begin
            own = [event for key, _ in selected for event in own_events[key]]
            own_syncs = [event for event in own if event[0] in ("fsync", "fdatasync")]
            foreground_tids = {key[0] for key, _ in selected}
            concurrent = [
                (tid, event) for tid, event in syncs if tid not in foreground_tids and overlap(event[3:5], (begin, end))
            ]
            row = dict(
                **label,
                workload=workload,
                interval=interval,
                wall_ns=end - begin,
                handler_ids=[list(key) for key, _ in selected],
                handler_kinds=dict(collections.Counter(r[0] for _, r in selected)),
                handler_wall_ns=sum(r[3] - r[2] for _, r in selected),
                handler_cpu_ns=sum(r[4] for _, r in selected),
                outside_handler_ns=end - begin - covered,
                caller_vm_steps=sum(r[9] for _, r in selected),
                exclusive_wall_ns={stage: sum(r[10][i][2] for _, r in selected) for i, stage in enumerate(stages)},
                exclusive_cpu_ns={stage: sum(r[10][i][4] for _, r in selected) for i, stage in enumerate(stages)},
                own_events=own,
                own_sync_ns_by_file={
                    file: sum(e[4] - e[3] for e in own_syncs if e[1] == file)
                    for file in ("payload", "wal", "database", "fence")
                },
                concurrent_syncs=[[tid, event] for tid, event in concurrent],
                concurrent_sync_overlap_ns=union((max(begin, e[3]), min(end, e[4])) for _, e in concurrent),
            )
            cases.append(row)
            return len(cases) - 1

        for phase in mount["phases"]:
            indices = []
            for interval in phase["application_requests"]:
                selected = [((tid, ident), roots[tid, ident]) for tid, ident in interval["handler_ids"]]
                indices.append(case(phase["workload"], interval, selected))
            phases.append(dict(**label, workload=phase["workload"], cases=indices))
        for phase in original["samples"]:
            if "diagnostic_intervals" not in phase:
                continue
            intervals = phase["diagnostic_intervals"]
            small = phase["workload"] == "small_file_fsync"
            assert len(intervals) == (128 if small else 513)
            indices = []
            for interval in intervals:
                begin, end = interval["begin_ns"], interval["end_ns"]
                selected = [
                    (key, r)
                    for key, r in roots.items()
                    if r[8] == phase["application_pid"]
                    and overlap(r[2:4], (begin, end))
                    and (small or r[0] == interval["kind"])
                ]
                if small:
                    assert sum(r[0] == "create" for _, r in selected) == 1
                    assert sum(r[0] == "write" and r[6] == 4096 for _, r in selected) == 1
                    assert sum(r[0] == "fsync" for _, r in selected) == 1
                elif interval["kind"] == "write":
                    assert len(selected) == 1
                    assert selected[0][1][5:7] == [interval["offset"], 4096]
                else:
                    assert len(selected) == 1
                indices.append(case(phase["workload"], interval, selected))
            phases.append(
                dict(
                    **label,
                    workload=phase["workload"],
                    cases=indices,
                    reported_seconds=phase["seconds"],
                    reported_rate=phase["mib_per_second"] if not small else phase["ops_per_second"],
                )
            )
    assert len(phases) == 64 and len(cases) == 16 * (32 + 1 + 64 + 1 + 513 + 128)
    save(out / "requests.json", cases)
    groups = []
    for state in ("fuse", "aged_fuse"):
        for variant in ("baseline", "updated"):
            for workload in ("stream_256", "write_64", "random_write_4k", "small_file_fsync"):
                values = [
                    (i, row)
                    for i, row in enumerate(cases)
                    if (row["state"], row["variant"], row["workload"]) == (state, variant, workload)
                ]
                kinds = sorted({row["interval"]["kind"] for _, row in values})
                for kind in kinds:
                    rows = [(i, row) for i, row in values if row["interval"]["kind"] == kind]
                    groups.append(
                        dict(
                            state=state,
                            variant=variant,
                            workload=workload,
                            kind=kind,
                            distributions={
                                key: distribution([row[key] for _, row in rows])
                                for key in (
                                    "wall_ns",
                                    "handler_wall_ns",
                                    "handler_cpu_ns",
                                    "outside_handler_ns",
                                    "caller_vm_steps",
                                    "concurrent_sync_overlap_ns",
                                )
                            },
                            total_exclusive_wall_ns={
                                stage: sum(row["exclusive_wall_ns"][stage] for _, row in rows) for stage in stages
                            },
                            total_own_sync_ns_by_file={
                                file: sum(row["own_sync_ns_by_file"][file] for _, row in rows)
                                for file in ("payload", "wal", "database", "fence")
                            },
                            operations_at_least_20ms=sum(row["wall_ns"] >= 20000000 for _, row in rows),
                            two_longest_case_indices=[
                                i for i, _ in sorted(rows, key=lambda pair: pair[1]["wall_ns"], reverse=True)[:2]
                            ],
                        )
                    )
    inputs = dict(analysis["inputs"])
    inputs.update(
        {
            str(p): digest(p)
            for p in [
                Path(__file__),
                out / "requests.json",
                out / "analysis.json",
                out / "analysis-command.json",
                out / "workload-proof.json",
                Path(build["parent"]) / "analysis.json",
            ]
        }
    )
    save(
        out / "findings.json",
        dict(
            status="PASS",
            started_epoch=started,
            ended_epoch=time.time(),
            inputs=inputs,
            mounts=16,
            total_roots=total_roots,
            total_events=total_events,
            stages=stages,
            requests=len(cases),
            request_file=str(out / "requests.json"),
            phases=phases,
            groups=groups,
            workers=workers,
            original_plain_failed_checks=plain["full_failed_checks"],
            interpretation="Instrumented attribution only; simultaneous threads overlap. Outside-handler intervals include application/transport/scheduling. Syscall timing does not identify a kernel/device cause.",
        ),
    )
    for row in groups:
        print(row["state"], row["variant"], row["workload"], row["kind"], row["distributions"]["wall_ns"])


if __name__ == "__main__":
    main()
