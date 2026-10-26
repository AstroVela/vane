# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute final FULL calls and reconcile timed reads with native stages."""

import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from prepare import TRACE, digest, save, validate


def overlap(first, second):
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def union(intervals):
    end = total = 0
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def main():
    out = Path(sys.argv[1]).resolve()
    started = time.time()
    argv = [sys.executable, "-u", str(TRACE / "analyze.py"), str(out)]
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
    assert plain["full_checks"] == 22 and len(plain["full_failed_checks"]) == 4
    spec = importlib.util.spec_from_file_location("read_cpu", TRACE.parent / "request_costs/analyze_cpu.py")
    cpu_helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu_helper)
    reads, finals, phases = [], [], []
    total_roots = total_events = 0
    for mount in analysis["mounts"]:
        original = next(
            row for row in raw[mount["state"]] if (row["variant"], row["rep"]) == (mount["variant"], mount["rep"])
        )
        trace = json.loads(Path(mount["trace_file"]).read_text())
        stats = validate(trace)
        total_roots += stats["roots"]
        total_events += stats["events"]
        stages = trace["stages"]
        roots = {(t["tid"], r[1]): r for t in trace["threads"] for r in t["roots"]}
        events = [(t["tid"], e) for t in trace["threads"] for e in t["events"]]
        syncs = [(tid, e) for tid, e in events if e[0] in ("fsync", "fdatasync")]
        assert all(e[6] == 0 for _, e in syncs)
        for root in roots.values():
            if root[0] == "prewriteback_worker":
                assert root[10][stages.index("publication_hold")][0] == 0
                assert root[10][stages.index("checkpoint_hold")][0] == 0
        label = {key: mount[key] for key in ("state", "variant", "rep")}
        for phase in mount["phases"]:
            app = phase["application_requests"][-1]
            assert app["kind"] == "fsync" and len(app["handler_ids"]) == 1
            tid, ident = app["handler_ids"][0]
            root = roots[tid, ident]
            own = [e for t, e in events if (t, e[2]) == (tid, ident)]
            own_syncs = [e for e in own if e[0] in ("fsync", "fdatasync")]
            concurrent = [(t, e) for t, e in syncs if t != tid and overlap(root[2:4], e[3:5])]
            finals.append(
                dict(
                    **label,
                    workload=phase["workload"],
                    root=root,
                    tid=tid,
                    application_wall_ns=app["end_ns"] - app["begin_ns"],
                    outside_handler_ns=app["outside_handler_ns"],
                    exclusive_wall_ns={stage: root[10][i][2] for i, stage in enumerate(stages)},
                    own_events=own,
                    own_syncs=own_syncs,
                    own_sync_ns_by_file={
                        file: sum(e[4] - e[3] for e in own_syncs if e[1] == file)
                        for file in ("payload", "wal", "database", "fence")
                    },
                    concurrent_syncs=[[t, e] for t, e in concurrent],
                    concurrent_sync_overlap_ns=union((max(root[2], e[3]), min(root[3], e[4])) for _, e in concurrent),
                )
            )
            phases.append(dict(**label, workload=phase["workload"], actual_syncs=phase["actual_syncs"]))
        sample = next(s for s in original["samples"] if s["workload"] == "read_64")
        begin, end = sample["application_begin_ns"], sample["application_end_ns"]
        selected = [
            (tid, root)
            for (tid, _), root in roots.items()
            if root[0] == "read" and root[8] == sample["application_pid"] and overlap(root[2:4], (begin, end))
        ]
        assert selected and all(begin <= r[2] <= r[3] <= end for _, r in selected)
        assert len({tid for tid, _ in selected}) == len({r[7] for _, r in selected}) == 1
        position = 0
        for _, root in sorted(selected, key=lambda item: item[1][5]):
            assert root[5] == position
            position += root[6]
        assert position == sample["bytes"] == 64 * 1024**2
        counters = [[sum(r[10][i][j] for _, r in selected) for j in range(5)] for i in range(len(stages))]
        wall = sum(r[3] - r[2] for _, r in selected)
        covered = union(r[2:4] for _, r in selected)
        assert covered == wall <= end - begin
        concurrent = [(tid, e) for tid, e in syncs if overlap(e[3:5], (begin, end))]
        cpu = [json.loads(line) for line in Path(mount["cpu_file"]).read_text().splitlines()]
        cpu = [dict(row, epoch_ns=row["end_monotonic_ns"]) for row in cpu]
        reads.append(
            dict(
                **label,
                begin_ns=begin,
                end_ns=end,
                application_wall_ns=end - begin,
                reported_seconds=sample["seconds"],
                native_requests=len(selected),
                native_request_bytes=position,
                native_wall_ns=wall,
                native_cpu_ns=sum(r[4] for _, r in selected),
                outside_handler_ns=end - begin - covered,
                caller_vm_steps=sum(r[9] for _, r in selected),
                stage_counters=dict(zip(stages, counters)),
                handler_ids=[[tid, r[1]] for tid, r in selected],
                concurrent_syncs=[[tid, e] for tid, e in concurrent],
                concurrent_worker_roots=[
                    [tid, r]
                    for (tid, _), r in roots.items()
                    if r[0] in ("prewriteback_worker", "checkpoint_worker") and overlap(r[2:4], (begin, end))
                ],
                scheduler=cpu_helper.metrics(cpu, selected[0][0], begin, end),
            )
        )
    assert len(reads) == 16 and len(finals) == len(phases) == 32
    groups = []
    for state in ("fuse", "aged_fuse"):
        for variant in ("baseline", "updated"):
            rows = [r for r in reads if (r["state"], r["variant"]) == (state, variant)]
            assert len(rows) == 4
            groups.append(
                dict(
                    state=state,
                    variant=variant,
                    medians={
                        key: statistics.median(r[key] for r in rows)
                        for key in (
                            "application_wall_ns",
                            "native_wall_ns",
                            "native_cpu_ns",
                            "outside_handler_ns",
                            "caller_vm_steps",
                        )
                    },
                    median_stage_counters={
                        stage: [statistics.median(r["stage_counters"][stage][j] for r in rows) for j in range(5)]
                        for stage in stages
                    },
                    read_concurrent_syncs=sum(len(r["concurrent_syncs"]) for r in rows),
                    read_concurrent_worker_roots=sum(len(r["concurrent_worker_roots"]) for r in rows),
                )
            )
    inputs = dict(analysis["inputs"])
    inputs.update(
        {
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path(build["parent"]) / "analysis.json",
                out / "analysis.json",
                out / "analysis-command.json",
                out / "read-workload-proof.json",
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
            phase_syncs=phases,
            final_fsyncs=finals,
            timed_reads=reads,
            read_groups=groups,
            longest_candidate_fsyncs=sorted(
                [r for r in finals if r["variant"] == "updated"],
                key=lambda r: r["root"][3] - r["root"][2],
                reverse=True,
            )[:3],
            original_plain_failed_checks=plain["full_failed_checks"],
            interpretation="Instrumented attribution only. Simultaneous thread intervals overlap; outside-handler time includes application, transport and scheduling. Syscall time does not identify a kernel/device cause.",
        ),
    )
    for group in groups:
        print(json.dumps(group))
    for r in sorted(finals, key=lambda r: r["root"][3] - r["root"][2], reverse=True)[:4]:
        print("long final", r["state"], r["variant"], r["rep"], r["workload"], r["own_sync_ns_by_file"])


if __name__ == "__main__":
    main()
