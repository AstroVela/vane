# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile application writes with disjoint handler costs and actual wait intervals."""

import argparse
import collections
import importlib.util
import json
import time
from pathlib import Path

from prepare import digest, save, validate


def union(intervals):
    end, total = -1, 0
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def overlap(first, second):
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    plan = json.loads((out / "plan.json").read_text())
    assert json.loads((out / "input-check.json").read_text())["status"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    result = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert result["status"] == "PASS" and result["monitor_stopped"]
    started = time.time()
    assert result["ended_epoch"] <= started
    helper = here.parent / "request_costs/analyze_cpu.py"
    spec = importlib.util.spec_from_file_location("cpu_analysis", helper)
    cpu_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu_analysis)
    report = dict(
        status="RUNNING",
        started_epoch=started,
        inputs={
            str(p): digest(p)
            for p in [
                Path(__file__),
                helper,
                out / "plan.json",
                out / "input-check.json",
                out / "ext4-diagnostic/results.json",
            ]
        },
        interpretation="Instrumented diagnosis only; original plain performance gate failures remain",
        mounts=[],
    )
    for state in ("fuse", "aged_fuse"):
        for mount in result[state]:
            trace = json.loads(Path(mount["waits_file"]).read_text())
            stats = validate(trace)
            stages = trace["stages"]
            roots = [(t["tid"], r) for t in trace["threads"] for r in t["roots"]]
            events = [(t["tid"], e) for t in trace["threads"] for e in t["events"]]
            main_tid = next(tid for tid, r in roots if r[0] == "write")
            cpu = [json.loads(line) for line in Path(mount["cpu_samples_file"]).read_text().splitlines()]
            # Reuse the unchanged interval estimator with monotonic sample endpoints.
            cpu = [dict(sample, epoch_ns=sample["end_monotonic_ns"]) for sample in cpu]
            row = dict(
                state=state,
                variant=mount["variant"],
                rep=mount["rep"],
                main_tid=main_tid,
                trace_file=mount["waits_file"],
                trace_sha256=digest(Path(mount["waits_file"])),
                cpu_file=mount["cpu_samples_file"],
                cpu_sha256=digest(Path(mount["cpu_samples_file"])),
                validation=stats,
                phases=[],
                slow_roots=[],
                slow_syscalls=[],
            )
            for phase in mount["samples"]:
                if "application_intervals" not in phase:
                    continue
                app = phase["application_intervals"]
                begin, end = app[0]["begin_ns"], app[-1]["end_ns"]
                caller = phase["application_pid"]
                selected = [
                    (tid, r)
                    for tid, r in roots
                    if tid == main_tid
                    and r[0] in ("write", "fsync")
                    and r[8] == caller
                    and overlap((r[2], r[3]), (begin, end))
                ]
                inodes = {r[7] for _, r in selected}
                assert len(inodes) == 1
                counters = [[sum(r[10][i][j] for _, r in selected) for j in range(5)] for i in range(len(stages))]
                requests = []
                for interval in app:
                    b, e = interval["begin_ns"], interval["end_ns"]
                    handlers = [
                        (tid, r) for tid, r in selected if r[0] == interval["kind"] and overlap((r[2], r[3]), (b, e))
                    ]
                    if interval["kind"] == "write":
                        handlers = [
                            (tid, r)
                            for tid, r in handlers
                            if interval["offset"] <= r[5] < interval["offset"] + interval["bytes"]
                        ]
                    assert handlers, (state, mount["variant"], mount["rep"], interval)
                    if interval["kind"] == "write":
                        ordered = sorted((r[5], r[6]) for _, r in handlers)
                        offset = interval["offset"]
                        for pos, size in ordered:
                            assert pos == offset
                            offset += size
                        assert offset == interval["offset"] + interval["bytes"]
                    else:
                        assert len(handlers) == 1
                    covered = union((max(b, r[2]), min(e, r[3])) for _, r in handlers)
                    request = dict(
                        interval,
                        handlers=len(handlers),
                        handler_ids=[[tid, r[1]] for tid, r in handlers],
                        handler_wall_ns=sum(r[3] - r[2] for _, r in handlers),
                        handler_cpu_ns=sum(r[4] for _, r in handlers),
                        handler_covered_ns=covered,
                        outside_handler_ns=e - b - covered,
                        stage_exclusive_wall_ns=[sum(r[10][i][2] for _, r in handlers) for i in range(len(stages))],
                        stage_exclusive_cpu_ns=[sum(r[10][i][4] for _, r in handlers) for i in range(len(stages))],
                    )
                    assert request["outside_handler_ns"] >= 0
                    requests.append(request)
                writes = [r for _, r in selected if r[0] == "write"]
                assert sum(r[6] for r in writes) == phase["validation"]["bytes"]
                sizes = collections.Counter(r[6] for r in writes)
                syncs = [
                    (tid, e)
                    for tid, e in events
                    if e[0] in ("fsync", "fdatasync") and overlap((e[3], e[4]), (begin, end))
                ]
                by_file = collections.defaultdict(list)
                for tid, event in syncs:
                    by_file[("foreground" if tid == main_tid else "worker") + "/" + event[1]].append(event)
                row["phases"].append(
                    dict(
                        workload=phase["workload"],
                        begin_ns=begin,
                        end_ns=end,
                        wall_ns=end - begin,
                        application_write_bytes=phase["application_write_bytes"],
                        fuse_write_requests=len(writes),
                        fuse_request_sizes=dict(sizes),
                        fuse_inodes=sorted(inodes),
                        stage_counters=counters,
                        caller_vm_steps=sum(r[9] for _, r in selected),
                        application_requests=requests,
                        scheduler=cpu_analysis.metrics(cpu, main_tid, begin, end),
                        actual_syncs={
                            key: dict(
                                calls=len(values),
                                summed_call_ns=sum(e[4] - e[3] for e in values),
                                maximum_call_ns=max(e[4] - e[3] for e in values),
                            )
                            for key, values in by_file.items()
                        },
                    )
                )
            for tid, root in roots:
                if root[3] - root[2] < 100000000:
                    continue
                own = [e for t, e in events if t == tid and e[2] == root[1]]
                waits = []
                for event in own:
                    if event[0] not in (
                        "database_wait",
                        "checkpoint_wait",
                        "publication_wait",
                        "lease_wait",
                        "payload_guard",
                        "admission",
                    ):
                        continue
                    concurrent_syncs = [
                        (t, e)
                        for t, e in events
                        if t != tid and e[0] in ("fsync", "fdatasync") and overlap((e[3], e[4]), (event[3], event[4]))
                    ]
                    waits.append(
                        dict(
                            event=event,
                            concurrent_other_thread_syncs=[[t, e] for t, e in concurrent_syncs],
                            concurrent_sync_covered_ns=union(
                                (max(event[3], e[3]), min(event[4], e[4])) for _, e in concurrent_syncs
                            ),
                        )
                    )
                row["slow_roots"].append(
                    dict(
                        tid=tid,
                        root=root,
                        events=own,
                        waits=waits,
                        scheduler=cpu_analysis.metrics(cpu, tid, root[2], root[3]),
                    )
                )
            row["slow_syscalls"] = [
                [tid, e]
                for tid, e in events
                if e[0] in ("pwrite", "pwrite64", "pread", "pread64", "fsync", "fdatasync", "flock")
                and e[4] - e[3] >= 100000000
            ]
            report["mounts"].append(row)
            print(
                state,
                mount["variant"],
                mount["rep"],
                "roots",
                stats["roots"],
                "slow roots",
                len(row["slow_roots"]),
                "slow syscalls",
                len(row["slow_syscalls"]),
                flush=True,
            )
    report["stages"] = stages
    report["counter_columns"] = trace["counter_columns"]
    report["root_columns"] = trace["root_columns"]
    report["event_columns"] = trace["event_columns"]
    report["status"] = "PASS"
    report["ended_epoch"] = time.time()
    save(out / "analysis.json", report)


if __name__ == "__main__":
    main()
