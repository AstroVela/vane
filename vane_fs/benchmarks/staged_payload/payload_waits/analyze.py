# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Count actual syncs and match write waits to recorded mutex holders and their I/O."""

import argparse
import collections
import hashlib
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    started = time.time()
    proof = json.loads((out / "wrapper-proof.json").read_text())
    canonical = subprocess.check_output(
        ["git", "show", "HEAD:vane_fs/benchmarks/staged_payload/fuse_waits/run.py"],
        cwd=TRACE,
    )
    assert proof["original_sha256"] == digest(TRACE / "run.py") == hashlib.sha256(canonical).hexdigest()
    assert (
        proof["exact_reverse"]
        and (out / "wrapper-proof.json").stat().st_mtime < json.loads((out / "plan.json").read_text())["frozen_epoch"]
    )
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
    phases, links, slow_writes, finals = [], [], [], []
    total_roots = total_events = 0
    kinds = {"checkpoint_wait": "checkpoint_hold", "publication_wait": "publication_hold"}
    for mount in analysis["mounts"]:
        trace = json.loads(Path(mount["trace_file"]).read_text())
        stats = validate(trace)
        total_roots += stats["roots"]
        total_events += stats["events"]
        roots = {(t["tid"], r[1]): r for t in trace["threads"] for r in t["roots"]}
        events = [(t["tid"], i, e) for t in trace["threads"] for i, e in enumerate(t["events"])]
        syncs = [(tid, i, e) for tid, i, e in events if e[0] in ("fsync", "fdatasync")]
        holds = [(tid, i, e) for tid, i, e in events if e[0] in kinds.values()]
        by_root = collections.defaultdict(list)
        for tid, i, event in events:
            by_root[tid, event[2]].append((i, event))
        for phase in mount["phases"]:
            label = {key: mount[key] for key in ("state", "variant", "rep")}
            label["workload"] = phase["workload"]
            first, last = phase["application_requests"][0], phase["application_requests"][-1]
            assert first["kind"] == "write" and last["kind"] == "fsync"
            bounds = (first["begin_ns"], last["end_ns"])
            writing = (bounds[0], last["begin_ns"])
            requests = [r for r in phase["application_requests"] if r["kind"] == "write"]
            ids = {tuple(ident) for request in requests for ident in request["handler_ids"]}
            selected = [roots[ident] for ident in ids]
            counters = {
                stage: [sum(r[10][i][j] for r in selected) for j in range(5)] for i, stage in enumerate(trace["stages"])
            }
            phase_links = []
            for tid, root_id in sorted(ids):
                for index, wait in by_root[tid, root_id]:
                    if wait[0] not in kinds:
                        continue
                    matches = []
                    hold_intervals, sync_intervals = [], []
                    by_file = collections.defaultdict(list)
                    for other_tid, hold_index, held in holds:
                        if other_tid == tid or held[0] != kinds[wait[0]] or not overlap(wait[3:5], held[3:5]):
                            continue
                        hold_intervals.append((max(wait[3], held[3]), min(wait[4], held[4])))
                        owner_syncs = []
                        for sync_tid, sync_index, event in syncs:
                            if sync_tid != other_tid or event[2] != held[2] or not overlap(wait[3:5], event[3:5]):
                                continue
                            assert held[3] <= event[3] <= event[4] <= held[4]
                            interval = (max(wait[3], event[3]), min(wait[4], event[4]))
                            sync_intervals.append(interval)
                            by_file[event[1]].append(interval)
                            owner_syncs.append(dict(tid=sync_tid, event_index=sync_index, event=event))
                        matches.append(
                            dict(
                                tid=other_tid,
                                event_index=hold_index,
                                event=held,
                                owner_root=roots[other_tid, held[2]][0],
                                syncs=owner_syncs,
                            )
                        )
                    link = dict(
                        id=len(links),
                        **label,
                        tid=tid,
                        root_id=root_id,
                        wait_event_index=index,
                        wait_event=wait,
                        holder_intervals=matches,
                        wait_ns=wait[4] - wait[3],
                        held_covered_ns=union(hold_intervals),
                        holder_sync_covered_ns=union(sync_intervals),
                        holder_sync_covered_ns_by_file={file: union(values) for file, values in by_file.items()},
                    )
                    assert link["holder_sync_covered_ns"] <= link["held_covered_ns"] <= link["wait_ns"]
                    phase_links.append(link)
                    links.append(link)
            phase_syncs = {}
            for window, interval in [("complete", bounds), ("writes", writing)]:
                grouped = collections.defaultdict(list)
                for tid, index, event in syncs:
                    if interval[0] <= event[3] < interval[1]:
                        role = "foreground" if tid == mount["main_tid"] else "worker"
                        grouped[role + "/" + event[1]].append(event)
                phase_syncs[window] = {
                    key: dict(
                        calls=len(values),
                        summed_call_ns=sum(e[4] - e[3] for e in values),
                        maximum_call_ns=max(e[4] - e[3] for e in values),
                        all_succeeded=all(e[6] == 0 for e in values),
                    )
                    for key, values in grouped.items()
                }
            worker_holds = {
                kind: [
                    e
                    for tid, _, e in holds
                    if tid != mount["main_tid"] and e[0] == kind and bounds[0] <= e[3] < bounds[1]
                ]
                for kind in kinds.values()
            }
            stages = trace["stages"]
            record = dict(
                **label,
                begin_ns=bounds[0],
                end_ns=bounds[1],
                writing_end_ns=writing[1],
                application_writes=len(requests),
                native_writes=len(ids),
                write_handler_wall_ns=sum(r[3] - r[2] for r in selected),
                write_handler_cpu_ns=sum(r[4] for r in selected),
                caller_vm_steps=sum(r[9] for r in selected),
                write_stage_counters=counters,
                actual_syncs=phase_syncs,
                worker_holds={
                    key: dict(
                        calls=len(values),
                        summed_wall_ns=sum(e[4] - e[3] for e in values),
                        summed_cpu_ns=sum(e[5] for e in values),
                    )
                    for key, values in worker_holds.items()
                },
                write_waits={
                    kind: dict(
                        recorded_calls=sum(link["wait_event"][0] == kind for link in phase_links),
                        total_exclusive_ns=counters[kind][2],
                        recorded_ns=sum(link["wait_ns"] for link in phase_links if link["wait_event"][0] == kind),
                        holder_sync_covered_ns=sum(
                            link["holder_sync_covered_ns"] for link in phase_links if link["wait_event"][0] == kind
                        ),
                        holder_sync_covered_ns_by_file={
                            file: sum(
                                link["holder_sync_covered_ns_by_file"].get(file, 0)
                                for link in phase_links
                                if link["wait_event"][0] == kind
                            )
                            for file in ("payload", "wal", "database", "fence")
                        },
                    )
                    for kind in kinds
                },
            )
            assert all(v["recorded_ns"] <= v["total_exclusive_ns"] for v in record["write_waits"].values())
            phases.append(record)
            for request in requests:
                if request["end_ns"] - request["begin_ns"] < 100000000:
                    continue
                selected_links = [
                    link for link in phase_links if (link["tid"], link["root_id"]) in map(tuple, request["handler_ids"])
                ]
                slow_writes.append(
                    dict(
                        **label,
                        begin_ns=request["begin_ns"],
                        end_ns=request["end_ns"],
                        offset=request["offset"],
                        bytes=request["bytes"],
                        handler_ids=request["handler_ids"],
                        wall_ns=request["end_ns"] - request["begin_ns"],
                        handler_wall_ns=request["handler_wall_ns"],
                        handler_cpu_ns=request["handler_cpu_ns"],
                        outside_handler_ns=request["outside_handler_ns"],
                        exclusive_wall_ns=dict(zip(stages, request["stage_exclusive_wall_ns"])),
                        lock_wait_ns=sum(link["wait_ns"] for link in selected_links),
                        holder_sync_covered_ns=sum(link["holder_sync_covered_ns"] for link in selected_links),
                        link_ids=[link["id"] for link in selected_links],
                    )
                )
            assert len(last["handler_ids"]) == 1
            ident = tuple(last["handler_ids"][0])
            root = roots[ident]
            finals.append(
                dict(
                    **label,
                    tid=ident[0],
                    root_id=ident[1],
                    wall_ns=root[3] - root[2],
                    cpu_ns=root[4],
                    exclusive_wall_ns={stage: root[10][i][2] for i, stage in enumerate(stages)},
                    syncs=[event for _, event in by_root[ident] if event[0] in ("fsync", "fdatasync")],
                )
            )
    assert len(phases) == len(finals) == 32
    groups = []
    for state in ("fuse", "aged_fuse"):
        for variant in ("baseline", "updated"):
            for workload in ("stream_256", "write_64"):
                rows = [p for p in phases if (p["state"], p["variant"], p["workload"]) == (state, variant, workload)]
                assert len(rows) == 4
                groups.append(
                    dict(
                        state=state,
                        variant=variant,
                        workload=workload,
                        worker_payload_syncs_during_writes=[
                            p["actual_syncs"]["writes"].get("worker/payload", {}).get("calls", 0) for p in rows
                        ],
                        payload_syncs_complete=[
                            sum(v["calls"] for k, v in p["actual_syncs"]["complete"].items() if k.endswith("/payload"))
                            for p in rows
                        ],
                        median_write_handler_wall_ns=statistics.median(p["write_handler_wall_ns"] for p in rows),
                        median_write_handler_cpu_ns=statistics.median(p["write_handler_cpu_ns"] for p in rows),
                        median_write_wait_ns={
                            kind: statistics.median(p["write_waits"][kind]["total_exclusive_ns"] for p in rows)
                            for kind in kinds
                        },
                        median_held_sync_covered_ns={
                            kind: statistics.median(p["write_waits"][kind]["holder_sync_covered_ns"] for p in rows)
                            for kind in kinds
                        },
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
            TRACE / "run.py",
            out / "wrapper-proof.json",
        ]
    }
    inputs.update({m["trace_file"]: m["trace_sha256"] for m in analysis["mounts"]})
    save(out / "links.json", dict(status="PASS", inputs=inputs, links=links))
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
            phases=phases,
            groups=groups,
            final_fsyncs=finals,
            slow_application_writes=slow_writes,
            examples=sorted(slow_writes, key=lambda r: r["wall_ns"], reverse=True)[:4],
            original_plain_failed_checks=11,
            interpretation="Instrumented sync counts and exact wait/held-interval intersections; overlapping threads are not independent costs. No adoption throughput or kernel/device causal claim.",
        ),
    )
    for group in groups:
        print(group["state"], group["variant"], group["workload"], json.dumps(group))


if __name__ == "__main__":
    main()
