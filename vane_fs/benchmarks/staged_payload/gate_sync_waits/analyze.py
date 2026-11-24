# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Join native stages, actual sync calls and conservative kernel points to application intervals."""

import bisect
import collections
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from match import rejection
from prepare import digest, save, validate


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def union(intervals):
    total = end = 0
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def belongs(root, app, pid):
    if root[8] != pid or not app["begin_ns"] <= root[2] < app["end_ns"]:
        return False
    if app["kind"] == "small_file":
        return True
    if root[0] != app["kind"]:
        return False
    return app["kind"] != "write" or app["offset"] <= root[5] < root[5] + root[6] <= app["offset"] + app["bytes"]


def main():
    root, started = Path(sys.argv[1]).resolve(), time.time()
    out = root / "native"
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    assert raw["status"] == "PASS" and raw["ended_epoch"] < started
    assert json.loads((out / "input-check.json").read_text())["status"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    cases, calls, mounts, gates = [], [], [], []
    accounting = collections.Counter()
    observer_cpu = observer_wall = accepted_points = rejected_points = 0
    symbols = {}

    def stack_names(trace, offsets):
        binary = trace["native_image"]
        if binary not in symbols:
            text = subprocess.check_output(["nm", "-anC", binary], text=True)
            variant = "updated" if "/updated/" in binary else "baseline"
            (root / (variant + "-symbols.txt")).write_text(text)
            rows = []
            for line in text.splitlines():
                match = re.fullmatch(r"([0-9a-f]+) [tTwW] (.+)", line)
                if match:
                    rows.append((int(match[1], 16), match[2]))
            rows.sort()
            symbols[binary] = ([row[0] for row in rows], rows)
        starts, rows = symbols[binary]
        names = []
        for offset in offsets:
            index = bisect.bisect_right(starts, offset - 1) - 1
            assert index >= 0
            address, name = rows[index]
            names.append(dict(offset=offset, symbol=name, displacement=offset - 1 - address))
        return names

    for state in ("fuse", "aged_fuse"):
        for mount in raw[state]:
            label = dict(state=state, variant=mount["variant"], rep=mount["rep"])
            trace_path = Path(mount["waits_file"])
            trace = json.loads(trace_path.read_text())
            stats = validate(trace)
            accounting.update(
                {k: stats[k] for k in ("roots", "events", "range_calls", "sleep_calls", "slow_sql_calls")}
            )
            stages = trace["stages"]
            roots = {(t["tid"], r[1]): r for t in trace["threads"] for r in t["roots"]}
            events = [(t["tid"], i, e) for t in trace["threads"] for i, e in enumerate(t["events"])]
            own = collections.defaultdict(list)
            statements = {t["tid"]: t["statements"] for t in trace["threads"]}
            for tid, index, event in events:
                own[tid, event[2]].append((index, event))
            samples = [json.loads(line) for line in Path(mount["cpu_samples_file"]).read_text().splitlines()]
            final = samples[-1]
            assert final["fatal"] is None and final["mapping_closed"] and final["mapping_read_only"]
            observer_cpu += final["total_sampler_cpu_ns"]
            observer_wall += final["observation_wall_ns"]
            by_call = collections.defaultdict(list)
            for sample_index, sample in enumerate(samples):
                assert not sample["threads"] and not sample["frequency_khz"]
                for point_index, point in enumerate(sample["targeted_waits"]):
                    m = point["marker_before"]
                    by_call[m["tid"], m["kind"], m["begin_ns"]].append((sample_index, point_index, point))
            matched_point_count = 0
            for tid, index, event in events:
                if event[0] in ("begin_restart_wait", "begin_restart_hold"):
                    gates.append(label | dict(tid=tid, event_index=index, event=event))
                if event[0] not in ("fsync", "fdatasync", "sync_file_range", "nanosleep"):
                    continue
                assert event[6:8] in ([0, 0], [-1, 4]), "Retain interrupted calls without treating them as successful"
                accepted, rejected, waits = [], collections.Counter(), collections.Counter()
                for si, pi, point in by_call.pop((tid, event[0], event[3]), []):
                    reason = rejection(point, event, tid)
                    if reason:
                        rejected[reason] += 1
                        rejected_points += 1
                    else:
                        accepted.append([si, pi])
                        waits.update([point["wait"]["wchan"].get("value", "unavailable")])
                        accepted_points += 1
                    matched_point_count += 1
                calls.append(
                    label
                    | dict(
                        tid=tid,
                        event_index=index,
                        event=event,
                        stack_names=stack_names(trace, event[18]),
                        sql=statements[tid][event[14]] if event[14] >= 0 else None,
                        root_kind=roots[tid, event[2]][0] if event[2] >= 0 else None,
                        accepted_points=accepted,
                        rejected_points=dict(rejected),
                        wait_counts=dict(waits),
                    )
                )
            assert not by_call, "A native marker has no recorded call"
            app_count = 0
            for phase in mount["samples"]:
                for app in phase.get("application_intervals", phase.get("diagnostic_intervals", [])):
                    selected = [(key, r) for key, r in roots.items() if belongs(r, app, phase["application_pid"])]
                    assert selected
                    if app["kind"] == "small_file":
                        assert sum(r[0] == "create" for _, r in selected) == 1
                        assert sum(r[0] == "write" and r[6] == 4096 for _, r in selected) == 1
                        assert sum(r[0] == "fsync" for _, r in selected) == 1
                    elif app["kind"] == "fsync" or app.get("bytes") == 4096:
                        assert len(selected) == 1
                    begin, end = app["begin_ns"], app["end_ns"]
                    owned_events = [(key[0], index, e) for key, _ in selected for index, e in own[key]]
                    syncs = [(tid, index, e) for tid, index, e in owned_events if e[0] in ("fsync", "fdatasync")]
                    covered = union((max(begin, r[2]), min(end, r[3])) for _, r in selected)
                    assert 0 <= covered <= end - begin
                    cases.append(
                        label
                        | dict(
                            workload=phase["workload"],
                            interval=app,
                            wall_ns=end - begin,
                            handler_ids=[list(key) for key, _ in selected],
                            handler_cpu_ns=sum(r[4] for _, r in selected),
                            outside_handler_ns=end - begin - covered,
                            exclusive_wall_ns={s: sum(r[10][i][2] for _, r in selected) for i, s in enumerate(stages)},
                            exclusive_cpu_ns={s: sum(r[10][i][4] for _, r in selected) for i, s in enumerate(stages)},
                            own_sync_ns_by_file={
                                f: sum(e[4] - e[3] for _, _, e in syncs if e[1] == f)
                                for f in ("payload", "wal", "database", "fence", "other")
                            },
                            own_events=[[tid, index, e] for tid, index, e in owned_events],
                            concurrent_checkpoint_events=[
                                [tid, index, e]
                                for tid, index, e in events
                                if e[0] == "checkpoint"
                                and overlap(e[3:5], (begin, end))
                                and (tid, e[2]) not in {key for key, _ in selected}
                            ],
                        )
                    )
                    app_count += 1
            assert app_count == 739
            if mount["variant"] == "baseline":
                assert all(
                    r[10][stages.index(s)][0] == 0
                    for r in roots.values()
                    for s in ("begin_restart_wait", "begin_restart_hold")
                )
            else:
                assert sum(r[10][stages.index("begin_restart_wait")][0] for r in roots.values()) > 0
            mounts.append(
                label
                | dict(
                    stats=stats,
                    application_intervals=app_count,
                    trace_file=str(trace_path),
                    observer_points=matched_point_count,
                    observer_cpu_ns=final["total_sampler_cpu_ns"],
                    observer_wall_ns=final["observation_wall_ns"],
                )
            )
    assert len(mounts) == 24 and len(cases) == 24 * 739
    save(root / "requests.json", cases)
    save(root / "calls.json", calls)
    save(root / "gate-events.json", gates)
    long_cases = sorted(
        [(i, c) for i, c in enumerate(cases) if c["wall_ns"] >= 1000000000], key=lambda p: p[1]["wall_ns"], reverse=True
    )
    groups = []
    for state in ("fuse", "aged_fuse"):
        for variant in ("baseline", "updated"):
            for workload in ("stream_256", "write_64", "random_write_4k", "small_file_fsync"):
                selected = [
                    (i, c)
                    for i, c in enumerate(cases)
                    if (c["state"], c["variant"], c["workload"]) == (state, variant, workload)
                ]
                for kind in sorted({c["interval"]["kind"] for _, c in selected}):
                    rows = [(i, c) for i, c in selected if c["interval"]["kind"] == kind]
                    groups.append(
                        dict(
                            state=state,
                            variant=variant,
                            workload=workload,
                            kind=kind,
                            count=len(rows),
                            median_wall_ns=statistics.median(c["wall_ns"] for _, c in rows),
                            maximum_wall_ns=max(c["wall_ns"] for _, c in rows),
                            cases_ge20ms=[i for i, c in rows if c["wall_ns"] >= 20000000],
                            total_exclusive_wall_ns={
                                s: sum(c["exclusive_wall_ns"][s] for _, c in rows) for s in stages
                            },
                            total_sync_ns_by_file={
                                f: sum(c["own_sync_ns_by_file"][f] for _, c in rows)
                                for f in ("payload", "wal", "database", "fence", "other")
                            },
                        )
                    )
    inputs = plan["inputs"] | {
        str(p): digest(p)
        for p in (
            Path(__file__).resolve(),
            out / "plan.json",
            out / "ext4-diagnostic/results.json",
            out / "measurement-command.json",
            root / "requests.json",
            root / "calls.json",
            root / "gate-events.json",
        )
    }
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(
        root / "findings.json",
        dict(
            status="PASS",
            started_epoch=started,
            ended_epoch=time.time(),
            inputs=inputs,
            decision="DIAGNOSTIC_ONLY_NO_ADOPTION",
            mounts=mounts,
            sequential_content_checks=48,
            accounting=dict(accounting),
            application_intervals=len(cases),
            long_case_indices=[i for i, _ in long_cases],
            groups=groups,
            observer=dict(
                cpu_ns=observer_cpu,
                wall_ns=observer_wall,
                percent_of_one_core=100 * observer_cpu / observer_wall,
                accepted_points=accepted_points,
                rejected_points=rejected_points,
                thread_frequency_census=False,
                native_recorder_cost_additional=True,
            ),
            ordinary_failed_checks_retained=9,
        ),
    )
    print(
        "PASS",
        len(cases),
        "application joins; observer CPU",
        100 * observer_cpu / observer_wall,
        "% of one core; long cases",
        len(long_cases),
        flush=True,
    )
    for index, case in long_cases:
        print(
            index,
            case["state"],
            case["variant"],
            case["rep"],
            case["workload"],
            case["interval"]["kind"],
            case["wall_ns"] / 1e9,
            case["own_sync_ns_by_file"],
            flush=True,
        )


if __name__ == "__main__":
    main()
