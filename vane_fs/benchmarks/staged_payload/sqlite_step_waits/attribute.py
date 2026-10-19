# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Separate proven busy-handler sleeps, other SQL waits and actual sync calls."""

import collections
import json
import sys
import time
from pathlib import Path

from match import NUMBERS, rejection
from prepare import digest, save, validate
from verify import decode, symbols

HERE = Path(__file__).resolve().parent


def distribution(values):
    if not values:
        return dict(count=0)
    values = sorted(values)
    position = (len(values) - 1) * 0.99
    lower = int(position)
    middle = len(values) // 2
    return dict(
        count=len(values),
        median=(values[middle] + values[~middle]) / 2,
        p99=values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower),
        max=values[-1],
        sum=sum(values),
    )


def main():
    out, started = Path(sys.argv[1]).resolve(), time.time()
    plan = json.loads((out / "plan.json").read_text())
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    base = json.loads((out / "findings.json").read_text())
    requests = json.loads((out / "requests.json").read_text())
    assert raw["status"] == base["status"] == "PASS" and raw["ended_epoch"] < started
    assert all(digest(Path(path)) == sha for path, sha in plan["inputs"].items())
    binaries, tables = {}, {}
    for variant in ("baseline", "updated"):
        binary = out / "diagnostic" / variant / "build/python-release/vane-fs-mount"
        text, table = symbols(binary)
        binaries[variant] = dict(path=str(binary), sha256=digest(binary), nm=text)
        tables[variant] = table
    save(out / "symbols.json", binaries)
    case_roots = {}
    for index, case in enumerate(requests):
        for tid, root in case["handler_ids"]:
            key = case["state"], case["variant"], case["rep"], tid, root
            assert key not in case_roots
            case_roots[key] = index
    calls, sql_calls, mounts = [], [], []
    for state in ("fuse", "aged_fuse"):
        for mount in raw[state]:
            variant, rep = mount["variant"], mount["rep"]
            label = dict(state=state, variant=variant, rep=rep)
            trace = json.loads(Path(mount["waits_file"]).read_text())
            stats = validate(trace)
            points = [
                (cycle, index, point)
                for cycle, row in enumerate(map(json.loads, Path(mount["cpu_samples_file"]).read_text().splitlines()))
                for index, point in enumerate(row["targeted_waits"])
            ]
            by_begin = collections.defaultdict(list)
            for cycle, index, point in points:
                marker = point["marker_before"]
                by_begin[marker["tid"], marker["kind"], marker["begin_ns"]].append((cycle, index, point))
            accepted_refs, rejected = set(), collections.Counter()
            for thread in trace["threads"]:
                tid = thread["tid"]
                roots = {root[1]: root for root in thread["roots"]}
                events = thread["events"]
                sleep_by_context = collections.defaultdict(list)
                for index, event in enumerate(events):
                    if event[0] not in NUMBERS:
                        continue
                    frames = decode(event[18], tables[variant])
                    busy = event[0] == "nanosleep" and any("sqliteDefaultBusyCallback" in name for name in frames)
                    vfs = event[0] == "nanosleep" and any(name == "unixSleep" for name in frames)
                    accepted, waits = [], collections.Counter()
                    for cycle, point_index, point in by_begin[tid, event[0], event[3]]:
                        reason = rejection(point, event, tid)
                        if reason:
                            rejected[reason] += 1
                        else:
                            accepted.append([cycle, point_index])
                            accepted_refs.add((cycle, point_index))
                            waits[point["wait"]["wchan"].get("value", "unavailable")] += 1
                    root = roots.get(event[2])
                    value = dict(
                        **label,
                        tid=tid,
                        event_index=index,
                        root_id=event[2],
                        root_name=root[0] if root else None,
                        case_index=case_roots.get((state, variant, rep, tid, event[2])),
                        kind=event[0],
                        file=event[1],
                        begin_ns=event[3],
                        end_ns=event[4],
                        wall_ns=event[4] - event[3],
                        cpu_ns=event[5],
                        result=event[6],
                        errno=event[7],
                        sql=thread["statements"][event[14]] if event[14] >= 0 else None,
                        sql_context=event[15],
                        sql_api=event[16],
                        frames=frames,
                        busy_handler_frame=busy,
                        unix_sleep_frame=vfs,
                        requested_sleep_ns=event[9] if event[0] == "nanosleep" else None,
                        observations=accepted,
                        wait_counts=dict(waits),
                    )
                    calls.append(value)
                    if event[0] == "nanosleep":
                        sleep_by_context[event[15]].append(value)
                for index, event in enumerate(events):
                    if event[0] not in ("sql_step", "sql_exec"):
                        continue
                    sleeps = sleep_by_context[event[15]]
                    assert all(event[3] <= call["begin_ns"] <= call["end_ns"] <= event[4] for call in sleeps)
                    root = roots[event[2]]
                    sql_calls.append(
                        dict(
                            **label,
                            tid=tid,
                            event_index=index,
                            root_id=event[2],
                            root_name=root[0],
                            case_index=case_roots.get((state, variant, rep, tid, event[2])),
                            api=event[0],
                            sql=thread["statements"][event[14]],
                            context=event[15],
                            result=event[6],
                            begin_ns=event[3],
                            end_ns=event[4],
                            wall_ns=event[4] - event[3],
                            cpu_ns=event[5],
                            vm_steps=event[17],
                            sleeps=len(sleeps),
                            busy_sleeps=sum(call["busy_handler_frame"] for call in sleeps),
                            vfs_sleeps=sum(call["unix_sleep_frame"] for call in sleeps),
                            sleep_ns=sum(call["wall_ns"] for call in sleeps),
                            requested_sleep_ns=sum(call["requested_sleep_ns"] for call in sleeps),
                            busy_sleep_ns=sum(call["wall_ns"] for call in sleeps if call["busy_handler_frame"]),
                            sleep_event_indices=[call["event_index"] for call in sleeps],
                        )
                    )
            samples = [json.loads(line) for line in Path(mount["cpu_samples_file"]).read_text().splitlines()]
            final = samples[-1]
            assert final["sampler_finished"] and final["fatal"] is None and final["marker"] and final["mapping_closed"]
            lifecycle = (
                mount["lifecycle_ended_epoch"] - mount["lifecycle_started_epoch"]
                if "lifecycle_ended_epoch" in mount
                else None
            )
            mounts.append(
                dict(
                    **label,
                    **stats,
                    targeted_points=len(points),
                    accepted_points=len(accepted_refs),
                    rejected_for_known_call=dict(rejected),
                    unassigned_points=len(points) - len(accepted_refs) - sum(rejected.values()),
                    sampler_cpu_ns=sum(row["sampler_cpu_ns"] for row in samples),
                    sampler_elapsed_ns=samples[-1]["end_monotonic_ns"] - samples[0]["begin_monotonic_ns"],
                    sampler_cycles=len(samples) - 1,
                    lifecycle_seconds=lifecycle,
                    stack_access=final["stack_access"],
                )
            )
    save(out / "native-calls.json", calls)
    save(out / "sql-calls.json", sql_calls)
    groups = []
    for state, variant, api, sql in sorted({(c["state"], c["variant"], c["api"], c["sql"]) for c in sql_calls}):
        selected = [
            c for c in sql_calls if (c["state"], c["variant"], c["api"], c["sql"]) == (state, variant, api, sql)
        ]
        groups.append(
            dict(
                state=state,
                variant=variant,
                api=api,
                sql=sql,
                wall_ns=distribution([c["wall_ns"] for c in selected]),
                cpu_ns=distribution([c["cpu_ns"] for c in selected]),
                sleep_calls=sum(c["sleeps"] for c in selected),
                busy_sleep_calls=sum(c["busy_sleeps"] for c in selected),
                sleep_ns=sum(c["sleep_ns"] for c in selected),
                busy_sleep_ns=sum(c["busy_sleep_ns"] for c in selected),
            )
        )
    focused = []
    for index, case in enumerate(requests):
        if case["workload"] not in ("random_write_4k", "small_file_fsync") or case["wall_ns"] < 20000000:
            continue
        own = [call for call in calls if call["case_index"] == index]
        sql = [call for call in sql_calls if call["case_index"] == index]
        focused.append(
            dict(
                case_index=index,
                **{
                    key: case[key]
                    for key in (
                        "state",
                        "variant",
                        "rep",
                        "workload",
                        "wall_ns",
                        "handler_cpu_ns",
                        "outside_handler_ns",
                        "exclusive_wall_ns",
                        "own_sync_ns_by_file",
                    )
                },
                own_calls=[
                    {
                        key: call[key]
                        for key in (
                            "tid",
                            "event_index",
                            "kind",
                            "file",
                            "wall_ns",
                            "cpu_ns",
                            "sql_context",
                            "busy_handler_frame",
                            "unix_sleep_frame",
                            "wait_counts",
                        )
                    }
                    for call in own
                ],
                slow_sql=sql,
            )
        )
    selected = lambda field: [
        dict(rank=index, **row)
        for index, row in enumerate(sorted(field, key=lambda c: c["wall_ns"], reverse=True)[:12])
    ]
    report = dict(
        status="PASS",
        started_epoch=started,
        ended_epoch=time.time(),
        original_plain_failed_checks=base["original_plain_failed_checks"],
        diagnostic_only=True,
        default_busy_handler_unchanged=True,
        native_policy_unchanged=True,
        mounts=mounts,
        native_calls=len(calls),
        slow_sql_calls=len(sql_calls),
        sql_groups=groups,
        longest_sql=selected(sql_calls),
        longest_native=selected(calls),
        targeted_cases=focused,
        sleep_totals=dict(
            calls=sum(c["kind"] == "nanosleep" for c in calls),
            busy_handler_calls=sum(c["busy_handler_frame"] for c in calls),
            vfs_sleep_calls=sum(c["unix_sleep_frame"] for c in calls),
        ),
        inputs=base["inputs"]
        | {
            str(path): digest(path)
            for path in [
                Path(__file__),
                HERE / "match.py",
                HERE / "verify.py",
                out / "native-calls.json",
                out / "sql-calls.json",
                out / "symbols.json",
            ]
        },
        attribution_limits="Exact syscall/FD/native-begin identity and containment are required. Stack frames prove observed call paths, not every internal reason. Sparse points cannot bound unsampled waits. Concurrent checkpoint calls alone do not prove causality.",
    )
    save(out / "diagnosis.json", report)
    print(
        "PASS",
        len(calls),
        "native calls;",
        len(sql_calls),
        "slow SQL calls;",
        len(focused),
        "targeted application tails",
    )
    print("Sleeps", report["sleep_totals"])
    print(
        "Observer CPU ms",
        sum(row["sampler_cpu_ns"] for row in mounts) / 1e6,
        "accepted points",
        sum(row["accepted_points"] for row in mounts),
    )
    for row in report["longest_sql"][:5]:
        print(
            row["state"],
            row["variant"],
            row["rep"],
            row["root_name"],
            row["sql"],
            row["wall_ns"],
            row["sleeps"],
            row["busy_sleeps"],
        )


if __name__ == "__main__":
    main()
