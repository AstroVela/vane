# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Join read-only observations to native calls by thread, bracket, syscall and FD."""

import bisect
import collections
import json
import statistics
import sys
import time
from pathlib import Path

from common import digest, save, validate

NUMBERS = dict(fsync=74, fdatasync=75, sync_file_range=277)
DEVICE_FIELDS = "reads reads_merged sectors_read read_ms writes writes_merged sectors_written write_ms inflight io_ms weighted_io_ms discards discards_merged sectors_discarded discard_ms flushes flush_ms".split()


def rejection(wait, event):
    if not event[3] <= wait["begin_ns"] <= wait["end_ns"] <= event[4]:
        return "cross_boundary"
    if not wait["stable"] or wait["syscall_before"] != wait["syscall_after"]:
        return "unstable_or_unavailable_syscall"
    call = wait["syscall_before"]
    if call.get("number") != NUMBERS[event[0]]:
        return "different_syscall"
    if call.get("fd") != event[8]:
        return "different_fd"
    if wait.get("file", {}).get("kind") != event[1]:
        return "unavailable_or_different_file"
    if event[0] == "sync_file_range" and [call.get(k) for k in ("offset", "bytes", "flags")] != event[9:12]:
        return "different_range"
    return None


def belongs(root, app, pid):
    if root[8] != pid or not app["begin_ns"] <= root[2] < app["end_ns"]:
        return False
    if app["kind"] == "small_file":
        return True
    if root[0] != app["kind"]:
        return False
    return app["kind"] != "write" or app["offset"] <= root[5] < root[5] + root[6] <= app["offset"] + app["bytes"]


def device_context(snapshots, begins, begin, end):
    lo, hi = bisect.bisect_left(begins, begin), bisect.bisect_right(begins, end)
    inside = [s for s in snapshots[lo:hi] if s["end_ns"] <= end]
    result = dict(interior_snapshots=len(inside), whole_device_only=True)
    if len(inside) >= 2:
        first, last = inside[0], inside[-1]
        delta = [b - a for a, b in zip(first["values"], last["values"])]
        result.update(
            first_begin_ns=first["begin_ns"],
            first_end_ns=first["end_ns"],
            last_begin_ns=last["begin_ns"],
            last_end_ns=last["end_ns"],
            observed_span_ns=last["end_ns"] - first["begin_ns"],
            delta=dict(zip(DEVICE_FIELDS, delta)),
            max_sampled_inflight=max(s["values"][8] for s in inside),
            counter_regression=any(v < 0 for i, v in enumerate(delta) if i != 8),
        )
    return result


def group(calls):
    result = []
    by = collections.defaultdict(list)
    for call in calls:
        by[(call["state"], call["variant"], call["kind"], call["file"])].append(call)
    for key, rows in sorted(by.items()):
        waits = collections.Counter()
        for row in rows:
            waits.update(row["wait_counts"])
        result.append(
            dict(zip(("state", "variant", "kind", "file"), key))
            | dict(
                calls=len(rows),
                observed_calls=sum(bool(r["matched_observations"]) for r in rows),
                calls_ge_20ms=sum(r["wall_ns"] >= 20000000 for r in rows),
                calls_ge_1s=sum(r["wall_ns"] >= 1000000000 for r in rows),
                wall_median_ns=statistics.median(r["wall_ns"] for r in rows),
                wall_max_ns=max(r["wall_ns"] for r in rows),
                summed_call_wall_ns=sum(r["wall_ns"] for r in rows),
                summed_call_cpu_ns=sum(r["cpu_ns"] for r in rows),
                wait_counts=dict(waits),
            )
        )
    return result


def main():
    out = Path(sys.argv[1]).resolve()
    started = time.time()
    plan = json.loads((out / "plan.json").read_text())
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert raw["status"] == "PASS" and raw["ended_epoch"] < started
    assert json.loads((out / "measurement-checks.json").read_text())["status"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    calls, mounts, cold, accounting, applications = [], [], [], collections.Counter(), 0
    for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
        for row in raw[key]:
            mount = f"{state}-{row['rep']}-{row['variant']}"
            trace = json.loads(Path(row["waits_file"]).read_text())
            stats = validate(trace)
            accounting.update({k: stats[k] for k in ("roots", "events", "range_calls")})
            samples = [json.loads(line) for line in Path(row["cpu_samples_file"]).read_text().splitlines()]
            assert samples[-1]["sampler_finished"] and samples[-1]["stop_requested"] and samples[-1]["fatal"] is None
            cycles = samples[:-1]
            by, devices, observations = collections.defaultdict(list), [], {}
            for index, sample in enumerate(cycles):
                if "values" in sample["device"]:
                    devices.append(sample["device"])
                for thread in sample["threads"]:
                    reference = (index, thread["tid"])
                    wait = thread["wait"]
                    by[thread["tid"]].append((wait["begin_ns"], wait, reference))
                    observations[reference] = wait
            devices.sort(key=lambda d: d["begin_ns"])
            device_begins = [d["begin_ns"] for d in devices]
            begins = {tid: [r[0] for r in rows] for tid, rows in by.items()}
            roots = {(thread["tid"], root[1]): root for thread in trace["threads"] for root in thread["roots"]}
            apps = [
                (sample["workload"], app, sample["application_pid"])
                for sample in row["samples"]
                for app in sample.get("application_intervals", sample.get("diagnostic_intervals", []))
            ]
            applications += len(apps)
            # Native callback identity prevents a previous callback's reply tail
            # being attributed to the next application's write interval.
            root_phases = {}
            for root_key, root in roots.items():
                phases = [f"{name}/{app['kind']}" for name, app, pid in apps if belongs(root, app, pid)]
                assert len(phases) <= 1, (mount, root_key, phases)
                root_phases[root_key] = phases[0] if phases else "outside_recorded_application_intervals"
            matched, rejects, mount_calls = set(), collections.Counter(), []
            for thread in trace["threads"]:
                tid = thread["tid"]
                for event_index, event in enumerate(thread["events"]):
                    if event[0] not in NUMBERS:
                        continue
                    assert event[6:8] == [0, 0]
                    lo, hi = (
                        bisect.bisect_left(begins.get(tid, []), event[3]),
                        bisect.bisect_right(begins.get(tid, []), event[4]),
                    )
                    accepted, waits, rejected = [], collections.Counter(), collections.Counter()
                    for _, wait, reference in by[tid][lo:hi]:
                        why = rejection(wait, event)
                        if why:
                            rejected[why] += 1
                        else:
                            accepted.append(list(reference))
                            matched.add(reference)
                            waits[wait["wchan"].get("value", "unavailable")] += 1
                    rejects.update(rejected)
                    root = roots.get((tid, event[2]))
                    call = dict(
                        mount=mount,
                        state=state,
                        variant=row["variant"],
                        rep=row["rep"],
                        tid=tid,
                        event_index=event_index,
                        root_id=event[2],
                        root_name=root[0] if root else None,
                        phase=root_phases.get((tid, event[2]), "unscoped_native_call"),
                        kind=event[0],
                        file=event[1],
                        fd=event[8],
                        begin_ns=event[3],
                        end_ns=event[4],
                        wall_ns=event[4] - event[3],
                        cpu_ns=event[5],
                        matched_observations=len(accepted),
                        observation_references=accepted,
                        wait_counts=dict(waits),
                        rejected_inside_candidates=dict(rejected),
                        device=device_context(devices, device_begins, event[3], event[4]),
                    )
                    mount_calls.append(call)
            calls.extend(mount_calls)
            for sample in row["samples"]:
                if sample["workload"] not in ("stream_256", "write_64"):
                    continue
                first = sample["application_intervals"][0]
                owned_roots = {k for k, r in roots.items() if belongs(r, first, sample["application_pid"])}
                selected = [r for k, r in roots.items() if k in owned_roots]
                assert sum(r[6] for r in selected) == first["bytes"]
                assert len(selected) == first["bytes"] // (1024 * 1024) + 1
                own_calls = [c for c in mount_calls if (c["tid"], c["root_id"]) in owned_roots]
                cold.append(
                    dict(
                        mount=mount,
                        workload=sample["workload"],
                        first_application=first,
                        application_ns=first["end_ns"] - first["begin_ns"],
                        roots=len(selected),
                        root_cpu_ns=sum(r[4] for r in selected),
                        root_wall_ns=sum(r[3] - r[2] for r in selected),
                        stages={name: sum(r[10][i][2] for r in selected) for i, name in enumerate(trace["stages"])},
                        own_calls=[
                            {
                                k: v
                                for k, v in c.items()
                                if k not in ("observation_references", "device", "rejected_inside_candidates")
                            }
                            for c in own_calls
                        ],
                        concurrent_range_calls=[
                            dict(
                                tid=c["tid"],
                                event_index=c["event_index"],
                                overlap_ns=min(c["end_ns"], first["end_ns"]) - max(c["begin_ns"], first["begin_ns"]),
                            )
                            for c in mount_calls
                            if c["kind"] == "sync_file_range"
                            and max(c["begin_ns"], first["begin_ns"]) < min(c["end_ns"], first["end_ns"])
                        ],
                    )
                )
            periods = [b["begin_monotonic_ns"] - a["begin_monotonic_ns"] for a, b in zip(cycles, cycles[1:])]
            durations = [s["end_monotonic_ns"] - s["begin_monotonic_ns"] for s in cycles]
            stable_io = {
                k
                for k, v in observations.items()
                if v["stable"] and v["syscall_before"].get("number") in NUMBERS.values()
            }
            mounts.append(
                dict(
                    mount=mount,
                    trace=stats,
                    cycles=len(cycles),
                    thread_observations=len(observations),
                    stable_sync_observations=len(stable_io),
                    matched_sync_observations=len(matched),
                    unmatched_stable_sync_observations=len(stable_io - matched),
                    rejected_inside_candidates=dict(rejects),
                    stack_access=samples[-1]["stack_access"],
                    device_errors=sum("error" in s["device"] for s in cycles),
                    period_median_ns=statistics.median(periods),
                    period_max_ns=max(periods),
                    cycle_median_ns=statistics.median(durations),
                    cycle_max_ns=max(durations),
                    sampler_cpu_ns=sum(s["sampler_cpu_ns"] for s in cycles),
                    lifecycle_seconds=row["lifecycle_seconds"],
                    device=device_context(devices, device_begins, devices[0]["begin_ns"], devices[-1]["end_ns"]),
                )
            )
            print(mount, "PASS", len(mount_calls), "sync calls", len(matched), "matched observations", flush=True)
    assert len(mounts) == 16 and applications == 11824 and len(cold) == 32
    save(out / "sync-calls.json", calls)
    inputs = plan["inputs"] | {
        str(p): digest(p)
        for p in [
            *Path(__file__).parent.glob("*.py"),
            out / "sync-calls.json",
            *sorted((out / "ext4-diagnostic").glob("*.json*")),
        ]
        if p.is_file()
    }
    findings = dict(
        status="PASS",
        started_epoch=started,
        ended_epoch=time.time(),
        inputs=inputs,
        mount_count=len(mounts),
        application_intervals=applications,
        accounting=dict(accounting),
        sync_calls=len(calls),
        successful_sync_calls=len(calls),
        groups=group(calls),
        mounts=mounts,
        cold_requests=cold,
        tails_ge_1s=[c for c in calls if c["wall_ns"] >= 1000000000],
        original_fuse_failed_checks=7,
        production_unchanged=True,
        interpretation="Point counts are not elapsed time. Summed call wall times can overlap across threads. Device deltas cover strictly interior whole-device snapshots only and cannot identify a request owner or per-call flush duration. Inaccessible full stacks prevent identifying the exact QoS policy or journal transaction owner.",
    )
    # Detailed references for every call are retained separately, keeping the
    # review summary small while preserving all unsampled and rejected calls.
    for tail in findings["tails_ge_1s"]:
        tail.pop("observation_references", None)
    save(out / "findings.json", findings)
    print("PASS complete trace accounting and conservative wait attribution", flush=True)


if __name__ == "__main__":
    main()
