# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile FULL barriers, checkpoint pause/resumption and complete native costs."""

import argparse
import collections
import importlib.util
import json
import statistics
import time
from pathlib import Path


def digest(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def distribution(values):
    values = sorted(values)
    assert values
    pos = (len(values) - 1) * 0.99
    lo = int(pos)
    return dict(
        count=len(values),
        median=statistics.median(values),
        total=sum(values),
        maximum=values[-1],
        p99=values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo),
    )


def validate_trace(trace):
    path = Path(__file__).resolve().parent.parent / "range_writeback_waits/prepare.py"
    spec = importlib.util.spec_from_file_location("pause_trace_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(trace)


def verify(row, trace):
    stats = validate_trace(trace)
    begin, end = row["begin_ns"], row["foreground_end_ns"]
    roots = {(thread["tid"], root[1]): root for thread in trace["threads"] for root in thread["roots"]}
    events = [(thread["tid"], event) for thread in trace["threads"] for event in thread["events"]]
    syncs = [(tid, e) for tid, e in events if e[0] in ("fsync", "fdatasync", "sync_file_range")]
    assert all(e[6:8] == [0, 0] for _, e in syncs)
    foreground = [(key, root) for key, root in roots.items() if begin <= root[2] < end and root[0] == "fsync"]
    assert len(foreground) == 129
    foreground_ids = {key for key, _ in foreground}
    wal = [(tid, e) for tid, e in syncs if (tid, e[2]) in foreground_ids and e[1] == "wal"]
    assert all(e[0] == "fsync" for _, e in wal)
    assert all(any((tid, e[2]) == key for tid, e in wal) for key in foreground_ids)
    active = [
        (key, root)
        for key, root in roots.items()
        if root[0] == "checkpoint_worker" and root[2] < end and begin < root[3]
    ]
    if row["arm"] == "paused":
        assert not active
        resumed = [
            root
            for root in roots.values()
            if root[0] == "checkpoint_worker" and row["resume_begin_ns"] <= root[2] < row["resume_end_ns"]
        ]
        assert resumed and row["wal_after"][0] > row["wal_after"][1]
    else:
        assert active
    assert not [e for _, e in syncs if e[0] == "sync_file_range" and e[3] < end and begin < e[4]]
    assert len(row["random_operations"]) == 512 and len(row["small_operations"]) == 128
    assert row["wal_before"][0] == row["wal_before"][1]
    assert row["wal_resumed"][0] == row["wal_resumed"][1]
    assert row["wal_highwater_bytes"] < 64 * 1024**2
    seconds = dict(
        foreground=(end - begin) / 1e9,
        random=(row["random_end_ns"] - row["random_begin_ns"]) / 1e9,
        random_final_sync=(row["random_end_ns"] - row["random_sync_begin_ns"]) / 1e9,
        small=(end - row["small_begin_ns"]) / 1e9,
        stop=(row["resume_begin_ns"] - end) / 1e9,
        resume=(row["resume_end_ns"] - row["resume_begin_ns"]) / 1e9,
        status=(row["close_begin_ns"] - row["resume_end_ns"]) / 1e9,
        close=(row["end_ns"] - row["close_begin_ns"]) / 1e9,
        total=(row["end_ns"] - begin) / 1e9,
        preparation=(begin - row["preparation_begin_ns"]) / 1e9,
        lifecycle=(row["end_ns"] - row["preparation_begin_ns"]) / 1e9,
    )
    assert abs(seconds["total"] - sum(seconds[k] for k in ("foreground", "stop", "resume", "status", "close"))) < 1e-8
    assert all(value >= 0 for value in seconds.values())
    selected = [(tid, e) for tid, e in syncs if e[3] < end and begin < e[4]]
    worker_syncs = [(tid, e) for tid, e in selected if roots.get((tid, e[2]), [""])[0] == "checkpoint_worker"]
    overlaps = [e for _, e in wal if any(e[3] < other[4] and other[3] < e[4] for _, other in worker_syncs)]
    return dict(
        trace_validation=stats,
        seconds=seconds,
        foreground_full_barriers=len(foreground),
        foreground_wal_sync_calls=len(wal),
        foreground_wal_sync_ns=distribution([e[4] - e[3] for _, e in wal]),
        worker_syncs=dict(collections.Counter(e[1] for _, e in worker_syncs)),
        foreground_syncs_overlapping_worker=len(overlaps),
        random_operation_ns=distribution([e - b for b, e in row["random_operations"]]),
        small_operation_ns=distribution([e - b for b, e in row["small_operations"]]),
        sync_tails=[[tid, e] for tid, e in syncs if begin <= e[3] < row["end_ns"] and e[4] - e[3] >= 100000000],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    out = parser.parse_args().artifacts.resolve()
    started = time.time()
    raw = json.loads((out / "results.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    assert raw["status"] == "PASS" and len(raw["runs"]) == 48 and raw["monitor_stopped"]
    assert raw["ended_epoch"] <= started
    assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    inputs = dict(plan["inputs"])
    inputs.update({str(p): digest(p) for p in [Path(__file__), out / "results.json", out / "plan.json"]})
    rows = []
    for run in raw["runs"]:
        trace_path = Path(run["trace"])
        values = verify(run, json.loads(trace_path.read_text()))
        inputs[str(trace_path)] = digest(trace_path)
        rows.append(
            dict(
                label=run["label"],
                state=run["state"],
                component=run["component"],
                arm=run["arm"],
                rep=run["rep"],
                wal_highwater_bytes=run["wal_highwater_bytes"],
                **values,
            )
        )
    groups = []
    pairs = []
    for state in plan["states"]:
        for component in plan["components"]:
            arms = {
                arm: [r for r in rows if (r["state"], r["component"], r["arm"]) == (state, component, arm)]
                for arm in ("active", "paused")
            }
            assert all(len(values) == 6 for values in arms.values())
            comparison = {}
            for metric in (
                "foreground",
                "random",
                "random_final_sync",
                "small",
                "stop",
                "resume",
                "close",
                "total",
                "preparation",
                "lifecycle",
            ):
                distributions = {
                    arm: distribution([r["seconds"][metric] for r in values]) for arm, values in arms.items()
                }
                comparison[metric] = dict(
                    distributions=distributions,
                    median_rate_ratio=distributions["active"]["median"] / distributions["paused"]["median"],
                    aggregate_rate_ratio=distributions["active"]["total"] / distributions["paused"]["total"],
                    paired_wins=sum(
                        next(r for r in arms["paused"] if r["rep"] == a["rep"])["seconds"][metric]
                        < a["seconds"][metric]
                        for a in arms["active"]
                    ),
                )
            groups.append(dict(state=state, component=component, comparison=comparison))
            for rep in range(6):
                values = {arm: next(r for r in arms[arm] if r["rep"] == rep) for arm in arms}
                pairs.append(
                    dict(
                        state=state,
                        component=component,
                        rep=rep,
                        foreground_ratio=values["active"]["seconds"]["foreground"]
                        / values["paused"]["seconds"]["foreground"],
                        total_ratio=values["active"]["seconds"]["total"] / values["paused"]["seconds"]["total"],
                    )
                )
    findings = dict(
        status="PASS",
        started_epoch=started,
        ended_epoch=time.time(),
        inputs=inputs,
        runs=rows,
        groups=groups,
        pairs=pairs,
        full_barriers=sum(r["foreground_full_barriers"] for r in rows),
        full_wal_sync_calls=sum(r["foreground_wal_sync_calls"] for r in rows),
        interpretation="Native checkpoint-pause intervention, not FUSE adoption throughput. Pausing also changes backfill batching/WAL reuse; overlap is not an isolated kernel cause. All resumed maintenance and original close are included in total.",
    )
    assert findings["full_barriers"] == 6192
    save(out / "findings.json", findings)
    for group in groups:
        print(group["state"], group["component"], {key: group["comparison"][key] for key in ("foreground", "total")})


if __name__ == "__main__":
    main()
