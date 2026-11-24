# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile first-OPEN ordering, file growth and complete native costs."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def distribution(values):
    values = sorted(values)
    assert values
    return dict(count=len(values), median=statistics.median(values), total=sum(values), maximum=max(values))


def verify(row, trace):
    path = Path(__file__).resolve().parent.parent / "range_writeback_waits/prepare.py"
    spec = importlib.util.spec_from_file_location("fence_trace_validation", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    accounting = validator.validate(trace)
    assert trace["event_columns"][14:] == ["size_before", "size_after", "allocated_before", "allocated_after"]
    assert all(len(e) == 18 for t in trace["threads"] for e in t["events"])
    writes = [
        (t["tid"], r)
        for t in trace["threads"]
        for r in t["roots"]
        if r[0] == "write" and row["writes_begin_ns"] <= r[2] < row["writes_end_ns"]
    ]
    writes.sort(key=lambda value: value[1][2])
    assert len(writes) == 9 and len({tid for tid, _ in writes}) == 1
    assert [r[6] for _, r in writes] == [1048528, *[1048576] * 7, 48]
    assert [r[5] for _, r in writes] == [0, *[1048528 + 1048576 * i for i in range(8)]]
    tid, first = writes[0]
    all_events = [(t["tid"], e) for t in trace["threads"] for e in t["events"]]
    events = [e for t, e in all_events if (t, e[2]) == (tid, first[1])]
    wal = [e for e in events if e[0] == "fdatasync" and e[1] == "wal"]
    fence = [e for e in events if e[0] == "fdatasync" and e[1] == "fence"]
    assert len(wal) == len(fence) == 1
    fence_writes = [e for e in events if e[0] == "pwrite" and e[1] == "fence"]
    assert len(fence_writes) == 1 and fence_writes[0][9:11] == [4096, 4096]
    assert fence_writes[0][6] == 4096
    assert wal[0][4] <= fence_writes[0][3] < fence_writes[0][4] <= fence[0][3]
    payload = [e for e in events if e[0] == "pwrite" and e[1] == "payload"]
    assert payload and min(e[3] for e in payload) >= fence[0][4]
    assert fence_writes[0][14] == row["fence_before"] and fence_writes[0][15] == 8192
    assert fence[0][14:16] == [8192, 8192]
    prefix = [e for e in events if e[0] == "checkpoint" and e[4] <= wal[0][3] and e[6] == 0]
    assert prefix and prefix[-1][12] >= 0
    assert all(e[6:8] == [0, 0] for _, e in all_events if e[0] in ("fsync", "fdatasync", "sync_file_range"))
    initial = [
        e
        for t, e in all_events
        if t == tid and e[0] == "pwrite" and e[1] == "fence" and row["begin_ns"] <= e[3] < row["writes_begin_ns"]
    ]
    if row["state"] == "fresh":
        assert [e[9] for e in initial] == ([0] if row["variant"] == "original" else [4096, 0])
        assert initial[-1][15] == row["fence_before"]
    else:
        assert not initial
    assert not [e for _, e in all_events if e[0] == "sync_file_range"]
    seconds = dict(
        initialization=(row["writes_begin_ns"] - row["begin_ns"]) / 1e9,
        first_request=(first[3] - first[2]) / 1e9,
        first_8m=(row["writes_end_ns"] - row["writes_begin_ns"]) / 1e9,
        final_sync=(row["sync_end_ns"] - row["writes_end_ns"]) / 1e9,
        close=(row["end_ns"] - row["sync_end_ns"]) / 1e9,
        total=(row["end_ns"] - row["begin_ns"]) / 1e9,
        preparation=(row["begin_ns"] - row["preparation_begin_ns"]) / 1e9,
        lifecycle=(row["end_ns"] - row["preparation_begin_ns"]) / 1e9,
        open_wal_sync=(wal[0][4] - wal[0][3]) / 1e9,
        open_fence_sync=(fence[0][4] - fence[0][3]) / 1e9,
    )
    assert abs(seconds["total"] - sum(seconds[k] for k in ("initialization", "first_8m", "final_sync", "close"))) < 1e-8
    return dict(
        accounting=accounting,
        seconds=seconds,
        first_request_cpu_ns=first[4],
        first_request_steps=first[9],
        prefix_frames=prefix[-1][12],
        prefix_copied=prefix[-1][13],
        wal_size=wal[0][14],
        wal_sync=wal[0],
        fence_sync=fence[0],
        fence_write=fence_writes[0],
        initial_fence_writes=initial,
        sync_tails=[[t, e] for t, e in all_events if e[0] in ("fsync", "fdatasync") and e[4] - e[3] >= 100000000],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    out = parser.parse_args().artifacts.resolve()
    started = time.time()
    raw = json.loads((out / "results.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    assert raw["status"] == "PASS" and len(raw["runs"]) == 48 and raw["monitor_stopped"]
    assert raw["ended_epoch"] < started
    assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    rows = []
    inputs = dict(plan["inputs"])
    inputs.update({str(p): digest(p) for p in (out / "results.json", out / "plan.json", Path(__file__))})
    for row in raw["runs"]:
        path = Path(row["trace"])
        inputs[str(path)] = digest(path)
        rows.append(
            dict(
                label=row["label"],
                state=row["state"],
                variant=row["variant"],
                rep=row["rep"],
                **verify(row, json.loads(path.read_text())),
            )
        )
    groups = []
    for state, reps in plan["repetitions"].items():
        arms = {arm: [r for r in rows if (r["state"], r["variant"]) == (state, arm)] for arm in plan["variants"]}
        assert all(len(values) == reps for values in arms.values())
        metrics = {}
        for metric in rows[0]["seconds"]:
            d = {arm: distribution([r["seconds"][metric] for r in values]) for arm, values in arms.items()}
            metrics[metric] = dict(
                distributions=d,
                median_ratio=d["original"]["median"] / d["materialized"]["median"],
                aggregate_ratio=d["original"]["total"] / d["materialized"]["total"],
                paired_wins=sum(
                    next(c for c in arms["materialized"] if c["rep"] == r["rep"])["seconds"][metric]
                    < r["seconds"][metric]
                    for r in arms["original"]
                ),
            )
        groups.append(dict(state=state, metrics=metrics))
    fresh = next(g for g in groups if g["state"] == "fresh")["metrics"]
    screen = (
        fresh["first_8m"]["aggregate_ratio"] >= 1.05
        and fresh["first_8m"]["paired_wins"] >= 9
        and fresh["total"]["aggregate_ratio"] >= 0.98
        and all(g["metrics"]["total"]["aggregate_ratio"] >= 0.98 for g in groups)
    )
    save(
        out / "findings.json",
        dict(
            status="PASS",
            started_epoch=started,
            ended_epoch=time.time(),
            inputs=inputs,
            runs=rows,
            groups=groups,
            native_screen_passed=screen,
            ordinary_fuse_gates_not_replaced=True,
            interpretation="Only fence slot materialization changes. Total includes initialization, all first-8MiB writes, mandatory FULL and close. Reused and post-GC fences already have both slots. No native result clears the original FUSE gates or proves a kernel/device cause.",
        ),
    )
    for group in groups:
        print(group["state"], {k: group["metrics"][k] for k in ("first_8m", "total", "open_fence_sync")})
    print("Native screen", screen)


if __name__ == "__main__":
    main()
