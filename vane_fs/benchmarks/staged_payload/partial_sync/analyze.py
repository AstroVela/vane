# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Separate publication/lock intervals from per-file durable syscall waits."""

import argparse
import collections
import json
from pathlib import Path


def union_seconds(intervals):
    total = 0
    end = -1
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    report = dict(
        status="PASS",
        config=raw["config"],
        samples=[],
        slow_syscalls=[],
        scope="Instrumented diagnosis, not a performance adoption window; nested or cross-thread intervals must not be added; whole-device counters do not identify per-file physical writes",
    )
    for state in ("fuse", "aged_fuse"):
        for row in raw[state]:
            trace = json.loads(Path(row["timeline_file"]).read_text())
            origin = trace["origin_epoch_ns"]
            events = [dict(tid=t["tid"], **e) for t in trace["threads"] for e in t["events"]]
            hints = [e for e in events if e["name"] == "sync_file_range_write"]
            if hints:
                assert row["variant"] == "updated"
                assert all(
                    e["flags"] == 2 and e["range_bytes"] >= 16 * 1024 * 1024 and e["offset"] % 4096 == 0 for e in hints
                )
                report.setdefault("hints", []).append(
                    dict(
                        state=state,
                        rep=row["rep"],
                        calls=len(hints),
                        max_seconds=max((e["end_ns"] - e["begin_ns"]) / 1e9 for e in hints),
                        requested_bytes=sum(e["range_bytes"] for e in hints),
                        errors=[e["errno"] for e in hints if e["rc"] != 0],
                    )
                )
            waits = [json.loads(line) for line in Path(row["wait_samples_file"]).read_text().splitlines()]
            for sample in row["samples"]:
                start = int(sample["started_epoch"] * 1e9) - origin
                finish = int(sample["ended_epoch"] * 1e9) - origin
                groups = collections.defaultdict(list)
                for e in events:
                    begin, end = max(e["begin_ns"], start), min(e["end_ns"], finish)
                    if end <= begin:
                        continue
                    label = e["name"]
                    if label in ("fsync", "fdatasync"):
                        file = e["file"]
                        label += (
                            "_payload"
                            if file.endswith(".payload")
                            else "_wal"
                            if file.endswith("-wal")
                            else "_fence"
                            if file.endswith(".stage-fence")
                            else "_db"
                            if file.endswith(".sqlite")
                            else "_other"
                        )
                    groups[label].append((begin, end))
                report["samples"].append(
                    dict(
                        state=state,
                        variant=row["variant"],
                        rep=row["rep"],
                        workload=sample["workload"],
                        seconds=sample["seconds"],
                        observed_union_seconds={key: union_seconds(values) for key, values in groups.items()},
                    )
                )
            for e in events:
                if (
                    e["name"] not in ("fsync", "fdatasync", "sync_file_range_write")
                    or e["end_ns"] - e["begin_ns"] < 100_000_000
                ):
                    continue
                epoch = (origin + e["begin_ns"]) / 1e9
                workload = next(
                    (s["workload"] for s in row["samples"] if s["started_epoch"] <= epoch <= s["ended_epoch"]),
                    "outside_workload_timers",
                )
                channels = collections.Counter(
                    w.get("wchan", "error")
                    for w in waits
                    if w["tid"] == e["tid"] and origin + e["begin_ns"] <= w["epoch_ns"] <= origin + e["end_ns"]
                )
                before, after = e["disk_before"]["values"], e["disk_after"]["values"]
                report["slow_syscalls"].append(
                    dict(
                        state=state,
                        variant=row["variant"],
                        rep=row["rep"],
                        workload=workload,
                        tid=e["tid"],
                        name=e["name"],
                        file=Path(e["file"]).name,
                        seconds=(e["end_ns"] - e["begin_ns"]) / 1e9,
                        waits=dict(channels),
                        completed_device_write_sectors=after[6] - before[6],
                        completed_device_flushes=after[15] - before[15],
                        device_flush_milliseconds=after[16] - before[16],
                    )
                )
    (out / "diagnostic-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for e in report["slow_syscalls"]:
        if e["seconds"] >= 1 and e["variant"] == "updated":
            print(e["state"], e["rep"], e["workload"], e["name"], e["file"], round(e["seconds"], 3), e["waits"])


if __name__ == "__main__":
    main()
