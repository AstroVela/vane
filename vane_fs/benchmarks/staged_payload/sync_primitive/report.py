# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Summarize both completed windows, WAL growth and every slow application call."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    out = parser.parse_args().artifacts.resolve()
    inputs = {}

    def read(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    findings = read(out / "findings.json")
    summary = read(out / "plain/summary.json")
    raw = read(out / "plain/ext4-plain/results.json")
    assert findings["status"] == summary["status"] == raw["status"] == "PASS"
    report = dict(status="PASS", inputs=inputs, native=[], plain=[], tails=[], host={})
    groups = {(g["state"], g["variant"]): g for g in findings["groups"]}
    for state in ("cold", "warm", "gc"):
        a, b = groups[state, "fsync"], groups[state, "fdatasync"]
        growth = {}
        for variant in ("fsync", "fdatasync"):
            growth[variant] = []
            for row in findings["runs"]:
                if row["state"] == state and row["variant"] == variant:
                    wal = sorted(
                        [s for s in row["measured_syncs"] if s["kind"] == "wal" and s["request"] == "fsync"],
                        key=lambda s: s["begin_ns"],
                    )
                    assert len(wal) == 128 and all(s["operation"] == variant for s in wal)
                    growth[variant].append(
                        dict(
                            rep=row["rep"],
                            distinct_sizes=len({s["file_size"] for s in wal}),
                            first_bytes=wal[0]["file_size"],
                            last_bytes=wal[-1]["file_size"],
                        )
                    )
        pairs = [r for r in findings["pairs"] if r["state"] == state]
        report["native"].append(
            dict(
                state=state,
                fsync_files_per_second=a["operations_per_second"]["median"],
                fdatasync_files_per_second=b["operations_per_second"]["median"],
                median_ratio=b["operations_per_second"]["median"] / a["operations_per_second"]["median"],
                aggregate_ratio=b["aggregate_operations_per_second"] / a["aggregate_operations_per_second"],
                paired_wins=sum(r["fdatasync_rate_ratio"] > 1 for r in pairs),
                median_128_full_wal_sync_ms={
                    "fsync": a["foreground_wal_sync_ns"]["median"] / 1e6,
                    "fdatasync": b["foreground_wal_sync_ns"]["median"] / 1e6,
                },
                wal_growth=growth,
            )
        )
    for state, data in summary["plain"]["ext4"].items():
        for name, value in data["workloads"].items():
            report["plain"].append(
                dict(
                    state=state,
                    workload=name,
                    median_rates={v: value[v]["rate"]["median"] for v in ("baseline", "updated")},
                    aggregate_rates={v: value[v]["aggregate_rate"] for v in ("baseline", "updated")},
                    median_ratio=value["median_rate_ratio"],
                    aggregate_ratio=value["aggregate_rate_ratio"],
                )
            )
        for mount in raw[state]:
            for row in mount["samples"]:
                calls = []
                for field in ("write_latencies", "operation_latencies"):
                    calls += [
                        dict(field=field, index=i, seconds=value)
                        for i, value in enumerate(row.get(field, []))
                        if value >= 1
                    ]
                calls += [
                    dict(field=field, seconds=row[field])
                    for field in ("fsync_seconds", "close_seconds", "final_directory_fsync_seconds")
                    if row.get(field, 0) >= 1
                ]
                if calls or row["seconds"] >= 4:
                    report["tails"].append(
                        dict(
                            state=state,
                            variant=mount["variant"],
                            rep=mount["rep"],
                            workload=row["workload"],
                            phase_seconds=row["seconds"],
                            calls=calls,
                        )
                    )
    for name, path in (("native", out / "host.jsonl"), ("plain", out / "plain/ext4-plain/host.jsonl")):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ticks = [b - a for a, b in zip(rows[0]["cpu_ticks"], rows[-1]["cpu_ticks"], strict=True)]
        total = sum(ticks[:8])  # Exclude guest fields already counted under user/nice.
        report["host"][name] = dict(
            samples=len(rows),
            seconds=rows[-1]["epoch"] - rows[0]["epoch"],
            whole_host_busy_percent=100 * (total - ticks[3] - ticks[4]) / total,
            iowait_percent=100 * ticks[4] / total,
            load_min=min(r["load"][0] for r in rows),
            load_max=max(r["load"][0] for r in rows),
            disks={
                disk: {
                    key: rows[-1]["diskstats"][disk][index] - rows[0]["diskstats"][disk][index]
                    for key, index in (
                        ("written_sectors", 6),
                        ("write_ms", 7),
                        ("busy_ms", 9),
                        ("flushes", 15),
                        ("flush_ms", 16),
                    )
                }
                for disk in ("sda", "sda2")
            },
        )
    report.update(
        kernel=platform.release(),
        logical_cpus=os.cpu_count(),
        adoption_gate_pass=summary["pass_all"],
        failed_gates=[r for r in summary["checks"] if not r["pass_gate"]],
        large_writes=summary["large_writes"],
    )
    inputs[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS complete windows;", len(report["tails"]), "retained slow phases; adoption", report["adoption_gate_pass"]
    )


if __name__ == "__main__":
    main()
