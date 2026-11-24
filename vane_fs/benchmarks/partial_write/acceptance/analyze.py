# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Summarize all predeclared samples without excluding slow runs."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def distribution(values):
    values = sorted(values)
    assert values

    def percentile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)

    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "sum": sum(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    report = {"native": {}, "fuse": {}, "diagnostic": {}, "controls": {}}
    data = json.loads((out / "native-results.json").read_text())
    for shift in (0, 1, 4095):
        row = {}
        cpus = {}
        for variant in ("baseline", "updated"):
            values = [r for r in data["plain"] if r["shift"] == shift and r["variant"] == variant]
            cpus[variant] = [r["metrics"][0]["cpu_seconds"] for r in values]
            row[variant] = {
                "cpu": distribution(cpus[variant]),
                "wall": distribution([r["metrics"][0]["seconds"] for r in values]),
            }
        row["cpu_ratio"] = row["updated"]["cpu"]["median"] / row["baseline"]["cpu"]["median"]
        row["paired_cpu_wins"] = sum(c < b for b, c in zip(cpus["baseline"], cpus["updated"]))
        row["gate"] = row["cpu_ratio"] <= (0.95 if shift else 1.05) and (not shift or row["paired_cpu_wins"] >= 6)
        report["native"][str(shift)] = row
    for path in sorted(out.glob("*-plain/results.json")):
        data = json.loads(path.read_text())
        states = {}
        for state in ("fuse", "aged_fuse"):
            groups = collections.defaultdict(lambda: collections.defaultdict(list))
            cycles = collections.defaultdict(list)
            for row in data[state]:
                cycles[row["variant"]].append(row["lifecycle_seconds"])
                for sample in row["samples"]:
                    groups[sample["workload"]][row["variant"]].append(sample)
            results = {}
            for workload, variants in groups.items():
                summary = {}
                for variant, samples in variants.items():
                    key = "mib_per_second" if "mib_per_second" in samples[0] else "ops_per_second"
                    metrics = {
                        "rate": distribution([s[key] for s in samples]),
                        "seconds": distribution([s["seconds"] for s in samples]),
                        "aggregate_rate": sum(s[key] * s["seconds"] for s in samples)
                        / sum(s["seconds"] for s in samples),
                    }
                    for name in (
                        "fsync_seconds",
                        "final_directory_fsync_seconds",
                        "requests_seconds",
                        "daemon_cpu_seconds",
                    ):
                        if name in samples[0]:
                            metrics[name] = distribution([s[name] for s in samples])
                    latencies = [t for s in samples for t in s.get("write_latencies", s.get("operation_latencies", []))]
                    if latencies:
                        metrics["latencies"] = distribution(latencies)
                        metrics["requests_over_50ms"] = sum(t > 0.05 for t in latencies)
                        metrics["requests_over_1s"] = sum(t > 1 for t in latencies)
                    summary[variant] = metrics
                if set(summary) == {"baseline", "updated"}:
                    b, c = summary["baseline"], summary["updated"]
                    summary["median_rate_ratio"] = c["rate"]["median"] / b["rate"]["median"]
                    summary["aggregate_rate_ratio"] = c["aggregate_rate"] / b["aggregate_rate"]
                    tail = not (
                        "latencies" in b
                        and c["latencies"]["p99"] > b["latencies"]["p99"] * 1.2
                        and c["latencies"]["p99"] - b["latencies"]["p99"] > 0.001
                    )
                    summary["gate"] = (
                        summary["median_rate_ratio"] >= 0.95 and summary["aggregate_rate_ratio"] >= 0.95 and tail
                    )
                results[workload] = summary
            results["lifecycle"] = {v: distribution(times) for v, times in cycles.items()}
            if set(cycles) == {"baseline", "updated"}:
                results["lifecycle"]["gate"] = sum(cycles["updated"]) <= 1.05 * sum(cycles["baseline"])
            states[state] = results
        report["fuse"][path.parent.name] = {"status": data["status"], "states": states}
    for path in out.glob("*-diagnostic/results.json"):
        data = json.loads(path.read_text())
        rows = []
        for state in ("fuse", "aged_fuse"):
            for row in data[state]:
                if "timeline_file" not in row:
                    continue
                trace = json.loads(Path(row["timeline_file"]).read_text())
                for sample in row["samples"]:
                    groups = collections.defaultdict(list)
                    writes = collections.Counter()
                    crossing = collections.Counter()
                    overlaps = collections.defaultdict(list)
                    for thread in trace["threads"]:
                        for event in thread["events"]:
                            epoch = (trace["origin_epoch_ns"] + event["begin_ns"]) / 1e9
                            end_epoch = (trace["origin_epoch_ns"] + event["end_ns"]) / 1e9
                            key = event["name"]
                            if key in ("fsync", "fdatasync"):
                                key += "_wal" if event["file"].endswith("-wal") else "_db"
                            if key.startswith("fuse_write_"):
                                if sample["started_epoch"] <= epoch <= sample["ended_epoch"]:
                                    writes[(int(key.split("_")[-1]), event["offset"] % 4096)] += 1
                                key = "fuse_write"
                            if key.startswith(("fsync", "fdatasync", "checkpoint", "admission", "fuse_write")):
                                overlap = min(end_epoch, sample["ended_epoch"]) - max(epoch, sample["started_epoch"])
                                if overlap > 0:
                                    overlaps[key].append(overlap)
                                    if epoch < sample["started_epoch"] or end_epoch > sample["ended_epoch"]:
                                        crossing[key] += 1
                                if sample["started_epoch"] <= epoch <= sample["ended_epoch"]:
                                    groups[key].append((event["end_ns"] - event["begin_ns"]) / 1e9)
                    rows.append(
                        {
                            "state": state,
                            "variant": row["variant"],
                            "rep": row["rep"],
                            "workload": sample["workload"],
                            "metrics": {k: distribution(v) for k, v in groups.items()},
                            "phase_overlap_seconds": {k: distribution(v) for k, v in overlaps.items()},
                            "crossing_events": dict(crossing),
                            "write_shapes": [
                                {"bytes": key[0], "offset_mod_4096": key[1], "count": count}
                                for key, count in sorted(writes.items())
                            ],
                        }
                    )
        report["diagnostic"][path.parent.name] = rows
    for path in out.glob("control-*/results.json"):
        data = json.loads(path.read_text())
        times = [
            s
            for row in data["samples"]
            for group in row["metrics"]["groups"]
            for s in (group["first_sync"], group["remaining_sync"], group["db_sync"])
        ]
        report["controls"][path.parent.name] = {
            "status": data["status"],
            "sync": distribution(times),
            "over_1s": sum(t > 1 for t in times),
        }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "native_gate": all(r["gate"] for r in report["native"].values()),
                "fuse": {
                    window: {
                        state: {k: v.get("gate") for k, v in results.items()}
                        for state, results in values["states"].items()
                    }
                    for window, values in report["fuse"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
