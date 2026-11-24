# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep all samples and compare complete write times against the frozen gates."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def distribution(values):
    values = sorted(values)
    assert values

    def percentile(p):
        i = (len(values) - 1) * p
        lo = int(i)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (i - lo)

    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "sum": sum(values),
    }


def plain(data):
    result = {}
    for state in ("fuse", "aged_fuse"):
        rows = data[state]
        grouped = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in rows:
            for sample in row["samples"]:
                grouped[sample["workload"]][row["variant"]].append((row["rep"], sample))
        workloads = {}
        for workload, variants in grouped.items():
            summary = {}
            for variant, pairs in variants.items():
                samples = [s for _, s in pairs]
                key = "mib_per_second" if "mib_per_second" in samples[0] else "ops_per_second"
                row = {
                    "rate": distribution([s[key] for s in samples]),
                    "seconds": distribution([s["seconds"] for s in samples]),
                    "aggregate_rate": sum(s[key] * s["seconds"] for s in samples) / sum(s["seconds"] for s in samples),
                }
                for name in (
                    "requests_seconds",
                    "fsync_seconds",
                    "close_seconds",
                    "daemon_cpu_seconds",
                    "final_directory_fsync_seconds",
                ):
                    if name in samples[0]:
                        row[name] = distribution([s[name] for s in samples])
                latencies = [t for s in samples for t in s.get("write_latencies", s.get("operation_latencies", []))]
                if latencies:
                    row["latencies"] = distribution(latencies)
                    row["requests_over_1s"] = sum(t > 1 for t in latencies)
                summary[variant] = row
            b, c = summary["baseline"], summary["updated"]
            summary["median_rate_ratio"] = c["rate"]["median"] / b["rate"]["median"]
            summary["aggregate_rate_ratio"] = c["aggregate_rate"] / b["aggregate_rate"]
            baseline = dict(variants["baseline"])
            candidate = dict(variants["updated"])
            assert baseline.keys() == candidate.keys()
            summary["paired_complete_write_wins"] = sum(
                candidate[rep]["seconds"] < baseline[rep]["seconds"] for rep in baseline
            )
            tail = not (
                "latencies" in b
                and c["latencies"]["p99"] > b["latencies"]["p99"] * 1.2
                and c["latencies"]["p99"] - b["latencies"]["p99"] > 0.001
            )
            summary["nonregression_gate"] = (
                summary["median_rate_ratio"] >= 0.95 and summary["aggregate_rate_ratio"] >= 0.95 and tail
            )
            workloads[workload] = summary
        lifecycle = {
            v: distribution([row["lifecycle_seconds"] for row in rows if row["variant"] == v])
            for v in ("baseline", "updated")
        }
        lifecycle["nonregression_gate"] = lifecycle["updated"]["sum"] <= 1.05 * lifecycle["baseline"]["sum"]
        result[state] = {"workloads": workloads, "lifecycle": lifecycle}
    return result


def diagnostics(data):
    result = []
    for state in ("fuse", "aged_fuse"):
        for row in data[state]:
            trace = json.loads(Path(row["timeline_file"]).read_text())
            events = [(thread["tid"], event) for thread in trace["threads"] for event in thread["events"]]
            for sample in row["samples"]:
                groups = collections.defaultdict(list)
                overlap = collections.defaultdict(list)
                origin = trace["origin_epoch_ns"] / 1e9
                start, end = sample["started_epoch"] - origin, sample["ended_epoch"] - origin
                repeated = 0
                for tid, event in events:
                    begin, finish = event["begin_ns"] / 1e9, event["end_ns"] / 1e9
                    name = event["name"]
                    if name in ("fsync", "fdatasync"):
                        name += "_wal" if event["file"].endswith("-wal") else "_db"
                    if not name.startswith(("admission", "checkpoint", "fsync", "fdatasync", "worker_yield")):
                        continue
                    if start <= begin <= end:
                        groups[name].append(finish - begin)
                        if name == "admission_mutex_wait":
                            repeated += sum(
                                other_tid != tid
                                and other["name"] == "checkpoint_0"
                                and event["begin_ns"] <= other["begin_ns"] < event["end_ns"]
                                for other_tid, other in events
                            )
                    seconds = min(finish, end) - max(begin, start)
                    if seconds > 0:
                        overlap[name].append(seconds)
                result.append(
                    {
                        "state": state,
                        "variant": row["variant"],
                        "rep": row["rep"],
                        "workload": sample["workload"],
                        "events": {name: distribution(values) for name, values in groups.items()},
                        "phase_overlap": {name: distribution(values) for name, values in overlap.items()},
                        "passive_starts_during_mutex_wait": repeated,
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    report = {"plain": {}, "diagnostic": {}, "controls": {}}
    for medium in ("tmpfs", "ext4"):
        data = json.loads((out / (medium + "-plain/results.json")).read_text())
        assert data["status"] == "PASS" and data["monitor_stopped"]
        report["plain"][medium] = plain(data)
    data = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert data["status"] == "PASS" and data["monitor_stopped"]
    report["diagnostic"]["ext4"] = diagnostics(data)
    for phase in ("before", "after"):
        data = json.loads((out / ("control-" + phase) / "results.json").read_text())
        times = [
            t
            for row in data["samples"]
            for group in row["metrics"]["groups"]
            for t in (group["first_sync"], group["remaining_sync"], group["db_sync"])
        ]
        report["controls"][phase] = {"fsync_seconds": distribution(times), "over_1s": sum(t > 1 for t in times)}
    nonregression = all(
        workload["nonregression_gate"]
        for medium in report["plain"].values()
        for state in medium.values()
        for workload in state["workloads"].values()
    )
    nonregression = nonregression and all(
        state["lifecycle"]["nonregression_gate"] for medium in report["plain"].values() for state in medium.values()
    )
    benefit = all(
        state["workloads"]["stream_256"]["aggregate_rate_ratio"] >= 1.05
        and state["workloads"]["stream_256"]["paired_complete_write_wins"] >= 5
        for state in report["plain"]["ext4"].values()
    )
    report["performance_gate"] = {"nonregression": nonregression, "benefit": benefit, "pass": nonregression and benefit}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
