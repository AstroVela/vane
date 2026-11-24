# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Recompute paired read distributions and the predeclared adoption gates."""

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    plan = json.loads((out / "plan.json").read_text())
    report = {"status": "PASS", "plan": plan, "native": {}, "fuse": {}, "gates": {}}
    for kind in ("native", "fuse"):
        raw = json.loads((out / kind / "results.json").read_text())
        assert raw["status"] == "PASS"
        for row in raw["samples"]:
            for sample in row["metrics"]["samples"] if kind == "native" else row["samples"]:
                key = "/".join((row["state"], row["variant"], sample["workload"]))
                entry = report[kind].setdefault(key, {"samples": []})
                values = dict(
                    rep=row["rep"],
                    seconds=sample["seconds"],
                    bytes=sample["bytes"],
                    throughput_mib=sample["bytes"] / sample["seconds"] / 1024**2,
                )
                if kind == "native":
                    values["cpu_seconds"] = sample["cpu_seconds"]
                else:
                    latencies = json.loads((out / kind / sample["latencies_file"]).read_text())["latencies"]
                    values.update(
                        p50=percentile(latencies, 0.5),
                        p95=percentile(latencies, 0.95),
                        p99=percentile(latencies, 0.99),
                        maximum=max(latencies),
                        latencies_file=sample["latencies_file"],
                    )
                entry["samples"].append(values)
        for key, entry in report[kind].items():
            values = entry["samples"]
            entry["median_mib_per_second"] = statistics.median(v["throughput_mib"] for v in values)
            entry["aggregate_mib_per_second"] = (
                sum(v["bytes"] for v in values) / sum(v["seconds"] for v in values) / 1024**2
            )
            entry["seconds_range"] = [min(v["seconds"] for v in values), max(v["seconds"] for v in values)]
            if kind == "native":
                entry["median_cpu_seconds"] = statistics.median(v["cpu_seconds"] for v in values)
                entry["sum_cpu_seconds"] = sum(v["cpu_seconds"] for v in values)
            else:
                pooled = []
                for v in values:
                    pooled.extend(json.loads((out / kind / v["latencies_file"]).read_text())["latencies"])
                entry["pooled_p99_seconds"] = percentile(pooled, 0.99)
    limits = plan["adoption_gates"]
    for base in ("blob", "external"):
        checks = []
        for kind in ("native", "fuse"):
            for key, before in report[kind].items():
                state, variant, workload = key.split("/")
                if variant != base:
                    continue
                after = report[kind]["/".join((state, base + "_point", workload))]
                check = dict(kind=kind, state=state, workload=workload)
                if kind == "native":
                    check["cpu_median_ratio"] = after["median_cpu_seconds"] / before["median_cpu_seconds"]
                    check["cpu_sum_ratio"] = after["sum_cpu_seconds"] / before["sum_cpu_seconds"]
                    ceiling = (
                        1 - limits["native_old_snapshot_cpu_improvement_minimum"]
                        if workload == "old_snapshot"
                        else 1 + limits["point_read_native_cpu_regression_maximum"]
                    )
                    check["pass"] = max(check["cpu_median_ratio"], check["cpu_sum_ratio"]) <= ceiling
                else:
                    check["throughput_median_ratio"] = after["median_mib_per_second"] / before["median_mib_per_second"]
                    check["throughput_aggregate_ratio"] = (
                        after["aggregate_mib_per_second"] / before["aggregate_mib_per_second"]
                    )
                    check["p99_ratio"] = after["pooled_p99_seconds"] / before["pooled_p99_seconds"]
                    check["p99_increase_seconds"] = after["pooled_p99_seconds"] - before["pooled_p99_seconds"]
                    check["pass"] = min(
                        check["throughput_median_ratio"], check["throughput_aggregate_ratio"]
                    ) >= limits["fuse_throughput_median_and_aggregate_ratio_minimum"] and (
                        check["p99_ratio"] <= limits["fuse_p99_latency_ratio_maximum"]
                        or check["p99_increase_seconds"] <= limits["fuse_p99_absolute_increase_allowance_seconds"]
                    )
                # Pair identifiers are preserved; order balancing is not a significance test.
                pairs = {s["rep"]: s for s in after["samples"]}
                check["paired_throughput_wins"] = sum(
                    pairs[s["rep"]]["throughput_mib"] > s["throughput_mib"] for s in before["samples"]
                )
                checks.append(check)
        report["gates"][base] = dict(pass_all=all(c["pass"] for c in checks), checks=checks)
    target = args.output or out / "summary.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    for name, gate in report["gates"].items():
        print(
            name,
            "PASS" if gate["pass_all"] else "REJECT",
            ["/".join((c["kind"], c["state"], c["workload"])) for c in gate["checks"] if not c["pass"]],
        )


if __name__ == "__main__":
    main()
