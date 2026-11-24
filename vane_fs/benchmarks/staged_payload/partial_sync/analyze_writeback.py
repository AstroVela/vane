# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate all samples against the frozen payload-writeback adoption gates."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("summaries", here.parents[1] / "checkpoint_priority/analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = json.loads((out / "plan.json").read_text())
    report = dict(status="PASS", plan=plan, plain={}, checks=[], large_writes={})
    for medium in plan["media"]:
        complete = out / (medium + "-plain/results.json")
        if complete.exists():
            raw = json.loads(complete.read_text())
        else:
            raw = json.loads((out / "windows.json").read_text())[medium]
            for state, label in (("fuse", "fresh"), ("aged_fuse", "aged")):
                raw[state] = [
                    r
                    for variant in ("baseline", "updated")
                    for r in json.loads((out / (medium + "-" + label + "-" + variant + ".json")).read_text())
                ]
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        result = module.plain(raw)
        report["plain"][medium] = result
        for state, state_data in result.items():
            for name, values in state_data["workloads"].items():
                passed = (
                    min(values["median_rate_ratio"], values["aggregate_rate_ratio"])
                    >= plan["regression_gates"]["all_workloads_median_and_aggregate_ratio_minimum"]
                )
                if "latencies" in values["baseline"]:
                    before = values["baseline"]["latencies"]["p99"]
                    after = values["updated"]["latencies"]["p99"]
                    passed = passed and (
                        after <= before * plan["regression_gates"]["pooled_p99_ratio_maximum"]
                        or after - before <= plan["regression_gates"]["pooled_p99_absolute_increase_allowance_seconds"]
                    )
                report["checks"].append(
                    dict(
                        medium=medium,
                        state=state,
                        workload=name,
                        median_ratio=values["median_rate_ratio"],
                        aggregate_ratio=values["aggregate_rate_ratio"],
                        pass_gate=passed,
                    )
                )
            rows = raw[state]
            durations = {
                v: {
                    r["rep"]: sum(s["seconds"] for s in r["samples"] if s["workload"] in ("stream_256", "write_64"))
                    for r in rows
                    if r["variant"] == v
                }
                for v in ("baseline", "updated")
            }
            rates = {v: len(durations[v]) * 320 / sum(durations[v].values()) for v in durations}
            wins = sum(durations["updated"][rep] < total for rep, total in durations["baseline"].items())
            ratio = rates["updated"] / rates["baseline"]
            report["large_writes"][medium + "/" + state] = dict(
                aggregate_mib_per_second=rates, aggregate_ratio=ratio, paired_wins=wins
            )
            if medium == "ext4":
                report["checks"].append(
                    dict(
                        medium=medium,
                        state=state,
                        workload="combined_large_writes",
                        aggregate_ratio=ratio,
                        paired_wins=wins,
                        pass_gate=ratio >= plan["primary_gates"]["ext4_combined_large_write_aggregate_ratio_minimum"]
                        and wins >= plan["primary_gates"]["ext4_combined_large_write_paired_wins_minimum"],
                    )
                )
            lifecycle = {
                v: [r["lifecycle_seconds"] for r in rows if r["variant"] == v] for v in ("baseline", "updated")
            }
            ratio = statistics.median(lifecycle["updated"]) / statistics.median(lifecycle["baseline"])
            report["checks"].append(
                dict(
                    medium=medium,
                    state=state,
                    workload="lifecycle",
                    median_ratio=ratio,
                    pass_gate=ratio <= plan["regression_gates"]["lifecycle_median_ratio_maximum"],
                )
            )
    report["pass_all"] = all(c["pass_gate"] for c in report["checks"])
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS" if report["pass_all"] else "REJECT",
        ["/".join((c["medium"], c["state"], c["workload"])) for c in report["checks"] if not c["pass_gate"]],
    )
    print(json.dumps(report["large_writes"], indent=2))


if __name__ == "__main__":
    main()
