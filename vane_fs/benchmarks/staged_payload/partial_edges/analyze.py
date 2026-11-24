# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Recompute complete-workload metrics with the frozen partial-edge gates."""

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
    helper = here.parents[1] / "checkpoint_priority/analyze.py"
    spec = importlib.util.spec_from_file_location("summaries", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = json.loads((out / "plan.json").read_text())
    report = dict(status="PASS", plan=plan, plain={}, checks=[])
    for medium in plan["media"]:
        complete = out / (medium + "-plain/results.json")
        if complete.exists():
            raw = json.loads(complete.read_text())
        else:
            raw = json.loads((out / "windows.json").read_text())[medium]
            for state, label in (("fuse", "fresh"), ("aged_fuse", "aged")):
                raw[state] = [
                    row
                    for variant in ("baseline", "updated")
                    for row in json.loads((out / (medium + "-" + label + "-" + variant + ".json")).read_text())
                ]
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        result = module.plain(raw)
        report["plain"][medium] = result
        for state, state_data in result.items():
            for name, values in state_data["workloads"].items():
                if name == "stream_256":
                    limit = plan["primary_gates"]["first_256_median_and_aggregate_minimum_ratio"]
                    passed = (
                        min(values["median_rate_ratio"], values["aggregate_rate_ratio"]) >= limit
                        and values["paired_complete_write_wins"]
                        >= plan["primary_gates"]["first_256_paired_wins_minimum"]
                    )
                else:
                    limit = plan["regression_gates"]["all_other_workloads_median_and_aggregate_ratio_minimum"]
                    passed = min(values["median_rate_ratio"], values["aggregate_rate_ratio"]) >= limit
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
                v: [row["lifecycle_seconds"] for row in rows if row["variant"] == v] for v in ("baseline", "updated")
            }
            ratio = statistics.median(durations["updated"]) / statistics.median(durations["baseline"])
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
    for medium in plan["media"]:
        for state in ("fuse", "aged_fuse"):
            values = report["plain"][medium][state]["workloads"]["stream_256"]
            print(
                medium,
                state,
                "first256 median/aggregate",
                [
                    (v, round(values[v]["rate"]["median"], 2), round(values[v]["aggregate_rate"], 2))
                    for v in ("baseline", "updated")
                ],
            )


if __name__ == "__main__":
    main()
