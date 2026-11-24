# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute matched native write costs without adding overlapping intervals."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

from prepare import digest, save, validate


def distribution(values):
    return dict(
        values=values,
        median=statistics.median(values),
        max=max(values),
        p99=sorted(values)[min(len(values) - 1, int(len(values) * 0.99))],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    source = here.parent / "request_costs/analyze_cpu.py"
    spec = importlib.util.spec_from_file_location("native_cpu_metrics", source)
    cpu_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu_metrics)
    report = dict(
        status="PASS",
        inputs={str(source): digest(source), str(Path(__file__)): digest(Path(__file__))},
        runs=[],
        groups=[],
        observer_comparisons=[],
    )
    for mode, count in [("plain", 24), ("diagnostic", 8)]:
        result = json.loads((out / mode / "results.json").read_text())
        assert result["status"] == "PASS" and len(result["runs"]) == count
        report["inputs"][str(out / mode / "results.json")] = digest(out / mode / "results.json")
        for row in result["runs"]:
            validate(row, mode == "diagnostic")
            samples = [json.loads(line) for line in Path(row["cpu_samples_file"]).read_text().splitlines()]
            # Reuse the existing interval estimator with monotonic endpoints;
            # the native probe and sampler share CLOCK_MONOTONIC's origin.
            monotonic = [dict(s, epoch_ns=s["end_monotonic_ns"]) for s in samples]
            phases = []
            for phase in row["phases"]:
                sampled = cpu_metrics.metrics(monotonic, row["main_tid"], phase["begin_ns"], phase["end_ns"])
                if phase["name"] == "writes":
                    assert sampled["complete_sample_intervals"] >= 1
                stages = {}
                for name, values in phase["stages"].items():
                    stage = dict(zip(row["counter_columns"], values, strict=True))
                    stage["exclusive_cpu_share"] = stage["exclusive_cpu_ns"] / phase["stages"]["root"][3]
                    stages[name] = stage
                syncs = [e for e in row["events"] if e[0] != "checkpoint" and e[2] == row["phases"].index(phase)]
                phases.append(
                    dict(
                        name=phase["name"],
                        wall_ns=phase["wall_ns"],
                        thread_cpu_ns=phase["thread_cpu_ns"],
                        process_cpu_ns=phase["process_cpu_ns"],
                        sampled_cpu=sampled,
                        stages=stages,
                        foreground_syncs=[
                            dict(zip(row["event_columns"], e, strict=True)) for e in syncs if e[3] == row["main_tid"]
                        ],
                        worker_syncs=[
                            dict(zip(row["event_columns"], e, strict=True)) for e in syncs if e[3] != row["main_tid"]
                        ],
                    )
                )
            report["runs"].append(
                dict(
                    mode=mode,
                    medium=row["medium"],
                    mib=row["mib"],
                    rep=row["rep"],
                    total_ns=row["total_ns"],
                    total_thread_cpu_ns=row["total_thread_cpu_ns"],
                    total_process_cpu_ns=row["total_process_cpu_ns"],
                    phases=phases,
                )
            )
    for mode in ("plain", "diagnostic"):
        for medium in ("ext4", "tmpfs"):
            for size in (64, 256):
                rows = [r for r in report["runs"] if (r["mode"], r["medium"], r["mib"]) == (mode, medium, size)]
                phases = {}
                for name in ("initialization", "create", "writes", "barrier", "release", "close"):
                    data = [next(p for p in row["phases"] if p["name"] == name) for row in rows]
                    stages = {}
                    if mode == "diagnostic":
                        for stage in data[0]["stages"]:
                            stages[stage] = {
                                metric: distribution([d["stages"][stage][metric] for d in data])
                                for metric in data[0]["stages"][stage]
                            }
                    frequency = [d["sampled_cpu"]["runtime_weighted_frequency_khz"] for d in data]
                    phases[name] = dict(
                        wall_ns=distribution([d["wall_ns"] for d in data]),
                        thread_cpu_ns=distribution([d["thread_cpu_ns"] for d in data]),
                        process_cpu_ns=distribution([d["process_cpu_ns"] for d in data]),
                        sampled_frequency_khz=frequency,
                        observed_frequency_khz_median=statistics.median(f for f in frequency if f is not None)
                        if any(f is not None for f in frequency)
                        else None,
                        stages=stages,
                    )
                group = dict(
                    mode=mode,
                    medium=medium,
                    mib=size,
                    count=len(rows),
                    median_mib_s=statistics.median(size * 1e9 / r["total_ns"] for r in rows),
                    aggregate_mib_s=size * 1e9 * len(rows) / sum(r["total_ns"] for r in rows),
                    total_ns=distribution([r["total_ns"] for r in rows]),
                    total_thread_cpu_ns=distribution([r["total_thread_cpu_ns"] for r in rows]),
                    total_process_cpu_ns=distribution([r["total_process_cpu_ns"] for r in rows]),
                    phases=phases,
                )
                report["groups"].append(group)
                print(
                    mode,
                    medium,
                    size,
                    "write wall/caller CPU/barrier ms",
                    *[
                        round(phases[n][k]["median"] / 1e6, 3)
                        for n, k in [("writes", "wall_ns"), ("writes", "thread_cpu_ns"), ("barrier", "wall_ns")]
                    ],
                    flush=True,
                )
    for medium in ("ext4", "tmpfs"):
        for size in (64, 256):
            groups = {r["mode"]: r for r in report["groups"] if r["medium"] == medium and r["mib"] == size}
            a, b = [groups[m]["phases"]["writes"] for m in ("plain", "diagnostic")]
            report["observer_comparisons"].append(
                dict(
                    medium=medium,
                    mib=size,
                    caller_cpu_median_ratio=b["thread_cpu_ns"]["median"] / a["thread_cpu_ns"]["median"],
                    interpretation="separate windows and sampled CPU states; not a causal overhead estimate or adjusted throughput",
                )
            )
    save(out / "analysis.json", report)


if __name__ == "__main__":
    main()
