# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the original FUSE gates and describe consecutive extent write cycles."""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from prepare import digest, save


def load_module(path):
    spec = importlib.util.spec_from_file_location("extent_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    run = json.loads((out / "run.json").read_text())
    assert build["status"] == run["status"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in build["inputs"].items())
    report = dict(status="RUNNING", started_epoch=time.time(), windows={}, inputs=dict(build["inputs"]))
    source = here.parent / "partial_sync/analyze_writeback.py"
    module = load_module(here.parents[1] / "checkpoint_priority/analyze.py")
    report["inputs"].update({str(p): digest(p) for p in [Path(__file__), source, out / "build.json", out / "run.json"]})
    begin = time.time()
    argv = [sys.executable, str(source), str(out / "full")]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    report["original_analyzer"] = dict(
        argv=argv,
        started_epoch=begin,
        ended_epoch=time.time(),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    assert result.returncode == 0, result.stderr
    gate = json.loads((out / "full/summary.json").read_text())
    assert gate["status"] == "PASS"
    report["full_acceptance_pass"] = gate["pass_all"]
    report["full_failed_checks"] = [c for c in gate["checks"] if not c["pass_gate"]]
    report["full_checks"] = len(gate["checks"])
    assert report["full_checks"] == 22
    failure_details = []
    for check in report["full_failed_checks"]:
        flags = []
        name = check["workload"]
        if name == "combined_large_writes":
            if check["aggregate_ratio"] < 1.1:
                flags.append("combined aggregate improvement below 10%")
            if check["paired_wins"] < 5:
                flags.append("fewer than five of six paired wins")
        elif name == "lifecycle":
            flags.append("median lifecycle increase above 5%")
        else:
            if check["median_ratio"] < 0.95:
                flags.append("median throughput loss above 5%")
            if check["aggregate_ratio"] < 0.95:
                flags.append("aggregate throughput loss above 5%")
            values = gate["plain"][check["medium"]][check["state"]]["workloads"][name]
            if "latencies" in values["baseline"]:
                before = values["baseline"]["latencies"]["p99"]
                after = values["updated"]["latencies"]["p99"]
                if after > before * 1.2 and after - before > 0.001:
                    flags.append("pooled p99 exceeds 20% and 1 ms allowances")
        assert flags
        failure_details.append(dict(check=check, reasons=flags))
    report["failure_details"] = failure_details
    total_mounts = total_phases = 0
    guard_hashes = set()
    for window, config in build["windows"].items():
        root = Path(config["root"])
        raw = json.loads((root / "ext4-plain/results.json").read_text())
        plan = json.loads((root / "plan.json").read_text())
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        assert raw["config"]["instrumented"] is False
        assert raw["config"]["continuous_cycles"] == plan["continuous_cycles"]
        assert raw["config"]["orders"] == plan["orders"]
        assert raw["config"]["format_versions"] == {"baseline": 1004, "updated": 1005}
        report["inputs"][str(root / "ext4-plain/results.json")] = digest(root / "ext4-plain/results.json")
        expected_samples = 7 + 2 * plan["continuous_cycles"]
        assert len(raw["fuse"]) == len(raw["aged_fuse"]) == config["mounts"] // 2
        summary = dict(
            started_epoch=raw["started_epoch"], ended_epoch=raw["ended_epoch"], mounts=config["mounts"], states={}
        )
        distributions = module.plain(raw)
        for state in ("fuse", "aged_fuse"):
            rows = raw[state]
            lifecycle_components = {
                variant: {
                    key: module.distribution([row[key] for row in rows if row["variant"] == variant])
                    for key in (
                        "initialization_seconds",
                        "mount_ready_seconds",
                        "settle_seconds",
                        "preparation_seconds",
                        "gc_wall_seconds",
                        "unmount_seconds",
                        "workload_seconds",
                    )
                    if all(key in row for row in rows if row["variant"] == variant)
                }
                for variant in ("baseline", "updated")
            }
            grouped = {}
            cycles = {}
            long_calls = []
            for row in rows:
                total_mounts += 1
                variant = row["variant"]
                assert len(row["samples"]) == expected_samples
                assert row["inspection"]["quick_check"] == [["ok"]]
                assert row["inspection"]["format"] == [[plan["format_versions"][variant], 4096]]
                assert row["inspection"]["block_high_index_present"]
                assert all(n in (8, 4096) for n, _ in row["inspection"]["payload_lengths"])
                if state == "aged_fuse":
                    assert row["after_gc_inspection"]["format"] == row["inspection"]["format"]
                    assert row["guard_validation"]["bytes"] == 33554432
                    guard_hashes.add(row["guard_validation"]["sha256"])
                phases = [s for s in row["samples"] if s["workload"].startswith(("stream_256", "write_64"))]
                assert len(phases) == 2 * plan["continuous_cycles"]
                total_phases += len(phases)
                durations = []
                for i, sample in enumerate(phases):
                    mib = 256 if i % 2 == 0 else 64
                    assert sample["validation"]["bytes"] == mib * 1048576
                    assert sample["buffer_page_offset"] == 48
                    assert len(sample["write_latencies"]) == (32 if mib == 256 else 64)
                    assert row["started_epoch"] <= sample["started_epoch"] < sample["ended_epoch"] <= row["ended_epoch"]
                    if window == "sustained":
                        assert sample["cycle"] == i // 2
                    durations.append(sample["seconds"])
                    cycle = i // 2
                    cycles.setdefault(cycle, {}).setdefault(variant, {}).setdefault(row["rep"], 0)
                    cycles[cycle][variant][row["rep"]] += sample["seconds"]
                    for call, seconds in enumerate(sample["write_latencies"]):
                        if seconds >= 1:
                            long_calls.append(
                                dict(
                                    variant=variant,
                                    rep=row["rep"],
                                    workload=sample["workload"],
                                    kind="write",
                                    index=call,
                                    seconds=seconds,
                                )
                            )
                    if sample["fsync_seconds"] >= 1:
                        long_calls.append(
                            dict(
                                variant=variant,
                                rep=row["rep"],
                                workload=sample["workload"],
                                kind="fsync",
                                seconds=sample["fsync_seconds"],
                            )
                        )
                grouped.setdefault(variant, {})[row["rep"]] = sum(durations)
            assert grouped["baseline"].keys() == grouped["updated"].keys()
            total_mib = 320 * plan["continuous_cycles"]
            aggregate = {v: len(values) * total_mib / sum(values.values()) for v, values in grouped.items()}
            combined = dict(
                complete_large_phase_seconds=grouped,
                aggregate_mib_per_second=aggregate,
                aggregate_ratio=aggregate["updated"] / aggregate["baseline"],
                paired_wins=sum(grouped["updated"][rep] < value for rep, value in grouped["baseline"].items()),
            )
            cycle_rows = []
            for cycle, variants in cycles.items():
                rates = {v: len(values) * 320 / sum(values.values()) for v, values in variants.items()}
                cycle_rows.append(
                    dict(
                        cycle=cycle,
                        complete_large_phase_seconds=variants,
                        aggregate_mib_per_second=rates,
                        aggregate_ratio=rates["updated"] / rates["baseline"],
                        paired_wins=sum(
                            variants["updated"][rep] < value for rep, value in variants["baseline"].items()
                        ),
                    )
                )
            lifecycle = {v: distributions[state]["lifecycle"][v] for v in ("baseline", "updated")}
            summary["states"][state] = dict(
                combined=combined,
                cycles=cycle_rows,
                workloads=distributions[state]["workloads"],
                lifecycle=lifecycle,
                lifecycle_components=lifecycle_components,
                long_large_calls=long_calls,
            )
        report["windows"][window] = summary
    assert total_mounts == 32 and total_phases == 112 and len(guard_hashes) == 1
    report.update(
        status="PASS",
        ended_epoch=time.time(),
        mounts=total_mounts,
        sequential_content_checks=total_phases,
        retained_guard_sha256=guard_hashes.pop(),
        limitations=[
            "Sustained window has only two pairs per state and is supplementary; it does not replace the original six-pair gates.",
            "Four cycles recreate/unlink the same logical file names in one live workspace; GC runs only during post-GC preparation.",
            "Application timings locate writes/fsync but do not attribute individual waits to a native mutex, checkpoint or device.",
            "No production adoption or format migration is implemented.",
        ],
    )
    save(out / "analysis.json", report)
    print(
        "PASS validation", total_mounts, "mounts", total_phases, "large-file checks; full acceptance", gate["pass_all"]
    )
    print("Failed checks", ["/".join((c["state"], c["workload"])) for c in report["full_failed_checks"]])
    for window, summary in report["windows"].items():
        for state, values in summary["states"].items():
            print(window, state, json.dumps(values["combined"]))


if __name__ == "__main__":
    main()
