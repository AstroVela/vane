# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Separate maintenance overlaps and final fsync costs in the frozen diagnostic."""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

OPERATIONS = {
    "stream_256": "write",
    "write_64": "write",
    "read_64": "read",
    "random_read_4k": "read",
    "random_write_4k": "write",
    "mkdir": "mkdir",
    "rename": "rename",
    "rmdir": "rmdir",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def totals(rows, stages):
    return dict(
        handlers=len(rows),
        wall_ns=sum(r["end_ns"] - r["begin_ns"] for r in rows),
        cpu_ns=sum(r["cpu_ns"] for r in rows),
        vm_steps=sum(r["vm_steps"] for r in rows),
        stage_ns={s: sum(r["stage_ns"][i] for r in rows) for i, s in enumerate(stages)},
        stage_counts={s: sum(r["counts"][i] for r in rows) for i, s in enumerate(stages)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    cpu_analyzer = Path(__file__).resolve().parent.parent / "request_costs/analyze_cpu.py"
    spec = importlib.util.spec_from_file_location("owned_cpu", cpu_analyzer)
    cpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu)
    source = out / "ext4-diagnostic/results.json"
    raw = json.loads(source.read_text())
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    assert json.loads((out / "input-check.json").read_text())["status"] == "PASS"
    result = dict(
        status="PASS",
        scope="Inclusive diagnostic costs and full overlapping worker spans; not additive or adoption rates",
        inputs={
            str(source): digest(source),
            str(Path(__file__).resolve()): digest(Path(__file__)),
            str(cpu_analyzer): digest(cpu_analyzer),
        },
        phases=[],
        groups=[],
        slow_overlapping_spans=[],
    )
    for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
        for mount in raw[key]:
            path = Path(mount["request_costs_file"])
            result["inputs"][str(path)] = digest(path)
            trace = json.loads(path.read_text())
            cpu_path = Path(mount["cpu_samples_file"])
            result["inputs"][str(cpu_path)] = digest(cpu_path)
            samples = [json.loads(line) for line in cpu_path.read_text().splitlines()]
            events = [dict(zip(trace["columns"], row, strict=True)) for t in trace["threads"] for row in t["rows"]]
            for phase in mount["samples"]:
                name = phase["workload"]
                if name not in OPERATIONS:
                    continue
                begin = round(phase["started_epoch"] * 10**9) - trace["origin_epoch_ns"]
                end = round(phase["ended_epoch"] * 10**9) - trace["origin_epoch_ns"]
                overlapping = [
                    r
                    for r in events
                    if r["begin_ns"] < end and r["end_ns"] > begin and r["end_ns"] - r["begin_ns"] >= 100000000
                ]
                if overlapping:
                    result["slow_overlapping_spans"].append(
                        dict(
                            state=state,
                            variant=mount["variant"],
                            rep=mount["rep"],
                            workload=name,
                            phase_begin_ns=begin,
                            phase_end_ns=end,
                            rows=overlapping,
                        )
                    )
                request_seconds = phase.get(
                    "requests_seconds", phase["seconds"] - phase.get("final_directory_fsync_seconds", 0)
                )
                epoch_begin = begin + trace["origin_epoch_ns"]
                cpu_state = cpu.metrics(
                    samples, mount["daemon_pid"], epoch_begin, epoch_begin + round(request_seconds * 10**9)
                )
                assert cpu_state["complete_sample_intervals"] >= 1
                workers = [
                    r
                    for r in events
                    if r["name"] == "checkpoint_worker" and r["begin_ns"] < end and r["end_ns"] > begin
                ]
                foreground = [r for r in events if r["name"] == OPERATIONS[name] and begin <= r["begin_ns"] <= end]
                barriers = [r for r in events if r["name"] == "fsync" and begin <= r["begin_ns"] <= end]
                if "fsync_seconds" in phase or "final_directory_fsync_seconds" in phase:
                    assert len(barriers) == 1, (state, mount["variant"], mount["rep"], name)
                result["phases"].append(
                    dict(
                        state=state,
                        variant=mount["variant"],
                        rep=mount["rep"],
                        workload=name,
                        phase_seconds=phase["seconds"],
                        application_barrier_seconds=phase.get(
                            "fsync_seconds", phase.get("final_directory_fsync_seconds", 0)
                        ),
                        foreground=totals(foreground, trace["stages"]),
                        barrier=totals(barriers, trace["stages"]),
                        worker=totals(workers, trace["stages"]),
                        request_cpu_state=cpu_state,
                        passive_worker_overlaps=sum(r["last_checkpoint"][0] == 0 for r in workers),
                        noop_only_worker_overlaps=sum(r["last_checkpoint"][0] == -1 for r in workers),
                        worker_error_overlaps=sum(r["last_checkpoint"][3] not in (0, 5) for r in workers),
                    )
                )
    for state, variant, name in sorted({(p["state"], p["variant"], p["workload"]) for p in result["phases"]}):
        selected = [p for p in result["phases"] if (p["state"], p["variant"], p["workload"]) == (state, variant, name)]
        result["groups"].append(
            dict(
                state=state,
                variant=variant,
                workload=name,
                phases=len(selected),
                application_barrier_seconds_median=statistics.median(
                    p["application_barrier_seconds"] for p in selected
                ),
                passive_worker_overlaps=[p["passive_worker_overlaps"] for p in selected],
                noop_only_worker_overlaps=[p["noop_only_worker_overlaps"] for p in selected],
                worker_error_overlaps=sum(p["worker_error_overlaps"] for p in selected),
                request_frequency_khz_median=statistics.median(
                    p["request_cpu_state"]["runtime_weighted_frequency_khz"] for p in selected
                ),
                medians={
                    scope: {
                        "wall_ns": statistics.median(p[scope]["wall_ns"] for p in selected),
                        "cpu_ns": statistics.median(p[scope]["cpu_ns"] for p in selected),
                        "stage_ns": {
                            s: statistics.median(p[scope]["stage_ns"][s] for p in selected)
                            for s in selected[0][scope]["stage_ns"]
                        },
                        "stage_counts": {
                            s: statistics.median(p[scope]["stage_counts"][s] for p in selected)
                            for s in selected[0][scope]["stage_counts"]
                        },
                    }
                    for scope in ("foreground", "barrier", "worker")
                },
            )
        )
    (out / "maintenance-analysis.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("maintenance analysis PASS", len(result["phases"]), "phases", flush=True)


if __name__ == "__main__":
    main()
