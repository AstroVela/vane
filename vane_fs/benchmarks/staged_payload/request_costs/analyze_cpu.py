# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Relate owned-thread scheduler accounting and sampled frequency to request phases."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(samples, pid, begin, end):
    runtime = queued = migrated = intervals = weighted_ns = frequency_sum = worker_ns = covered_ns = 0
    frequencies = {}
    cpus = set()
    for first, second in zip(samples, samples[1:]):
        if first["epoch_ns"] < begin or second["epoch_ns"] > end:
            continue
        a = {t["tid"]: t for t in first["threads"]}
        b = {t["tid"]: t for t in second["threads"]}
        if pid not in a or pid not in b:
            continue
        intervals += 1
        covered_ns += second["epoch_ns"] - first["epoch_ns"]
        for tid in a.keys() & b.keys():
            assert b[tid]["runtime_ns"] >= a[tid]["runtime_ns"]
            if tid != pid:
                worker_ns += b[tid]["runtime_ns"] - a[tid]["runtime_ns"]
        delta = b[pid]["runtime_ns"] - a[pid]["runtime_ns"]
        runtime += delta
        queued += b[pid]["runqueue_ns"] - a[pid]["runqueue_ns"]
        cpus.update([a[pid]["cpu"], b[pid]["cpu"]])
        if a[pid]["cpu"] != b[pid]["cpu"]:
            migrated += 1
            continue
        freq = second["frequency_khz"].get(str(b[pid]["cpu"]))
        if freq is None or delta == 0:
            continue
        weighted_ns += delta
        frequency_sum += freq * delta
        frequencies[str(freq)] = frequencies.get(str(freq), 0) + delta
    return dict(
        complete_sample_intervals=intervals,
        covered_wall_ns=covered_ns,
        scheduler_runtime_ns_observed=runtime,
        scheduler_runqueue_ns_observed=queued,
        background_runtime_ns_observed=worker_ns,
        migrated_intervals=migrated,
        cpus=sorted(cpus),
        same_cpu_frequency_weight_ns=weighted_ns,
        runtime_weighted_frequency_khz=frequency_sum / weighted_ns if weighted_ns else None,
        frequency_khz_runtime_weights_ns=frequencies,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    path = out / "ext4-diagnostic/results.json"
    raw = json.loads(path.read_text())
    assert raw["status"] == "PASS"
    assert json.loads((out / "input-check.json").read_text())["status"] == "PASS"
    result = dict(
        status="PASS",
        scope="Diagnostic estimates from sysfs and scheduler accounting, not cycle counts or adoption throughput",
        inputs={str(path): digest(path)},
        mounts=[],
        groups=[],
        controlled_pairs=[],
    )
    for state, key in [("fresh", "fuse"), ("post_gc", "aged_fuse")]:
        for mount in raw[key]:
            sample_file = Path(mount["cpu_samples_file"])
            result["inputs"][str(sample_file)] = digest(sample_file)
            samples = [json.loads(line) for line in sample_file.read_text().splitlines()]
            assert all(a["epoch_ns"] <= b["epoch_ns"] for a, b in zip(samples, samples[1:]))
            entries = []
            for phase in mount["samples"]:
                name = phase["workload"]
                if name not in ("read_64", "random_read_4k", "random_write_4k", "mkdir", "rename", "rmdir"):
                    continue
                begin = round(phase["started_epoch"] * 10**9)
                duration = phase.get(
                    "requests_seconds", phase["seconds"] - phase.get("final_directory_fsync_seconds", 0)
                )
                end = begin + round(duration * 10**9)
                value = metrics(samples, mount["daemon_pid"], begin, end)
                assert value["complete_sample_intervals"] >= 1, (state, mount["variant"], mount["rep"], name)
                entries.append(
                    dict(
                        workload=name,
                        phase_seconds=phase["seconds"],
                        request_phase_seconds=duration,
                        fsync_seconds=phase.get("fsync_seconds", phase.get("final_directory_fsync_seconds")),
                        **value,
                    )
                )
            preparation = [r for r in raw.get("cpu_preparation", []) if r["daemon_pid"] == mount["daemon_pid"]]
            if preparation:
                assert len(preparation) == 6
                assert all(r["ended_epoch"] - r["started_epoch"] >= 2 for r in preparation)
                if mount["variant"].endswith("-warm"):
                    assert all(r["warmup_child"] and r["warmup_child"]["seconds"] >= 2 for r in preparation)
                else:
                    assert all(r["warmup_child"] is None for r in preparation)
            assert mount["inspection"]["quick_check"] == [["ok"]]
            result["mounts"].append(
                dict(
                    state=state,
                    variant=mount["variant"],
                    rep=mount["rep"],
                    daemon_pid=mount["daemon_pid"],
                    preparation=preparation,
                    phases=entries,
                )
            )
    phases = [
        dict(state=m["state"], variant=m["variant"], rep=m["rep"], **p) for m in result["mounts"] for p in m["phases"]
    ]
    for state, variant, name in sorted({(p["state"], p["variant"], p["workload"]) for p in phases}):
        selected = [p for p in phases if (p["state"], p["variant"], p["workload"]) == (state, variant, name)]
        result["groups"].append(
            dict(
                state=state,
                variant=variant,
                workload=name,
                phases=len(selected),
                request_phase_seconds_median=statistics.median(p["request_phase_seconds"] for p in selected),
                runtime_ns_median=statistics.median(p["scheduler_runtime_ns_observed"] for p in selected),
                runqueue_ns_median=statistics.median(p["scheduler_runqueue_ns_observed"] for p in selected),
                frequency_khz_median=statistics.median(p["runtime_weighted_frequency_khz"] for p in selected),
            )
        )
    if "cpu_preparation" in raw:
        for mount in result["mounts"]:
            if not mount["variant"].endswith("-idle"):
                continue
            warm = next(
                m
                for m in result["mounts"]
                if m["state"] == mount["state"]
                and m["rep"] == mount["rep"]
                and m["variant"] == mount["variant"].replace("-idle", "-warm")
            )
            for idle_phase, warm_phase in zip(mount["phases"], warm["phases"], strict=True):
                assert idle_phase["workload"] == warm_phase["workload"]
                result["controlled_pairs"].append(
                    dict(
                        state=mount["state"],
                        variant=mount["variant"].split("-")[0],
                        rep=mount["rep"],
                        workload=idle_phase["workload"],
                        idle_requests_seconds=idle_phase["request_phase_seconds"],
                        warm_requests_seconds=warm_phase["request_phase_seconds"],
                        warm_speed_ratio=idle_phase["request_phase_seconds"] / warm_phase["request_phase_seconds"],
                        idle_frequency_khz=idle_phase["runtime_weighted_frequency_khz"],
                        warm_frequency_khz=warm_phase["runtime_weighted_frequency_khz"],
                    )
                )
    (out / "cpu-summary.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("CPU diagnostic analysis PASS", len(result["mounts"]), "mounts", flush=True)


if __name__ == "__main__":
    main()
