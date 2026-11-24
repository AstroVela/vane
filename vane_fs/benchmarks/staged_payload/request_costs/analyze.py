# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute short requests without using diagnostic rates as adoption evidence."""

import argparse
import hashlib
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


def distribution(values):
    values = sorted(values)
    return {
        "count": len(values),
        "sum": sum(values),
        "median": statistics.median(values) if values else 0,
        "max": max(values, default=0),
        "p99": values[min(len(values) - 1, int(len(values) * 0.99))] if values else 0,
    }


def overlap(row, begin, end):
    return row["begin_ns"] < end and row["end_ns"] > begin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    result = {
        "status": "PASS",
        "scope": "Diagnostic costs, never plain adoption throughput",
        "inputs": {},
        "mounts": [],
        "groups": [],
    }
    for medium in ("tmpfs", "ext4"):
        path = out / (medium + "-diagnostic/results.json")
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        assert raw["status"] == "PASS"
        result["inputs"][str(path)] = digest(path)
        for state, key in [("fresh", "fuse"), ("post_gc", "aged_fuse")]:
            for mount in raw[key]:
                trace_path = Path(mount["request_costs_file"])
                trace = json.loads(trace_path.read_text())
                result["inputs"][str(trace_path)] = digest(trace_path)
                events = []
                for thread in trace["threads"]:
                    for values in thread["rows"]:
                        row = dict(zip(trace["columns"], values, strict=True))
                        assert row["end_ns"] >= row["begin_ns"] and row["cpu_ns"] >= 0
                        assert len(row["counts"]) == len(row["stage_ns"]) == len(trace["stages"])
                        row["tid"] = thread["tid"]
                        row["epoch_begin_ns"] = trace["origin_epoch_ns"] + row["begin_ns"]
                        events.append(row)
                entries = []
                for sample in mount["samples"]:
                    name = sample["workload"]
                    if name not in OPERATIONS:
                        continue
                    begin = round(sample["started_epoch"] * 10**9) - trace["origin_epoch_ns"]
                    end = round(sample["ended_epoch"] * 10**9) - trace["origin_epoch_ns"]
                    selected = [r for r in events if r["name"] == OPERATIONS[name] and begin <= r["begin_ns"] <= end]
                    if name in ("random_read_4k", "random_write_4k", "mkdir", "rename", "rmdir"):
                        assert len(selected) == sample["requests"], (
                            medium,
                            state,
                            mount["variant"],
                            name,
                            len(selected),
                        )
                    if name in ("stream_256", "write_64"):
                        assert sum(r["bytes"] for r in selected) == (256 if name == "stream_256" else 64) * 1024**2
                    worker = [r for r in events if r["name"] == "checkpoint_worker" and overlap(r, begin, end)]
                    hint = [r for r in events if r["name"] == "hint" and overlap(r, begin, end)]
                    entry = dict(
                        medium=medium,
                        state=state,
                        variant=mount["variant"],
                        rep=mount["rep"],
                        workload=name,
                        phase_seconds=sample["seconds"],
                        requests_seconds=sample.get("requests_seconds"),
                        fsync_seconds=sample.get("fsync_seconds", sample.get("final_directory_fsync_seconds")),
                        handlers=len(selected),
                        wall_ns=distribution([r["end_ns"] - r["begin_ns"] for r in selected]),
                        cpu_ns=distribution([r["cpu_ns"] for r in selected]),
                        off_cpu_ns=distribution([max(0, r["end_ns"] - r["begin_ns"] - r["cpu_ns"]) for r in selected]),
                        vm_steps=sum(r["vm_steps"] for r in selected),
                        stage_ns={s: sum(r["stage_ns"][i] for r in selected) for i, s in enumerate(trace["stages"])},
                        stage_counts={s: sum(r["counts"][i] for r in selected) for i, s in enumerate(trace["stages"])},
                        checkpoint_worker_overlaps=len(worker),
                        checkpoint_worker_wall_ns=sum(r["end_ns"] - r["begin_ns"] for r in worker),
                        checkpoint_worker_cpu_ns=sum(r["cpu_ns"] for r in worker),
                        hint_overlaps=len(hint),
                        hint_bytes=sum(r["bytes"] for r in hint),
                        hint_wall_ns=sum(r["end_ns"] - r["begin_ns"] for r in hint),
                        sqlite_busy_queries=sum(r["last_checkpoint"][3] == 5 for r in selected),
                        wal_frame_resets=sum(
                            b["committed_frames"] >= 0 and a["committed_frames"] > b["committed_frames"]
                            for a, b in zip(selected, selected[1:])
                        ),
                        slowest=[
                            {
                                k: r[k]
                                for k in (
                                    "begin_ns",
                                    "end_ns",
                                    "cpu_ns",
                                    "vm_steps",
                                    "stage_ns",
                                    "last_checkpoint",
                                    "committed_frames",
                                )
                            }
                            for r in sorted(selected, key=lambda r: r["end_ns"] - r["begin_ns"], reverse=True)[:5]
                        ],
                    )
                    entries.append(entry)
                assert mount["inspection"]["quick_check"] == [["ok"]]
                result["mounts"].append(
                    dict(
                        medium=medium,
                        state=state,
                        variant=mount["variant"],
                        rep=mount["rep"],
                        stages=trace["stages"],
                        phases=entries,
                    )
                )
    all_entries = [entry for mount in result["mounts"] for entry in mount["phases"]]
    keys = sorted({(e["medium"], e["state"], e["variant"], e["workload"]) for e in all_entries})
    for medium, state, variant, workload in keys:
        rows = [
            e
            for e in all_entries
            if (e["medium"], e["state"], e["variant"], e["workload"]) == (medium, state, variant, workload)
        ]
        result["groups"].append(
            dict(
                medium=medium,
                state=state,
                variant=variant,
                workload=workload,
                phases=len(rows),
                phase_seconds=distribution([e["phase_seconds"] for e in rows]),
                handler_wall_ns=distribution([e["wall_ns"]["sum"] for e in rows]),
                handler_cpu_ns=distribution([e["cpu_ns"]["sum"] for e in rows]),
                handler_off_cpu_ns=distribution([e["off_cpu_ns"]["sum"] for e in rows]),
                vm_steps=distribution([e["vm_steps"] for e in rows]),
                stage_ns={s: distribution([e["stage_ns"][s] for e in rows]) for s in rows[0]["stage_ns"]},
                worker_overlaps=sum(e["checkpoint_worker_overlaps"] for e in rows),
                hint_overlaps=sum(e["hint_overlaps"] for e in rows),
                sqlite_busy_queries=sum(e["sqlite_busy_queries"] for e in rows),
                wal_frame_resets=sum(e["wal_frame_resets"] for e in rows),
            )
        )
    (out / "request-costs-summary.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("request cost analysis PASS", len(result["mounts"]), "mounts", flush=True)


if __name__ == "__main__":
    main()
