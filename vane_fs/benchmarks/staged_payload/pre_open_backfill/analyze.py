# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile typed syncs and distinguish prefix preparation from midstream publication."""

import argparse
import collections
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return dict(
        count=len(values),
        total=sum(values),
        median=statistics.median(values) if values else 0,
        maximum=max(values, default=0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    args = parser.parse_args()
    out, here = args.window.resolve(), Path(__file__).resolve().parent
    old = here.parent / "checkpoint_followthrough/analyze.py"
    subprocess.run([sys.executable, str(old), str(out)], check=True)
    findings_path, raw_path = out / "findings.json", out / "ext4-diagnostic/results.json"
    findings, raw = (json.loads(path.read_text()) for path in (findings_path, raw_path))
    report = dict(
        status="PASS",
        interpretation="Inclusive and overlapping intervals; preparation includes real synchronous I/O, not a new latency bound",
        inputs={str(path): digest(path) for path in (Path(__file__), old, findings_path, raw_path)},
        preparation_calls=[],
        phases=[],
        groups=[],
        sync_tails_at_least_one_second=[],
        slow_full_phases=[],
    )
    for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
        for mount in raw[key]:
            ident = dict(state=state, variant=mount["variant"], rep=mount["rep"])
            path = Path(mount["request_costs_file"])
            report["inputs"][str(path)] = digest(path)
            trace = json.loads(path.read_text())
            syncs, prepares = [], []
            for thread in trace["threads"]:
                rows = [dict(zip(trace["columns"], row, strict=True)) for row in thread["rows"]]
                calls = collections.defaultdict(list)
                for value in thread["checkpoints"]:
                    calls[value[0]].append(
                        dict(
                            zip(
                                ("request_index", "mode", "frames", "copied", "code", "begin_ns", "end_ns"),
                                value,
                                strict=True,
                            )
                        )
                    )
                thread_syncs = []
                for value in thread["syncs"]:
                    sync = dict(zip(trace["sync_columns"], value, strict=True), tid=thread["tid"])
                    sync["request"] = rows[sync["request_index"]]["name"] if sync["request_index"] >= 0 else "unscoped"
                    syncs.append(sync)
                    thread_syncs.append(sync)
                    if sync["end_ns"] - sync["begin_ns"] >= 1_000_000_000:
                        report["sync_tails_at_least_one_second"].append(dict(**ident, **sync))
                for index, row in enumerate(rows):
                    if row["name"] != "write":
                        continue
                    selected = [
                        sync
                        for sync in thread_syncs
                        if sync["request_index"] == index and sync["site"] == "clean_prepare"
                    ]
                    if not selected:
                        continue
                    passive = [call for call in calls[index] if call["mode"] == 0]
                    assert len(passive) == 1 and mount["variant"] == "updated"
                    call = passive[0]
                    status = [c for c in calls[index] if c["mode"] == -1 and c["end_ns"] <= call["begin_ns"]][-1]
                    assert status["frames"] > status["copied"] >= 0
                    assert call["code"] == 0 and call["copied"] <= call["frames"]
                    assert call["copied"] >= status["frames"], "Preparation did not copy its observed prefix"
                    prepares.append(
                        dict(
                            **ident,
                            request_begin_ns=row["begin_ns"],
                            request_end_ns=row["end_ns"],
                            bytes=row["bytes"],
                            prefix_frames=status["frames"] - status["copied"],
                            checkpoint=call,
                            syncs=selected,
                        )
                    )
            report["preparation_calls"].extend(prepares)
            for sample in mount["samples"]:
                begin = round(sample["started_epoch"] * 1e9) - trace["origin_epoch_ns"]
                end = round(sample["ended_epoch"] * 1e9) - trace["origin_epoch_ns"]
                chosen = [p for p in prepares if begin <= p["request_begin_ns"] < end]
                for tail in report["sync_tails_at_least_one_second"]:
                    if all(tail[name] == ident[name] for name in ident) and begin <= tail["begin_ns"] < end:
                        tail["measured_workload"] = sample["workload"]
                if sample["workload"] == "small_file_fsync" and sample["seconds"] >= 1:
                    wal = [
                        s
                        for s in syncs
                        if s["request"] != "checkpoint_worker" and s["kind"] == "wal" and begin <= s["begin_ns"] < end
                    ]
                    database = [
                        s
                        for s in syncs
                        if s["request"] == "checkpoint_worker"
                        and s["kind"] == "database"
                        and begin <= s["begin_ns"] < end
                    ]
                    wal_overlap = [
                        s
                        for s in wal
                        if any(s["begin_ns"] < d["end_ns"] and d["begin_ns"] < s["end_ns"] for d in database)
                    ]
                    phase = next(
                        p
                        for p in findings["phases"]
                        if all(p[name] == ident[name] for name in ident) and p["workload"] == sample["workload"]
                    )
                    report["slow_full_phases"].append(
                        dict(
                            **ident,
                            seconds=sample["seconds"],
                            cpu_ns=phase["cpu_ns"],
                            begin_ns=phase["begin_ns"],
                            publication_wait_ns=phase["publication_wait_ns"],
                            pwrite_ns=phase["pwrite_ns"],
                            foreground_wal_sync_ns=distribution([s["end_ns"] - s["begin_ns"] for s in wal]),
                            worker_database_sync_ns=distribution([s["end_ns"] - s["begin_ns"] for s in database]),
                            foreground_wal_syncs_overlapping_worker_database=len(wal_overlap),
                            syncs=[*wal, *database],
                            interpretation="Overlap is observed; it does not establish the cause of slower device completion",
                        )
                    )
                report["phases"].append(
                    dict(
                        **ident,
                        workload=sample["workload"],
                        seconds=sample["seconds"],
                        preparation_calls=len(chosen),
                        preparation_checkpoint_ns=sum(
                            p["checkpoint"]["end_ns"] - p["checkpoint"]["begin_ns"] for p in chosen
                        ),
                        midstream_publications=sum(
                            all(p[name] == ident[name] for name in ident) and p["phase_begin_ns"] == begin
                            for p in findings["midstream_publications"]
                        ),
                        actual_sync_ns=sum(s["end_ns"] - s["begin_ns"] for s in syncs if begin <= s["begin_ns"] < end),
                    )
                )
    for tail in report["sync_tails_at_least_one_second"]:
        tail.setdefault("measured_workload", None)
    groups = collections.defaultdict(list)
    for phase in report["phases"]:
        groups[(phase["state"], phase["variant"], phase["workload"])].append(phase)
    for (state, variant, workload), phases in sorted(groups.items()):
        report["groups"].append(
            dict(
                state=state,
                variant=variant,
                workload=workload,
                **{
                    name: distribution([phase[name] for phase in phases])
                    for name in (
                        "seconds",
                        "preparation_calls",
                        "preparation_checkpoint_ns",
                        "midstream_publications",
                        "actual_sync_ns",
                    )
                },
            )
        )
    (out / "preparation-findings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS",
        len(report["preparation_calls"]),
        "preparations;",
        len(report["sync_tails_at_least_one_second"]),
        "syncs >=1s",
        flush=True,
    )


if __name__ == "__main__":
    main()
