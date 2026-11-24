# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain complete phase comparisons and every slow rename lookup."""

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    result = dict(
        status="PASS",
        scope="Diagnostic attribution only; neither window changes the original adoption decision",
        inputs={str(Path(__file__).resolve()): digest(Path(__file__))},
        comparisons=[],
        lookup_off_cpu_threshold_ns=500_000,
        lookup_groups=[],
        slow_lookups=[],
    )
    for label, directory in (("original-control", out), ("wake-latch-causal", out / "causal")):
        inputs = {}
        for name in (
            "complete-phases.json",
            "request-costs-summary.json",
            "sync-analysis.json",
            "cpu-summary.json",
            "ext4-diagnostic/results.json",
        ):
            path = directory / name
            inputs[name] = read(path)
            result["inputs"][str(path)] = digest(path)
        for state in ("fresh", "post_gc"):
            for workload in ("random_write_4k", "small_file_fsync", "rename"):
                entry = dict(window=label, state=state, workload=workload, variants={})
                for variant in ("baseline", "updated"):
                    groups = {
                        name: next(
                            g
                            for g in value["groups"]
                            if (g["state"], g["variant"], g["workload"]) == (state, variant, workload)
                        )
                        for name, value in inputs.items()
                        if "groups" in value
                    }
                    complete = groups["complete-phases.json"]
                    request = groups["request-costs-summary.json"]
                    cpu = groups["cpu-summary.json"]
                    raw = inputs["ext4-diagnostic/results.json"]
                    samples = [
                        s
                        for m in raw["fuse" if state == "fresh" else "aged_fuse"]
                        if m["variant"] == variant
                        for s in m["samples"]
                        if s["workload"] == workload
                    ]
                    entry["variants"][variant] = dict(
                        phases=complete["phases"],
                        phase_seconds_median=complete["seconds"]["median"],
                        requests_seconds_median=statistics.median(
                            s.get("requests_seconds", s["seconds"] - s.get("final_directory_fsync_seconds", 0))
                            for s in samples
                        ),
                        final_fsync_seconds_median=statistics.median(
                            s.get("fsync_seconds", s.get("final_directory_fsync_seconds", 0)) for s in samples
                        ),
                        complete_handler_cpu_ns_median=complete["cpu_ns"]["median"],
                        handler_counts=complete["handler_counts"],
                        primary_vm_steps_median=request["vm_steps"]["median"],
                        primary_off_cpu_ns_median=request["handler_off_cpu_ns"]["median"],
                        primary_stage_ns={n: d["median"] for n, d in request["stage_ns"].items()},
                        syncs=groups["sync-analysis.json"]["by_role_kind"],
                        sampled_frequency_khz_median=cpu["frequency_khz_median"],
                        observed_runqueue_ns_median=cpu["runqueue_ns_median"],
                    )
                entry["candidate_phase_seconds_ratio"] = (
                    entry["variants"]["updated"]["phase_seconds_median"]
                    / entry["variants"]["baseline"]["phase_seconds_median"]
                )
                result["comparisons"].append(entry)
        groups = collections.defaultdict(list)
        raw = inputs["ext4-diagnostic/results.json"]
        for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
            for mount in raw[key]:
                path = Path(mount["request_costs_file"])
                result["inputs"][str(path)] = digest(path)
                trace = read(path)
                events, restarts = [], []
                for thread in trace["threads"]:
                    rows = [dict(zip(trace["columns"], r, strict=True), tid=thread["tid"]) for r in thread["rows"]]
                    events.extend(rows)
                    for values in thread["syncs"]:
                        call = dict(zip(trace["sync_columns"], values, strict=True), tid=thread["tid"])
                        if call["site"] == "wal_restart":
                            assert rows[call["request_index"]]["name"] == "checkpoint_worker"
                            assert call["result"] == call["errno"] == 0
                            restarts.append(call)
                phase = next(s for s in mount["samples"] if s["workload"] == "rename")
                begin = round(phase["started_epoch"] * 1e9) - trace["origin_epoch_ns"]
                end = round(phase["ended_epoch"] * 1e9) - trace["origin_epoch_ns"]
                lookups = [r for r in events if r["name"] == "lookup" and begin <= r["begin_ns"] < end]
                assert len(lookups) == 128
                slow = [
                    r
                    for r in lookups
                    if r["end_ns"] - r["begin_ns"] - r["cpu_ns"] > result["lookup_off_cpu_threshold_ns"]
                ]
                for row in slow:
                    overlaps = [c for c in restarts if c["begin_ns"] < row["end_ns"] and c["end_ns"] > row["begin_ns"]]
                    result["slow_lookups"].append(
                        dict(
                            window=label,
                            state=state,
                            variant=mount["variant"],
                            rep=mount["rep"],
                            **{k: v for k, v in row.items() if k != "stage_ns"},
                            stage_ns=dict(zip(trace["stages"], row["stage_ns"], strict=True)),
                            restart_sync_overlaps=overlaps,
                        )
                    )
                groups[(state, mount["variant"])].append(
                    dict(
                        count=len(lookups),
                        slow_count=len(slow),
                        begin_ns=sum(r["stage_ns"][trace["stages"].index("begin")] for r in lookups),
                        statement_step_ns=sum(r["stage_ns"][trace["stages"].index("statement_step")] for r in lookups),
                    )
                )
        for (state, variant), rows in sorted(groups.items()):
            result["lookup_groups"].append(
                dict(
                    window=label,
                    state=state,
                    variant=variant,
                    lookups=sum(r["count"] for r in rows),
                    slow_lookups=sum(r["slow_count"] for r in rows),
                    begin_ns_median=statistics.median(r["begin_ns"] for r in rows),
                    statement_step_ns_median=statistics.median(r["statement_step_ns"] for r in rows),
                )
            )
    result["every_slow_lookup_overlaps_restart_sync"] = all(r["restart_sync_overlaps"] for r in result["slow_lookups"])
    (out / "findings.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("diagnostic findings PASS", len(result["slow_lookups"]), "slow lookups retained", flush=True)


if __name__ == "__main__":
    main()
