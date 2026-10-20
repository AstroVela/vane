# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile typed syncs and summarize a native progress screen without adoption."""

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    raw_path = out / "probe-results.json"
    raw = json.loads(raw_path.read_text())
    plan = json.loads((out / "probe-plan.json").read_text())
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    assert len(raw["runs"]) == plan["runs"] == 54
    result = dict(
        status="PASS",
        scope=plan["scope"],
        no_adoption_decision=True,
        groups=[],
        checks=[],
        phases=[],
        inputs={str(p): digest(p) for p in (raw_path, out / "probe-plan.json", Path(__file__).resolve())},
        sync_calls=0,
        slow_calls=[],
        aggregate_reconciliation=True,
    )
    for run in raw["runs"]:
        path = Path(run["trace_path"])
        result["inputs"][str(path)] = digest(path)
        trace = json.loads(path.read_text())
        syncs, rows = [], []
        for thread in trace["threads"]:
            events = [dict(zip(trace["columns"], values, strict=True)) for values in thread["rows"]]
            children = collections.defaultdict(list)
            for values in thread["syncs"]:
                call = dict(zip(trace["sync_columns"], values, strict=True))
                assert call["kind"] in ("payload", "wal", "fence", "database", "directory", "other")
                assert call["result"] == call["errno"] == 0 and call["end_ns"] >= call["begin_ns"]
                call["role"] = "foreground" if thread["tid"] == run["main_tid"] else "background"
                parent = call["request_index"]
                if parent >= 0:
                    event = events[parent]
                    assert event["begin_ns"] <= call["begin_ns"] <= call["end_ns"] <= event["end_ns"]
                    children[parent].append(call)
                syncs.append(call)
            for index, row in enumerate(events):
                for name in ("fsync", "fdatasync"):
                    stage = trace["stages"].index(name)
                    selected = [c for c in children[index] if c["operation"] == name]
                    assert row["counts"][stage] == len(selected)
                    assert row["stage_ns"][stage] == sum(c["end_ns"] - c["begin_ns"] for c in selected)
            rows.extend(events)
        result["sync_calls"] += len(syncs)
        identity = {name: run[name] for name in ("state", "variant", "rep")}
        for call in syncs:
            if call["end_ns"] - call["begin_ns"] >= 100_000_000:
                result["slow_calls"].append(dict(**identity, **call))
        for phase in run["phases"]:
            name = phase["workload"]
            events = [r for r in rows if r["name"] == name and phase["begin_ns"] <= r["begin_ns"] < phase["end_ns"]]
            barriers = [
                r for r in rows if r["name"] == "fsync" and phase["begin_ns"] <= r["begin_ns"] < phase["end_ns"]
            ]
            assert len(events) == 128 and len(barriers) == 1
            selected = [c for c in syncs if c["begin_ns"] < phase["end_ns"] and c["end_ns"] > phase["begin_ns"]]
            buckets = []
            for role, kind in sorted({(c["role"], c["kind"]) for c in selected}):
                calls = [c for c in selected if (c["role"], c["kind"]) == (role, kind)]
                buckets.append(
                    dict(
                        role=role,
                        kind=kind,
                        count=len(calls),
                        inclusive_ns=sum(c["end_ns"] - c["begin_ns"] for c in calls),
                    )
                )
            result["phases"].append(
                dict(
                    **identity,
                    workload=name,
                    seconds=phase["seconds"],
                    final_barrier_seconds=phase["fsync_seconds"],
                    foreground_cpu_ns=sum(r["cpu_ns"] for r in events),
                    syncs=buckets,
                )
            )
    for state, variant, name in sorted({(p["state"], p["variant"], p["workload"]) for p in result["phases"]}):
        selected = [p for p in result["phases"] if (p["state"], p["variant"], p["workload"]) == (state, variant, name)]
        assert len(selected) == 6
        result["groups"].append(
            dict(
                state=state,
                variant=variant,
                workload=name,
                runs=6,
                median_ops_per_second=statistics.median(128 / p["seconds"] for p in selected),
                median_final_barrier_ms=statistics.median(p["final_barrier_seconds"] for p in selected) * 1000,
                median_foreground_cpu_ms=statistics.median(p["foreground_cpu_ns"] for p in selected) / 1e6,
                background_wal_sync_counts=[
                    sum(b["count"] for b in p["syncs"] if b["role"] == "background" and b["kind"] == "wal")
                    for p in selected
                ],
            )
        )
    for state in plan["states"]:
        for name in ("mkdir", "rename", "rmdir"):
            groups = {g["variant"]: g for g in result["groups"] if (g["state"], g["workload"]) == (state, name)}
            ratio = groups["updated"]["median_ops_per_second"] / groups["orphan-control"]["median_ops_per_second"]
            result["checks"].append(
                dict(
                    state=state,
                    workload=name,
                    check="complete_phase_to_orphan",
                    ratio=ratio,
                    passed=ratio
                    >= plan[
                        "reused_complete_phase_reference_ratio_min"
                        if state == "reused"
                        else "cold_and_retained_complete_phase_reference_ratio_min"
                    ],
                )
            )
            if state == "reused":
                ratio = (
                    groups["updated"]["median_final_barrier_ms"] / groups["restart-control"]["median_final_barrier_ms"]
                )
                result["checks"].append(
                    dict(
                        state=state,
                        workload=name,
                        check="final_barrier_to_restart",
                        ratio=ratio,
                        passed=ratio <= plan["reused_final_barrier_ratio_to_restart_max"],
                    )
                )
    result["screen_passed"] = all(c["passed"] for c in result["checks"])
    (out / "probe-summary.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("native screen", "PASS" if result["screen_passed"] else "REJECT", "syncs", result["sync_calls"], flush=True)
    for check in result["checks"]:
        if not check["passed"]:
            print(check, flush=True)


if __name__ == "__main__":
    main()
