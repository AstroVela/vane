# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute the cleanup improvement separately from uninstrumented adoption rates."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residual(sample):
    return sample["seconds"] - sum(sample["write_latencies"]) - sample["fsync_seconds"] - sample["close_seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    out, diagnostic = args.artifacts.resolve(), args.diagnostic.resolve()
    result = dict(
        status="PASS",
        inputs={},
        native={},
        plain=[],
        diagnostic=[],
        groups=[],
        residual_scope="Complete time less write calls, final fsync and close: includes create/queue and loop overhead",
        diagnostic_scope="Inclusive handler costs; instrumented wall time does not decide adoption",
    )
    for variant in ("baseline", "updated"):
        path = out / (variant + "-volume.log")
        result["inputs"][str(path)] = digest(path)
        result["native"][variant] = json.loads(path.read_text().splitlines()[0])
    for medium in ("tmpfs", "ext4"):
        path = out / (medium + "-plain/results.json")
        result["inputs"][str(path)] = digest(path)
        raw = json.loads(path.read_text())
        for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
            for mount in raw[key]:
                sample = next(s for s in mount["samples"] if s["workload"] == "write_64")
                result["plain"].append(
                    dict(
                        medium=medium,
                        state=state,
                        variant=mount["variant"],
                        rep=mount["rep"],
                        create_and_loop_seconds=residual(sample),
                        complete_seconds=sample["seconds"],
                        write_calls_seconds=sum(sample["write_latencies"]),
                        fsync_seconds=sample["fsync_seconds"],
                        close_seconds=sample["close_seconds"],
                    )
                )
    path = diagnostic / "ext4-diagnostic/results.json"
    result["inputs"][str(path)] = digest(path)
    raw = json.loads(path.read_text())
    for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
        for mount in raw[key]:
            path = Path(mount["request_costs_file"])
            result["inputs"][str(path)] = digest(path)
            trace = json.loads(path.read_text())
            events = [dict(zip(trace["columns"], row, strict=True)) for t in trace["threads"] for row in t["rows"]]
            first = next(s for s in mount["samples"] if s["workload"] == "stream_256")
            second = next(s for s in mount["samples"] if s["workload"] == "write_64")
            after_first = round(first["ended_epoch"] * 10**9) - trace["origin_epoch_ns"]
            begin = round(second["started_epoch"] * 10**9) - trace["origin_epoch_ns"]
            end = round(second["ended_epoch"] * 10**9) - trace["origin_epoch_ns"]
            cleanup = [
                r
                for r in events
                if r["name"] in ("forget", "forget_multi")
                and r["vm_steps"] > 1000000
                and after_first <= r["begin_ns"] < end
            ]
            assert len(cleanup) == 1, (state, mount["variant"], mount["rep"], len(cleanup))
            event = cleanup[0]
            result["diagnostic"].append(
                dict(
                    state=state,
                    variant=mount["variant"],
                    rep=mount["rep"],
                    row=event,
                    handler_seconds=(event["end_ns"] - event["begin_ns"]) / 10**9,
                    handler_cpu_seconds=event["cpu_ns"] / 10**9,
                    handler_tail_after_write64_begin_seconds=max(0, event["end_ns"] - begin) / 10**9,
                    create_and_loop_seconds=residual(second),
                )
            )
    for kind in ("plain", "diagnostic"):
        combinations = sorted({(r.get("medium", "ext4"), r["state"], r["variant"]) for r in result[kind]})
        for medium, state, variant in combinations:
            rows = [
                r
                for r in result[kind]
                if (r.get("medium", "ext4"), r["state"], r["variant"]) == (medium, state, variant)
            ]
            fields = (
                ("create_and_loop_seconds", "complete_seconds", "write_calls_seconds", "fsync_seconds")
                if kind == "plain"
                else (
                    "create_and_loop_seconds",
                    "handler_seconds",
                    "handler_cpu_seconds",
                    "handler_tail_after_write64_begin_seconds",
                )
            )
            result["groups"].append(
                dict(
                    kind=kind,
                    medium=medium,
                    state=state,
                    variant=variant,
                    samples=len(rows),
                    medians={f: statistics.median(r[f] for r in rows) for f in fields},
                    vm_steps=[r["row"]["vm_steps"] for r in rows] if kind == "diagnostic" else [],
                )
            )
    (out / "orphan-analysis.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("orphan analysis PASS", len(result["plain"]), "plain mounts", len(result["diagnostic"]), "diagnostic mounts")


if __name__ == "__main__":
    main()
