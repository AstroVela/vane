# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Extract request shapes and deterministic worst-case examples from the full diagnosis."""

import argparse
import collections
import json
from pathlib import Path

from prepare import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    path = out / "analysis.json"
    analysis = json.loads(path.read_text())
    assert analysis["status"] == "PASS"
    names = analysis["stages"]
    all_requests, shapes, phases = [], collections.defaultdict(list), []
    slow_categories = collections.Counter()
    for mount in analysis["mounts"]:
        for phase in mount["phases"]:
            shapes[phase["workload"]].append(phase["fuse_request_sizes"])
            applications = phase["application_requests"]
            phases.append(
                dict(
                    state=mount["state"],
                    variant=mount["variant"],
                    rep=mount["rep"],
                    workload=phase["workload"],
                    wall_ns=phase["wall_ns"],
                    handler_cpu_ns=sum(r["handler_cpu_ns"] for r in applications),
                    checkpoint_wait_ns=phase["stage_counters"][names.index("checkpoint_wait")][2],
                    foreground_sync_ns=sum(
                        phase["stage_counters"][names.index(name)][2] for name in ("fsync", "fdatasync")
                    ),
                    outside_handlers_ns=sum(r["outside_handler_ns"] for r in applications),
                    scheduler=phase["scheduler"],
                )
            )
        for item in mount["slow_roots"]:
            root = item["root"]
            phase = next((p for p in mount["phases"] if p["begin_ns"] <= root[2] <= p["end_ns"]), None)
            if root[0] not in ("write", "fsync") or phase is None:
                continue
            exclusive = {name: root[10][i][2] for i, name in enumerate(names)}
            request = dict(
                state=mount["state"],
                variant=mount["variant"],
                rep=mount["rep"],
                workload=phase["workload"],
                tid=item["tid"],
                root_id=root[1],
                operation=root[0],
                offset=root[5],
                bytes=root[6],
                begin_ns=root[2],
                end_ns=root[3],
                wall_ns=root[3] - root[2],
                caller_cpu_ns=root[4],
                exclusive_wall_ns=exclusive,
                exclusive_cpu_ns={name: root[10][i][4] for i, name in enumerate(names)},
                actual_events=item["events"],
                waits=item["waits"],
                scheduler=item["scheduler"],
            )
            all_requests.append(request)
            if request["wall_ns"] >= 1000000000:
                costs = dict(
                    checkpoint_wait=exclusive["checkpoint_wait"],
                    foreground_sync=exclusive["fsync"] + exclusive["fdatasync"],
                    other=sum(
                        v for name, v in exclusive.items() if name not in ("checkpoint_wait", "fsync", "fdatasync")
                    ),
                )
                slow_categories[max(costs, key=costs.get)] += 1
    request_shapes = {}
    for workload, values in shapes.items():
        assert all(value == values[0] for value in values)
        request_shapes[workload] = dict(mounts=len(values), histogram=values[0], handlers=sum(values[0].values()))
    examples = {
        "worker_checkpoint_wait": max(all_requests, key=lambda r: r["exclusive_wall_ns"]["checkpoint_wait"]),
        "foreground_preparation": max(
            (r for r in all_requests if r["operation"] == "write"),
            key=lambda r: r["exclusive_wall_ns"]["fsync"] + r["exclusive_wall_ns"]["fdatasync"],
        ),
        "final_fsync": max((r for r in all_requests if r["operation"] == "fsync"), key=lambda r: r["wall_ns"]),
    }
    report = dict(
        status="PASS",
        inputs={str(p): digest(p) for p in [path, Path(__file__)]},
        mounts=len(analysis["mounts"]),
        phases=len(phases),
        total_roots=sum(m["validation"]["roots"] for m in analysis["mounts"]),
        total_events=sum(m["validation"]["events"] for m in analysis["mounts"]),
        total_slow_roots_100ms=sum(len(m["slow_roots"]) for m in analysis["mounts"]),
        total_slow_syscalls_100ms=sum(len(m["slow_syscalls"]) for m in analysis["mounts"]),
        request_shapes=request_shapes,
        large_write_handlers_at_least_1s=dict(slow_categories),
        phases_at_least_4s=[p for p in phases if p["wall_ns"] >= 4000000000],
        all_phases=phases,
        examples=examples,
    )
    save(out / "findings.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "mounts",
                    "phases",
                    "total_roots",
                    "total_events",
                    "total_slow_roots_100ms",
                    "total_slow_syscalls_100ms",
                    "request_shapes",
                    "large_write_handlers_at_least_1s",
                )
            },
            indent=2,
        )
    )
    for name, example in examples.items():
        print(
            name,
            example["state"],
            example["variant"],
            example["rep"],
            example["operation"],
            example["wall_ns"] / 1e9,
            "seconds",
            example["caller_cpu_ns"] / 1e6,
            "CPU ms",
        )


if __name__ == "__main__":
    main()
