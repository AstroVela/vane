# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare complete lifecycle and write-phase counts without timing claims."""

import argparse
import json
import statistics
import time
from pathlib import Path

from generate import digest, save


def metrics(data, phases):
    selected = [p for p in data["phases"] if p["name"] in phases]
    result = {
        f"{role}_{key}": sum(p["io"][role][index] for p in selected)
        for role in data["phases"][0]["io"]
        for index, key in enumerate(data["io_columns"])
    }
    result.update(
        {
            f"{key}_{column}": sum(p[key][index] for p in selected)
            for key in ("steps", "vm_steps")
            for index, column in enumerate(data["sql_columns"])
        }
    )
    result["direct_vm_total"] = sum(sum(p["vm_steps"]) for p in selected)
    result.update(
        {
            "checkpoint_" + mode: sum(p["checkpoint_calls"][i] for p in selected)
            for i, mode in enumerate(data["checkpoint_columns"])
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    started = time.time()
    plan = json.loads((out / "count-plan.json").read_text())
    counts = json.loads((out / "counts.json").read_text())
    assert counts["status"] == "PASS" and counts["host_monitor_stopped"]
    assert len(counts["runs"]) == len(plan["runs"]) == 30
    assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    hashes, rows, summaries = {}, [], []
    for run in counts["runs"]:
        data = run["result"]
        identity = str(run["mib"])
        hashes.setdefault(identity, data["content_fnv1a64"])
        assert hashes[identity] == data["content_fnv1a64"]
        assert data["content_bytes_verified"] == run["mib"] * 1048576
        expected_requests = (
            run["mib"]
            if run["shape"] == "aligned"
            else run["mib"] * 2
            if run["shape"] == "fuse1"
            else run["mib"] + (run["mib"] + 7) // 8
        )
        assert data["requests"] == expected_requests
        rows.append(
            dict(
                mib=run["mib"],
                shape=run["shape"],
                variant=run["variant"],
                rep=run["rep"],
                validation=run["validation"],
                versions=data["versions"],
                payloads=data["payloads"],
                external_payloads=data["external_payloads"],
                logical_blocks=data["logical_blocks"],
                lifecycle=metrics(data, ("setup", "writes", "barrier", "validation", "close")),
                write_and_barrier=metrics(data, ("writes", "barrier")),
                writes=metrics(data, ("writes",)),
                barrier=metrics(data, ("barrier",)),
            )
        )
    for mib in (64, 256):
        for shape in ("aligned", "fuse8", "fuse1"):
            variants = {}
            for variant in ("baseline", "extent"):
                selected = [r for r in rows if (r["mib"], r["shape"], r["variant"]) == (mib, shape, variant)]
                assert len(selected) == 2 and {r["rep"] for r in selected} == {0, 1}
                aggregate = {
                    key: statistics.median(r[key] for r in selected)
                    for key in ("versions", "payloads", "external_payloads", "logical_blocks")
                }
                aggregate.update(
                    {
                        scope: {key: statistics.median(r[scope][key] for r in selected) for key in selected[0][scope]}
                        for scope in ("lifecycle", "write_and_barrier", "writes", "barrier")
                    }
                )
                variants[variant] = aggregate
            baseline, extent = variants["baseline"], variants["extent"]
            reductions = {key: 1 - extent[key] / baseline[key] for key in ("versions", "payloads", "external_payloads")}
            for scope in ("lifecycle", "write_and_barrier", "writes"):
                reductions[scope] = {
                    key: 1 - extent[scope][key] / baseline[scope][key]
                    for key in (
                        "wal_successful_bytes",
                        "database_successful_bytes",
                        "payload_successful_bytes",
                        "direct_vm_total",
                        "vm_steps_insert_versions",
                        "vm_steps_insert_payloads",
                    )
                    if baseline[scope][key]
                }
            summaries.append(dict(mib=mib, shape=shape, median=variants, reduction=reductions))
    inputs = dict(plan["inputs"])
    inputs.update({str(p): digest(p) for p in [Path(__file__), out / "counts.json", out / "count-plan.json"]})
    save(
        out / "analysis.json",
        dict(
            status="PASS",
            started_epoch=started,
            ended_epoch=time.time(),
            inputs=inputs,
            count_runs=30,
            matched_ab_ba_repetitions=2,
            full_content_checks=30,
            content_fnv1a64=hashes,
            expected_content_sha256=plan["expected_content_sha256"],
            rows=rows,
            summaries=summaries,
            interpretation="Count-only native replay. No latency/throughput adoption claim or upper bound on payload/device sync duration.",
            limitations=[
                "Fresh workspace sequential writes only; no full FUSE run or aged/fragmented throughput screen.",
                "Logical block mutations, truncate, merge and tree operations can materialize scalar ranges.",
                "Direct sqlite3_step VM counts exclude SQLite-internal sqlite3_exec statements.",
                "All successful pwrite/pwrite64 bytes are counted, including repeated/overwritten WAL frames, not physical device writes.",
                "All samples retained; worker scheduling changes checkpoint and sync counts with identical policies.",
            ],
        ),
    )
    for row in summaries:
        base, extent = row["median"]["baseline"], row["median"]["extent"]
        print(
            row["mib"],
            row["shape"],
            "version rows",
            base["versions"],
            "->",
            extent["versions"],
            "WAL bytes",
            base["lifecycle"]["wal_successful_bytes"],
            "->",
            extent["lifecycle"]["wal_successful_bytes"],
            "write VM reduction",
            f"{100 * row['reduction']['writes']['direct_vm_total']:.2f}%",
        )


if __name__ == "__main__":
    main()
