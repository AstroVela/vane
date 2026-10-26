# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Attribute each sync, verify aggregates and retain every slow overlapping call."""

import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path

KINDS = ("payload", "wal", "fence", "database", "directory", "other")
ROLES = ("foreground", "barrier", "worker", "unscoped")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def distribution(values):
    values = sorted(values)
    return dict(
        count=len(values),
        sum=sum(values),
        median=statistics.median(values) if values else 0,
        p99=values[min(len(values) - 1, int(len(values) * 0.99))] if values else 0,
        max=max(values, default=0),
    )


def union(intervals):
    end, total = -1, 0
    for begin, current_end in sorted(intervals):
        total += max(0, current_end - max(begin, end))
        end = max(end, current_end)
    return total


def totals(calls, begin=None, end=None):
    intervals = collections.defaultdict(list)
    for call in calls:
        mount = tuple(call.get(k) for k in ("state", "variant", "rep"))
        intervals[mount].append(
            (
                max(call["begin_ns"], begin) if begin is not None else call["begin_ns"],
                min(call["end_ns"], end) if end is not None else call["end_ns"],
            )
        )
    return dict(
        duration_ns=distribution([c["end_ns"] - c["begin_ns"] for c in calls]),
        cpu_ns=distribution([c["cpu_ns"] for c in calls]),
        probe_ns=distribution([c["probe_ns"] for c in calls]),
        interval_union_ns=sum(union(rows) for rows in intervals.values()),
        file_size_min=min((c["file_size"] for c in calls), default=-1),
        file_size_max=max((c["file_size"] for c in calls), default=-1),
        at_least_100ms=sum(c["end_ns"] - c["begin_ns"] >= 100_000_000 for c in calls),
        at_least_1s=sum(c["end_ns"] - c["begin_ns"] >= 1_000_000_000 for c in calls),
    )


def source_proof(out, prior):
    build, old = read(out / "build.json"), read(prior / "build.json")
    proof = dict(status="PASS", variants={}, changed_runtime_logic=False)
    for name, inputs in build["source_inputs"].items():
        target, reference = out / "diagnostic" / name, prior / "diagnostic" / name
        assert inputs.keys() == old["source_inputs"][name].keys()
        entries = {}
        for relative, expected in inputs.items():
            path, before = target / relative, reference / relative
            assert digest(path) == expected and digest(before) == old["source_inputs"][name][relative]
            source = path.read_bytes()
            if relative in ("src/staging.hpp", "src/payload_file.hpp"):
                source = re.sub(rb"\n\t\trequest_trace::SyncSite sync_site\([^\n]*\);", b"", source)
                if relative == "src/payload_file.hpp":
                    prefix = b'#include "request_trace.hpp"\n'
                    assert source.startswith(prefix)
                    source = source[len(prefix) :]
            if relative != "src/request_trace.hpp":
                assert source == before.read_bytes(), relative
            entries[relative] = dict(
                sha256=expected,
                prior_sha256=digest(before),
                equivalence="recorder header"
                if relative == "src/request_trace.hpp"
                else "byte-identical after removing sync-site tags",
            )
        proof["variants"][name] = entries
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--prior-diagnostic", type=Path, required=True)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    write(out / "source-proof.json", source_proof(out, args.prior_diagnostic.resolve()))
    raw_path = out / "ext4-diagnostic/results.json"
    raw = read(raw_path)
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    assert read(out / "input-check.json")["status"] == read(out / "sync-tests.json")["status"] == "PASS"
    result = dict(
        status="PASS",
        scope="Instrumented sync costs only; overlapping calls are not additive; no adoption decision",
        inputs={str(raw_path): digest(raw_path), str(Path(__file__)): digest(Path(__file__))},
        mounts=[],
        groups=[],
        kind_totals=[],
        site_totals=[],
        aggregate_reconciliation=True,
        all_slow_calls_retained=True,
        slow_threshold_ns=100_000_000,
    )
    phases, slow_calls, all_calls = [], [], []
    for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
        for mount in raw[key]:
            trace_path = Path(mount["request_costs_file"])
            trace = read(trace_path)
            result["inputs"][str(trace_path)] = digest(trace_path)
            calls = []
            for thread in trace["threads"]:
                rows = [dict(zip(trace["columns"], row, strict=True)) for row in thread["rows"]]
                parents = collections.defaultdict(list)
                for index, values in enumerate(thread["syncs"]):
                    call = dict(zip(trace["sync_columns"], values, strict=True))
                    assert call["kind"] in KINDS and call["operation"] in ("fsync", "fdatasync")
                    assert call["result"] == call["errno"] == 0
                    assert call["end_ns"] >= call["begin_ns"] and call["cpu_ns"] >= 0 and call["probe_ns"] >= 0
                    call.update(tid=thread["tid"], id=f"{thread['tid']}:{index}")
                    parent = call["request_index"]
                    if parent < 0:
                        call.update(request="unscoped", role="unscoped")
                    else:
                        row = rows[parent]
                        assert row["begin_ns"] <= call["begin_ns"] <= call["end_ns"] <= row["end_ns"]
                        name = row["name"]
                        role = (
                            "worker" if name == "checkpoint_worker" else "barrier" if name == "fsync" else "foreground"
                        )
                        call.update(
                            request=name,
                            role=role,
                            request_begin_ns=row["begin_ns"],
                            request_end_ns=row["end_ns"],
                            request_cpu_ns=row["cpu_ns"],
                        )
                        parents[parent].append(call)
                    calls.append(call)
                for index, row in enumerate(rows):
                    for operation in ("fsync", "fdatasync"):
                        stage = trace["stages"].index(operation)
                        selected = [c for c in parents[index] if c["operation"] == operation]
                        assert row["counts"][stage] == len(selected)
                        assert row["stage_ns"][stage] == sum(c["end_ns"] - c["begin_ns"] for c in selected)
            identity = dict(state=state, variant=mount["variant"], rep=mount["rep"])
            bounds = []
            for phase in mount["samples"]:
                begin = round(phase["started_epoch"] * 10**9) - trace["origin_epoch_ns"]
                end = round(phase["ended_epoch"] * 10**9) - trace["origin_epoch_ns"]
                bounds.append((phase["workload"], begin, end))
                selected = [c for c in calls if c["begin_ns"] < end and c["end_ns"] > begin]
                phases.append(
                    dict(
                        **identity,
                        workload=phase["workload"],
                        begin_ns=begin,
                        end_ns=end,
                        phase_seconds=phase["seconds"],
                        syncs=totals(selected, begin, end),
                        by_role_kind=[
                            dict(
                                role=role,
                                kind=kind,
                                **totals([c for c in selected if c["role"] == role and c["kind"] == kind], begin, end),
                            )
                            for role in ROLES
                            for kind in KINDS
                        ],
                        by_site=[
                            dict(site=site, **totals([c for c in selected if c["site"] == site], begin, end))
                            for site in sorted({c["site"] for c in selected})
                        ],
                    )
                )
            assert mount["inspection"]["quick_check"] == [["ok"]] and mount["inspection"]["format"] == [[1004, 4096]]
            for call in calls:
                all_calls.append(dict(**identity, **call))
                if call["end_ns"] - call["begin_ns"] < result["slow_threshold_ns"]:
                    continue
                overlapping = []
                for other in calls:
                    overlap = min(call["end_ns"], other["end_ns"]) - max(call["begin_ns"], other["begin_ns"])
                    if other["id"] != call["id"] and overlap > 0:
                        overlapping.append(
                            {k: other[k] for k in ("id", "kind", "site", "operation", "role", "begin_ns", "end_ns")}
                            | dict(overlap_ns=overlap)
                        )
                slow_calls.append(
                    dict(
                        **identity,
                        **call,
                        workloads=[
                            name for name, begin, end in bounds if call["begin_ns"] < end and call["end_ns"] > begin
                        ],
                        overlapping=overlapping,
                    )
                )
            result["mounts"].append(
                dict(
                    **identity,
                    calls=len(calls),
                    slow_calls=sum(c["end_ns"] - c["begin_ns"] >= result["slow_threshold_ns"] for c in calls),
                    probe_ns=distribution([c["probe_ns"] for c in calls]),
                )
            )
    for state, variant, workload in sorted({(p["state"], p["variant"], p["workload"]) for p in phases}):
        selected = [p for p in phases if (p["state"], p["variant"], p["workload"]) == (state, variant, workload)]
        buckets = []
        for role in ROLES:
            for kind in KINDS:
                rows = [next(b for b in p["by_role_kind"] if (b["role"], b["kind"]) == (role, kind)) for p in selected]
                buckets.append(
                    dict(
                        role=role,
                        kind=kind,
                        duration_sum_ns_median=statistics.median(r["duration_ns"]["sum"] for r in rows),
                        call_count_median=statistics.median(r["duration_ns"]["count"] for r in rows),
                        max_ns=max(r["duration_ns"]["max"] for r in rows),
                        at_least_1s=sum(r["at_least_1s"] for r in rows),
                    )
                )
        result["groups"].append(
            dict(
                state=state,
                variant=variant,
                workload=workload,
                phases=len(selected),
                phase_seconds=distribution([p["phase_seconds"] for p in selected]),
                by_role_kind=buckets,
            )
        )
    for kind in KINDS:
        result["kind_totals"].append(dict(kind=kind, **totals([c for c in all_calls if c["kind"] == kind])))
    for site in sorted({c["site"] for c in all_calls}):
        result["site_totals"].append(dict(site=site, **totals([c for c in all_calls if c["site"] == site])))
    result.update(
        total_calls=len(all_calls),
        total_slow_calls=len(slow_calls),
        total_at_least_1s=sum(c["end_ns"] - c["begin_ns"] >= 1_000_000_000 for c in all_calls),
        probe_ns=distribution([c["probe_ns"] for c in all_calls]),
    )
    write(out / "sync-analysis.json", result)
    write(out / "sync-phases.json", phases)
    write(out / "slow-syncs.json", slow_calls)
    print(
        "sync attribution PASS",
        len(result["mounts"]),
        "mounts",
        len(all_calls),
        "calls",
        len(slow_calls),
        "slow calls",
        flush=True,
    )


if __name__ == "__main__":
    main()
