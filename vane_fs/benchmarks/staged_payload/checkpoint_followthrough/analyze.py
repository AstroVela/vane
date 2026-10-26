# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Pair lookup with rename and locate CLEAN deferrals in actual WAL generations."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from prepare import digest

START = 16 * 1024 * 1024


def distribution(values):
    values = sorted(values)
    return dict(
        count=len(values),
        sum=sum(values),
        median=statistics.median(values) if values else 0,
        p99=values[min(len(values) - 1, int(len(values) * 0.99))] if values else 0,
        max=max(values, default=0),
    )


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    args = parser.parse_args()
    out = args.window.resolve()
    raw_path = out / "ext4-diagnostic/results.json"
    raw = json.loads(raw_path.read_text())
    result = dict(
        status="PASS",
        interpretation="Diagnostic intervals; inclusive stages and overlapping syncs are not additive",
        inputs={str(Path(__file__)): digest(Path(__file__)), str(raw_path): digest(raw_path)},
        mounts=[],
        phases=[],
        groups=[],
        slow_pairs=[],
        large_clean_episodes=[],
        midstream_publications=[],
        sync_tails=[],
    )
    pairs_report = []
    for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
        for mount in raw[key]:
            ident = dict(state=state, variant=mount["variant"], rep=mount["rep"])
            trace_path = Path(mount["request_costs_file"])
            trace = json.loads(trace_path.read_text())
            result["inputs"][str(trace_path)] = digest(trace_path)
            stages = trace["stages"]
            events, syncs, checkpoints = [], [], []
            for thread in trace["threads"]:
                rows = [
                    dict(zip(trace["columns"], r, strict=True), tid=thread["tid"], index=i)
                    for i, r in enumerate(thread["rows"])
                ]
                selected = collections.defaultdict(list)
                wal = collections.defaultdict(list)
                for v in thread["syncs"]:
                    c = dict(zip(trace["sync_columns"], v, strict=True), tid=thread["tid"])
                    c["request"] = rows[c["request_index"]]["name"] if c["request_index"] >= 0 else "unscoped"
                    if c["request_index"] >= 0:
                        row = rows[c["request_index"]]
                        assert row["begin_ns"] <= c["begin_ns"] <= c["end_ns"] <= row["end_ns"]
                        selected[c["request_index"]].append(c)
                    assert c["result"] == c["errno"] == 0
                    syncs.append(c)
                    if c["end_ns"] - c["begin_ns"] >= 100_000_000:
                        result["sync_tails"].append(dict(**ident, **c))
                for v in thread["checkpoints"]:
                    c = dict(
                        zip(
                            ("request_index", "mode", "frames", "copied", "code", "begin_ns", "end_ns"), v, strict=True
                        ),
                        tid=thread["tid"],
                    )
                    c["request"] = rows[c["request_index"]]["name"] if c["request_index"] >= 0 else "unscoped"
                    if c["request_index"] >= 0:
                        row = rows[c["request_index"]]
                        assert row["begin_ns"] <= c["begin_ns"] <= c["end_ns"] <= row["end_ns"]
                        wal[c["request_index"]].append(c)
                    checkpoints.append(c)
                for row in rows:
                    for operation in ("fsync", "fdatasync"):
                        index = stages.index(operation)
                        calls = [c for c in selected[row["index"]] if c["operation"] == operation]
                        assert row["counts"][index] == len(calls)
                        assert row["stage_ns"][index] == sum(c["end_ns"] - c["begin_ns"] for c in calls)
                    index = stages.index("checkpoint")
                    assert row["counts"][index] == len(wal[row["index"]])
                    assert row["stage_ns"][index] >= sum(c["end_ns"] - c["begin_ns"] for c in wal[row["index"]])
                    row["wal_calls"] = wal[row["index"]]
                events.extend(rows)
            events.sort(key=lambda r: r["begin_ns"])
            generation_resets = []
            last_frames = -1
            for row in events:
                frames = row["committed_frames"]
                if frames >= 0:
                    if last_frames >= 0 and frames < last_frames:
                        generation_resets.append(row["begin_ns"])
                    last_frames = frames
            workers = [r for r in events if r["name"] == "checkpoint_worker"]
            deferred = [r for r in workers if r["decision"][0] >= 0 and r["decision"][1] == 0 and r["decision"][5] == 1]
            large = [r for r in deferred if r["decision"][2] * 4120 + 32 >= START]
            small = [r for r in deferred if r not in large]
            entry = dict(
                **ident,
                syncs=len(syncs),
                checkpoints=len(checkpoints),
                clean_deferrals=len(deferred),
                large_generation_clean_deferrals=len(large),
                small_generation_clean_deferrals=len(small),
                decision_remaining_ns=distribution([r["decision"][6] for r in deferred]),
            )
            result["mounts"].append(entry)
            episode = []
            for row in workers:
                d = row["decision"]
                if d[0] >= 0 and d[1] == 0 and d[5] == 1:
                    episode.append(row)
                    continue
                if episode and any(r["decision"][2] * 4120 + 32 >= START for r in episode):
                    begin = episode[0]["decision"][0]
                    end = row["begin_ns"]
                    following = [
                        r for r in events if r["begin_ns"] >= begin and r["name"] in ("write", "create", "rename")
                    ]
                    result["large_clean_episodes"].append(
                        dict(
                            **ident,
                            begin_ns=begin,
                            end_ns=end,
                            decisions=[r["decision"] for r in episode],
                            exit_decision=d,
                            exit_checkpoints=row["wal_calls"],
                            overlapping_writer_admission_ns=sum(
                                r["stage_ns"][stages.index("admission")] for r in following if r["begin_ns"] <= end
                            ),
                            next_writer={k: following[0][k] for k in ("name", "begin_ns", "end_ns", "stage_ns")}
                            if following
                            else None,
                        )
                    )
                episode = []
            for sample in mount["samples"]:
                begin = round(sample["started_epoch"] * 1e9) - trace["origin_epoch_ns"]
                end = round(sample["ended_epoch"] * 1e9) - trace["origin_epoch_ns"]
                rows = [r for r in events if begin <= r["begin_ns"] < end and r["name"] != "checkpoint_worker"]
                phase = dict(
                    **ident,
                    workload=sample["workload"],
                    seconds=sample["seconds"],
                    observed_generation_resets=sum(begin <= t < end for t in generation_resets),
                    cpu_ns=sum(r["cpu_ns"] for r in rows),
                    begin_ns=sum(r["stage_ns"][stages.index("begin")] for r in rows),
                    admission_ns=sum(r["stage_ns"][stages.index("admission")] for r in rows),
                    large_clean_deferrals=sum(begin <= r["decision"][0] < end for r in large),
                    small_clean_deferrals=sum(begin <= r["decision"][0] < end for r in small),
                    worker_database_syncs=sum(
                        c["request"] == "checkpoint_worker" and c["kind"] == "database" and begin <= c["begin_ns"] < end
                        for c in syncs
                    ),
                    restart_calls=sum(
                        c["request"] == "checkpoint_worker" and c["mode"] == 2 and begin <= c["begin_ns"] < end
                        for c in checkpoints
                    ),
                    handler_counts=dict(collections.Counter(r["name"] for r in rows)),
                )
                for stage in (
                    "publication_wait",
                    "publication",
                    "payload_guard",
                    "commit",
                    "wal_hook",
                    "pread",
                    "pwrite",
                ):
                    phase[stage + "_ns"] = sum(r["stage_ns"][stages.index(stage)] for r in rows)
                for role in ("foreground", "worker"):
                    for kind in ("database", "wal", "payload", "fence"):
                        calls = [
                            c
                            for c in syncs
                            if (c["request"] == "checkpoint_worker") == (role == "worker")
                            and c["kind"] == kind
                            and begin <= c["begin_ns"] < end
                        ]
                        phase[role + "_" + kind + "_sync_ns"] = sum(c["end_ns"] - c["begin_ns"] for c in calls)
                        phase[role + "_" + kind + "_sync_count"] = len(calls)
                if sample["workload"] == "write_64":
                    for worker in workers:
                        d = worker["decision"]
                        if not (begin <= worker["begin_ns"] < end and d[1] == 1 and d[5] == 0):
                            continue
                        calls = [
                            c for c in syncs if c["tid"] == worker["tid"] and c["request_index"] == worker["index"]
                        ]
                        if not any(c["kind"] == "payload" for c in calls):
                            continue
                        delayed = [
                            r
                            for r in rows
                            if r["name"] == "write"
                            and r["stage_ns"][stages.index("publication_wait")] > 1_000_000
                            and overlap((r["begin_ns"], r["end_ns"]), (worker["begin_ns"], worker["end_ns"]))
                        ]
                        clean = [
                            w for w in workers if begin <= w["begin_ns"] < worker["begin_ns"] and w["decision"][1] == 0
                        ]
                        result["midstream_publications"].append(
                            dict(
                                **ident,
                                phase_begin_ns=begin,
                                preceding_clean=[
                                    dict(begin_ns=w["begin_ns"], decision=w["decision"], checkpoints=w["wal_calls"])
                                    for w in clean
                                ],
                                worker={k: worker[k] for k in ("begin_ns", "end_ns", "decision", "wal_calls")},
                                syncs=calls,
                                blocked_writes=[
                                    {k: r[k] for k in ("begin_ns", "end_ns", "cpu_ns", "stage_ns", "bytes")}
                                    for r in delayed
                                ],
                                stages=stages,
                            )
                        )
                if sample["workload"] == "rename":
                    lookups = [r for r in rows if r["name"] == "lookup"]
                    renames = [r for r in rows if r["name"] == "rename"]
                    assert len(lookups) == len(renames) == 128
                    pairs = []
                    for i, (lookup, rename) in enumerate(zip(lookups, renames, strict=True)):
                        assert lookup["end_ns"] <= rename["begin_ns"]
                        if i:
                            assert renames[i - 1]["end_ns"] <= lookup["begin_ns"]
                        pair = dict(
                            **ident,
                            sequence=i,
                            begin_ns=lookup["begin_ns"],
                            end_ns=rename["end_ns"],
                            elapsed_ns=rename["end_ns"] - lookup["begin_ns"],
                            lookup_ns=lookup["end_ns"] - lookup["begin_ns"],
                            rename_ns=rename["end_ns"] - rename["begin_ns"],
                            lookup_cpu_ns=lookup["cpu_ns"],
                            rename_cpu_ns=rename["cpu_ns"],
                            lookup_begin_ns=lookup["stage_ns"][stages.index("begin")],
                            rename_begin_ns=rename["stage_ns"][stages.index("begin")],
                        )
                        pairs.append(pair)
                        if pair["lookup_begin_ns"] + pair["rename_begin_ns"] >= 500_000:
                            interval = (lookup["begin_ns"], rename["end_ns"])
                            result["slow_pairs"].append(
                                dict(
                                    **pair,
                                    checkpoints=[
                                        c
                                        for c in checkpoints
                                        if c["mode"] == 2 and overlap(interval, (c["begin_ns"], c["end_ns"]))
                                    ],
                                    syncs=[
                                        c
                                        for c in syncs
                                        if c["site"] == "wal_restart"
                                        and overlap(interval, (c["begin_ns"], c["end_ns"]))
                                    ],
                                )
                            )
                    pairs_report.extend(pairs)
                    for metric in (
                        "elapsed_ns",
                        "lookup_ns",
                        "rename_ns",
                        "lookup_cpu_ns",
                        "rename_cpu_ns",
                        "lookup_begin_ns",
                        "rename_begin_ns",
                    ):
                        phase["pair_" + metric] = sum(p[metric] for p in pairs)
                    phase["slow_lookup"] = sum(p["lookup_begin_ns"] >= 500_000 for p in pairs)
                    phase["slow_rename"] = sum(p["rename_begin_ns"] >= 500_000 for p in pairs)
                result["phases"].append(phase)
    for state, variant, workload in sorted({(p["state"], p["variant"], p["workload"]) for p in result["phases"]}):
        selected = [
            p for p in result["phases"] if (p["state"], p["variant"], p["workload"]) == (state, variant, workload)
        ]
        result["groups"].append(
            dict(
                state=state,
                variant=variant,
                workload=workload,
                **{
                    k: distribution([p[k] for p in selected])
                    for k, v in selected[0].items()
                    if isinstance(v, (int, float)) and k != "rep"
                },
            )
        )
    for filename, content in (("findings.json", result), ("rename-pairs.json", pairs_report)):
        (out / filename).write_text(json.dumps(content, separators=(",", ":")) + "\n")
    print(
        "PASS",
        len(result["mounts"]),
        "mounts;",
        len(pairs_report),
        "rename pairs;",
        len(result["large_clean_episodes"]),
        "large CLEAN episodes",
        flush=True,
    )


if __name__ == "__main__":
    main()
