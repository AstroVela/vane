# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep first-overwrite identities, p99 tails and their native wait evidence."""

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

from prepare import digest, save, validate

HERE = Path(__file__).resolve().parent


def distribution(values):
    values = sorted(values)
    position = (len(values) - 1) * 0.99
    lower = int(position)
    return dict(
        count=len(values),
        median=statistics.median(values),
        p99=values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower),
        max=values[-1],
        sum=sum(values),
    )


def overlap(first, second):
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def main():
    out, started = Path(sys.argv[1]).resolve(), time.time()
    findings = json.loads((out / "findings.json").read_text())
    cases = json.loads((out / "requests.json").read_text())
    analysis = json.loads((out / "analysis.json").read_text())
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert findings["status"] == raw["status"] == "PASS" and raw["ended_epoch"] < started
    assert len(findings["original_plain_failed_checks"]) == 9
    helper = HERE.parent / "request_costs/analyze_cpu.py"
    spec = importlib.util.spec_from_file_location("scheduler_costs", helper)
    cpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu)
    report = dict(
        status="PASS",
        started_epoch=started,
        inputs={
            str(path): digest(path) for path in [Path(__file__), helper, out / "findings.json", out / "requests.json"]
        },
        original_plain_failed_checks=findings["original_plain_failed_checks"],
        diagnostic_only=True,
        native_policy_unchanged=True,
        historical_first_overwrite_ns=1617439321,
        historical_sample_cannot_be_retroactively_attributed=True,
        groups=[],
        first_overwrites=[],
        tail_cases=[],
        worker_checks=[],
        mount_checks=[],
    )
    first = {phase["cases"][0] for phase in findings["phases"] if phase["workload"] == "random_write_4k"}
    for state in ("fuse", "aged_fuse"):
        for variant in ("baseline", "updated"):
            for name in ("first_overwrite", "all_overwrites", "random_final_fsync", "small_file_fsync"):
                selected = [
                    (index, case)
                    for index, case in enumerate(cases)
                    if (case["state"], case["variant"]) == (state, variant)
                    and (
                        name == "first_overwrite"
                        and index in first
                        or name == "all_overwrites"
                        and case["workload"] == "random_write_4k"
                        and case["interval"]["kind"] == "write"
                        or name == "random_final_fsync"
                        and case["workload"] == "random_write_4k"
                        and case["interval"]["kind"] == "fsync"
                        or name == "small_file_fsync"
                        and case["workload"] == name
                    )
                ]
                report["groups"].append(
                    dict(
                        state=state,
                        variant=variant,
                        workload=name,
                        wall_ns=distribution([case["wall_ns"] for _, case in selected]),
                        handler_cpu_ns=distribution([case["handler_cpu_ns"] for _, case in selected]),
                        own_sync_ns_by_file={
                            file: sum(case["own_sync_ns_by_file"][file] for _, case in selected)
                            for file in ("payload", "wal", "database", "fence")
                        },
                        exclusive_wall_ns={
                            stage: sum(case["exclusive_wall_ns"][stage] for _, case in selected)
                            for stage in findings["stages"]
                        },
                        cases_ge_20ms=[index for index, case in selected if case["wall_ns"] >= 20000000],
                        p99_bracket_case_indices=[
                            selected_index
                            for selected_index, _ in sorted(selected, key=lambda pair: pair[1]["wall_ns"])[
                                int((len(selected) - 1) * 0.99) : int((len(selected) - 1) * 0.99) + 2
                            ]
                        ],
                    )
                )
    for mount in analysis["mounts"]:
        label = {key: mount[key] for key in ("state", "variant", "rep")}
        original = next(
            row for row in raw[mount["state"]] if (row["variant"], row["rep"]) == (mount["variant"], mount["rep"])
        )
        trace = json.loads(Path(mount["trace_file"]).read_text())
        stats = validate(trace)
        roots = {(thread["tid"], root[1]): root for thread in trace["threads"] for root in thread["roots"]}
        events = [(thread["tid"], event) for thread in trace["threads"] for event in thread["events"]]
        samples = [
            dict(sample, epoch_ns=sample["end_monotonic_ns"])
            for sample in map(json.loads, Path(original["cpu_samples_file"]).read_text().splitlines())
        ]
        report["mount_checks"].append(dict(**label, **stats))
        for thread in trace["threads"]:
            hints = [event for event in thread["events"] if event[0] == "sync_file_range"]
            assert all(0 < event[10] <= 4 * 1024 * 1024 for event in hints)
            gaps = [b[3] - a[4] for a, b in zip(hints, hints[1:])]
            assert all(gap >= 16000000 for gap in gaps)
            for event in hints:
                root = roots[thread["tid"], event[2]]
                assert root[0] == "prewriteback_worker"
                assert all(
                    root[10][trace["stages"].index(stage)][0] == 0 for stage in ("publication_hold", "checkpoint_hold")
                )
            if hints:
                report["worker_checks"].append(
                    dict(
                        **label,
                        tid=thread["tid"],
                        calls=len(hints),
                        maximum_bytes=max(e[10] for e in hints),
                        minimum_completion_gap_ns=min(gaps) if gaps else None,
                    )
                )
        for index, case in enumerate(cases):
            if any(case[key] != value for key, value in label.items()):
                continue
            wanted = index in first or (
                case["workload"] in ("random_write_4k", "small_file_fsync") and case["wall_ns"] >= 20000000
            )
            if not wanted:
                continue
            begin, end = case["interval"]["begin_ns"], case["interval"]["end_ns"]
            owned = {tuple(key) for key in case["handler_ids"]}
            foreground = {key[0] for key in owned}
            busy = [
                dict(
                    tid=tid,
                    event=event,
                    root_name=roots[tid, event[2]][0] if (tid, event[2]) in roots else None,
                    overlap_ns=overlap(event[3:5], (begin, end)),
                )
                for tid, event in events
                if tid not in foreground
                and event[0]
                in ("fsync", "fdatasync", "sync_file_range", "publication_hold", "checkpoint_hold", "checkpoint")
                and overlap(event[3:5], (begin, end))
            ]
            value = dict(
                case_index=index,
                **case,
                roots=[[tid, roots[tid, ident]] for tid, ident in owned],
                background_evidence=busy,
                scheduler=[dict(tid=tid, **cpu.metrics(samples, tid, begin, end)) for tid in sorted(foreground)],
            )
            if index in first:
                report["first_overwrites"].append(value)
            if case["wall_ns"] >= 20000000:
                report["tail_cases"].append(value)
    assert len(report["first_overwrites"]) == len(report["mount_checks"]) == 16
    report["inputs"].update(findings["inputs"])
    report["ended_epoch"] = time.time()
    report["largest_first_overwrite_ns"] = max(case["wall_ns"] for case in report["first_overwrites"])
    save(out / "diagnosis.json", report)
    for group in report["groups"]:
        print(group["state"], group["variant"], group["workload"], group["wall_ns"], flush=True)


if __name__ == "__main__":
    main()
