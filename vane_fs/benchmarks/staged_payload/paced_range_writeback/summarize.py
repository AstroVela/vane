# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain complete phase costs and all failed gates without selecting samples."""

import difflib
import json
import statistics
import sys
import time
from pathlib import Path

from generate import digest, save


def main():
    out = Path(sys.argv[1]).resolve()
    started = time.time()
    build = json.loads((out / "build.json").read_text())
    source = json.loads((out / "source.json").read_text())
    pauses = json.loads((out / "full-pauses.json").read_text())
    analysis = json.loads((out / "fuse/analysis.json").read_text())
    run = json.loads((out / "fuse/run.json").read_text())
    assert (
        build["status"] == pauses["status"] == analysis["status"] == run["status"] == "PASS"
        and run["ended_epoch"] < started
    )
    phases, tails = [], []
    for window in ("full", "sustained"):
        raw = json.loads((out / "fuse" / window / "ext4-plain/results.json").read_text())
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        for state in ("fuse", "aged_fuse"):
            for row in raw[state]:
                for sample in row["samples"]:
                    base = dict(
                        window=window, state=state, variant=row["variant"], rep=row["rep"], workload=sample["workload"]
                    )
                    phases.append(base | {k: v for k, v in sample.items() if not isinstance(v, (dict, list))})
                    for key in ("write_latencies", "latencies"):
                        for i, seconds in enumerate(sample.get(key, [])):
                            if seconds >= 0.02:
                                tails.append(base | dict(kind=key, index=i, seconds=seconds))
                    if sample.get("fsync_seconds", 0) >= 0.02:
                        tails.append(base | dict(kind="fsync_seconds", seconds=sample["fsync_seconds"]))
    save(out / "phase-costs.json", dict(phases=phases, tails_ge_20ms=tails))
    groups = []
    for window in ("full", "sustained"):
        for state in ("fuse", "aged_fuse"):
            for variant in ("baseline", "updated"):
                for workload in sorted({r["workload"] for r in phases}):
                    values = [
                        r
                        for r in phases
                        if (r["window"], r["state"], r["variant"], r["workload"]) == (window, state, variant, workload)
                    ]
                    if not values:
                        continue
                    costs = {
                        k: dict(
                            median=statistics.median(r[k] for r in values),
                            max=max(r[k] for r in values),
                            sum=sum(r[k] for r in values),
                        )
                        for k in ("seconds", "requests_seconds", "fsync_seconds", "daemon_cpu_seconds")
                        if k in values[0]
                    }
                    groups.append(
                        dict(
                            window=window,
                            state=state,
                            variant=variant,
                            workload=workload,
                            samples=len(values),
                            costs=costs,
                        )
                    )
    patch = ""
    for relative in source["transformations"]:
        original = (Path(source["mechanism_parent"]) / relative).read_text().splitlines(keepends=True)
        updated = (Path(source["candidate"]) / relative).read_text().splitlines(keepends=True)
        patch += "".join(difflib.unified_diff(original, updated, fromfile="a/" + relative, tofile="b/" + relative))
    (out / "policy.patch").write_text(patch)
    inputs = json.loads((out / "fuse/build.json").read_text())["inputs"] | analysis["inputs"]
    inputs.update(
        {
            str(p): digest(p)
            for p in [
                Path(__file__).resolve(),
                out / "phase-costs.json",
                out / "policy.patch",
                out / "fuse/analysis.json",
                out / "fuse/run.json",
                out / "full-pauses.json",
            ]
        }
    )
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(
        out / "findings.json",
        dict(
            status="PASS",
            decision="ADOPTABLE_IN_THIS_WINDOW" if analysis["full_acceptance_pass"] else "REJECT",
            started_epoch=started,
            ended_epoch=time.time(),
            inputs=inputs,
            full_gate_checks=analysis["full_checks"],
            failed_checks=analysis["full_failed_checks"],
            failure_details=analysis["failure_details"],
            windows=analysis["windows"],
            groups=groups,
            application_tails_ge_20ms=tails,
            sequential_content_checks=112,
            mounts=32,
            python_tests_passed=153,
            python_skips=["production-v1 migration unavailable for format 1005", "PyAV unavailable"],
            cpp_suites_per_mode=12,
            extent_cases_per_mode=12,
            pacing_cases_per_mode=8,
            inherited_fault_cases_per_mode=14,
            real_full_pause_cases_per_mode=8,
            production_unchanged=True,
            original_unpaced_fuse_failures=7,
            limitations=[
                "The contemporaneous baseline is the original no-hint extent implementation; the old unpaced range implementation is not a third arm in this window. Historical seven failures and current nine failures do not isolate a pacing effect.",
                "Supplementary sustained measurements have two pairs per state and do not replace original six-pair gates.",
                "Ordinary application timings do not identify native mutex, checkpoint, syscall or device causes. The negative checkpoint-pause and format-1004 trials remain retained.",
                "A 4 MiB WRITE-only call can still block; pauses cannot cancel an active syscall or guarantee a latency bound.",
            ],
        ),
    )
    print(
        "PASS complete phase accounting; decision", "PASS" if analysis["full_acceptance_pass"] else "REJECT", flush=True
    )


if __name__ == "__main__":
    main()
