# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep the original gate analysis and expose incremental paired effects."""

import json
import statistics
import sys
import time
from pathlib import Path

from prepare import digest, save

if __name__ == "__main__":
    here, out = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve()
    parent = here.parent / "paced_range_writeback/analyze.py"
    exec(compile(parent.read_text(), str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
    analysis = json.loads((out / "analysis.json").read_text())
    run = json.loads((out / "run.json").read_text())
    started = time.time()
    assert analysis["status"] == run["status"] == "PASS" and run["ended_epoch"] < started
    effects = {}
    for window, summary in analysis["windows"].items():
        effects[window] = {}
        for state, values in summary["states"].items():
            combined = values["combined"]
            cycles = 1 if window == "full" else 4
            median_rates = {
                variant: statistics.median(320 * cycles / seconds for seconds in durations.values())
                for variant, durations in combined["complete_large_phase_seconds"].items()
            }
            workloads = {}
            for name, stats in values["workloads"].items():
                row = dict(
                    median_rates={v: stats[v]["rate"]["median"] for v in ("baseline", "updated")},
                    aggregate_rates={v: stats[v]["aggregate_rate"] for v in ("baseline", "updated")},
                    median_ratio=stats["median_rate_ratio"],
                    aggregate_ratio=stats["aggregate_rate_ratio"],
                    paired_rate_wins=stats["paired_complete_write_wins"],
                )
                if "latencies" in stats["baseline"]:
                    row["pooled_p99_ms"] = {v: stats[v]["latencies"]["p99"] * 1000 for v in ("baseline", "updated")}
                    row["maximum_latency_ms"] = {
                        v: stats[v]["latencies"]["max"] * 1000 for v in ("baseline", "updated")
                    }
                if "fsync_seconds" in stats["baseline"]:
                    row["median_final_fsync_ms"] = {
                        v: stats[v]["fsync_seconds"]["median"] * 1000 for v in ("baseline", "updated")
                    }
                    row["maximum_final_fsync_ms"] = {
                        v: stats[v]["fsync_seconds"]["max"] * 1000 for v in ("baseline", "updated")
                    }
                workloads[name] = row
            effects[window][state] = dict(
                combined=combined
                | dict(
                    median_mib_per_second=median_rates,
                    median_rate_ratio=median_rates["updated"] / median_rates["baseline"],
                ),
                workloads=workloads,
                lifecycle=values["lifecycle"],
                long_large_calls=values["long_large_calls"],
            )
    inputs = analysis["inputs"] | {
        str(p): digest(p)
        for p in (Path(__file__).resolve(), out / "analysis.json", out / "comparison.json", out / "tests.json")
    }
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(
        out / "findings.json",
        dict(
            status="PASS",
            started_epoch=started,
            ended_epoch=time.time(),
            inputs=inputs,
            decision="MEASURED_ONLY_NO_ADOPTION",
            matched_gate_only=True,
            native_changed_files=["src/checkpoint.hpp", "src/workspace.cpp"],
            effects=effects,
            original_full_checks=analysis["full_checks"],
            original_full_acceptance_pass=analysis["full_acceptance_pass"],
            original_failed_checks=analysis["full_failed_checks"],
            original_failure_details=analysis["failure_details"],
            mounts=32,
            sequential_content_checks=112,
            limitations=[
                "The contemporaneous baseline already contains the same paced range hints and FULL pauses; only BEGIN/RESTART coordination differs.",
                "The original 22 checks, including the 10% combined-large-write improvement target, remain unchanged and fully reported. No new relaxed adoption threshold is introduced.",
                "Six primary AB/BA pairs per state are one local window, not a multi-host or long-term guarantee. Supplementary four-cycle windows have only two pairs per state.",
                "Ordinary application timings do not assign individual pauses to SQLite backoff, a checkpoint or a particular syscall/device. Controlled native tests establish the specific contention mechanism separately.",
                "No sample is dropped and no production code, SDK, format, durability or tuning policy is changed.",
            ],
        ),
    )
    print("PASS matched gate-only effect summary; all original failed checks retained", flush=True)
