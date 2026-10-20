# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify separation of hint submission from transactions and publication."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here.parent / "partial_sync/analyze.py"), str(out)], check=True)
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    report = dict(
        status="PASS",
        rows=[],
        span_filter_threshold_seconds=0.001,
        scope="Separate instrumented window; thread separation compares retained markers; short spans below 1 ms are filtered; blocking isolation is independently verified by fault tests; no throughput adoption inference",
    )
    for state in ("fuse", "aged_fuse"):
        for row in raw[state]:
            trace = json.loads(Path(row["timeline_file"]).read_text())
            events = [dict(tid=t["tid"], **e) for t in trace["threads"] for e in t["events"]]
            hints = [e for e in events if e["name"] == "sync_file_range_write"]
            queue = [e for e in events if e["name"] == "enqueue_payload_writeback"]
            shutdown = [e for e in events if e["name"] == "writeback_shutdown"]
            foreground = [
                e
                for e in events
                if e["name"] in ("admission", "begin_batch", "payload_publication", "publish", "publication_mutex_wait")
            ]
            hint_threads = {e["tid"] for e in hints}
            assert hint_threads.isdisjoint(e["tid"] for e in foreground + queue + shutdown)
            assert bool(hints) == (row["variant"] == "updated")
            assert not queue or row["variant"] == "updated"
            report["rows"].append(
                dict(
                    state=state,
                    variant=row["variant"],
                    rep=row["rep"],
                    hint_calls=len(hints),
                    retained_enqueue_calls=len(queue),
                    hint_threads=sorted(hint_threads),
                    enqueue_threads=sorted({e["tid"] for e in queue}),
                    observed_foreground_threads=sorted({e["tid"] for e in foreground}),
                    hint_max_seconds=max([(e["end_ns"] - e["begin_ns"]) / 1e9 for e in hints], default=0),
                    maximum_hint_range_bytes=max([e["range_bytes"] for e in hints], default=0),
                    enqueue_max_seconds_upper_bound=max(
                        [(e["end_ns"] - e["begin_ns"]) / 1e9 for e in queue], default=0.001
                    )
                    if row["variant"] == "updated"
                    else None,
                    shutdown_max_seconds_upper_bound=max(
                        [(e["end_ns"] - e["begin_ns"]) / 1e9 for e in shutdown], default=0.001
                    )
                    if row["variant"] == "updated"
                    else None,
                    hints_with_overlapping_retained_enqueue=sum(
                        any(q["begin_ns"] < h["end_ns"] and h["begin_ns"] < q["end_ns"] for q in queue) for h in hints
                    ),
                    syscall_errors=[e["errno"] for e in hints if e["rc"] != 0],
                )
            )
    (out / "background-diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: background hint thread separation", len(report["rows"]), "mounts")


if __name__ == "__main__":
    main()
