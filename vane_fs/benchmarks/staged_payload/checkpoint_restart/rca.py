# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the existing orphan-control checkpoint/sync evidence before comparing restart."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--prior-diagnostic", type=Path, required=True)
    args = parser.parse_args()
    out, prior = args.artifacts.resolve(), args.prior_diagnostic.resolve()
    result = dict(
        status="PASS", scope="Existing instrumented orphan control; not adoption throughput", inputs={}, groups=[]
    )
    path = prior / "ext4-diagnostic/results.json"
    raw = json.loads(path.read_text())
    result["inputs"] = {str(Path(__file__).resolve()): digest(Path(__file__)), str(path): digest(path)}
    for key, state in (("fuse", "fresh"), ("aged_fuse", "post_gc")):
        counts = collections.Counter()
        for mount in raw[key]:
            if mount["variant"] != "baseline":
                continue
            path = Path(mount["request_costs_file"])
            result["inputs"][str(path)] = digest(path)
            trace = json.loads(path.read_text())
            origin = trace["origin_epoch_ns"] / 1e9
            phase = next(s for s in mount["samples"] if s["workload"] == "random_write_4k")
            begin, end = [(phase[name] - origin) * 1e9 for name in ("started_epoch", "ended_epoch")]
            for thread in trace["threads"]:
                for index, row in enumerate(thread["rows"]):
                    if row[0] != "checkpoint_worker" or row[9][0] != 0 or row[2] < begin or row[1] > end:
                        continue
                    counts["passive_calls"] += 1
                    counts["reported_frames_equal_copied"] += row[9][1] == row[9][2]
                    for call in thread["syncs"]:
                        if call[0] == index:
                            counts[call[2] + "_sync_calls"] += 1
        result["groups"].append(dict(state=state, workload="random_write_4k", counts=dict(counts)))
    result["hypothesis"] = (
        "Concurrent commits can advance the live WAL head during PASSIVE; a lock-nonblocking RESTART may finish the tail under the writer lock and allow WAL reuse. It can still perform real I/O and may regress barriers or writes."
    )
    (out / "rca.json").write_text(json.dumps(result, indent=2) + "\n")
    print("existing checkpoint/sync evidence frozen", result["groups"])


if __name__ == "__main__":
    main()
