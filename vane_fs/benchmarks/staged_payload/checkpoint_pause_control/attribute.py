# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explain final FULL costs and retained tails after all measurements finish."""

import argparse
import json
import statistics
import time
from pathlib import Path

from analyze import digest, distribution, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    out = parser.parse_args().artifacts.resolve()
    started = time.time()
    raw = json.loads((out / "results.json").read_text())
    findings = json.loads((out / "findings.json").read_text())
    assert raw["status"] == findings["status"] == "PASS" and raw["monitor_stopped"]
    assert raw["ended_epoch"] < findings["started_epoch"] <= findings["ended_epoch"] < started
    rows = []
    inputs = {str(p): digest(p) for p in (Path(__file__), out / "results.json", out / "findings.json")}
    for run in raw["runs"]:
        path = Path(run["trace"])
        inputs[str(path)] = digest(path)
        trace = json.loads(path.read_text())
        roots = {(t["tid"], r[1]): r for t in trace["threads"] for r in t["roots"]}
        events = [(t["tid"], e) for t in trace["threads"] for e in t["events"]]
        final = [
            (key, root)
            for key, root in roots.items()
            if root[0] == "fsync" and run["random_sync_begin_ns"] <= root[2] < run["random_end_ns"]
        ]
        assert len(final) == 1
        key, root = final[0]
        own = [e for tid, e in events if (tid, e[2]) == key and e[0] == "fsync" and e[1] == "wal"]
        assert own and all(e[6:8] == [0, 0] for e in own)
        foreground_wal = [
            e
            for tid, e in events
            if tid == key[0]
            and e[0] == "fsync"
            and e[1] == "wal"
            and run["begin_ns"] <= e[3] < run["foreground_end_ns"]
        ]
        worker = [
            e
            for tid, e in events
            if roots.get((tid, e[2]), [""])[0] == "checkpoint_worker"
            and e[0] in ("fsync", "fdatasync")
            and e[3] < root[3]
            and root[2] < e[4]
        ]
        metrics = next(r for r in findings["runs"] if r["label"] == run["label"])
        rows.append(
            dict(
                label=run["label"],
                state=run["state"],
                component=run["component"],
                arm=run["arm"],
                final_full_ns=root[3] - root[2],
                final_full_cpu_ns=root[4],
                final_wal_sync_ns=sum(e[4] - e[3] for e in own),
                final_wal_sync_cpu_ns=sum(e[5] for e in own),
                final_wal_sync_calls=len(own),
                final_worker_syncs=worker,
                final_worker_overlap_ns=sum(max(0, min(root[3], e[4]) - max(root[2], e[3])) for e in worker),
                foreground_wal_sync_ns=metrics["foreground_wal_sync_ns"]["total"],
                foreground_all_wal_sync_ns=sum(e[4] - e[3] for e in foreground_wal),
                foreground_all_wal_sync_calls=len(foreground_wal),
                foreground_seconds=metrics["seconds"]["foreground"],
                foreground_wal_fraction=metrics["foreground_wal_sync_ns"]["total"]
                / (metrics["seconds"]["foreground"] * 1e9),
                preparation_seconds=metrics["seconds"]["preparation"],
                lifecycle_seconds=metrics["seconds"]["lifecycle"],
                foreground_all_wal_fraction=sum(e[4] - e[3] for e in foreground_wal)
                / (metrics["seconds"]["foreground"] * 1e9),
                sync_tails=metrics["sync_tails"],
            )
        )
    groups = []
    for state in ("warm", "gc"):
        for component in ("baseline", "updated"):
            for arm in ("active", "paused"):
                values = [r for r in rows if (r["state"], r["component"], r["arm"]) == (state, component, arm)]
                assert len(values) == 6
                groups.append(
                    dict(
                        state=state,
                        component=component,
                        arm=arm,
                        final_full_ns=distribution([r["final_full_ns"] for r in values]),
                        final_wal_sync_ns=distribution([r["final_wal_sync_ns"] for r in values]),
                        final_full_cpu_ns=distribution([r["final_full_cpu_ns"] for r in values]),
                        foreground_wal_fraction_median=statistics.median(r["foreground_wal_fraction"] for r in values),
                        foreground_all_wal_fraction_median=statistics.median(
                            r["foreground_all_wal_fraction"] for r in values
                        ),
                        total_seconds_median=statistics.median(
                            next(m for m in findings["runs"] if m["label"] == r["label"])["seconds"]["total"]
                            for r in values
                        ),
                    )
                )
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(
        out / "attribution.json",
        dict(
            status="PASS",
            started_epoch=started,
            ended_epoch=time.time(),
            inputs=inputs,
            groups=groups,
            runs=rows,
            interpretation="Pausing removes checkpoint overlap but increases typical final FULL and total cost in this prepared workload. Active and paused arms both retain sync tails. Timing overlap alone does not establish a kernel cause. Preparation tails and complete lifecycle remain visible.",
        ),
    )
    print("PASS final FULL and tail attribution for all 48 unchanged measurements")


if __name__ == "__main__":
    main()
