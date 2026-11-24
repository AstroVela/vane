# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run a frozen three-variant native replay; screening evidence, never adoption."""

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    for name in ("tests.json", "sanitize.json", "restart-tests.json", "progress-tests.json"):
        assert json.loads((out / name).read_text())["status"] == "PASS"

    def check():
        assert all(digest(Path(path)) == expected for path, expected in build["inputs"].items())
        assert all(digest(out / path) == expected for path, expected in build["binaries"].items())
        assert digest(Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a") == build["sdk"]["sqlite_sha256"]
        source = out / "updated"
        assert all(digest(source / path) == expected for path, expected in build["source_inputs"]["updated"].items())

    check()
    dest = out / "probes"
    dest.mkdir()
    assert subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(dest)], text=True).strip() == "ext4"
    variants = ("orphan-control", "restart-control", "updated")
    orders = list(itertools.permutations(variants))
    monitor_path = here.parents[1] / "io_interference/monitor.py"
    plan = dict(
        status="FROZEN_BEFORE_NATIVE_SCREEN",
        frozen_epoch=time.time(),
        scope="Instrumented native API replay only; does not replace original FUSE adoption workload or gates",
        states=["cold", "reused", "retained"],
        orders=orders,
        runs=54,
        directory_operations_per_phase=128,
        final_barrier="Workspace::Sync() inside each phase timer",
        warmup="Real 20 MiB SQL padding commit and one inline seed; forced reusable generation or pinned old reader",
        instrumentation="Frozen typed sync recorder and native request scopes; runtime libraries uninstrumented",
        phase_seconds="All requests plus final barrier; preparation, inspection, close and cleanup outside timers",
        no_adoption_decision=True,
        cold_and_retained_complete_phase_reference_ratio_min=0.95,
        reused_complete_phase_reference_ratio_min=0.95,
        reused_final_barrier_ratio_to_restart_max=0.80,
        frequency_or_affinity_changes=False,
        global_cache_eviction=False,
        inputs={
            str(p): digest(p)
            for p in (Path(__file__).resolve(), here / "probe.cpp", here / "fixture.hpp", monitor_path)
        },
        binaries={str(out / name / "probe"): digest(out / name / "probe") for name in variants},
    )
    (out / "probe-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    report = dict(
        status="RUNNING",
        scope=plan["scope"],
        started_epoch=time.time(),
        runs=[],
        cleanup=[],
        free_bytes_before=shutil.disk_usage(dest).free,
        monitor_stopped=False,
    )

    def save():
        (out / "probe-results.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")

    monitor = subprocess.Popen(
        [sys.executable, str(monitor_path), str(dest / "host.jsonl")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for state in plan["states"]:
            for rep, order in enumerate(orders):
                for variant in order:
                    assert monitor.poll() is None
                    assert shutil.disk_usage(dest).free > 2 * 1024**3
                    work = Path(tempfile.mkdtemp(prefix="owned-wal-progress-", dir=dest))
                    database = work / "db.sqlite"
                    label = f"{state}-{rep}-{variant}"
                    log, trace = dest / (label + ".log"), dest / (label + ".json")
                    process = None
                    started = time.time()
                    try:
                        with log.open("w") as stream:
                            process = subprocess.Popen(
                                [str(out / variant / "probe"), str(database), state], stdout=stream, stderr=stream
                            )
                            code = process.wait(timeout=60)
                        assert code == 0, label
                        row = json.loads(log.read_text())
                        assert row["status"] == "PASS" and row["validation"]
                        shutil.copy2(Path(str(database) + ".request-costs.json"), trace)
                        report["runs"].append(
                            dict(
                                **row,
                                variant=variant,
                                rep=rep,
                                pid=process.pid,
                                started_epoch=started,
                                ended_epoch=time.time(),
                                trace_path=str(trace),
                            )
                        )
                        print(
                            label,
                            {p["workload"]: round(p["fsync_seconds"] * 1000, 3) for p in row["phases"]},
                            flush=True,
                        )
                    finally:
                        if process and process.poll() is None:
                            process.kill()
                            process.wait(timeout=10)
                        assert str(work) not in Path("/proc/mounts").read_text()
                        files = [
                            dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
                            for p in work.rglob("*")
                            if p.is_file()
                        ]
                        shutil.rmtree(work)
                        report["cleanup"].append(
                            dict(
                                path=str(work),
                                removed=not work.exists(),
                                files=files,
                                pid=process.pid if process else None,
                                process_stopped=True,
                            )
                        )
                        save()
        check()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        report.update(ended_epoch=time.time(), monitor_stopped=True, free_bytes_after=shutil.disk_usage(dest).free)
        save()


if __name__ == "__main__":
    main()
