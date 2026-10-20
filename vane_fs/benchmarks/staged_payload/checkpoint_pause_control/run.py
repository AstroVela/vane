# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run a balanced two-component/two-worker control with resumed costs included."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from analyze import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    out = parser.parse_args().artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS" and len(build["tests"]) == 20
    inputs = build["inputs"] | build["binaries"]
    sampler = here.parent / "request_costs/sample_cpu.py"
    monitor_path = here.parents[1] / "io_interference/monitor.py"
    for path in (Path(__file__), out / "build.json", sampler, monitor_path):
        inputs[str(path)] = digest(path)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    # Rotate/reverse all four settings so each occupies early and late positions.
    a = ["baseline", "active"]
    b = ["baseline", "paused"]
    c = ["updated", "active"]
    d = ["updated", "paused"]
    orders = [[a, b, c, d], [d, c, b, a], [c, d, a, b], [b, a, d, c], [a, c, d, b], [b, d, c, a]]
    plan = dict(
        status="FROZEN_BEFORE_MEASUREMENT",
        frozen_epoch=time.time(),
        states=["warm", "gc"],
        components=["baseline", "updated"],
        orders=orders,
        runs=48,
        random_writes=512,
        random_seed=892,
        small_files=128,
        full_barriers_per_run=129,
        wal_limit_bytes=64 * 1024**2,
        warm_clean_padding_bytes=20 * 1024**2,
        total_includes_stop_resume_backfill_and_close=True,
        workload_is_native_not_fuse=True,
        diagnostic_only=True,
        hypothesis="Pausing background checkpoint work during explicit FULL-heavy phases may improve the phase and total after all deferred maintenance. It also changes batching and WAL reuse, so this is not an isolated kernel-concurrency test.",
        descriptive_gain_screen=dict(
            minimum_aggregate_ratio=1.1, minimum_paired_wins=5, apply_to_foreground_and_total_in_all_four_groups=True
        ),
        no_production_adoption=True,
        no_cache_eviction_or_tuning=True,
        inputs=inputs,
    )
    save(out / "plan.json", plan)
    runs = out / "runs"
    runs.mkdir()
    report = dict(status="RUNNING", started_epoch=time.time(), runs=[], cleanup=[])
    spec = importlib.util.spec_from_file_location("pause_cpu", sampler)
    cpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu)
    monitor = subprocess.Popen([sys.executable, str(monitor_path), str(out / "host.jsonl")])
    try:
        time.sleep(20)
        assert monitor.poll() is None
        for state in plan["states"]:
            for rep, order in enumerate(orders):
                for component, arm in order:
                    label = f"{state}-{rep}-{component}-{arm}"
                    owned = runs / ("owned-" + label)
                    owned.mkdir()
                    assert shutil.disk_usage(owned).free > 8 * 1024**3
                    result = runs / (label + ".json")
                    log = runs / (label + ".log")
                    argv = [
                        str(out / (component + "-release-probe")),
                        str(owned / "db.sqlite"),
                        state,
                        arm,
                        str(result),
                    ]
                    process = None
                    stop = threading.Event()
                    samples = []
                    observer = None
                    begin = time.time()
                    try:
                        with log.open("w") as stream:
                            process = subprocess.Popen(argv, stdout=stream, stderr=stream)
                            observer = threading.Thread(target=cpu.sample, args=(process.pid, stop, samples))
                            observer.start()
                            try:
                                code = process.wait(timeout=180)
                            finally:
                                if process.poll() is None:
                                    process.terminate()
                                    process.wait(timeout=30)
                                stop.set()
                                observer.join(timeout=10)
                                assert not observer.is_alive()
                        assert code == 0, label
                        row = json.loads(result.read_text())
                        assert row["status"] == "PASS" and row["content_and_integrity"]
                        trace = runs / (label + "-trace.json")
                        shutil.copy2(owned / "db.sqlite.waits.json", trace)
                        row.update(
                            label=label,
                            component=component,
                            rep=rep,
                            argv=argv,
                            pid=process.pid,
                            started_epoch=begin,
                            ended_epoch=time.time(),
                            trace=str(trace),
                        )
                        report["runs"].append(row)
                        print(
                            label,
                            "foreground",
                            round((row["foreground_end_ns"] - row["begin_ns"]) / 1e9, 6),
                            "total",
                            round((row["end_ns"] - row["begin_ns"]) / 1e9, 6),
                            flush=True,
                        )
                    finally:
                        assert process is None or process.poll() is not None
                        trace_path = owned / "db.sqlite.waits.json"
                        if trace_path.exists() and not (runs / (label + "-trace.json")).exists():
                            shutil.copy2(trace_path, runs / (label + "-trace.json"))
                        (runs / (label + "-cpu.jsonl")).write_text(
                            "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in samples)
                        )
                        assert str(owned) not in Path("/proc/mounts").read_text()
                        files = {
                            str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
                            for p in owned.rglob("*")
                            if p.is_file()
                        }
                        shutil.rmtree(owned)
                        report["cleanup"].append(
                            dict(
                                path=str(owned),
                                removed=not owned.exists(),
                                process_stopped=True,
                                mount_absent=True,
                                files=files,
                            )
                        )
                        save(out / "results.json", report)
        assert all(digest(Path(p)) == sha for p, sha in inputs.items())
        report.update(status="PASS", ended_epoch=time.time(), free_bytes=shutil.disk_usage(out).free)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), ended_epoch=time.time())
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        report["monitor_stopped"] = True
        save(out / "results.json", report)


if __name__ == "__main__":
    main()
