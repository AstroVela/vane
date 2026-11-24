# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run serial first-OPEN controls with initialization and close inside total."""

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
    assert build["status"] == "PASS"
    inputs = build["inputs"] | build["sources"] | build["binaries"]
    sampler = here.parent / "request_costs/sample_cpu.py"
    monitor_path = here.parents[1] / "io_interference/monitor.py"
    for path in [out / "build.json", sampler, monitor_path, *here.glob("*.py"), *here.glob("*.cpp")]:
        inputs[str(path)] = digest(path)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    plan = dict(
        status="FROZEN_BEFORE_MEASUREMENT",
        frozen_epoch=time.time(),
        repetitions=dict(fresh=12, reused=6, gc=6),
        variants=["original", "materialized"],
        orders=[["original", "materialized"], ["materialized", "original"]],
        total_runs=48,
        bytes_per_run=8 * 1024**2,
        request_sizes=[1048528, *[1048576] * 7, 48],
        total_includes_initialization_full_and_close=True,
        lifecycle_includes_preparation=True,
        native_only=True,
        production_unchanged=True,
        wal_limit_unchanged_bytes=64 * 1024**2,
        full_barriers_and_ordering_unchanged=True,
        screen=dict(
            fresh_first_8m_aggregate_gain=1.05,
            fresh_first_8m_paired_wins=9,
            minimum_total_aggregate_ratio_all_states=0.98,
        ),
        ordinary_fuse_gates_not_replaced=True,
        hypothesis="Materialize the invalid second fence slot under the existing initialization sync to avoid allocating/extending it at first OPEN. Include initialization to distinguish a net improvement from moved waiting. Reused and GC fences already have both slots.",
        inputs=inputs,
    )
    save(out / "plan.json", plan)
    runs = out / "runs"
    runs.mkdir()
    spec = importlib.util.spec_from_file_location("fence_cpu", sampler)
    cpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu)
    report = dict(status="RUNNING", started_epoch=time.time(), runs=[], cleanup=[])
    monitor = subprocess.Popen([sys.executable, str(monitor_path), str(out / "host.jsonl")])
    try:
        time.sleep(20)
        assert monitor.poll() is None
        for state, reps in plan["repetitions"].items():
            for rep in range(reps):
                for variant in plan["orders"][rep % 2]:
                    label = f"{state}-{rep}-{variant}"
                    owned = runs / ("owned-" + label)
                    owned.mkdir()
                    assert shutil.disk_usage(owned).free > 8 * 1024**3
                    result = runs / (label + ".json")
                    log = runs / (label + ".log")
                    argv = [
                        str(out / ("release-" + variant + "-probe")),
                        str(owned / "db.sqlite"),
                        state,
                        variant,
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
                            "first8",
                            round((row["writes_end_ns"] - row["writes_begin_ns"]) / 1e9, 6),
                            "total",
                            round((row["end_ns"] - row["begin_ns"]) / 1e9, 6),
                            flush=True,
                        )
                    finally:
                        assert process is None or process.poll() is not None
                        trace = owned / "db.sqlite.waits.json"
                        if trace.exists() and not (runs / (label + "-trace.json")).exists():
                            shutil.copy2(trace, runs / (label + "-trace.json"))
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
                                processes_stopped=True,
                                mount_absent=True,
                                files=files,
                            )
                        )
                        save(out / "results.json", report)
        assert len(report["runs"]) == 48
        assert all(digest(Path(p)) == sha for p, sha in inputs.items())
        report.update(status="PASS", ended_epoch=time.time())
    except BaseException as error:
        report.update(status="FAIL", ended_epoch=time.time(), error=repr(error))
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        report["monitor_stopped"] = True
        save(out / "results.json", report)


if __name__ == "__main__":
    main()
