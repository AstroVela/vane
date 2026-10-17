# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run cold initialization-inclusive comparisons before separate syscall diagnostics."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS" and len(build["tests"]) == 36
    mode = "diagnostic" if args.diagnostic else "plain"
    dest = out / mode
    dest.mkdir()
    inputs = dict(build["inputs"], **build["probes"])
    for variant in build["variants"].values():
        inputs.update(variant["sources"])
        inputs.update(variant["binaries"])
    monitor_source = here.parents[1] / "io_interference/monitor.py"
    for path in (Path(__file__), out / "build.json", monitor_source):
        inputs[str(path)] = digest(path)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    plan = dict(
        status="FROZEN_BEFORE_COMPARISON",
        epoch=time.time(),
        mode=mode,
        sdk_variants=["fsync", "fdatasync"],
        workloads=["small128", "write64"],
        orders=[["baseline", "reserved"], ["reserved", "baseline"]] * (1 if args.diagnostic else 3),
        independent_new_workspaces=True,
        all_workers_active=True,
        reserve_bytes=16 * 1024**2,
        primary_cost="Workspace initialization and complete first durable workload, not a warmed phase",
        primary_gates=dict(total_median_and_aggregate_rate_minimum=1.10, paired_wins_minimum=5),
        regression_gates=dict(
            first_durable_file_median_seconds_ratio_maximum=1.05, close_median_seconds_ratio_maximum=1.05
        ),
        no_owned_build_or_test_overlap=True,
        no_cache_eviction_or_system_tuning=True,
        diagnostic_only=args.diagnostic,
        inputs=inputs,
    )
    (dest / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    result = dict(status="RUNNING", started_epoch=time.time(), runs=[], cleanup=[])

    def save():
        (dest / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    monitor = subprocess.Popen([sys.executable, str(monitor_source), str(dest / "host.jsonl")])
    try:
        time.sleep(20)
        assert monitor.poll() is None
        for sdk in plan["sdk_variants"]:
            for workload in plan["workloads"]:
                for rep, order in enumerate(plan["orders"]):
                    for variant in order:
                        label = f"{sdk}-{workload}-{rep}-{variant}"
                        owned = dest / ("owned-" + label)
                        owned.mkdir()
                        assert shutil.disk_usage(owned).free >= 2 * 1024**3
                        binary = out / f"probe-{sdk}-{variant}-{mode}"
                        command = [str(binary), str(owned / "db.sqlite"), workload, str(dest / (label + ".json"))]
                        process = None
                        start = time.time()
                        try:
                            with (dest / (label + ".log")).open("w") as log:
                                process = subprocess.Popen(command, stdout=log, stderr=log)
                                code = process.wait(timeout=180)
                            assert code == 0, label
                            row = json.loads((dest / (label + ".json")).read_text())
                            assert row["status"] == "PASS" and row["content_and_integrity"]
                            row.update(
                                sdk=sdk,
                                variant=variant,
                                rep=rep,
                                pid=process.pid,
                                command=command,
                                start=start,
                                end=time.time(),
                            )
                            result["runs"].append(row)
                            print(
                                label,
                                "total",
                                round(row["total_seconds"], 6),
                                "first",
                                round(row["first_durable_seconds"], 6),
                                flush=True,
                            )
                        finally:
                            if process is not None and process.poll() is None:
                                process.kill()
                                process.wait(timeout=20)
                            assert process is None or process.poll() is not None
                            for path in owned.glob("*.request-costs.json"):
                                shutil.copy2(path, dest / (label + "-trace.json"))
                            files = {
                                str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
                                for p in owned.rglob("*")
                                if p.is_file()
                            }
                            shutil.rmtree(owned)
                            result["cleanup"].append(
                                dict(
                                    path=str(owned),
                                    removed=not owned.exists(),
                                    process_stopped=True,
                                    mount_absent=True,
                                    files=files,
                                )
                            )
                            save()
        assert all(digest(Path(path)) == sha for path, sha in inputs.items())
        result.update(status="PASS", ended_epoch=time.time(), free_bytes_after=shutil.disk_usage(dest).free)
    except BaseException as error:
        result.update(status="FAIL", error=repr(error), ended_epoch=time.time())
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        result["monitor_stopped"] = True
        save()


if __name__ == "__main__":
    main()
