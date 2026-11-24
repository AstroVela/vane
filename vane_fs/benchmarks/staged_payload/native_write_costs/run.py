# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run matched fresh native writes on ext4 and tmpfs without changing the core."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from prepare import clean, digest, save, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS" and len(build["tests"]) == 3
    mode = "profile" if args.profile else "plain"
    dest = out / mode.replace("profile", "diagnostic")
    # The source component lives at out/profile; measurement evidence is separate.
    dest.mkdir()
    sampler_source = here.parent / "request_costs/sample_cpu.py"
    monitor_source = here.parents[1] / "io_interference/monitor.py"
    spec = importlib.util.spec_from_file_location("native_cpu_sampler", sampler_source)
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    inputs = dict(build["inputs"], **build["binaries"], **build["default"], **build["diagnostic_sources"])
    inputs.update(build["component"]["sources"])
    inputs.update(build["component"]["binaries"])
    inputs.update(
        {
            str(p): digest(p)
            for p in [Path(__file__), out / "build.json", out / "source-proof.json", sampler_source, monitor_source]
        }
    )
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    mounts = {
        medium: subprocess.check_output(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS", "-T", str(path)], text=True
        ).strip()
        for medium, path in [("ext4", out), ("tmpfs", Path("/dev/shm"))]
    }
    assert all(value.split()[1] == medium for medium, value in mounts.items())
    plan = dict(
        status="FROZEN_BEFORE_MEASUREMENT",
        frozen_epoch=time.time(),
        mode=mode,
        scope="unchanged format-1004 native core attribution; CPU-clock measurement, not FUSE adoption",
        sizes_mib=[64, 256],
        request_bytes=1048576,
        orders=[["tmpfs", "ext4"], ["ext4", "tmpfs"]] * (1 if args.profile else 3),
        independent_new_workspaces=True,
        durability="fsync; original final FULL Workspace::Sync",
        worker_enabled=True,
        sdk="unchanged default-fsync SQLite SDK",
        interpretation="exclusive diagnostic stages partition caller thread CPU; worker CPU is separately present in process time",
        all_samples_retained=True,
        no_owned_build_or_test_overlap=True,
        no_cache_eviction_affinity_or_device_setting_changes=True,
        cpu_sampler_interval_ms=5,
        host_sampler_interval_seconds=1,
        initial_host_baseline_seconds=20,
        post_cleanup_idle_seconds=1,
        frequency_policy_before=sampler.policy(),
        mounts=mounts,
        inputs=inputs,
        free_bytes_before={
            name: shutil.disk_usage(path).free for name, path in [("ext4", out), ("tmpfs", Path("/dev/shm"))]
        },
    )
    save(dest / "plan.json", plan)
    report = dict(status="RUNNING", started_epoch=time.time(), runs=[], cleanup=[])
    monitor = subprocess.Popen([sys.executable, str(monitor_source), str(dest / "host.jsonl")])
    try:
        time.sleep(20)
        for size in plan["sizes_mib"]:
            for rep, order in enumerate(plan["orders"]):
                for medium in order:
                    root = dest if medium == "ext4" else Path("/dev/shm")
                    assert shutil.disk_usage(root).free > 2 * 1024**3
                    assert monitor.poll() is None
                    label = f"{size}-{rep}-{medium}"
                    owned = Path(tempfile.mkdtemp(prefix="owned-native-write-costs-", dir=root))
                    process, thread = None, None
                    stop = threading.Event()
                    samples, errors = [], []

                    def sample():
                        try:
                            sampler.sample(process.pid, stop, samples)
                        except BaseException as error:
                            errors.append(repr(error))

                    command = [
                        str(out / ("probe-" + mode)),
                        str(owned / "db.sqlite"),
                        str(size),
                        str(dest / (label + ".json")),
                    ]
                    started = time.time()
                    try:
                        with (dest / (label + ".log")).open("w") as log:
                            process = subprocess.Popen(command, stdout=log, stderr=log)
                            thread = threading.Thread(target=sample)
                            thread.start()
                            code = process.wait(timeout=180)
                        assert code == 0, (dest / (label + ".log")).read_text()
                        stop.set()
                        thread.join(timeout=10)
                        assert not thread.is_alive() and not errors and samples
                        row = json.loads((dest / (label + ".json")).read_text())
                        validate(row, args.profile)
                        row.update(
                            medium=medium,
                            rep=rep,
                            mode=mode,
                            command=command,
                            pid=process.pid,
                            started_epoch=started,
                            ended_epoch=time.time(),
                            cpu_samples_file=str(dest / (label + "-cpu.jsonl")),
                        )
                        with Path(row["cpu_samples_file"]).open("w") as stream:
                            for sample_row in samples:
                                stream.write(json.dumps(sample_row, separators=(",", ":")) + "\n")
                        report["runs"].append(row)
                        print(
                            label,
                            round(size * 1e9 / row["total_ns"], 2),
                            "MiB/s; caller CPU",
                            round(row["total_thread_cpu_ns"] / 1e6, 2),
                            "ms",
                            flush=True,
                        )
                    finally:
                        if process is not None and process.poll() is None:
                            process.kill()
                            process.wait(timeout=20)
                        stop.set()
                        if thread is not None:
                            thread.join(timeout=10)
                            assert not thread.is_alive()
                        assert process is None or process.poll() is not None
                        clean(owned, report)
                        save(dest / "results.json", report)
                        time.sleep(1)
        assert all(digest(Path(p)) == sha for p, sha in inputs.items())
        assert sampler.policy() == plan["frequency_policy_before"]
        report.update(status="PASS", ended_epoch=time.time(), inputs_unchanged=True, frequency_policy_unchanged=True)
    except BaseException as error:
        report.update(status="FAIL", ended_epoch=time.time(), error=repr(error))
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        report["monitor_stopped"] = True
        save(dest / "results.json", report)


if __name__ == "__main__":
    main()
