# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run six alternating native pairs per state with unchanged FULL barriers."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS" and len(build["tests"]) == 12
    inputs = dict(build["inputs"], **build["runtime_source_inputs"], **build["binaries"])
    sampler = here.parent / "request_costs/sample_cpu.py"
    for path in (Path(__file__), sampler, out / "build.json"):
        inputs[str(path)] = digest(path)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    plan = dict(
        status="FROZEN_BEFORE_MEASUREMENT",
        epoch=time.time(),
        states=["cold", "warm", "gc"],
        orders=[["active", "quiesced"], ["quiesced", "active"]] * 3,
        diagnostic_only=True,
        full_barriers_per_run=128,
        files_per_run=128,
        bytes_per_file=4096,
        warm_clean_padding_bytes=20 * 1024 * 1024,
        wal_admission_bytes=64 * 1024 * 1024,
        same_native_binary_for_both_arms=True,
        independent_workspaces=True,
        preparation_excluded_from_phase=True,
        no_owned_build_or_test_overlap=True,
        no_cache_eviction_or_affinity_changes=True,
        inputs=inputs,
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    result = dict(status="RUNNING", started_epoch=time.time(), runs=[], cleanup=[])
    spec = importlib.util.spec_from_file_location("cpu_sampler", sampler)
    cpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu)
    runs = out / "runs"
    runs.mkdir()

    def save():
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        for state in plan["states"]:
            for rep, order in enumerate(plan["orders"]):
                for variant in order:
                    label = f"{state}-{rep}-{variant}"
                    work = runs / ("owned-" + label)
                    work.mkdir()
                    assert shutil.disk_usage(work).free >= 2 * 1024**3
                    report = runs / (label + ".json")
                    command = [str(out / "probe-release"), str(work / "db.sqlite"), state, variant, str(report)]
                    start, process, samples = time.time(), None, []
                    stop = threading.Event()
                    try:
                        with (runs / (label + ".log")).open("w") as log:
                            process = subprocess.Popen(command, stdout=log, stderr=log)
                            thread = threading.Thread(target=cpu.sample, args=(process.pid, stop, samples))
                            thread.start()
                            try:
                                code = process.wait(timeout=120)
                            finally:
                                if process.poll() is None:
                                    process.kill()
                                    process.wait(timeout=20)
                                stop.set()
                                thread.join(timeout=10)
                                assert not thread.is_alive()
                        assert code == 0, label
                        row = json.loads(report.read_text())
                        assert row["status"] == "PASS" and row["content_and_integrity"]
                        row.update(rep=rep, command=command, pid=process.pid, start=start, end=time.time())
                        result["runs"].append(row)
                        print(label, round(row["seconds"], 6), "seconds", flush=True)
                    finally:
                        assert process is None or process.poll() is not None
                        for path in work.glob("*.request-costs.json"):
                            shutil.copy2(path, runs / (label + "-trace.json"))
                        with (runs / (label + "-cpu.jsonl")).open("w") as stream:
                            for sample in samples:
                                stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
                        assert str(work) not in Path("/proc/mounts").read_text()
                        removed = {
                            str(path.relative_to(work)): dict(bytes=path.stat().st_size, sha256=digest(path))
                            for path in work.rglob("*")
                            if path.is_file()
                        }
                        shutil.rmtree(work)
                        result["cleanup"].append(
                            dict(
                                path=str(work),
                                removed=not work.exists(),
                                process_stopped=True,
                                mount_absent=True,
                                files=removed,
                            )
                        )
                        save()
        assert all(digest(Path(path)) == sha for path, sha in inputs.items())
        result.update(status="PASS", ended_epoch=time.time(), free_bytes_after=shutil.disk_usage(out).free)
    except BaseException as error:
        result.update(status="FAIL", error=repr(error), ended_epoch=time.time())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
