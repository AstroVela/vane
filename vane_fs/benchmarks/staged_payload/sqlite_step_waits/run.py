# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen complete paced workloads with the targeted SQL observer."""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from prepare import digest, save, validate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    out = Path(sys.argv[1]).resolve()
    prior = ROOT / "vane_fs/build/paced-request-waits-20261019"
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    assert json.loads((out / "observer-tests.json").read_text())["status"] == "PASS"
    source = (prior / "run_fuse.py").read_text()
    before = str(HERE.parent / "request_costs/sample_cpu.py")
    after = str(HERE / "sample.py")
    assert source.count(before) == 1
    modified = source.replace(before, after)
    assert modified.replace(after, before) == source
    compile(modified, str(out / "run_fuse.py"), "exec")
    (out / "run_fuse.py").write_text(modified)
    (out / "workload.py").write_bytes((prior / "workload.py").read_bytes())
    tests = dict(
        status="PASS",
        related_trace_tests=[
            validate(json.loads((out / f"{mode}-trace.json").read_text())) for mode in ("release", "asan")
        ],
        observer_tests=json.loads((out / "observer-tests.json").read_text()),
        original_correctness_tests_reused_by_exact_reverse=True,
    )
    save(out / "tests.json", tests)
    inputs = build["inputs"] | build["sources"] | build["binaries"]
    for path in [
        *HERE.glob("*.py"),
        *HERE.glob("*.hpp"),
        *HERE.glob("*.cpp"),
        prior / "run_fuse.py",
        prior / "workload.py",
        HERE.parent / "fuse_sync_waits/sample.py",
        out / "build.json",
        out / "observer-tests.json",
        out / "run_fuse.py",
        out / "workload.py",
        out / "tests.json",
    ]:
        inputs[str(path)] = digest(path)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    modules = {}
    for variant in ("baseline", "updated"):
        path = next((out / "diagnostic" / variant).glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"))
        modules[variant] = dict(path=str(path), sha256=digest(path))
    save(
        out / "plan.json",
        dict(
            status="FROZEN_BEFORE_DIAGNOSTIC_WINDOW",
            frozen_epoch=time.time(),
            mounts=16,
            orders=[["baseline", "updated"], ["updated", "baseline"]] * 2,
            states=["fresh", "post_gc"],
            medium="ext4",
            kernel=platform.release(),
            architecture=platform.machine(),
            inputs=inputs,
            variant_gc_python_modules=modules,
            role="SQL/sleep and targeted wait attribution only",
            original_plain_failed_checks=9,
            native_logic_unchanged=True,
            runner_reverse_exact=True,
            runner_only_change=[before, after],
            workload_bytes_unchanged=True,
            active_minimum_age_ns=20000000,
            kernel_observation_minimum_interval_ns=20000000,
            mapping_read_only=True,
            global_cache_eviction=False,
            cpu_device_tuning=False,
            owned_build_or_test_overlap=False,
        ),
    )
    command = [sys.executable, str(out / "run_fuse.py"), str(out), "--medium", "ext4", "--diagnostic"]
    begin = time.time()
    with (out / "measurement.log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    save(
        out / "measurement-command.json",
        dict(argv=command, started_epoch=begin, ended_epoch=time.time(), returncode=code),
    )
    assert code == 0
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    checks = []
    for state in ("fuse", "aged_fuse"):
        assert len(raw[state]) == 8
        for row in raw[state]:
            stats = validate(json.loads(Path(row["waits_file"]).read_text()))
            samples = [json.loads(line) for line in Path(row["cpu_samples_file"]).read_text().splitlines()]
            final = samples[-1]
            assert final["sampler_finished"] and final["stop_requested"] and final["fatal"] is None
            assert final["marker"] is not None and final["mapping_closed"] and final["mapping_read_only"]
            checks.append(dict(state=state, variant=row["variant"], rep=row["rep"], **stats))
    save(out / "input-check.json", dict(status="PASS", checked_inputs=len(inputs), mounts=checks))


if __name__ == "__main__":
    main()
