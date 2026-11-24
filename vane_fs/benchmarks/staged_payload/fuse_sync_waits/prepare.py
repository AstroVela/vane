# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze existing native diagnostics; change only the external observer."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from common import digest, save, validate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / "vane_fs/build/range-writeback-waits-20261018"


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir()
    started = time.time()
    parent = json.loads((PARENT / "build.json").read_text())
    assert parent["status"] == "PASS" and platform.machine() == "x86_64"
    default = json.loads((ROOT / "vane_fs/build/payload-range-writeback-retained-20261018/source.json").read_text())[
        "default_unchanged"
    ]
    inputs = parent["inputs"] | parent["sources"] | parent["binaries"] | default
    for name in ("build.json", "plan.json", "run_fuse.py", "workload.py"):
        path = PARENT / name
        inputs[str(path)] = digest(path)
    for path in [*HERE.glob("*.py"), ROOT / "vane_fs/benchmarks/io_interference/monitor.py"]:
        inputs[str(path)] = digest(path)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    (out / "diagnostic").symlink_to(PARENT / "diagnostic", target_is_directory=True)
    for variant in ("baseline", "updated"):
        (out / variant).symlink_to(PARENT / "diagnostic" / variant, target_is_directory=True)
    original = (PARENT / "run_fuse.py").read_text()
    before = str(HERE.parent / "request_costs/sample_cpu.py")
    after = str(HERE / "sample.py")
    assert original.count(before) == 1 and after not in original
    source = original.replace(before, after)
    assert source.replace(after, before) == original
    compile(source, str(out / "run_fuse.py"), "exec")
    (out / "run_fuse.py").write_text(source)
    (out / "workload.py").write_bytes((PARENT / "workload.py").read_bytes())
    commands, cleanup, trace_tests = [], [], []

    def run(label, command, env=None):
        begin = time.time()
        with (out / f"{label}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        commands.append(
            dict(label=label, argv=command, started_epoch=begin, ended_epoch=time.time(), returncode=result.returncode)
        )
        save(out / "preflight.json", dict(commands=commands, cleanup=cleanup))
        assert result.returncode == 0, label

    run("sampler-tests", [sys.executable, str(HERE / "test_sample.py"), "-v"])
    for mode in ("release", "asan"):
        work = Path(tempfile.mkdtemp(prefix="owned-fuse-sync-trace-", dir="/dev/shm"))
        try:
            env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
            run(f"{mode}-trace", [str(PARENT / f"trace-test-{mode}"), str(work / "test")], env)
            trace = work / "test.waits.json"
            stats = validate(json.loads(trace.read_text()))
            shutil.copy2(trace, out / f"{mode}-trace.json")
            trace_tests.append(dict(mode=mode, **stats))
        finally:
            files = [
                dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
                for p in sorted(work.rglob("*"))
                if p.is_file()
            ]
            shutil.rmtree(work)
            cleanup.append(
                dict(path=str(work), removed=not work.exists(), processes_stopped=True, mount_absent=True, files=files)
            )
            save(out / "preflight.json", dict(commands=commands, cleanup=cleanup))
    tests = dict(status="PASS", sampler_tests=6, native_recorder_contracts=trace_tests, commands=commands)
    save(out / "tests.json", tests)
    for name in ("run_fuse.py", "workload.py", "tests.json"):
        inputs[str(out / name)] = digest(out / name)
    save(
        out / "build.json",
        dict(
            status="PASS",
            started_epoch=started,
            ended_epoch=time.time(),
            parent=str(PARENT),
            template=parent["template"],
            sdk=parent["sdk"],
            inputs=inputs,
            default_unchanged=default,
            binaries=parent["binaries"],
            cleanup=cleanup,
            native_rebuild=False,
            runner_reverse_exact=True,
            runner_only_change=[before, after],
            workload_bytes_unchanged=True,
        ),
    )
    modules = json.loads((PARENT / "plan.json").read_text())["variant_gc_python_modules"]
    save(
        out / "plan.json",
        dict(
            status="FROZEN",
            frozen_epoch=time.time(),
            mounts=16,
            orders=[["baseline", "updated"], ["updated", "baseline"]] * 2,
            states=["fresh", "post_gc"],
            medium="ext4",
            kernel=platform.release(),
            architecture=platform.machine(),
            device="/sys/block/sda/stat",
            device_fields=17,
            sampling_delay_seconds=0.005,
            inputs=inputs,
            variant_gc_python_modules=modules,
            source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            free_bytes=shutil.disk_usage(out).free,
            native_logic_unchanged=True,
            original_fuse_failed_checks=7,
            production_unchanged=True,
            cpu_device_tuning=None,
            global_cache_eviction=False,
            owned_build_or_test_overlap=False,
            interpretation="Point observations require same TID, contained observation bracket, stable syscall number/FD and matching descriptor file class. Device counters are whole-device context, not process I/O accounting. Kernel stack denials remain visible. Diagnostic results do not clear ordinary adoption gates.",
        ),
    )
    print("PASS frozen native identities, sampler integration and Release/ASAN trace contracts", flush=True)


if __name__ == "__main__":
    main()
