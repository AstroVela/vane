# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check idle versus CPU-warm preparation without changing host frequency policy."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sample_cpu import policy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True, help="Completed owned-CPU sampling window")
    args = parser.parse_args()
    out, parent = args.output.resolve(), args.parent.resolve()
    out.mkdir()
    here = Path(__file__).resolve().parent
    identity = json.loads((parent / "build.json").read_text())
    assert identity["status"] == "PASS"
    assert json.loads((parent / "input-check.json").read_text())["status"] == "PASS"
    original = Path(identity["parent"])
    build = json.loads((original / "build.json").read_text())
    inputs = {
        str(p): digest(p)
        for p in (
            here / "run_controlled.py",
            here / "sample_cpu.py",
            here / "warmup.py",
            parent / "run_fuse.py",
            parent / "workload.py",
        )
    }
    inputs.update(identity["inputs"])
    allowed = sorted(os.sched_getaffinity(0))
    # Four distinct physical cores for caller, main FUSE thread, checkpoint and
    # hint workers. Only these owned processes/threads have their affinity set.
    cores = []
    topology = {}
    physical = set()
    for cpu in allowed:
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((directory / "physical_package_id").read_text().strip(), (directory / "core_id").read_text().strip())
        topology[str(cpu)] = list(key)
        if key not in physical:
            physical.add(key)
            cores.append(cpu)
        if len(cores) == 4:
            break
    assert len(cores) == 4
    variants = ["baseline-idle", "updated-idle", "baseline-warm", "updated-warm"]
    orders = [variants, list(reversed(variants))]
    for variant in variants:
        name = variant.split("-")[0]
        component = original / "updated" if name == "updated" else Path(build["parent"]) / "updated"
        assert all(digest(component / p) == h for p, h in build["source_inputs"][name].items())
        (out / variant).mkdir()
        (out / variant / "venv").symlink_to(original / "updated/venv", target_is_directory=True)
        alias = out / "diagnostic" / variant
        alias.mkdir(parents=True)
        (alias / "build").symlink_to(component / "build/python-release", target_is_directory=True)
    shutil.copy2(original / "tests.json", out / "tests.json")
    (out / "build.json").write_text(
        json.dumps(dict(status="PASS", parent=str(parent), inputs=inputs, native_binaries_unchanged=True), indent=2)
        + "\n"
    )
    runner = (parent / "run_fuse.py").read_text()
    runner = replace_once(runner, 'm.VARIANTS = ["baseline", "updated"]', "m.VARIANTS = " + repr(variants))
    runner = replace_once(
        runner,
        'm.ORDERS = m.FIRST = [["baseline", "updated"], ["updated", "baseline"]] * (2 if args.diagnostic else 3)',
        "m.ORDERS = m.FIRST = " + repr(orders),
    )
    runner = replace_once(
        runner,
        '    state = "fresh"\n\n    def start',
        '    state = "fresh"\n    active_variant = None\n    m.REPORT["cpu_preparation"] = []\n\n    def start',
    )
    runner = replace_once(runner, "        nonlocal active_pid\n", "        nonlocal active_pid, active_variant\n")
    runner = replace_once(
        runner,
        "        active_pid = process.pid",
        "        active_pid = process.pid\n        active_variant = Path(command[0]).parent.parent.name\n        os.sched_setaffinity(process.pid, {"
        + str(cores[1])
        + "})",
    )
    prepare = """
    os.sched_setaffinity(0, {CALLER_CPU})
    def cpu_prepare(label):
        workers = sorted(int(p.name) for p in Path(f"/proc/{active_pid}/task").glob("[0-9]*") if int(p.name) != active_pid)
        roles = {str(active_pid): FUSE_CPU}
        for index, tid in enumerate(workers):
            core = WORKER_CPUS[min(index, len(WORKER_CPUS) - 1)]
            os.sched_setaffinity(tid, {core})
            roles[str(tid)] = core
        before = int(Path("FREQUENCY_PATH").read_text())
        begin = time.time()
        if active_variant.endswith("-warm"):
            completed = subprocess.run([sys.executable, WARMUP_PATH, "--cpu", str(FUSE_CPU), "--seconds", "2"], check=True, capture_output=True, text=True, timeout=10)
            child = json.loads(completed.stdout)
        else:
            time.sleep(2)
            child = None
        m.REPORT["cpu_preparation"].append(dict(variant=active_variant, daemon_pid=active_pid, label=label, started_epoch=begin, ended_epoch=time.time(), frequency_before_khz=before, frequency_after_khz=int(Path("FREQUENCY_PATH").read_text()), thread_cpu_roles=roles, warmup_child=child))
    m.cpu_prepare = cpu_prepare
"""
    prepare = (
        prepare.replace("CALLER_CPU", str(cores[0]))
        .replace("FUSE_CPU", str(cores[1]))
        .replace("WORKER_CPUS", repr(cores[2:]))
    )
    prepare = prepare.replace(
        "FREQUENCY_PATH", f"/sys/devices/system/cpu/cpu{cores[1]}/cpufreq/scaling_cur_freq"
    ).replace("WARMUP_PATH", repr(str(here / "warmup.py")))
    runner = replace_once(runner, "    def cpu():", prepare + "\n    def cpu():")
    runner = runner.replace(
        'diagnostic_scope="Unchanged native binaries plus read-only owned-thread CPU sampling; not a plain adoption window"',
        'diagnostic_scope="Owned-process affinity and idle/CPU-warm preparation; no host frequency setting changes; not an adoption window"',
    )
    workload = (parent / "workload.py").read_text()
    for needle, label in [
        ("        t = time.perf_counter()\n        epoch = time.time()\n        count = 0", "read_64"),
        ("        t = time.perf_counter()\n        epoch = time.time()\n        answers =", "random_read_4k"),
        ("        epoch = time.time()\n        t = time.perf_counter()\n        latencies = []", "random_write_4k"),
    ]:
        workload = replace_once(workload, needle, '        cpu_prepare("' + label + '")\n' + needle)
    workload = replace_once(
        workload,
        '        for op in ["mkdir", "rename", "rmdir"]:\n            samples = []',
        '        for op in ["mkdir", "rename", "rmdir"]:\n            cpu_prepare(op)\n            samples = []',
    )
    frozen = dict(
        status="FROZEN_BEFORE_CONTROLLED_WINDOW",
        frozen_epoch=time.time(),
        parent=str(parent),
        medium="ext4",
        states=["fresh", "post_gc"],
        orders=orders,
        mounts=16,
        native_binaries_unchanged=True,
        preparation_seconds=2,
        preparation_conditions=["idle", "CPU-only warm"],
        cpu_roles=dict(caller=cores[0], fuse=cores[1], checkpoint=cores[2], hint=cores[3]),
        selected_cpu_topology=topology,
        frequency_policy_before=policy(),
        host_settings_unchanged=True,
        global_cache_eviction=False,
        owned_build_or_test_overlap=False,
        submission_budget_bytes=4194304,
        minimum_submission_gap_ms=16,
        gates_unchanged_and_not_applied_to_diagnostic=True,
        inputs=inputs,
        prepared_runner_sha256=hashlib.sha256(runner.encode()).hexdigest(),
        prepared_workload_sha256=hashlib.sha256(workload.encode()).hexdigest(),
    )
    (out / "plan.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (out / "run_fuse.py").write_text(runner)
    (out / "workload.py").write_text(workload)
    subprocess.run([sys.executable, str(out / "run_fuse.py"), str(out), "--medium", "ext4", "--diagnostic"], check=True)
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    assert policy() == frozen["frequency_policy_before"]
    (out / "input-check.json").write_text(
        json.dumps(dict(status="PASS", native_inputs_and_host_frequency_policy_unchanged=True), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
