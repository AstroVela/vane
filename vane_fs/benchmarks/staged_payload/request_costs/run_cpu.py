# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Sample owned-thread CPU frequency around unchanged, uninstrumented native workloads."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sample_cpu import policy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    out, parent = args.output.resolve(), args.parent.resolve()
    out.mkdir()
    here = Path(__file__).resolve().parent
    identity = json.loads((parent / "build.json").read_text())
    old_plan = json.loads((parent / "plan.json").read_text())
    assert identity["status"] == "PASS"
    assert json.loads((parent / "writeback-tests.json").read_text())["status"] == "PASS"
    inputs = {
        str(p): digest(p)
        for p in (here / "run_cpu.py", here / "sample_cpu.py", parent / "run_fuse.py", parent / "workload.py")
    }
    for name in ("baseline", "updated"):
        component = parent / "updated" if name == "updated" else Path(identity["parent"]) / "updated"
        assert all(digest(component / p) == h for p, h in identity["source_inputs"][name].items())
        (out / name).mkdir()
        (out / name / "venv").symlink_to(parent / "updated/venv", target_is_directory=True)
        alias = out / "diagnostic" / name
        alias.mkdir(parents=True)
        (alias / "build").symlink_to(component / "build/python-release", target_is_directory=True)
        for binary in ("vane-fs", "vane-fs-mount"):
            path = component / "build/python-release" / binary
            inputs[str(path)] = digest(path)
    shutil.copy2(parent / "tests.json", out / "tests.json")
    (out / "build.json").write_text(
        json.dumps(dict(status="PASS", parent=str(parent), inputs=inputs, native_binaries_unchanged=True), indent=2)
        + "\n"
    )
    assert digest(parent / "run_fuse.py") == old_plan["prepared_runner_sha256"]
    assert digest(parent / "workload.py") == old_plan["prepared_workload_sha256"]
    runner = (parent / "run_fuse.py").read_text().replace("import types\n", "import types\nimport threading\n")
    needle = "(1 if args.diagnostic else 3)"
    assert runner.count(needle) == 1
    runner = runner.replace(needle, "(2 if args.diagnostic else 3)")
    setup = """
    spec = importlib.util.spec_from_file_location("cpu_probe", Path(SAMPLER_PATH))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    samplers = []
"""
    runner = runner.replace(
        "    owned = []\n", setup.replace("SAMPLER_PATH", repr(str(here / "sample_cpu.py"))) + "    owned = []\n"
    )
    needle = "        process = subprocess.Popen(command, **kwargs)"
    assert runner.count(needle) == 1
    runner = runner.replace(
        needle,
        needle
        + """
        samples = []
        stop = threading.Event()
        thread = threading.Thread(target=probe.sample, args=(process.pid, stop, samples))
        thread.start()
        samplers.append((process, thread, stop, samples, Path(command[1]).parent))
""",
    )
    needle = "    def cleanup(work, extra=None):"
    assert runner.count(needle) == 1
    runner = runner.replace(
        needle,
        needle
        + """
        for process, thread, stop, samples, owned_work in samplers:
            if owned_work == work:
                assert process.poll() is not None
                stop.set()
                thread.join(timeout=10)
                assert not thread.is_alive()
                path = dest / (work.name + "-cpu.jsonl")
                with path.open("w") as stream:
                    for sample in samples:
                        stream.write(json.dumps(sample, separators=(",", ":")) + "\\n")
                record = (m.REPORT["fuse"] if state == "fresh" else m.REPORT["aged_fuse"])[-1]
                record["cpu_samples_file"] = str(path)
                record["daemon_pid"] = process.pid
""",
    )
    runner = runner.replace(
        "instrumented=args.diagnostic,",
        'instrumented=False,\n        diagnostic_scope="Unchanged native binaries plus read-only owned-thread CPU sampling; not a plain adoption window",',
    )
    plan = dict(
        status="FROZEN_BEFORE_CPU_DIAGNOSTIC_WINDOW",
        frozen_epoch=time.time(),
        parent=str(parent),
        medium="ext4",
        states=["fresh", "post_gc"],
        orders=[["baseline", "updated"], ["updated", "baseline"]] * 2,
        mounts=16,
        native_binaries_unchanged=True,
        workload_unchanged=True,
        nominal_sample_interval_ms=5,
        frequency_policy_before=policy(),
        frequency_or_affinity_changes=False,
        global_cache_eviction=False,
        owned_build_or_test_overlap=False,
        submission_budget_bytes=4194304,
        minimum_submission_gap_ms=16,
        interpretation="Sysfs reports sampled frequency, not per-request cycle counts; scheduler runtime weights approximate phase CPU frequency",
        inputs=inputs,
        prepared_runner_sha256=hashlib.sha256(runner.encode()).hexdigest(),
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "run_fuse.py").write_text(runner)
    (out / "workload.py").write_bytes((parent / "workload.py").read_bytes())
    subprocess.run([sys.executable, str(out / "run_fuse.py"), str(out), "--medium", "ext4", "--diagnostic"], check=True)
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    assert policy() == plan["frequency_policy_before"]
    (out / "input-check.json").write_text(
        json.dumps(dict(status="PASS", native_inputs_and_frequency_policy_unchanged=True), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
