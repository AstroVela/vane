# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run a fixed, separate diagnostic window with owned-thread wait sampling."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--pairs", type=int, choices=(1, 2), default=2, help="Number of AB/BA pair cycles per state")
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    parent = Path(build["parent"])
    old_plan = json.loads((parent / "plan.json").read_text())
    runner = (parent / "run_fuse.py").read_text()
    workload = parent / "workload.py"
    assert digest(parent / "run_fuse.py") == old_plan["prepared_runner_sha256"]
    assert digest(workload) == old_plan["prepared_workload_sha256"]
    sampler = here.parent / "block_sync/raw_probe.py"
    runner = runner.replace("import types\n", "import types\nimport threading\n")
    orders = f"({args.pairs} if args.diagnostic else 3)"
    runner = runner.replace("(1 if args.diagnostic else 3)", orders)
    assert runner.count(orders) == 1
    setup = """
    spec = importlib.util.spec_from_file_location("wait_probe", Path(SAMPLER_PATH))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    samplers = []
"""
    runner = runner.replace("    owned = []\n", setup.replace("SAMPLER_PATH", repr(str(sampler))) + "    owned = []\n")
    needle = "        process = subprocess.Popen(command, **kwargs)"
    runner = runner.replace(
        needle,
        needle
        + """
        rows = []
        stop = threading.Event()
        thread = threading.Thread(target=probe.sample_waits, args=(process.pid, stop, rows))
        thread.start()
        samplers.append((process, thread, stop, rows, Path(command[1]).parent))
""",
    )
    needle = "    def cleanup(work, extra=None):"
    runner = runner.replace(
        needle,
        needle
        + """
        for process, thread, stop, rows, owned_work in samplers:
            if owned_work == work:
                assert process.poll() is not None
                stop.set()
                thread.join(timeout=10)
                assert not thread.is_alive()
                path = dest / (work.name + "-waits.jsonl")
                with path.open("w") as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\\n")
                record = (m.REPORT["fuse"] if state == "fresh" else m.REPORT["aged_fuse"])[-1]
                record["wait_samples_file"] = str(path)
""",
    )
    runner = runner.replace(
        "instrumented=args.diagnostic,",
        'instrumented=args.diagnostic,\n        diagnostic_scope="Publication and durable-syscall intervals; not plain throughput",',
    )
    plan = dict(
        status="FROZEN_BEFORE_DIAGNOSTIC_WINDOW",
        frozen_epoch=time.time(),
        parent=str(parent),
        medium="ext4",
        states=["fresh", "post_gc"],
        orders=[["baseline", "updated"], ["updated", "baseline"]] * args.pairs,
        mounts=8 * args.pairs,
        syscall_sequence_unchanged=True,
        global_cache_eviction=False,
        owned_build_or_correctness_test_overlap=False,
        inputs={
            str(p): digest(p)
            for p in (parent / "run_fuse.py", workload, sampler, repo / "vane_fs/benchmarks/io_interference/monitor.py")
        },
        prepared_runner_sha256=hashlib.sha256(runner.encode()).hexdigest(),
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "run_fuse.py").write_text(runner)
    (out / "workload.py").write_bytes(workload.read_bytes())
    subprocess.run([sys.executable, str(out / "run_fuse.py"), str(out), "--medium", "ext4", "--diagnostic"], check=True)


if __name__ == "__main__":
    main()
