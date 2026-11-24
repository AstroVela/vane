# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Diagnose nonblocking checkpoint restart and sync contention against preceding orphan batching."""

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
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    parent = Path(build["parent"])
    plan = json.loads((parent / "plan.json").read_text())

    def check():
        for record in build["original_sources"].values():
            assert all(digest(Path(record["path"]) / p) == h for p, h in record["sha256"].items())
        assert all(
            digest(out / "diagnostic" / n / p) == h
            for n, entries in build["source_inputs"].items()
            for p, h in entries.items()
        )
        assert all(digest(out / p) == h for p, h in build["binaries"].items())
        assert all(digest(Path(p)) == h for p, h in build["inputs"].items())

    check()
    assert digest(parent / "run_fuse.py") == plan["prepared_runner_sha256"]
    assert digest(parent / "workload.py") == plan["prepared_workload_sha256"]
    sampler = here.parent / "request_costs/sample_cpu.py"
    runner = (parent / "run_fuse.py").read_text().replace("import types\n", "import types\nimport threading\n")
    assert runner.count("(1 if args.diagnostic else 3)") == 1
    runner = runner.replace("(1 if args.diagnostic else 3)", "(2 if args.diagnostic else 3)")
    runner = (
        runner.replace(".timeline.json", ".request-costs.json")
        .replace("-timeline.json", "-request-costs.json")
        .replace('row["timeline_file"]', 'row["request_costs_file"]')
    )
    setup = """
    spec = importlib.util.spec_from_file_location("cpu_probe", Path(SAMPLER_PATH))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    samplers = []
"""
    runner = runner.replace("    owned = []\n", setup.replace("SAMPLER_PATH", repr(str(sampler))) + "    owned = []\n")
    runner = runner.replace(
        "        process = subprocess.Popen(command, **kwargs)",
        """        process = subprocess.Popen(command, **kwargs)
        samples = []
        stop = threading.Event()
        thread = threading.Thread(target=probe.sample, args=(process.pid, stop, samples))
        thread.start()
        samplers.append((process, thread, stop, samples, Path(command[1]).parent))
""",
    )
    runner = runner.replace(
        "    def cleanup(work, extra=None):",
        """    def cleanup(work, extra=None):
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
        'instrumented=args.diagnostic,\n        diagnostic_scope="C++ request costs and read-only CPU sampling; no affinity/frequency changes; not adoption throughput",',
    )
    frozen = dict(
        status="FROZEN_BEFORE_DIAGNOSTIC_WINDOW",
        frozen_epoch=time.time(),
        parent=str(parent),
        medium="ext4",
        states=["fresh", "post_gc"],
        orders=[["baseline", "updated"], ["updated", "baseline"]] * 2,
        mounts=16,
        baseline_role="Preceding no-hint orphan batching; same baseline as the separate causal comparison",
        variant_gc_python_modules={
            "baseline": json.loads((parent / "build.json").read_text())["python_modules"]["orphan-control"],
            "updated": json.loads((parent / "build.json").read_text())["python_modules"]["updated"],
        },
        variant_logic_unchanged_by_instrumentation=True,
        nominal_cpu_sample_interval_ms=5,
        frequency_or_affinity_changes=False,
        global_cache_eviction=False,
        owned_build_or_test_overlap=False,
        inputs={
            str(p): digest(p)
            for p in (here / "run_diagnostic.py", parent / "run_fuse.py", parent / "workload.py", sampler)
        },
        prepared_runner_sha256=hashlib.sha256(runner.encode()).hexdigest(),
        prepared_workload_sha256=digest(parent / "workload.py"),
    )
    (out / "plan.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (out / "run_fuse.py").write_text(runner)
    (out / "workload.py").write_bytes((parent / "workload.py").read_bytes())
    subprocess.run([sys.executable, str(out / "run_fuse.py"), str(out), "--medium", "ext4", "--diagnostic"], check=True)
    check()
    (out / "input-check.json").write_text(json.dumps(dict(status="PASS"), indent=2) + "\n")


if __name__ == "__main__":
    main()
