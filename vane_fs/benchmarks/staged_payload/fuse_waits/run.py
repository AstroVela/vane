# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Associate unchanged FUSE workloads with request, worker and syscall timelines."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from prepare import digest, save, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    template = Path(build["parent"]) / "fuse-screen"
    original_plan = json.loads((template / "plan.json").read_text())
    assert digest(template / "run_fuse.py") == original_plan["prepared_runner_sha256"]
    assert digest(template / "workload.py") == original_plan["prepared_workload_sha256"]
    identities = dict(build["inputs"], **build["sources"], **build["binaries"])
    sampler = here.parent / "request_costs/sample_cpu.py"
    identities.update(
        {
            str(p): digest(p)
            for p in [Path(__file__), sampler, template / "run_fuse.py", template / "workload.py", out / "build.json"]
        }
    )
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    source = (template / "run_fuse.py").read_text()
    changes = []

    def replace(old, new, count=1):
        nonlocal source
        assert source.count(old) == count, (old, source.count(old), count)
        source = source.replace(old, new)
        changes.append((old, new, count))

    replace("import types", "import types\nimport threading")
    replace("(1 if args.diagnostic else 3)", "(2 if args.diagnostic else 3)")
    replace(
        'out / "diagnostic" / variant / "build" if args.diagnostic',
        'out / "diagnostic" / variant / "build/python-release" if args.diagnostic',
    )
    replace(
        "instrumented=args.diagnostic,",
        'instrumented=args.diagnostic,\n        diagnostic_scope="Exclusive request stages, typed synchronization and checkpoint events; no adoption throughput claim",',
    )
    replace(
        'sqlite_primitive="unchanged default SDK fsync",',
        'sqlite_primitive="unchanged default SDK fsync; request diagnostics",',
    )
    replace(
        'interpretation="Borrowed input buffers in matched format-1004 FUSE components on ext4; native CPU screen is separate",',
        'interpretation="Wait attribution for both frozen buffer implementations; original FUSE gate failures remain",',
    )
    replace(
        "    owned = []\n",
        f"""    cpu_spec = importlib.util.spec_from_file_location("owned_cpu", {str(sampler)!r})
    cpu_module = importlib.util.module_from_spec(cpu_spec)
    cpu_spec.loader.exec_module(cpu_module)
    cpu_observers = {{}}
    owned = []
""",
    )
    replace(
        "        active_pid = process.pid\n",
        """        active_pid = process.pid
        work = Path(command[1]).parent
        stop, samples = threading.Event(), []
        observer = threading.Thread(target=cpu_module.sample, args=(process.pid, stop, samples))
        observer.start()
        cpu_observers[work] = (stop, observer, samples)
""",
    )
    replace("        latencies, wal = [], []", "        latencies, wal, app = [], [], []")
    replace(
        '            "application_write_bytes": len(data),',
        '            "application_write_bytes": len(data),\n            "application_pid": os.getpid(),',
    )
    replace(
        """                t = time.perf_counter()
                assert os.write(fd, data) == len(data)
                latencies.append(time.perf_counter() - t)
""",
        """                begin_ns = time.monotonic_ns()
                t = time.perf_counter()
                assert os.write(fd, data) == len(data)
                latencies.append(time.perf_counter() - t)
                app.append(dict(kind="write", begin_ns=begin_ns, end_ns=time.monotonic_ns(), offset=len(app)*len(data), bytes=len(data)))
""",
    )
    replace(
        """            t = time.perf_counter()
            os.fsync(fd)
            row["fsync_seconds"] = time.perf_counter() - t
""",
        """            begin_ns = time.monotonic_ns()
            t = time.perf_counter()
            os.fsync(fd)
            row["fsync_seconds"] = time.perf_counter() - t
            app.append(dict(kind="fsync", begin_ns=begin_ns, end_ns=time.monotonic_ns()))
""",
    )
    replace(
        "            write_latencies=latencies,",
        "            write_latencies=latencies,\n            application_intervals=app,",
    )
    for old, new in [
        (".timeline.json", ".waits.json"),
        ("-timeline.json", "-waits.json"),
        ('row["timeline_file"]', 'row["waits_file"]'),
    ]:
        # A timeline suffix occurs twice in the existing cleanup wrapper.
        count = source.count(old)
        assert count in (1, 2)
        replace(old, new, count)
    replace(
        "        original_cleanup(work, extra)",
        """        if work in cpu_observers:
            stop, observer, samples = cpu_observers.pop(work)
            stop.set()
            observer.join(timeout=10)
            assert not observer.is_alive() and samples
            target = dest / (work.name + "-cpu.jsonl")
            target.write_text("".join(json.dumps(sample, separators=(",", ":")) + "\\n" for sample in samples))
            row = (m.REPORT["fuse"] if state == "fresh" else m.REPORT["aged_fuse"])[-1]
            row["cpu_samples_file"] = str(target)
            row["cpu_sampler_stopped"] = True
        original_cleanup(work, extra)""",
    )
    reverse = source
    for old, new, count in reversed(changes):
        assert reverse.count(new) == count
        reverse = reverse.replace(new, old)
    assert reverse.encode() == (template / "run_fuse.py").read_bytes()
    (out / "run_fuse.py").write_text(source)
    (out / "workload.py").write_bytes((template / "workload.py").read_bytes())
    identities.update({str(p): digest(p) for p in [out / "run_fuse.py", out / "workload.py"]})
    modules = {}
    for name in ("baseline", "updated"):
        path = next((out / "diagnostic" / name).glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"))
        modules[name] = dict(path=str(path), sha256=digest(path))
    plan = dict(
        status="FROZEN_BEFORE_DIAGNOSTIC_WINDOW",
        frozen_epoch=time.time(),
        orders=[["baseline", "updated"], ["updated", "baseline"]] * 2,
        mounts=16,
        states=["fresh", "post_gc"],
        medium="ext4",
        native_logic_unchanged=True,
        original_fuse_failures_retained=True,
        source_and_runner_reverse_exact=True,
        role="wait attribution only",
        stage_counters="Exclusive wall/CPU totals partition each request; inclusive spans overlap and must not be summed",
        events="Every sync/checkpoint plus I/O/lock spans >=1ms and syscall failures",
        application_gaps="Kernel copying, transport and scheduling combined; not an exact queue-delay measure",
        global_cache_eviction=False,
        cpu_device_tuning=False,
        owned_build_or_test_overlap=False,
        variant_gc_python_modules=modules,
        inputs=identities,
        runner_changes=changes,
    )
    save(out / "plan.json", plan)
    save(
        out / "tests.json",
        dict(
            status="PASS",
            related_trace_tests=[
                validate(json.loads((out / (mode + "-trace.json")).read_text())) for mode in ("release", "asan")
            ],
            original_correctness_tests_reused_by_exact_reverse=True,
        ),
    )
    subprocess.run([sys.executable, str(out / "run_fuse.py"), str(out), "--medium", "ext4", "--diagnostic"], check=True)
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    results = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert results["status"] == "PASS" and results["monitor_stopped"]
    checks = []
    for state in ("fuse", "aged_fuse"):
        assert len(results[state]) == 8
        for row in results[state]:
            trace = json.loads(Path(row["waits_file"]).read_text())
            assert row["cpu_sampler_stopped"]
            checks.append(dict(state=state, variant=row["variant"], rep=row["rep"], **validate(trace)))
    save(out / "input-check.json", dict(status="PASS", checked_inputs=len(identities), mounts=checks))


if __name__ == "__main__":
    main()
