# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the original 22 gates and full/sustained workloads without tuning."""

import json
import shutil
import sys
import time
from pathlib import Path

from generate import digest, save


def main():
    component, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    source = json.loads((component / "source.json").read_text())
    build = json.loads((component / "build.json").read_text())
    assert source["status"] == build["status"] == "PASS" and build["stages"]["paced_range_writeback"] == "PASS"
    full = json.loads((component / "full-pauses.json").read_text())
    assert full["status"] == "PASS"
    inputs = (
        source["inputs"] | source["sources"] | build["inputs"] | build["binaries"] | full["inputs"] | full["binaries"]
    )
    preflight = json.loads((component / "initial-preflight.json").read_text())
    inputs.update(preflight["inputs"])
    inputs.update(json.loads((component / "full-pause-preflight.json").read_text())["inputs"])
    for p in [
        component / "source.json",
        component / "build.json",
        component / "initial-preflight.json",
        component / "full-pauses.json",
        component / "full-pause-preflight.json",
        *here.glob("*.py"),
        *here.glob("*.hpp"),
        *here.glob("*.cpp"),
        here.parent / "extent_fuse/run.py",
        here.parent / "extent_fuse/analyze.py",
        here.parent / "partial_sync/analyze_writeback.py",
        here.parents[1] / "checkpoint_priority/analyze.py",
        here.parents[1] / "io_interference/monitor.py",
    ]:
        inputs[str(p)] = digest(p)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    out = component / "fuse"
    out.mkdir()
    report = dict(status="RUNNING", inputs=inputs, component=str(component), windows={})
    template_parent = Path(source["parent"]) / "fuse"
    for window, mounts in (("full", 24), ("sustained", 8)):
        template = template_parent / window
        root = out / window
        root.mkdir()
        plan = json.loads((template / "plan.json").read_text())
        modules = {}
        assert (
            digest(template / "run_fuse.py") == plan["prepared_runner_sha256"]
            and digest(template / "workload.py") == plan["prepared_workload_sha256"]
        )
        for variant, path in (("baseline", Path(source["original"])), ("updated", Path(source["candidate"]))):
            (root / variant).symlink_to(path, target_is_directory=True)
            module = next(path.glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"))
            modules[variant] = dict(path=str(module), sha256=digest(module))
            for binary in [
                module,
                *[path / "build/python-release" / name for name in ("vane-fs", "vane-fs-mount", "libvane_fs.a")],
            ]:
                inputs[str(binary)] = digest(binary)
        original = (template / "run_fuse.py").read_text()
        before = f"Finite-range 64 MiB payload writeback: {window} ordinary FUSE window; both formats 1005"
        after = f"Paced 4 MiB range claims, 16 ms completion delay and FULL pause: {window} ordinary FUSE window; both formats 1005"
        assert original.count(before) == 1
        code = original.replace(before, after)
        assert code.replace(after, before) == original
        (root / "run_fuse.py").write_text(code)
        shutil.copy2(template / "workload.py", root / "workload.py")
        assert (root / "workload.py").read_bytes() == (template / "workload.py").read_bytes()
        plan.update(
            frozen_epoch=time.time(),
            variant_gc_python_modules=modules,
            prepared_runner_sha256=digest(root / "run_fuse.py"),
            prepared_workload_sha256=digest(root / "workload.py"),
            template_runner_sha256=digest(template / "run_fuse.py"),
            template_workload_sha256=digest(template / "workload.py"),
            runner_reverse_exact=True,
            runner_only_change=[before, after],
            batch_bytes=4 * 1024**2,
            completion_delay_ms=16,
            interpretation=source["description"],
            primary_gates_unchanged=True,
            regression_gates_unchanged=True,
        )
        save(root / "plan.json", plan)
        save(root / "build.json", dict(status="PASS", component=str(component)))
        save(
            root / "tests.json",
            dict(status="PASS", component=str(component / "build.json"), all_related_checks_passed=True),
        )
        for p in [
            template / "plan.json",
            template / "run_fuse.py",
            template / "workload.py",
            *[root / name for name in ("plan.json", "run_fuse.py", "workload.py", "build.json", "tests.json")],
        ]:
            inputs[str(p)] = digest(p)
        report["windows"][window] = dict(root=str(root), mounts=mounts)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert shutil.disk_usage(out).free > 8 * 1024**3
    report.update(status="PASS", prepared_epoch=time.time(), free_bytes=shutil.disk_usage(out).free)
    save(out / "build.json", report)
    print("PASS frozen 24 complete-gate mounts and 8 supplementary four-cycle mounts", flush=True)


if __name__ == "__main__":
    main()
