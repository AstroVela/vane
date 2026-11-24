# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the original complete FUSE gates against the unchanged extent control."""

import argparse
import json
import shutil
import time
from pathlib import Path

from generate import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", type=Path)
    args = parser.parse_args()
    component, here = args.component.resolve(), Path(__file__).resolve().parent
    repo = here.parents[3]
    source = json.loads((component / "source.json").read_text())
    build = json.loads((component / "build.json").read_text())
    assert source["status"] == build["status"] == "PASS"
    assert build["stages"]["payload_range_writeback"] == "PASS"
    inputs = dict(source["inputs"], **source["sources"], **build["binaries"], **build["inputs"])
    history = json.loads((component / "initial-preflight.json").read_text())
    assert history["status"] == "PASS" and not history["measurement_started"]
    for path, sha in history["inputs"].items():
        assert inputs.get(path, sha) == sha
        inputs[path] = sha
    for path in [
        component / "source.json",
        component / "build.json",
        component / "regression-before.json",
        component / "regression-before-async.json",
        component / "initial-preflight.json",
        *map(Path, history["retained_inputs"]),
        *here.glob("*.py"),
        repo / "vane_fs/benchmarks/io_interference/monitor.py",
        here.parent / "extent_fuse/run.py",
        here.parent / "extent_fuse/analyze.py",
        here.parent / "extent_fuse/publish.py",
        here.parent / "partial_sync/analyze_writeback.py",
        repo / "vane_fs/benchmarks/checkpoint_priority/analyze.py",
    ]:
        inputs[str(path)] = digest(path)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    out = component / "fuse"
    out.mkdir()
    report = dict(status="RUNNING", inputs=inputs, component=str(component), windows={}, commands=build["commands"])
    for window, mounts in [("full", 24), ("sustained", 8)]:
        template = repo / "vane_fs/build/extent-fuse-20261018" / window
        plan = json.loads((template / "plan.json").read_text())
        assert digest(template / "run_fuse.py") == plan["prepared_runner_sha256"]
        assert digest(template / "workload.py") == plan["prepared_workload_sha256"]
        root = out / window
        root.mkdir()
        modules = {}
        for variant, path in [("baseline", Path(source["original"])), ("updated", Path(source["candidate"]))]:
            (root / variant).symlink_to(path, target_is_directory=True)
            module = next(path.glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"))
            modules[variant] = dict(path=str(module), sha256=digest(module))
            for binary in [
                module,
                *(path / "build/python-release" / f for f in ("vane-fs", "vane-fs-mount", "libvane_fs.a")),
            ]:
                inputs[str(binary)] = digest(binary)
        runner = (template / "run_fuse.py").read_text()
        transformations = [
            (
                'format_versions={"baseline": 1004, "updated": 1005},',
                'format_versions={"baseline": 1005, "updated": 1005},',
            ),
            ('version = 1004 if rows[-1]["variant"] == "baseline" else 1005', "version = 1005"),
            (
                f"Versioned extents: {window} ordinary FUSE window; per-variant GC modules",
                f"Finite-range 64 MiB payload writeback: {window} ordinary FUSE window; both formats 1005",
            ),
        ]
        for before, after in transformations:
            assert runner.count(before) == 1, (window, before)
            runner = runner.replace(before, after)
        reverse = runner
        for before, after in reversed(transformations):
            assert reverse.count(after) == 1
            reverse = reverse.replace(after, before)
        assert reverse.encode() == (template / "run_fuse.py").read_bytes()
        compile(runner, str(root / "run_fuse.py"), "exec")
        (root / "run_fuse.py").write_text(runner)
        shutil.copy2(template / "workload.py", root / "workload.py")
        plan.update(
            frozen_epoch=time.time(),
            format_versions=dict(baseline=1005, updated=1005),
            variant_gc_python_modules=modules,
            payload_start_bytes=dict(baseline=None, updated=64 * 1024**2),
            interpretation="Same original gates and workload; candidate adds finite-range background payload hints and retains the FULL payload barrier descriptor from before writes; publication and barrier ordering unchanged.",
            prepared_runner_sha256=digest(root / "run_fuse.py"),
            prepared_workload_sha256=digest(root / "workload.py"),
            template_runner_sha256=digest(template / "run_fuse.py"),
            template_workload_sha256=digest(template / "workload.py"),
            transformations=transformations,
            runner_reverse_exact=True,
        )
        save(root / "plan.json", plan)
        save(root / "build.json", dict(status="PASS", inherited_component_manifest=str(component / "build.json")))
        save(
            root / "tests.json",
            dict(
                status="PASS",
                inherited_by_source_binary_identity=True,
                component=str(component / "build.json"),
            ),
        )
        for path in [
            template / "plan.json",
            template / "run_fuse.py",
            template / "workload.py",
            root / "plan.json",
            root / "run_fuse.py",
            root / "workload.py",
            root / "build.json",
            root / "tests.json",
        ]:
            inputs[str(path)] = digest(path)
        report["windows"][window] = dict(root=str(root), mounts=mounts)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert shutil.disk_usage(out).free > 8 * 1024**3
    report.update(status="PASS", prepared_epoch=time.time(), free_bytes=shutil.disk_usage(out).free)
    save(out / "build.json", report)
    print("PASS: frozen 24 complete-gate mounts plus 8 supplementary sustained mounts; no retuned thresholds")


if __name__ == "__main__":
    main()
