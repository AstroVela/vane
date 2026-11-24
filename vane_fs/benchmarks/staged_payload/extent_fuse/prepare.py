# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze ordinary and sustained FUSE comparisons of formats 1004 and 1005."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--component", type=Path, required=True)
    args = parser.parse_args()
    out, component, here = args.artifacts.resolve(), args.component.resolve(), Path(__file__).resolve().parent
    build = json.loads((component / "build.json").read_text())
    source = json.loads((component / "source.json").read_text())
    assert build["status"] == source["status"] == "PASS"
    template = Path(source["parent"]) / "fuse-screen"
    old_plan = json.loads((template / "plan.json").read_text())
    assert digest(template / "run_fuse.py") == old_plan["prepared_runner_sha256"]
    assert digest(template / "workload.py") == old_plan["prepared_workload_sha256"]
    inputs = dict(source["inputs"], **source["sources"], **build["binaries"], **build["inputs"])
    repo = here.parents[3]
    for path in [
        component / "build.json",
        component / "source.json",
        template / "plan.json",
        template / "run_fuse.py",
        template / "workload.py",
        here / "run.py",
        Path(__file__),
        repo / "vane_fs/benchmarks/io_interference/monitor.py",
        here.parent / "partial_sync/analyze_writeback.py",
        repo / "vane_fs/benchmarks/checkpoint_priority/analyze.py",
        Path(source["sdk"]) / "lib/libsqlite3.a",
    ]:
        inputs[str(path)] = digest(path)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert shutil.disk_usage(component).free > 8 * 1024**3
    out.mkdir()
    components = dict(baseline=Path(source["original"]), updated=Path(source["candidate"]))
    manifest = dict(status="RUNNING", inputs=inputs, component=str(component), windows={}, commands=build["commands"])
    for window, repetitions, cycles in (("full", 6, 1), ("sustained", 2, 4)):
        root = out / window
        root.mkdir()
        modules = {}
        for name, path in components.items():
            (root / name).symlink_to(path, target_is_directory=True)
            module = next(path.glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"))
            modules[name] = dict(path=str(module), sha256=digest(module))
            inputs[str(module)] = digest(module)
            for file in ("vane-fs", "vane-fs-mount", "libvane_fs.a"):
                binary = path / "build/python-release" / file
                inputs[str(binary)] = digest(binary)
        (root / "workload.py").write_bytes((template / "workload.py").read_bytes())
        runner = (template / "run_fuse.py").read_text()
        transformations = []

        def replace(before, after):
            nonlocal runner
            assert runner.count(before) == 1, (window, before)
            runner = runner.replace(before, after)
            transformations.append([before, after])

        replace(
            "    args = parser.parse_args()",
            '    args = parser.parse_args()\n    assert not args.diagnostic, "Ordinary uninstrumented components only"',
        )
        replace(
            '[["baseline", "updated"], ["updated", "baseline"]] * (1 if args.diagnostic else 3)',
            f'[["baseline", "updated"], ["updated", "baseline"]] * {repetitions // 2}',
        )
        replace("m.SEQUENTIAL_SIZES = [(256, 8), (64, 1)]", f"m.SEQUENTIAL_SIZES = [(256, 8), (64, 1)] * {cycles}")
        replace("format_version=1004,", 'format_versions={"baseline": 1004, "updated": 1005},')
        replace(
            'interpretation="Borrowed input buffers in matched format-1004 FUSE components on ext4; native CPU screen is separate",',
            f'interpretation="Versioned extents: {window} ordinary FUSE window; per-variant GC modules",\n        continuous_cycles={cycles},',
        )
        replace(
            '            assert quick == [("ok",)] and schema == [(1004, 4096)] and all(n in (8, 4096) for n, _ in lengths)',
            """            rows = m.REPORT["fuse"] if state == "fresh" else m.REPORT["aged_fuse"]
            version = 1004 if rows[-1]["variant"] == "baseline" else 1005
            assert quick == [("ok",)] and schema == [(version, 4096)] and all(n in (8, 4096) for n, _ in lengths)""",
        )
        replace(
            '                "block_high_index_present": present,',
            """                "block_high_index_present": present,
                "version_rows": conn.execute("SELECT count(*) FROM block_versions").fetchone()[0],
                "covered_blocks": conn.execute("SELECT " + ("sum(count)" if version == 1005 else "count(*)") + " FROM block_versions WHERE deleted=0").fetchone()[0],
                "max_extent_blocks": conn.execute("SELECT max(count) FROM block_versions").fetchone()[0] if version == 1005 else 1,
                "payload_file": m.file_state(Path(str(db) + ".payload")),""",
        )
        if cycles > 1:
            replace(
                "    m.sequential = sequential",
                """    sequence_counts = {}

    def continuous(point, db, dimensions):
        row = sequential(point, db, dimensions)
        key = (str(point), row["workload"])
        index = sequence_counts.get(key, 0)
        sequence_counts[key] = index + 1
        row["base_workload"] = row["workload"]
        row["cycle"] = index
        row["workload"] += "_cycle_" + str(index)
        return row

    m.sequential = continuous""",
            )
        reverse = runner
        for before, after in reversed(transformations):
            assert reverse.count(after) == 1, (window, after)
            reverse = reverse.replace(after, before)
        assert reverse.encode() == (template / "run_fuse.py").read_bytes()
        compile(runner, str(root / "run_fuse.py"), "exec")
        (root / "run_fuse.py").write_text(runner)
        plan = dict(
            status="FROZEN_BEFORE_PLAIN_COMPARISON",
            frozen_epoch=time.time(),
            window=window,
            media=["ext4"],
            orders=[["baseline", "updated"], ["updated", "baseline"]] * (repetitions // 2),
            states=["fresh", "post_gc"],
            large_sequence=[dict(mib=256, application_mib=8), dict(mib=64, application_mib=1)] * cycles,
            buffer_page_offset=48,
            continuous_cycles=cycles,
            format_versions=dict(baseline=1004, updated=1005),
            durability="fsync",
            sqlite_sdk=source["sdk"],
            gc_preparation="Same logical retained-guard zeroing/GC sequence; each format uses its own frozen module.",
            variant_gc_python_modules=modules,
            primary_gates=old_plan["primary_gates"] if window == "full" else None,
            regression_gates=old_plan["regression_gates"] if window == "full" else None,
            supplementary=window == "sustained",
            interpretation="Original complete gates for full; sustained describes four consecutive cycles, without replacing gates.",
            checkpoint_thresholds_mib=[16, 64],
            final_barriers_unchanged=True,
            global_cache_eviction=False,
            cpu_or_device_tuning=False,
            owned_build_or_test_overlap=False,
            prepared_runner_sha256=digest(root / "run_fuse.py"),
            prepared_workload_sha256=digest(root / "workload.py"),
            template_runner_sha256=digest(template / "run_fuse.py"),
            template_workload_sha256=digest(template / "workload.py"),
            transformations=transformations,
            runner_reverse_exact=True,
        )
        for path in (root / "run_fuse.py", root / "workload.py"):
            inputs[str(path)] = digest(path)
        save(root / "plan.json", plan)
        save(root / "build.json", dict(status="PASS", inherited_component_manifest=str(component / "build.json")))
        save(
            root / "tests.json",
            dict(status="PASS", inherited_by_source_binary_identity=True, component=str(component / "build.json")),
        )
        inputs[str(root / "plan.json")] = digest(root / "plan.json")
        inputs[str(root / "build.json")] = digest(root / "build.json")
        inputs[str(root / "tests.json")] = digest(root / "tests.json")
        manifest["windows"][window] = dict(root=str(root), mounts=4 * repetitions)
    manifest.update(status="PASS", prepared_epoch=time.time(), inputs=inputs)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(out / "build.json", manifest)
    print("PASS frozen 24 ordinary and 8 sustained mounts; exact runner reversals; inherited component tests")


if __name__ == "__main__":
    main()
