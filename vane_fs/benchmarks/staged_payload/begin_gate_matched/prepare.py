# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify unchanged binaries, rerun gate controls and freeze the matched comparison."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
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
    out.mkdir()
    gate = json.loads((component / "source.json").read_text())
    parent_path = Path(gate["parent"])
    parent = json.loads((parent_path / "source.json").read_text())
    roots = dict(baseline=Path(parent["candidate"]), updated=Path(gate["candidate"]))
    assert gate["mechanism_parent"] == str(roots["baseline"])
    assert parent["format"] == gate["format"] == 1005 and parent["sdk"] == gate["sdk"]
    assert parent["default_unchanged"] == gate["default_unchanged"]
    for key in ("payload_start_bytes", "batch_bytes", "completion_delay_ms"):
        assert parent[key] == gate[key]
    sources = {}
    inputs = {}
    for variant, source, folder in (("baseline", parent, parent_path), ("updated", gate, component)):
        build = json.loads((folder / "build.json").read_text())
        assert source["status"] == build["status"] == "PASS"
        assert all(value == "PASS" for value in build["stages"].values())
        assert build["source_manifest"]["sources"] == source["sources"]
        inputs.update(source["inputs"] | source["sources"] | build["inputs"] | build["binaries"])
        sources[variant] = {
            str(Path(p).relative_to(roots[variant])): dict(path=p, sha256=sha) for p, sha in source["sources"].items()
        }
        for p in (folder / "source.json", folder / "build.json"):
            inputs[str(p)] = digest(p)
    assert sources["baseline"].keys() == sources["updated"].keys()
    different = [
        name
        for name in sources["baseline"]
        if sources["baseline"][name]["sha256"] != sources["updated"][name]["sha256"]
    ]
    assert set(different) == {"src/checkpoint.hpp", "src/workspace.cpp"}
    for name, proof in gate["transformations"].items():
        text = (roots["updated"] / name).read_text()
        for before, after, count in reversed(proof["changes"]):
            assert text.count(after) == count
            text = text.replace(after, before)
        assert text == (roots["baseline"] / name).read_text()
    original_tests = json.loads((component / "gate-tests.json").read_text())
    assert original_tests["status"] == original_tests["parent_negative_control"] == "PASS"
    inputs.update(original_tests["inputs"] | original_tests["binaries"] | gate["default_unchanged"])
    for p in [
        component / "gate-tests.json",
        *here.glob("*.py"),
        here.parent / "extent_fuse/run.py",
        here.parent / "extent_fuse/analyze.py",
        here.parent / "paced_range_writeback/analyze.py",
        here.parent / "partial_sync/analyze_writeback.py",
        here.parents[1] / "checkpoint_priority/analyze.py",
        here.parents[1] / "io_interference/monitor.py",
    ]:
        inputs[str(p)] = digest(p)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    comparison = dict(
        status="PASS",
        verified_epoch=time.time(),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        component=str(component),
        parent=str(parent_path),
        roots={k: str(v) for k, v in roots.items()},
        sources=sources,
        native_changed_files=different,
        reverse_exact=True,
        sdk=gate["sdk"],
        default_unchanged=gate["default_unchanged"],
        native_builds_reused_by_source_and_binary_identity=True,
        format=1005,
        original_gates_unchanged=True,
    )
    save(out / "comparison.json", comparison)
    tests = dict(
        status="RUNNING",
        started_epoch=time.time(),
        commands=[],
        cleanup=[],
        cases_per_build=7,
        parent_negative_cases_per_build=2,
        reused_binary_manifest=str(component / "gate-tests.json"),
    )
    try:
        for mode in ("release", "asan"):
            for variant in ("parent", "candidate"):
                binary = next(
                    Path(p) for p in original_tests["binaries"] if Path(p).name.endswith(mode + "-gate-" + variant)
                )
                owned = Path(tempfile.mkdtemp(prefix="owned-gate-matched-", dir="/dev/shm"))
                log, started = out / (mode + "-" + variant + ".log"), time.time()
                argv = [str(binary), str(owned), *(["--parent-control"] if variant == "parent" else [])]
                code, failure = None, None
                try:
                    with log.open("w") as stream:
                        try:
                            code = subprocess.run(
                                argv,
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                                timeout=180,
                                env=dict(
                                    os.environ,
                                    ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                                    UBSAN_OPTIONS="halt_on_error=1",
                                ),
                            ).returncode
                        except BaseException as error:
                            failure = repr(error)
                    tests["commands"].append(
                        dict(
                            argv=argv,
                            started_epoch=started,
                            ended_epoch=time.time(),
                            returncode=code,
                            error=failure,
                            log=str(log),
                            sha256=digest(log),
                        )
                    )
                    assert code == 0 and failure is None, log
                    print("PASS", mode, variant, flush=True)
                finally:
                    assert str(owned) not in Path("/proc/mounts").read_text()
                    files = {
                        str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
                        for p in owned.rglob("*")
                        if p.is_file()
                    }
                    shutil.rmtree(owned)
                    tests["cleanup"].append(
                        dict(
                            path=str(owned),
                            files=files,
                            removed=not owned.exists(),
                            processes_stopped=True,
                            mount_absent=True,
                        )
                    )
                    save(out / "tests.json", tests)
        tests["status"] = "PASS"
    except BaseException as error:
        tests.update(status="FAIL", error=repr(error))
        raise
    finally:
        tests["ended_epoch"] = time.time()
        save(out / "tests.json", tests)
    for p in (out / "tests.json", out / "comparison.json"):
        inputs[str(p)] = digest(p)
    windows = {}
    for window, mounts in (("full", 24), ("sustained", 8)):
        template, root = component / "fuse" / window, out / window
        root.mkdir()
        plan = json.loads((template / "plan.json").read_text())
        assert digest(template / "run_fuse.py") == plan["prepared_runner_sha256"]
        assert digest(template / "workload.py") == plan["prepared_workload_sha256"]
        modules = {}
        for variant, path in roots.items():
            (root / variant).symlink_to(path, target_is_directory=True)
            module = next(path.glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"))
            modules[variant] = dict(path=str(module), sha256=digest(module))
            for p in [
                module,
                *[path / "build/python-release" / name for name in ("vane-fs", "vane-fs-mount", "libvane_fs.a")],
            ]:
                inputs[str(p)] = digest(p)
        before = f"Paced range claims plus local BEGIN/RESTART gate: {window} ordinary FUSE window; both formats 1005"
        after = f"Matched paced parent versus BEGIN/RESTART gate: {window} ordinary FUSE window; both formats 1005"
        original = (template / "run_fuse.py").read_text()
        assert original.count(before) == 1
        code = original.replace(before, after)
        assert code.replace(after, before) == original
        compile(code, str(root / "run_fuse.py"), "exec")
        (root / "run_fuse.py").write_text(code)
        shutil.copy2(template / "workload.py", root / "workload.py")
        plan.update(
            frozen_epoch=time.time(),
            variant_gc_python_modules=modules,
            prepared_runner_sha256=digest(root / "run_fuse.py"),
            prepared_workload_sha256=digest(root / "workload.py"),
            template_runner_sha256=digest(template / "run_fuse.py"),
            runner_only_change=[before, after],
            runner_reverse_exact=True,
            interpretation="Incremental local gate only; both variants already use 4 MiB/16 ms paced hints and identical FULL pauses. Original 22 checks remain unmodified; no adoption is performed.",
            payload_start_bytes=dict(baseline=gate["payload_start_bytes"], updated=gate["payload_start_bytes"]),
            both_variants_paced=True,
            native_changed_files=different,
        )
        save(root / "plan.json", plan)
        save(root / "build.json", dict(status="PASS", component=str(out / "comparison.json")))
        save(
            root / "tests.json",
            dict(
                status="PASS",
                related_gate_rerun=str(out / "tests.json"),
                remaining_suites_inherited_by_source_binary_identity=True,
            ),
        )
        for p in [
            template / "plan.json",
            template / "run_fuse.py",
            template / "workload.py",
            *[root / n for n in ("plan.json", "run_fuse.py", "workload.py", "build.json", "tests.json")],
        ]:
            inputs[str(p)] = digest(p)
        windows[window] = dict(root=str(root), mounts=mounts)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert shutil.disk_usage(out).free > 8 * 1024**3
    save(
        out / "build.json",
        dict(
            status="PASS",
            prepared_epoch=time.time(),
            inputs=inputs,
            component=str(component),
            windows=windows,
            free_bytes=shutil.disk_usage(out).free,
            policy="Serial complete windows; no owned build/test/analysis overlap, cache eviction, device/CPU tuning or load-based sample filtering.",
        ),
    )
    print("PASS only gate differs; frozen 24 full and 8 sustained mounts; related native controls rerun", flush=True)


if __name__ == "__main__":
    main()
