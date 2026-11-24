# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualify and characterize the unchanged default implementation once at closeout."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from experiment import HERE, REPO, digest, save


def tests(out):
    out.mkdir(exist_ok=True)
    inventory = json.loads((HERE / "inventory.json").read_text())
    inputs = dict(inventory["default"]["identities"])
    environment = out / "test-env"
    if not environment.exists():
        result = subprocess.run(
            ["uv", "venv", "--python", str(REPO / ".venv/bin/python"), str(environment)],
            capture_output=True,
            text=True,
            check=True,
        )
        (out / "test-environment.log").write_text(result.stdout + result.stderr)
        site = environment / "lib/python3.12/site-packages"
        (site / "vane_fs").symlink_to(REPO / ".venv/lib/python3.12/site-packages/vane_fs", target_is_directory=True)
        control = Path(inventory["mechanism_control"]["component"])
        dependencies = json.loads((control / "source.json").read_text())["dependency_site_paths"]
        (site / "support.pth").write_text("".join(path + "\n" for path in dependencies))
    python = environment / "bin/python"
    observed = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import hashlib,json,pathlib,vane_fs._native as n; p=pathlib.Path(n.__file__).resolve(); print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))",
            ],
            text=True,
        )
    )
    assert inputs[observed["path"]] == observed["sha256"]
    save(out / "test-native-identity.json", observed)
    inputs[str(environment / "lib/python3.12/site-packages/support.pth")] = digest(
        environment / "lib/python3.12/site-packages/support.pth"
    )
    build = REPO / "vane_fs/build/python-release"
    for p in [
        *build.glob("vane_fs_*_test"),
        *[build / n for n in ("vane-fs", "vane-fs-mount", "libvane_fs.a")],
        Path(__file__),
    ]:
        inputs[str(p)] = digest(p)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    report = dict(status="RUNNING", started_epoch=time.time(), commands=[], cleanup=[], inputs=inputs)

    def run(label, argv, env):
        started = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, cwd="/tmp", env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        report["commands"].append(
            dict(
                argv=argv,
                started_epoch=started,
                ended_epoch=time.time(),
                returncode=result.returncode,
                log=str(out / (label + ".log")),
            )
        )
        assert result.returncode == 0, label

    owned = Path(tempfile.mkdtemp(prefix="owned-closeout-default-", dir="/dev/shm"))
    fixed = build / "owned-sync-syscalls"
    assert not fixed.exists()
    try:
        env = dict(os.environ, TMPDIR=str(owned))
        run("default-cpp", ["ctest", "--test-dir", str(build), "--output-on-failure"], env)
        run(
            "default-wal-fsync",
            [str(build / "vane_fs_sync_syscalls_test"), str(owned / "explicit-fsync"), "fsync"],
            env,
        )
        run(
            "default-python",
            [
                str(python),
                "-I",
                "-m",
                "pytest",
                "--rootdir",
                str(REPO / "vane_fs"),
                "--confcutdir",
                str(REPO / "vane_fs"),
                "--basetemp",
                str(owned / "pytest"),
                "-q",
                str(REPO / "vane_fs/tests"),
            ],
            dict(env, VANE_FS_MOUNT_BINARY=str(build / "vane-fs-mount")),
        )
        assert all(digest(Path(p)) == sha for p, sha in inputs.items())
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        for path in (owned, fixed):
            assert str(path) not in Path("/proc/mounts").read_text()
            existed = path.exists()
            files = (
                {
                    str(p.relative_to(path)): dict(bytes=p.stat().st_size, sha256=digest(p))
                    for p in path.rglob("*")
                    if p.is_file()
                }
                if existed
                else {}
            )
            if existed:
                shutil.rmtree(path)
            report["cleanup"].append(
                dict(
                    path=str(path),
                    files=files,
                    created_by_run=existed,
                    removed=not path.exists(),
                    processes_stopped=True,
                    mount_absent=True,
                )
            )
        report["ended_epoch"] = time.time()
        save(out / "default-tests.json", report)
    print("PASS default native/Python suites and actual WAL fsync", flush=True)


def freeze(out):
    verified = json.loads((out / "default-tests.json").read_text())
    assert verified["status"] == "PASS"
    inventory = json.loads((HERE / "inventory.json").read_text())
    parent = Path(inventory["mechanism_control"]["component"]) / "fuse"
    shipping = out / "shipping"
    (shipping / "build").mkdir(parents=True)
    (shipping / "build/python-release").symlink_to(REPO / "vane_fs/build/python-release", target_is_directory=True)
    (shipping / "venv").symlink_to(REPO / ".venv", target_is_directory=True)
    native = next((REPO / ".venv/lib/python3.12/site-packages/vane_fs").glob("_native*.so"))
    root = out / "fuse"
    root.mkdir()
    inputs = verified["inputs"] | {str(out / "default-tests.json"): digest(out / "default-tests.json")}
    windows = {}
    for window, repeats in (("full", 3), ("sustained", 2)):
        target, template = root / window, parent / window
        target.mkdir()
        (target / "default").symlink_to(shipping, target_is_directory=True)
        original = text = (template / "run_fuse.py").read_text()
        old_orders = 'm.ORDERS = m.FIRST = [["baseline", "updated"], ["updated", "baseline"]] * ' + (
            "3" if window == "full" else "1"
        )
        changes = [
            ('m.VARIANTS = ["baseline", "updated"]', 'm.VARIANTS = ["default"]'),
            (old_orders, f'm.ORDERS = m.FIRST = [["default"]] * {repeats}'),
            ('format_versions={"baseline": 1005, "updated": 1005}', 'format_versions={"default": 2}'),
            ("version = 1005", "version = 2"),
            ("all(n in (8, 4096) for n, _ in lengths)", "all(n == 4096 for n, _ in lengths)"),
            (
                f'interpretation="Paced 4 MiB range claims, 16 ms completion delay and FULL pause: {window} ordinary FUSE window; both formats 1005"',
                f'interpretation="Shipping format-2 BLOB characterization, {window}; one unchanged default implementation, no candidate comparison"',
            ),
        ]
        for before, after in changes:
            assert text.count(before) == 1, before
            text = text.replace(before, after)
        reverse = text
        for before, after in reversed(changes):
            reverse = reverse.replace(after, before)
        assert reverse == original
        (target / "run_fuse.py").write_text(text)
        workload = (template / "workload.py").read_text()
        before_fence = 'assert Path(str(db) + ".stage-fence").read_bytes()[40:48] == bytes(8)'
        after_fence = 'assert not Path(str(db) + ".stage-fence").exists()'
        assert workload.count(before_fence) == 2
        workload = workload.replace(before_fence, after_fence)
        assert workload.replace(after_fence, before_fence) == (template / "workload.py").read_text()
        (target / "workload.py").write_text(workload)
        plan = json.loads((template / "plan.json").read_text())
        plan.update(
            frozen_epoch=time.time(),
            orders=[["default"]] * repeats,
            variant_gc_python_modules=dict(default=dict(path=str(native), sha256=digest(native))),
            format_versions=dict(default=2),
            runner_reverse_exact=True,
            runner_changes=changes,
            prepared_runner_sha256=digest(target / "run_fuse.py"),
            prepared_workload_sha256=digest(target / "workload.py"),
            role="Default implementation characterization, not an adoption comparison",
            workload_only_change=[before_fence, after_fence, 2],
            timed_workload_unchanged=True,
        )
        save(target / "plan.json", plan)
        save(target / "build.json", dict(status="PASS", shipping=True))
        save(target / "tests.json", dict(status="PASS", default_tests=str(out / "default-tests.json")))
        for p in [*target.glob("*.py"), *target.glob("*.json"), template / "run_fuse.py", template / "workload.py"]:
            inputs[str(p)] = digest(p)
        windows[window] = dict(root=str(target), mounts=repeats * 2)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(
        root / "build.json",
        dict(
            status="PASS",
            prepared_epoch=time.time(),
            inputs=inputs,
            windows=windows,
            free_bytes=shutil.disk_usage(out).free,
        ),
    )
    print("PASS frozen 6 full + 4 sustained default mounts; exact runner changes retained", flush=True)


def analyze(out):
    root = out / "fuse"
    build, run = [json.loads((root / name).read_text()) for name in ("build.json", "run.json")]
    assert build["status"] == run["status"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in build["inputs"].items())
    path = HERE.parent / "checkpoint_priority/analyze.py"
    spec = importlib.util.spec_from_file_location("closeout_distribution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results, cleanup, hashes = {}, [], 0
    for window, config in build["windows"].items():
        raw = json.loads((Path(config["root"]) / "ext4-plain/results.json").read_text())
        assert (
            raw["status"] == "PASS" and raw["monitor_stopped"] and raw["config"]["format_versions"] == dict(default=2)
        )
        results[window] = {}
        for state in ("fuse", "aged_fuse"):
            mounts = raw[state]
            assert len(mounts) == config["mounts"] // 2 and all(m["variant"] == "default" for m in mounts)
            phases = [p for m in mounts for p in m["samples"]]
            workloads = {}
            for name in sorted({p["workload"] for p in phases}):
                rows = [p for p in phases if p["workload"] == name]
                key = "mib_per_second" if "mib_per_second" in rows[0] else "ops_per_second"
                workloads[name] = dict(
                    rate=module.distribution([p[key] for p in rows]),
                    aggregate_rate=sum(p[key] * p["seconds"] for p in rows) / sum(p["seconds"] for p in rows),
                    count=len(rows),
                )
                for metric in ("fsync_seconds", "write_latencies", "operation_latencies"):
                    if metric in rows[0]:
                        values = (
                            [n for p in rows for n in p[metric]]
                            if isinstance(rows[0][metric], list)
                            else [p[metric] for p in rows]
                        )
                        workloads[name][metric] = module.distribution(values)
                if name.startswith(("stream_256", "write_64")):
                    assert all(
                        p["validation"]["bytes"] == (256 if name.startswith("stream_256") else 64) * 1024**2
                        for p in rows
                    )
                    hashes += len(rows)
            results[window][state] = workloads
        cleanup.extend(raw["cleanup"])
    assert hashes == 44 and len(cleanup) == 10
    assert all(row["removed"] and not Path(row["path"]).exists() for row in cleanup)
    save(
        out / "shipping-results.json",
        dict(
            status="PASS",
            recorded_epoch=time.time(),
            format=2,
            implementation="unchanged default SQLite BLOB",
            candidate_comparison=False,
            mounts=10,
            sequential_content_checks=hashes,
            windows=results,
            cleanup=cleanup,
            inputs=build["inputs"] | {str(path): digest(path), str(Path(__file__)): digest(Path(__file__))},
        ),
    )
    print("PASS default characterization, 10 mounts and 44 content hashes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tests", "freeze", "analyze"))
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    globals()[args.action](args.artifacts.resolve())
