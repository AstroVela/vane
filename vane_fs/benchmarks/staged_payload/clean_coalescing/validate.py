# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate lookup pins, publication, bounded CLEAN progress and recovery."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    sdk = repo / "vane_fs/vcpkg_installed/x64-linux-release"
    report = dict(status="RUNNING", runs=[], cleanup=[], inputs={})

    def save():
        (out / "tests.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, argv, env=None, expected=0, marker=None):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            done = subprocess.run(argv, env=env, cwd="/tmp", stdout=log, stderr=log, timeout=600)
        report["runs"].append(dict(label=label, argv=argv, start=start, end=time.time(), returncode=done.returncode))
        save()
        assert done.returncode == expected, label
        if marker:
            assert marker in (out / (label + ".log")).read_text(), label
        print(label, "PASS", flush=True)

    def clean(path):
        assert str(path) not in Path("/proc/mounts").read_text()
        for log in path.rglob("*.log"):
            dest = out / "test-logs" / path.name / log.relative_to(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log, dest)
        shutil.rmtree(path)
        report["cleanup"].append(dict(path=str(path), removed=not path.exists()))
        save()

    helpers = {
        "progress": (
            here.parent / "checkpoint_progress/test_progress.cpp",
            ("cold", "reset", "live", "reader", "stream"),
            ("sqlite3_wal_checkpoint_v2", "fsync", "fdatasync"),
        ),
        "restart": (
            here.parent / "checkpoint_restart/test_restart.cpp",
            ("progress", "reader", "partial", "writer", "external", "error", "small", "crash"),
            ("sqlite3_wal_checkpoint_v2", "sqlite3_busy_timeout", "flock"),
        ),
        "policy": (
            here.parent / "checkpoint_batching/test_batch.cpp",
            ("clean", "continuous", "no_query", "stop"),
            (
                "sqlite3_step",
                "sqlite3_wal_checkpoint_v2",
                "_ZNSt18condition_variable10notify_oneEv",
                "_ZNSt18condition_variable10notify_allEv",
            ),
        ),
        "clean": (here / "test_clean.cpp", ("burst", "stop"), ("sqlite3_wal_checkpoint_v2",)),
    }
    try:
        for name in ("lookup", "coalesced"):
            run(
                name + "-native",
                ["ctest", "--test-dir", str(out / name / "build/python-release"), "--output-on-failure"],
            )
        component = out / "coalesced"
        owned = out / "coalesced-pytest-data"
        try:
            run(
                "coalesced-python",
                [
                    str(component / "venv/bin/python"),
                    "-I",
                    "-m",
                    "pytest",
                    "--rootdir",
                    str(component),
                    "--confcutdir",
                    str(component),
                    "--basetemp",
                    str(owned),
                    "-q",
                    str(component / "tests"),
                ],
                dict(
                    os.environ,
                    VANE_FS_MOUNT_BINARY=str(component / "build/python-release/vane-fs-mount"),
                    VANE_FS_PRODUCTION_PYTHON=str(repo / ".venv/bin/python"),
                ),
            )
        finally:
            if owned.exists():
                clean(owned)
        for name, source in (("production", repo / "vane_fs"), ("coalesced", component)):
            build = out / (name + "-sanitize")
            run(
                name + "-asan-configure",
                [
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                    "-DCMAKE_PREFIX_PATH=" + str(sdk),
                    "-DVANE_FS_BUILD_PYTHON=OFF",
                    "-DVANE_FS_BUILD_TOOLS=OFF",
                    "-DVANE_FS_BUILD_FUSE=OFF",
                    "-DVANE_FS_BUILD_TESTS=ON",
                    "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer",
                    "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
                    "-DVANE_FS_TEST_SQLITE_SYNC=fsync",
                ],
            )
            run(name + "-asan-build", ["cmake", "--build", str(build), "-j", "2"])
            run(
                name + "-asan-tests",
                ["ctest", "--test-dir", str(build), "--output-on-failure"],
                dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"),
            )
        for variant, sanitized in (("lookup", False), ("coalesced", False), ("coalesced", True)):
            source = out / variant
            build = out / "coalesced-sanitize" if sanitized else source / "build/python-release"
            prefix = variant + ("-asan" if sanitized else "")
            for name, (test, cases, wrappers) in helpers.items():
                if variant == "lookup" and name != "clean":
                    continue
                report["inputs"][str(test)] = hashlib.sha256(test.read_bytes()).hexdigest()
                binary = out / (prefix + "-" + name)
                argv = [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-DNDEBUG",
                    "-I" + str(source / "src"),
                    "-I" + str(source / "include"),
                    "-I" + str(sdk / "include"),
                    str(test),
                    str(build / "libvane_fs.a"),
                    str(sdk / "lib/libsqlite3.a"),
                    "-ldl",
                    "-pthread",
                ]
                if sanitized:
                    argv += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
                argv += ["-Wl,--wrap=" + symbol for symbol in wrappers] + ["-o", str(binary)]
                run(prefix + "-" + name + "-build", argv)
                for case in cases if variant == "coalesced" else ("burst",):
                    work = Path(tempfile.mkdtemp(prefix="owned-clean-coalescing-", dir="/dev/shm"))
                    try:
                        run(
                            prefix + "-" + name + "-" + case,
                            [str(binary), str(work / "db.sqlite"), case],
                            dict(
                                os.environ,
                                ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                                UBSAN_OPTIONS="halt_on_error=1",
                            ),
                            expected=1 if variant == "lookup" else 0,
                            marker="Small FULL commits were not coalesced" if variant == "lookup" else None,
                        )
                    finally:
                        clean(work)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
