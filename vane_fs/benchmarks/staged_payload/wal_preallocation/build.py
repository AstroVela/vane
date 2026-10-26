# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build a one-time WAL reservation in isolated, non-editable prototype components."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse successful validation after an archived harness failure"
    )
    args = parser.parse_args()
    out, parent, here = args.output.resolve(), args.parent.resolve(), Path(__file__).resolve().parent
    repo = here.parents[3]
    original = json.loads((parent / "build.json").read_text())
    previous = json.loads((out / "build.json").read_text()) if args.resume else None
    assert original["status"] == "PASS"
    if previous:
        assert previous["status"] == "FAIL"
        for variant in previous["variants"].values():
            for field in ("sources", "binaries"):
                assert all(digest(Path(p)) == sha for p, sha in variant[field].items())
    else:
        assert not (out / "build.json").exists()
    assert json.loads((out / "feasibility.json").read_text())["status"] == "PASS"
    recorder = repo / "vane_fs/build/pre-open-backfill-20261013/trace-generator"
    report = dict(
        status="RUNNING",
        parent=str(parent),
        commands=previous["commands"] if previous else [],
        cleanup=previous["cleanup"] if previous else [],
        variants={},
        probes={},
        tests=previous["tests"] if previous else [],
        inputs={
            str(p): digest(p)
            for p in [
                Path(__file__),
                here / "reserve.patch",
                here / "test_reserve.cpp",
                here / "probe.cpp",
                here.parent / "checkpoint_progress/fixture.hpp",
                parent / "build.json",
                parent / "tests.json",
                out / "feasibility.json",
                recorder / "trace.cpp",
                recorder / "trace.hpp",
            ]
        },
        policy=dict(
            new_fsync_workspaces_only=True,
            reserve_bytes=16 * 1024**2,
            sql_writer_lock=True,
            initial_full_registration_unchanged=True,
            restore_chunk_size=0,
            background_and_admission_unchanged=True,
        ),
    )

    completed = list(report["commands"])

    def save():
        (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, argv, cwd=None, env=None, expected=0):
        if not label.startswith("probe-") and any(
            row["label"] == label and row["argv"] == argv and row["cwd"] == str(cwd) and row["returncode"] == expected
            for row in completed
        ):
            print(label, "REUSE passed command", flush=True)
            return
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            dict(label=label, argv=argv, cwd=str(cwd), start=start, end=time.time(), returncode=result.returncode)
        )
        save()
        assert result.returncode == expected, label
        print(label, "PASS", flush=True)

    def clean(path):
        assert str(path) not in Path("/proc/mounts").read_text()
        files = {
            str(p.relative_to(path)): dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in path.rglob("*")
            if p.is_file()
        }
        shutil.rmtree(path)
        report["cleanup"].append(dict(path=str(path), removed=not path.exists(), mount_absent=True, files=files))
        save()

    try:
        for sdk_name in ("fsync", "fdatasync"):
            old = original["components"][sdk_name]
            baseline, sdk = Path(old["path"]), Path(old["sdk"])
            for field in ("sources", "binaries"):
                for path, sha in old[field].items():
                    assert digest(Path(path)) == sha
                    report["inputs"][path] = sha
            for path in [sdk / "include/sqlite3.h", sdk / "lib/libsqlite3.a"]:
                report["inputs"][str(path)] = digest(path)
            expected = {str(Path(path).relative_to(baseline)): sha for path, sha in old["sources"].items()}
            component = out / sdk_name
            if not component.exists():
                shutil.copytree(
                    baseline,
                    component,
                    ignore=shutil.ignore_patterns("build", "venv", "sanitize", "__pycache__", "*.orig"),
                )
            if digest(component / "src/workspace.cpp") == expected["src/workspace.cpp"]:
                run(
                    sdk_name + "-patch",
                    ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(here / "reserve.patch")],
                    component,
                )
            changed = {path for path, sha in expected.items() if digest(component / path) != sha}
            assert changed == {"src/workspace.cpp"}
            proof = out / (sdk_name + "-reverse-proof")
            (proof / "src").mkdir(parents=True, exist_ok=True)
            if not (proof / "src/workspace.cpp").exists():
                shutil.copy2(component / "src/workspace.cpp", proof / "src/workspace.cpp")
            run(
                sdk_name + "-reverse-patch",
                ["patch", "--batch", "--reverse", "--fuzz=0", "-p1", "-i", str(here / "reserve.patch")],
                proof,
            )
            assert digest(proof / "src/workspace.cpp") == expected["src/workspace.cpp"]
            python = component / "venv/bin/python"
            run(sdk_name + "-venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(component / "venv")])
            (component / "venv/lib/python3.12/site-packages/support.pth").write_text(
                str(repo / ".venv/lib/python3.12/site-packages")
                + "\n"
                + str(repo.parent / "astrovela_vane-batch-udf-20260930/.venv-gpu/lib/python3.12/site-packages")
                + "\n"
            )
            fuse = repo / ".cache/fuse-sdk/root/usr"
            env = dict(
                os.environ,
                SKBUILD_BUILD_DIR=str(component / "build/python-release"),
                SKBUILD_CMAKE_BUILD_TYPE="Release",
                CMAKE_BUILD_PARALLEL_LEVEL="2",
                PKG_CONFIG_PATH=str(fuse / "lib/x86_64-linux-gnu/pkgconfig"),
                PKG_CONFIG_SYSROOT_DIR=str(fuse.parent),
            )
            run(
                sdk_name + "-install",
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    ".",
                    "--no-build-isolation",
                    "-Ccmake.define.unofficial-sqlite3_DIR=" + str(sdk / "share/unofficial-sqlite3"),
                    "-Ccmake.define.VANE_FS_BUILD_TOOLS=ON",
                    "-Ccmake.define.VANE_FS_BUILD_FUSE=ON",
                    "-Ccmake.define.VANE_FS_BUILD_TESTS=ON",
                    "-Ccmake.define.VANE_FS_TEST_SQLITE_SYNC=" + sdk_name,
                ],
                component,
                env,
            )
            release, sanitized = component / "build/python-release", component / "sanitize"
            run(sdk_name + "-native", ["ctest", "--test-dir", str(release), "--output-on-failure"])
            owned = out / (sdk_name + "-pytest-data")
            try:
                run(
                    sdk_name + "-python",
                    [
                        str(python),
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
                    "/tmp",
                    dict(
                        os.environ,
                        VANE_FS_MOUNT_BINARY=str(release / "vane-fs-mount"),
                        VANE_FS_PRODUCTION_PYTHON=str(repo / ".venv/bin/python"),
                    ),
                )
            finally:
                if owned.exists():
                    clean(owned)
            run(
                sdk_name + "-asan-configure",
                [
                    "cmake",
                    "-S",
                    str(component),
                    "-B",
                    str(sanitized),
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                    "-DCMAKE_PREFIX_PATH=" + str(sdk),
                    "-DVANE_FS_BUILD_PYTHON=OFF",
                    "-DVANE_FS_BUILD_TOOLS=OFF",
                    "-DVANE_FS_BUILD_FUSE=OFF",
                    "-DVANE_FS_BUILD_TESTS=ON",
                    "-DVANE_FS_TEST_SQLITE_SYNC=" + sdk_name,
                    "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer",
                    "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
                ],
            )
            run(sdk_name + "-asan-build", ["cmake", "--build", str(sanitized), "-j", "2"])
            asan_env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
            run(sdk_name + "-asan-tests", ["ctest", "--test-dir", str(sanitized), "--output-on-failure"], env=asan_env)
            report["variants"][sdk_name] = dict(
                sdk=str(sdk),
                baseline=str(baseline),
                reserved=str(component),
                sources={str(component / path): digest(component / path) for path in expected},
                binaries={
                    str(p): digest(p)
                    for p in [
                        release / "libvane_fs.a",
                        release / "vane-fs-mount",
                        sanitized / "libvane_fs.a",
                        component / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so",
                    ]
                },
            )
            for mode, library in (("release", release / "libvane_fs.a"), ("asan", sanitized / "libvane_fs.a")):
                binary = out / (sdk_name + "-test-" + mode)
                command = [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-DNDEBUG",
                    "-I" + str(component / "src"),
                    "-I" + str(component / "include"),
                    "-I" + str(sdk / "include"),
                    str(here / "test_reserve.cpp"),
                    str(library),
                    str(sdk / "lib/libsqlite3.a"),
                    "-ldl",
                    "-pthread",
                    "-Wl,--wrap=pwrite64",
                    "-o",
                    str(binary),
                ]
                if mode == "asan":
                    command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
                run(sdk_name + "-test-" + mode + "-build", command)
                report["probes"][str(binary)] = digest(binary)
                for case in (
                    "normal",
                    "strict",
                    "reopen",
                    "unsupported",
                    "hint_unsupported",
                    "error",
                    "partial",
                    "crash_prepare",
                    "crash_ack",
                ):
                    if dict(sdk=sdk_name, mode=mode, case=case) in report["tests"]:
                        continue
                    owned = Path(tempfile.mkdtemp(prefix="owned-wal-reserve-test-", dir="/dev/shm"))
                    try:
                        run(
                            sdk_name + "-" + mode + "-" + case,
                            [str(binary), str(owned / "db.sqlite"), case],
                            env=asan_env,
                        )
                        report["tests"].append(dict(sdk=sdk_name, mode=mode, case=case))
                    finally:
                        clean(owned)
            for variant, source in (("baseline", baseline), ("reserved", component)):
                library = source / "build/python-release/libvane_fs.a"
                for mode in ("plain", "diagnostic", "asan"):
                    binary = out / ("probe-" + sdk_name + "-" + variant + "-" + mode)
                    command = [
                        "c++",
                        "-std=c++17",
                        "-O2",
                        "-DNDEBUG",
                        "-I" + str(source / "src"),
                        "-I" + str(source / "include"),
                        "-I" + str(sdk / "include"),
                        str(here / "probe.cpp"),
                    ]
                    if mode == "diagnostic":
                        command += ["-DVANE_WAL_TRACE=1", "-I" + str(recorder), str(recorder / "trace.cpp")]
                        command += [
                            "-Wl,--wrap=" + symbol
                            for symbol in (
                                "sqlite3_wal_checkpoint_v2",
                                "fsync",
                                "fdatasync",
                                "pread",
                                "pwrite",
                                "pread64",
                                "pwrite64",
                                "sync_file_range",
                            )
                        ]
                    if mode == "asan":
                        library = source / "sanitize/libvane_fs.a"
                        command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
                    command += [
                        str(library),
                        str(sdk / "lib/libsqlite3.a"),
                        "-ldl",
                        "-pthread",
                        "-Wl,--wrap=sqlite3_wal_hook",
                        "-o",
                        str(binary),
                    ]
                    run(binary.name + "-build", command)
                    report["probes"][str(binary)] = digest(binary)
                    for workload in ("small128", "write64"):
                        owned = Path(tempfile.mkdtemp(prefix="owned-wal-probe-test-", dir="/dev/shm"))
                        label = binary.name + "-" + workload
                        try:
                            run(
                                label,
                                [str(binary), str(owned / "db.sqlite"), workload, str(out / (label + ".json"))],
                                env=asan_env,
                            )
                        finally:
                            for path in owned.glob("*.request-costs.json"):
                                shutil.copy2(path, out / (label + "-trace.json"))
                            clean(owned)
            # Demonstrate that the new allocation test detects the old implementation.
            owned = Path(tempfile.mkdtemp(prefix="owned-wal-reserve-negative-", dir="/dev/shm"))
            try:
                binary = out / (sdk_name + "-test-negative")
                run(
                    sdk_name + "-negative-build",
                    [
                        "c++",
                        "-std=c++17",
                        "-O2",
                        "-DNDEBUG",
                        "-I" + str(baseline / "src"),
                        "-I" + str(baseline / "include"),
                        "-I" + str(sdk / "include"),
                        str(here / "test_reserve.cpp"),
                        str(baseline / "build/python-release/libvane_fs.a"),
                        str(sdk / "lib/libsqlite3.a"),
                        "-ldl",
                        "-pthread",
                        "-Wl,--wrap=pwrite64",
                        "-o",
                        str(binary),
                    ],
                )
                run(sdk_name + "-negative", [str(binary), str(owned / "db.sqlite"), "normal"], expected=1)
                assert "New filesystem omitted WAL reservation" in (out / (sdk_name + "-negative.log")).read_text()
            finally:
                clean(owned)
        assert digest(out / "fsync/build/python-release/libvane_fs.a") == digest(
            out / "fdatasync/build/python-release/libvane_fs.a"
        )
        assert all(digest(Path(p)) == sha for p, sha in report["inputs"].items())
        report.update(status="PASS", candidate_core_libraries_identical=True)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
