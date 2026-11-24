# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Incrementally install and validate the isolated extent component."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from generate import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--stage", choices=("release", "python", "sanitize", "custom", "all"), default="all")
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    repo = here.parents[3]
    source = json.loads((out / "source.json").read_text())
    assert source["status"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in dict(source["inputs"], **source["sources"]).items())
    target, sdk = Path(source["candidate"]), Path(source["sdk"])
    fingerprint = hashlib.sha256(json.dumps(source["sources"], sort_keys=True).encode()).hexdigest()
    previous = json.loads((out / "build.json").read_text()) if (out / "build.json").exists() else None
    report = dict(
        status="RUNNING",
        started_epoch=time.time(),
        source_fingerprint=fingerprint,
        source_manifest=source,
        commands=previous["commands"] if previous else [],
        cleanup=previous["cleanup"] if previous else [],
        stages=previous["stages"] if previous and previous["source_fingerprint"] == fingerprint else {},
        binaries={},
        inputs={},
    )
    attempt = len(list(out.glob("build-harness-*.py")))
    shutil.copy2(Path(__file__), out / f"build-harness-{attempt:03}.py")
    shutil.copy2(out / "source.json", out / f"build-source-{attempt:03}.json")
    if (out / "build.json").exists():
        shutil.copy2(out / "build.json", out / f"build-before-{attempt:03}.json")

    def run(label, argv, cwd=None, env=None):
        log = out / f"{len(report['commands']):03}-{label}.log"
        begin = time.time()
        with log.open("w") as stream:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=stream, timeout=900)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                cwd=str(cwd),
                source_fingerprint=fingerprint,
                started_epoch=begin,
                ended_epoch=time.time(),
                returncode=result.returncode,
                log=str(log),
                log_sha256=digest(log),
            )
        )
        save(out / "build.json", report)
        assert result.returncode == 0, log
        print(label, "PASS", flush=True)

    def clean(path):
        assert str(path) not in Path("/proc/mounts").read_text()
        files = {
            str(p.relative_to(path)): dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in path.rglob("*")
            if p.is_file()
        }
        shutil.rmtree(path)
        report["cleanup"].append(
            dict(path=str(path), files=files, removed=not path.exists(), processes_stopped=True, mount_absent=True)
        )

    def ctest(label, directory, env=None):
        owned = Path(tempfile.mkdtemp(prefix="owned-extent-" + label + "-", dir="/dev/shm"))
        try:
            run(
                label,
                ["ctest", "--test-dir", str(directory), "--output-on-failure"],
                env=dict(os.environ, **(env or {}), TMPDIR=str(owned)),
            )
        finally:
            clean(owned)
            for path in directory.glob("owned*"):
                if path.is_dir():
                    clean(path)

    try:
        release, sanitize = target / "build/python-release", target / "sanitize"
        python = target / "venv/bin/python"
        if args.stage in ("release", "all"):
            if not python.exists():
                run("venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(target / "venv")])
            support = [
                Path(source["original"]) / "venv/lib/python3.12/site-packages",
                repo / ".venv/lib/python3.12/site-packages",
            ]
            (target / "venv/lib/python3.12/site-packages/support.pth").write_text(
                "".join(str(p) + "\n" for p in support)
            )
            fuse = repo / ".cache/fuse-sdk/root/usr"
            env = dict(
                os.environ,
                SKBUILD_BUILD_DIR=str(release),
                SKBUILD_CMAKE_BUILD_TYPE="Release",
                CMAKE_BUILD_PARALLEL_LEVEL="2",
                PKG_CONFIG_PATH=str(fuse / "lib/x86_64-linux-gnu/pkgconfig"),
                PKG_CONFIG_SYSROOT_DIR=str(fuse.parent),
            )
            run(
                "install",
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
                    "-Ccmake.define.VANE_FS_TEST_SQLITE_SYNC=fsync",
                ],
                target,
                env,
            )
            ctest("release-components", release)
            report["stages"]["release"] = "PASS"
        if args.stage in ("python", "all"):
            assert report["stages"].get("release") == "PASS"
            owned = Path(tempfile.mkdtemp(prefix="owned-extent-python-", dir="/dev/shm"))
            try:
                run(
                    "python",
                    [
                        str(python),
                        "-I",
                        "-m",
                        "pytest",
                        "--rootdir",
                        str(target),
                        "--confcutdir",
                        str(target),
                        "--basetemp",
                        str(owned),
                        "-q",
                        str(target / "tests"),
                    ],
                    "/tmp",
                    dict(
                        os.environ,
                        VANE_FS_MOUNT_BINARY=str(release / "vane-fs-mount"),
                        VANE_FS_PRODUCTION_PYTHON=str(repo / ".venv/bin/python"),
                    ),
                )
            finally:
                clean(owned)
            report["stages"]["python"] = "PASS"
        if args.stage in ("sanitize", "all"):
            run(
                "asan-configure",
                [
                    "cmake",
                    "-S",
                    str(target),
                    "-B",
                    str(sanitize),
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                    "-DCMAKE_PREFIX_PATH=" + str(sdk),
                    "-DVANE_FS_BUILD_PYTHON=OFF",
                    "-DVANE_FS_BUILD_TOOLS=OFF",
                    "-DVANE_FS_BUILD_FUSE=OFF",
                    "-DVANE_FS_BUILD_TESTS=ON",
                    "-DVANE_FS_TEST_SQLITE_SYNC=fsync",
                    "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer",
                    "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
                ],
            )
            run("asan-build", ["cmake", "--build", str(sanitize), "-j", "2"])
            ctest(
                "asan-components",
                sanitize,
                dict(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"),
            )
            report["stages"]["sanitize"] = "PASS"
        if args.stage in ("custom", "all"):
            helper = here / "test_extents.cpp"
            assert report["stages"].get("release") == report["stages"].get("sanitize") == "PASS"
            for mode, directory in [("release", release), ("asan", sanitize)]:
                binary = out / ("test-extents-" + mode)
                flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if mode == "asan" else []
                run(
                    mode + "-extent-compile",
                    [
                        "c++",
                        "-std=c++17",
                        "-O2",
                        "-I" + str(target / "include"),
                        "-I" + str(sdk / "include"),
                        str(helper),
                        str(directory / "libvane_fs.a"),
                        str(sdk / "lib/libsqlite3.a"),
                        "-ldl",
                        "-pthread",
                        *flags,
                        "-o",
                        str(binary),
                    ],
                )
                owned = Path(tempfile.mkdtemp(prefix="owned-extent-cases-", dir="/dev/shm"))
                try:
                    run(
                        mode + "-extent-cases",
                        [str(binary), str(owned)],
                        env=dict(
                            os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"
                        ),
                    )
                finally:
                    clean(owned)
                report["binaries"][str(binary)] = digest(binary)
            report["inputs"][str(helper)] = digest(helper)
            report["stages"]["custom"] = "PASS"
        report["binaries"].update(
            {
                str(path): digest(path)
                for path in [
                    release / "libvane_fs.a",
                    release / "vane-fs-mount",
                    release / "vane-fs",
                    *target.glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"),
                    *sanitize.glob("libvane_fs.a"),
                ]
                if path.is_file()
            }
        )
        report["inputs"][str(Path(__file__))] = digest(Path(__file__))
        assert all(digest(Path(p)) == sha for p, sha in dict(source["inputs"], **source["sources"]).items())
        report["status"] = (
            "PASS"
            if all(report["stages"].get(stage) == "PASS" for stage in ("release", "python", "sanitize", "custom"))
            else "PARTIAL"
        )
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "build.json", report)


if __name__ == "__main__":
    main()
