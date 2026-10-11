# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Install an isolated payload-writeback candidate; reuse the frozen control."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    parser.add_argument("--test-support", type=Path, required=True)
    args = parser.parse_args()
    out, parent = args.output.resolve(), args.parent.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "build.json").exists()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    prior = json.loads((parent / "build.json").read_text())
    assert prior["status"] == "PASS"
    control = parent / "updated"
    expected = prior["source_inputs"]["updated"]
    assert all(digest(control / p) == h for p, h in expected.items())
    report = dict(status="RUNNING", parent=str(parent), commands=[], source_inputs={}, binaries={})

    def save():
        (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, argv, env=None, cwd=None):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, env=env, cwd=cwd, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            dict(label=label, argv=argv, returncode=result.returncode, started_epoch=start, ended_epoch=time.time())
        )
        save()
        result.check_returncode()
        print(label, "PASS", flush=True)

    try:
        check = out / "patch-check"
        shutil.copytree(
            control,
            check,
            ignore=shutil.ignore_patterns("build", "venv", "sanitize", "allocation", "*.orig", "__pycache__"),
        )
        run(
            "writeback-patch",
            ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", str(here / "background.patch")],
            cwd=check,
        )
        candidate = out / "updated"
        if not candidate.exists():
            check.rename(candidate)
        else:
            assert all(
                digest(candidate / p.relative_to(check)) == digest(p)
                for directory in ("src", "include", "python", "tests")
                for p in (check / directory).rglob("*")
                if p.is_file()
            )
            shutil.rmtree(check)
        report["source_inputs"]["baseline"] = expected
        report["source_inputs"]["updated"] = {
            str(p.relative_to(candidate)): digest(p)
            for directory in ("src", "include", "python", "tests")
            for p in (candidate / directory).rglob("*")
            if p.is_file()
        }
        assert {p for p in expected if expected[p] != report["source_inputs"]["updated"][p]} == {
            "src/staging.hpp",
            "src/workspace.cpp",
        }
        sdk, fuse = args.sqlite_prefix.resolve(), args.fuse_prefix.resolve()
        report["sdk"] = dict(sqlite=str(sdk), fuse=str(fuse), sqlite_sha256=digest(sdk / "lib/libsqlite3.a"))
        build = candidate / "build/python-release"
        env = dict(
            os.environ,
            PKG_CONFIG_PATH=str(fuse / "lib/x86_64-linux-gnu/pkgconfig"),
            PKG_CONFIG_SYSROOT_DIR=str(fuse.parent),
            CMAKE_BUILD_PARALLEL_LEVEL="2",
            SKBUILD_BUILD_DIR=str(build),
            SKBUILD_CMAKE_BUILD_TYPE="Release",
        )
        run("updated-venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(candidate / "venv")])
        (candidate / "venv/lib/python3.12/site-packages/build-support.pth").write_text(
            str(repo / ".venv/lib/python3.12/site-packages") + "\n" + str(args.test_support.resolve()) + "\n"
        )
        run(
            "updated-install",
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(candidate / "venv/bin/python"),
                str(candidate),
                "--no-build-isolation",
                "-Ccmake.define.unofficial-sqlite3_DIR=" + str(sdk / "share/unofficial-sqlite3"),
                "-Ccmake.define.VANE_FS_BUILD_TOOLS=ON",
                "-Ccmake.define.VANE_FS_BUILD_FUSE=ON",
                "-Ccmake.define.VANE_FS_BUILD_TESTS=ON",
                "-Ccmake.define.VANE_FS_TEST_SQLITE_SYNC=fsync",
            ],
            env,
        )
        baseline = out / "baseline"
        baseline.mkdir(exist_ok=True)
        (baseline / "build").mkdir(exist_ok=True)
        old_build = baseline / "build/python-release"
        if not old_build.exists():
            old_build.symlink_to(control / "build/python-release", target_is_directory=True)
        (baseline / "venv").symlink_to(candidate / "venv", target_is_directory=True)
        for name, component in (("baseline", control), ("updated", candidate)):
            run(
                name + "-writeback-test",
                [
                    "c++",
                    "-std=c++17",
                    "-O3",
                    "-DNDEBUG",
                    "-I" + str(component / "include"),
                    str(here / "test_background.cpp"),
                    str(component / "build/python-release/libvane_fs.a"),
                    str(sdk / "lib/libsqlite3.a"),
                    "-ldl",
                    "-pthread",
                    "-Wl,--wrap=sync_file_range",
                    "-Wl,--wrap=fdatasync",
                    "-Wl,--wrap=fsync",
                    "-Wl,--wrap=close",
                    "-o",
                    str(out / name / "writeback-test"),
                ],
            )
        for name in ("baseline", "updated"):
            report["binaries"][name + "/build/python-release/vane-fs-mount"] = digest(
                out / name / "build/python-release/vane-fs-mount"
            )
            report["binaries"][name + "/writeback-test"] = digest(out / name / "writeback-test")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
