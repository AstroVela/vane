# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build an isolated orphan batching candidate over the no-hint adoption control."""

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
    out.mkdir()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    control = parent / "updated"
    identity = json.loads((parent / "build.json").read_text())
    assert identity["status"] == "PASS"
    expected = identity["source_inputs"]["updated"]
    assert all(digest(control / p) == h for p, h in expected.items())
    sdk, fuse = args.sqlite_prefix.resolve(), args.fuse_prefix.resolve()
    report = dict(
        status="RUNNING",
        parent=str(parent),
        commands=[],
        source_inputs={"baseline": expected},
        binaries={},
        sdk=dict(sqlite=str(sdk), fuse=str(fuse), sqlite_sha256=digest(sdk / "lib/libsqlite3.a")),
        inputs={str(p): digest(p) for p in (here / "prepare.py", here / "batch.patch", here / "test_orphans.cpp")},
    )

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
        candidate = out / "updated"
        shutil.copytree(
            control,
            candidate,
            ignore=shutil.ignore_patterns("build", "venv", "sanitize", "allocation", "*.orig", "__pycache__"),
        )
        run(
            "orphan-patch",
            ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", str(here / "batch.patch")],
            cwd=candidate,
        )
        report["source_inputs"]["updated"] = {
            str(p.relative_to(candidate)): digest(p)
            for d in ("src", "include", "python", "tests")
            for p in (candidate / d).rglob("*")
            if p.is_file()
        }
        assert {p for p, h in expected.items() if h != report["source_inputs"]["updated"][p]} == {"src/workspace.cpp"}
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
        (out / "baseline/build").mkdir(parents=True)
        (out / "baseline/build/python-release").symlink_to(control / "build/python-release", target_is_directory=True)
        (out / "baseline/venv").symlink_to(candidate / "venv", target_is_directory=True)
        for name, source in (("baseline", control), ("updated", candidate)):
            run(
                name + "-orphan-test",
                [
                    "c++",
                    "-std=c++17",
                    "-O3",
                    "-DNDEBUG",
                    "-I" + str(source / "include"),
                    "-I" + str(sdk / "include"),
                    str(here / "test_orphans.cpp"),
                    str(source / "build/python-release/libvane_fs.a"),
                    str(sdk / "lib/libsqlite3.a"),
                    "-ldl",
                    "-pthread",
                    "-Wl,--wrap=sqlite3_step",
                    "-o",
                    str(out / name / "orphan-test"),
                ],
            )
            for binary in ("orphan-test", "build/python-release/vane-fs-mount", "build/python-release/vane-fs"):
                report["binaries"][name + "/" + binary] = digest(out / name / binary)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
