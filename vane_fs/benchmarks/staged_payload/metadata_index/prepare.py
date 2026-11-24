# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build a block high-index removal over frozen orphan batching, with unchanged adoption control."""

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
    parser.add_argument("--orphan-parent", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    parser.add_argument("--test-support", type=Path, required=True)
    args = parser.parse_args()
    out, parent = args.output.resolve(), args.parent.resolve()
    out.mkdir()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    control = parent / "updated"
    orphan_parent = args.orphan_parent.resolve()
    orphan = orphan_parent / "updated"
    identity = json.loads((parent / "build.json").read_text())
    assert identity["status"] == "PASS"
    expected = identity["source_inputs"]["updated"]
    orphan_expected = json.loads((orphan_parent / "build.json").read_text())["source_inputs"]["updated"]
    assert all(digest(orphan / p) == h for p, h in orphan_expected.items())
    assert all(digest(control / p) == h for p, h in expected.items())
    sdk, fuse = args.sqlite_prefix.resolve(), args.fuse_prefix.resolve()
    report = dict(
        status="RUNNING",
        parent=str(parent),
        orphan_parent=str(orphan_parent),
        commands=[],
        source_inputs={"baseline": expected, "orphan-control": orphan_expected},
        binaries={},
        sdk=dict(sqlite=str(sdk), fuse=str(fuse), sqlite_sha256=digest(sdk / "lib/libsqlite3.a")),
        inputs={
            str(p): digest(p)
            for p in (
                here / "prepare.py",
                here / "index.patch",
                here / "test_index.cpp",
                here.parent / "orphan_batching/test_orphans.cpp",
                here.parent / "checkpoint_batching/test_batch.cpp",
            )
        },
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
            orphan,
            candidate,
            ignore=shutil.ignore_patterns("build", "venv", "sanitize", "allocation", "*.orig", "__pycache__"),
        )
        run(
            "index-patch",
            ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", str(here / "index.patch")],
            cwd=candidate,
        )
        report["source_inputs"]["updated"] = {
            str(p.relative_to(candidate)): digest(p)
            for d in ("src", "include", "python", "tests")
            for p in (candidate / d).rglob("*")
            if p.is_file()
        }
        assert {p for p, h in orphan_expected.items() if h != report["source_inputs"]["updated"][p]} == {
            "src/workspace.cpp",
        }
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
        for name, source in (("baseline", control), ("orphan-control", orphan)):
            (out / name / "build").mkdir(parents=True)
            (out / name / "build/python-release").symlink_to(source / "build/python-release", target_is_directory=True)
            (out / name / "venv").symlink_to(source / "venv", target_is_directory=True)
        for name, source in (
            ("baseline", control),
            ("orphan-control", orphan),
            ("updated", candidate),
        ):
            common = [
                "c++",
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "-I" + str(source / "include"),
                "-I" + str(source / "src"),
                "-I" + str(sdk / "include"),
            ]
            libs = [
                str(source / "build/python-release/libvane_fs.a"),
                str(sdk / "lib/libsqlite3.a"),
                "-ldl",
                "-pthread",
            ]
            if name != "baseline":
                run(
                    name + "-index-test",
                    common
                    + (["-DVANE_TEST_INDEX_REMOVED"] if name == "updated" else [])
                    + [str(here / "test_index.cpp")]
                    + libs
                    + [
                        "-Wl,--wrap=sqlite3_step",
                        "-o",
                        str(out / name / "index-test"),
                    ],
                )
                report["binaries"][name + "/index-test"] = digest(out / name / "index-test")
            if name in ("baseline", "orphan-control", "updated"):
                run(
                    name + "-policy-test",
                    common
                    + [str(here.parent / "checkpoint_batching/test_batch.cpp")]
                    + libs
                    + [
                        "-Wl,--wrap=" + symbol
                        for symbol in (
                            "sqlite3_step",
                            "sqlite3_wal_checkpoint_v2",
                            "_ZNSt18condition_variable10notify_oneEv",
                            "_ZNSt18condition_variable10notify_allEv",
                        )
                    ]
                    + ["-o", str(out / name / "policy-test")],
                )
                report["binaries"][name + "/policy-test"] = digest(out / name / "policy-test")
                if name != "baseline":
                    run(
                        name + "-orphan-test",
                        common
                        + [str(here.parent / "orphan_batching/test_orphans.cpp")]
                        + libs
                        + ["-Wl,--wrap=sqlite3_step", "-o", str(out / name / "orphan-test")],
                    )
                    report["binaries"][name + "/orphan-test"] = digest(out / name / "orphan-test")
                for binary in ("build/python-release/vane-fs-mount", "build/python-release/vane-fs"):
                    report["binaries"][name + "/" + binary] = digest(out / name / binary)
        report["python_modules"] = {}
        for name, source in (("baseline", control), ("orphan-control", orphan), ("updated", candidate)):
            native = (
                source / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so"
            ).resolve()
            report["python_modules"][name] = dict(path=str(native), sha256=digest(native))
            report["inputs"][str(native)] = digest(native)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
