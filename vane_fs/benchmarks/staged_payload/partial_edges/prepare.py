# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build an external-payload candidate with safe partial-edge batching."""

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
    parser.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="Frozen point-read artifacts, containing the original external candidate",
    )
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    parser.add_argument("--test-support", type=Path, required=True)
    args = parser.parse_args()
    out, parent = args.output.resolve(), args.parent.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "build.json").exists()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    identity = json.loads((parent / "build.json").read_text())
    assert identity["status"] == "PASS"
    report = dict(status="RUNNING", parent=str(parent), parent_revision=identity["revision"], commands=[])

    def save():
        (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, argv, env=None, cwd=None):
        epoch = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, env=env, cwd=cwd, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            dict(label=label, argv=argv, returncode=result.returncode, started_epoch=epoch, ended_epoch=time.time())
        )
        save()
        result.check_returncode()
        print(label, "PASS", flush=True)

    try:
        for name in ("baseline", "updated"):
            target = out / name
            if not target.exists():
                shutil.copytree(
                    parent / "external",
                    target,
                    ignore=shutil.ignore_patterns("build", "venv", "probe", "regression", "*.orig", "__pycache__"),
                )
                if name == "updated":
                    run(
                        "partial-patch",
                        [
                            "patch",
                            "--batch",
                            "--fuzz=0",
                            "--no-backup-if-mismatch",
                            "-p1",
                            "-i",
                            str(here / "partial.patch"),
                        ],
                        cwd=target,
                    )
        expected = {k: v for k, v in identity["source_inputs"]["external"].items() if not k.endswith(".orig")}
        report["source_inputs"] = {}
        for name in ("baseline", "updated"):
            target = out / name
            report["source_inputs"][name] = {
                str(p.relative_to(target)): digest(p)
                for directory in ("src", "include", "python", "tests")
                for p in (target / directory).rglob("*")
                if p.is_file()
            }
        assert report["source_inputs"]["baseline"] == expected
        candidate = out / "updated"
        # Verify exactly the published patch, including when a diagnosis has
        # already prepared the candidate; do not trust an arbitrary local copy.
        check = out / "patch-check"
        shutil.copytree(out / "baseline", check)
        run(
            "patch-check",
            ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", str(here / "partial.patch")],
            cwd=check,
        )
        actual = {
            str(p.relative_to(check)): digest(p)
            for directory in ("src", "include", "python", "tests")
            for p in (check / directory).rglob("*")
            if p.is_file()
        }
        assert actual == report["source_inputs"]["updated"]
        shutil.rmtree(check)
        assert {k for k in expected if expected[k] != actual[k]} == {
            "src/workspace.cpp",
            "tests/test_core.cpp",
            "tests/test_workspace.py",
            "tests/test_staging.cpp",
        }
        sdk, fuse = args.sqlite_prefix.resolve(), args.fuse_prefix.resolve()
        env = dict(
            os.environ,
            PKG_CONFIG_PATH=str(fuse / "lib/x86_64-linux-gnu/pkgconfig"),
            PKG_CONFIG_SYSROOT_DIR=str(fuse.parent),
            CMAKE_BUILD_PARALLEL_LEVEL="2",
        )
        report["sdk"] = dict(sqlite=str(sdk), fuse=str(fuse), sqlite_sha256=digest(sdk / "lib/libsqlite3.a"))
        baseline_build = out / "baseline/build/python-release"
        baseline_build.parent.mkdir()
        baseline_build.symlink_to(parent / "external/build/python-release", target_is_directory=True)
        run("updated-venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(candidate / "venv")])
        (candidate / "venv/lib/python3.12/site-packages/build-support.pth").write_text(
            str(repo / ".venv/lib/python3.12/site-packages") + "\n" + str(args.test_support.resolve()) + "\n"
        )
        env.update(SKBUILD_BUILD_DIR=str(candidate / "build/python-release"), SKBUILD_CMAKE_BUILD_TYPE="Release")
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
        for name in ("baseline", "updated"):
            common = [
                "c++",
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "-I" + str(out / name / "include"),
                "-I" + str(sdk / "include"),
            ]
            libs = [
                str(out / name / "build/python-release/libvane_fs.a"),
                str(sdk / "lib/libsqlite3.a"),
                "-ldl",
                "-pthread",
            ]
            run(
                name + "-allocation",
                [
                    *common,
                    str(here / "allocation_probe.cpp"),
                    *libs,
                    "-Wl,--wrap=pwrite",
                    "-Wl,--wrap=pwrite64",
                    "-o",
                    str(out / name / "allocation"),
                ],
            )
        report["binaries"] = {
            str(p.relative_to(out)): digest(p)
            for name in ("baseline", "updated")
            for p in (out / name / "allocation", out / name / "build/python-release/vane-fs-mount")
        }
        # CollectGarbage is identical in both source trees. The shared wheel is
        # used only for unmounted fixture preparation, never measured I/O.
        (out / "baseline/venv").symlink_to(candidate / "venv", target_is_directory=True)
        shutil.copy2(here / "plan.json", out / "plan.json")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
