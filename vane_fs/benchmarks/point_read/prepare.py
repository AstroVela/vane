# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build frozen production/external pairs with a single-block equality query."""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

BASELINE = "4e83455308faea9d5f6fc45200df20a608ec6f25"
VARIANTS = ("blob", "blob_point", "external", "external_point")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    parser.add_argument("--test-support", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "build.json").exists()
    report = {"status": "RUNNING", "revision": BASELINE, "commands": []}

    def save():
        (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command, env=None, cwd=None):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(command, env=env, cwd=cwd, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            dict(label=label, argv=command, returncode=result.returncode, started_epoch=start, ended_epoch=time.time())
        )
        save()
        result.check_returncode()
        print(label, "PASS", flush=True)

    try:
        names = (
            "src",
            "include",
            "python",
            "tests",
            "LICENSES",
            "LICENSE",
            "README.md",
            "pyproject.toml",
            "CMakeLists.txt",
            "benchmarks/external_payload",
            "benchmarks/staged_payload",
        )
        archive = subprocess.check_output(
            ["git", "archive", BASELINE, "--", *["vane_fs/" + n for n in names]], cwd=repo
        )
        inputs = out / "input"
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for member in source:
                parts = Path(member.name).parts
                assert parts[0] == "vane_fs" and ".." not in parts
                if member.isdir():
                    continue
                assert member.isfile()
                path = inputs.joinpath(*parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream:
                    path.write_bytes(stream.read())
                path.chmod(member.mode & 0o777)
        generated = out / "components"
        run(
            "parent-prepare",
            [
                str(repo / ".venv/bin/python"),
                str(inputs / "vane_fs/benchmarks/staged_payload/random_io/prepare.py"),
                str(generated),
            ],
        )
        for name, parent in (("blob", "baseline"), ("external", "cached")):
            shutil.copytree(generated / parent, out / name)
            shutil.copytree(out / name, out / (name + "_point"))
            run(
                name + "-patch",
                ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", str(here / "point.patch")],
                cwd=out / (name + "_point"),
            )
        # Only newly prepared, unused source copies; no measured data exists yet.
        shutil.rmtree(generated)
        sdk, fuse = args.sqlite_prefix.resolve(), args.fuse_prefix.resolve()
        env = dict(
            os.environ,
            PKG_CONFIG_PATH=str(fuse / "lib/x86_64-linux-gnu/pkgconfig"),
            PKG_CONFIG_SYSROOT_DIR=str(fuse.parent),
            CMAKE_BUILD_PARALLEL_LEVEL="2",
        )
        report["environment"] = {
            k: env[k] for k in ("PKG_CONFIG_PATH", "PKG_CONFIG_SYSROOT_DIR", "CMAKE_BUILD_PARALLEL_LEVEL")
        }
        report["sdk"] = {"sqlite": str(sdk), "fuse": str(fuse), "sqlite_sha256": digest(sdk / "lib/libsqlite3.a")}
        report["source_inputs"] = {}
        for name in VARIANTS:
            component = out / name
            report["source_inputs"][name] = {
                str(p.relative_to(component)): digest(p)
                for directory in ("src", "include", "python", "tests")
                for p in (component / directory).rglob("*")
                if p.is_file()
            }
            build = component / "build/python-release"
            if name.endswith("_point"):
                run(name + "-venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(component / "venv")])
                (component / "venv/lib/python3.12/site-packages/build-support.pth").write_text(
                    str(repo / ".venv/lib/python3.12/site-packages") + "\n" + str(args.test_support.resolve()) + "\n"
                )
                package_env = dict(env, SKBUILD_BUILD_DIR=str(build), SKBUILD_CMAKE_BUILD_TYPE="Release")
                run(
                    name + "-install",
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(component / "venv/bin/python"),
                        str(component),
                        "--no-build-isolation",
                        "-Ccmake.define.unofficial-sqlite3_DIR=" + str(sdk / "share/unofficial-sqlite3"),
                        "-Ccmake.define.VANE_FS_BUILD_TOOLS=ON",
                        "-Ccmake.define.VANE_FS_BUILD_FUSE=ON",
                        "-Ccmake.define.VANE_FS_BUILD_TESTS=ON",
                        "-Ccmake.define.VANE_FS_TEST_SQLITE_SYNC=fsync",
                    ],
                    package_env,
                )
            else:
                run(
                    name + "-configure",
                    [
                        "cmake",
                        "-S",
                        str(component),
                        "-B",
                        str(build),
                        "-G",
                        "Ninja",
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DCMAKE_PREFIX_PATH=" + str(sdk),
                        "-DVANE_FS_BUILD_PYTHON=OFF",
                        "-DVANE_FS_BUILD_TOOLS=ON",
                        "-DVANE_FS_BUILD_FUSE=ON",
                        "-DVANE_FS_BUILD_TESTS=ON",
                        "-DVANE_FS_TEST_SQLITE_SYNC=fsync",
                    ],
                    env,
                )
                run(name + "-build", ["cmake", "--build", str(build), "-j", "2"], env)
            common = [
                "c++",
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "-I" + str(component / "include"),
                "-I" + str(sdk / "include"),
            ]
            libs = [str(build / "libvane_fs.a"), str(sdk / "lib/libsqlite3.a"), "-ldl", "-pthread"]
            run(name + "-probe", [*common, str(here / "probe.cpp"), *libs, "-o", str(component / "probe")])
            run(
                name + "-regression",
                [
                    *common,
                    str(here / "test_point_read.cpp"),
                    *libs,
                    "-Wl,--wrap=sqlite3_step",
                    "-o",
                    str(component / "regression"),
                ],
            )
        for pair in (("blob", "blob_point"), ("external", "external_point")):
            left, right = (report["source_inputs"][x] for x in pair)
            assert left.keys() == right.keys()
            assert {k for k in left if left[k] != right[k]} == {"src/workspace.cpp"}
        report["binaries"] = {
            str(p.relative_to(out)): digest(p)
            for name in VARIANTS
            for p in (
                out / name / "probe",
                out / name / "regression",
                out / name / "build/python-release/vane-fs-mount",
            )
        }
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
