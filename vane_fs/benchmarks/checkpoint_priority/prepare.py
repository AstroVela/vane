# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build an isolated checkpoint-priority comparison from a frozen revision."""

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

BASELINE = "8e21f48ea64c7811893d847116ccc4fbf204f81d"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    parser.add_argument(
        "--test-support", type=Path, required=True, help="Existing Python test dependency site-packages"
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    out = args.output.resolve()
    if out.exists():
        parser.error("Choose a new output directory")
    out.mkdir(parents=True)
    report = {"status": "RUNNING", "baseline_revision": BASELINE, "started_epoch": time.time(), "commands": []}

    def save():
        (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command, env=None, cwd=None):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(command, env=env, cwd=cwd, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            {
                "label": label,
                "argv": command,
                "returncode": result.returncode,
                "started_epoch": start,
                "ended_epoch": time.time(),
            }
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
            "triplets",
            "LICENSES",
            "CMakeLists.txt",
            "pyproject.toml",
            "vcpkg.json",
            "README.md",
            "DESIGN.md",
            "LICENSE",
        )
        archive = subprocess.check_output(
            ["git", "archive", BASELINE, "--", *["vane_fs/" + n for n in names]], cwd=repo
        )
        baseline = out / "baseline"
        baseline.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for member in source:
                parts = Path(member.name).parts
                assert parts[0] == "vane_fs" and ".." not in parts
                path = baseline.joinpath(*parts[1:])
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    assert member.isfile()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as stream:
                        path.write_bytes(stream.read())
                    path.chmod(member.mode & 0o777)
        candidate = out / "updated"
        shutil.copytree(baseline, candidate)
        run("patch", ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(here / "priority.patch")], cwd=candidate)
        identities = {}
        for name in ("baseline", "updated"):
            component = out / name
            identities[name] = {
                str(p.relative_to(component)): digest(p)
                for directory in ("src", "include", "python", "tests")
                for p in (component / directory).rglob("*")
                if p.is_file()
            }
        assert {p for p in identities["baseline"] if identities["baseline"][p] != identities["updated"][p]} == {
            "src/checkpoint.hpp"
        }
        report["source_inputs"] = identities
        sdk, fuse = args.sqlite_prefix.resolve(), args.fuse_prefix.resolve()
        env = dict(
            os.environ,
            PKG_CONFIG_PATH=str(fuse / "lib/x86_64-linux-gnu/pkgconfig"),
            PKG_CONFIG_SYSROOT_DIR=str(fuse.parent),
        )
        build = baseline / "build/python-release"
        run(
            "baseline-configure",
            [
                "cmake",
                "-S",
                str(baseline),
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
        run("baseline-build", ["cmake", "--build", str(build), "-j", "2"], env)
        (baseline / "venv").symlink_to(repo / ".venv", target_is_directory=True)
        run("candidate-venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(candidate / "venv")])
        support = candidate / "venv/lib/python3.12/site-packages/build-support.pth"
        support.write_text(
            str(repo / ".venv/lib/python3.12/site-packages") + "\n" + str(args.test_support.resolve()) + "\n"
        )
        report["test_support"] = str(args.test_support.resolve())
        env.update(SKBUILD_BUILD_DIR=str(candidate / "build/python-release"), SKBUILD_CMAKE_BUILD_TYPE="Release")
        run(
            "candidate-install",
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
        # Only the scheduling probes pause admission and observe a worker yield.
        # Plain and storage-diagnostic libraries contain neither test hook.
        for name in ("baseline", "updated"):
            source = (out / name / "src/checkpoint.hpp").read_text()
            source = source.replace(
                "namespace vane_fs {",
                "void checkpoint_priority_before_lock();\nvoid checkpoint_priority_yield();\n\nnamespace vane_fs {",
                1,
            )
            needle = "\n\t\tstd::lock_guard<std::mutex> checkpoint_lock(checkpoint_mutex);"
            assert source.count(needle) == 1
            source = source.replace(needle, "\n\t\tcheckpoint_priority_before_lock();" + needle)
            if name == "updated":
                needle = "\t\t\tif (yielded) {"
                assert source.count(needle) == 1
                source = source.replace(needle, needle + "\n\t\t\t\tcheckpoint_priority_yield();")
            target = out / "scheduling" / name
            target.mkdir(parents=True)
            (target / "checkpoint.hpp").write_text(source)
            run(
                name + "-scheduling-build",
                [
                    "g++",
                    "-std=c++17",
                    "-O3",
                    "-DNDEBUG",
                    "-I" + str(target),
                    "-I" + str(out / name / "include"),
                    "-I" + str(sdk / "include"),
                    str(here / "test_priority.cpp"),
                    str(sdk / "lib/libsqlite3.a"),
                    "-ldl",
                    "-pthread",
                    "-Wl,--wrap=sqlite3_wal_checkpoint_v2",
                    "-o",
                    str(target / "priority-test"),
                ],
            )
        acceptance = here.parent / "partial_write/acceptance"
        # Reuse the published probe builder with an additional diagnostic-only
        # yield event. Keep the original builder and generated adapter frozen.
        frozen = out / "frozen"
        frozen.mkdir()
        original = (acceptance / "prepare.py").read_text()
        (frozen / "acceptance-prepare.py").write_text(original)
        adapter = original.replace(
            "here = Path(__file__).resolve().parent", "here = Path(" + repr(str(acceptance)) + ")", 1
        )
        needle = "            header.write_text(source)"
        assert adapter.count(needle) == 1
        adapter = adapter.replace(
            needle,
            """            if name == "updated":
                needle = "\\t\\t\\tif (yielded) {"
                assert source.count(needle) == 1
                source = source.replace(needle, needle + '\\n\\t\\t\\t\\tpartial_trace("worker_yield", std::chrono::steady_clock::now());')
"""
            + needle,
        )
        probe_preparer = frozen / "priority-probes.py"
        probe_preparer.write_text(adapter)
        run(
            "probe-build",
            [
                str(repo / ".venv/bin/python"),
                str(probe_preparer),
                str(out),
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--sqlite-prefix",
                str(sdk),
                "--fuse-prefix",
                str(fuse),
            ],
        )
        report["inputs"] = {
            str(p): digest(p)
            for p in (
                here / "priority.patch",
                here / "test_priority.cpp",
                probe_preparer,
                sdk / "lib/libsqlite3.a",
                fuse / "lib/x86_64-linux-gnu/libfuse3.so",
            )
        }
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save()


if __name__ == "__main__":
    main()
