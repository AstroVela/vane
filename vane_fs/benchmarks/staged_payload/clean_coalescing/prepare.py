# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build lookup-only and CLEAN-coalesced candidates from one frozen payload parent."""

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
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    out, parent = args.output.resolve(), args.parent.resolve()
    out.mkdir(exist_ok=True)
    sdk, fuse = repo / "vane_fs/vcpkg_installed/x64-linux-release", repo / ".cache/fuse-sdk/root/usr"
    expected = json.loads((parent / "build.json").read_text())["source_inputs"]["updated"]
    assert all(digest(parent / "updated" / p) == h for p, h in expected.items())
    lookup = here.parents[1] / "lookup_reads/lookup.patch"
    report = dict(status="RUNNING", parent=str(parent), commands=[], variants={}, inputs={})
    for path in (Path(__file__).resolve(), lookup, here / "clean.patch", sdk / "lib/libsqlite3.a"):
        report["inputs"][str(path)] = digest(path)

    def save():
        (out / "candidates.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, argv, cwd=None, env=None):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            dict(label=label, argv=argv, cwd=str(cwd), start=start, end=time.time(), returncode=result.returncode)
        )
        save()
        result.check_returncode()
        print(label, "PASS", flush=True)

    try:
        for name in ("lookup", "coalesced"):
            component = out / name
            if not component.exists():
                shutil.copytree(
                    parent / "updated",
                    component,
                    ignore=shutil.ignore_patterns("build", "venv", "sanitize", "allocation", "__pycache__", "*.orig"),
                )
                for patch in [lookup] if name == "lookup" else [lookup, here / "clean.patch"]:
                    run(
                        name + "-" + patch.stem,
                        ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", str(patch)],
                        cwd=component,
                    )
            python = component / "venv/bin/python"
            if not python.exists():
                run(name + "-venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(component / "venv")])
                (component / "venv/lib/python3.12/site-packages/support.pth").write_text(
                    str(repo / ".venv/lib/python3.12/site-packages")
                    + "\n"
                    + str(repo.parent / "astrovela_vane-batch-udf-20260930/.venv-gpu/lib/python3.12/site-packages")
                    + "\n"
                )
            env = dict(
                os.environ,
                SKBUILD_BUILD_DIR=str(component / "build/python-release"),
                SKBUILD_CMAKE_BUILD_TYPE="Release",
                CMAKE_BUILD_PARALLEL_LEVEL="2",
                PKG_CONFIG_PATH=str(fuse / "lib/x86_64-linux-gnu/pkgconfig"),
                PKG_CONFIG_SYSROOT_DIR=str(fuse.parent),
            )
            run(
                name + "-install",
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
                cwd=component,
                env=env,
            )
            report["variants"][name] = {
                "path": str(component),
                "sources": {
                    str(p.relative_to(component)): digest(p)
                    for folder in ("src", "include", "tests", "python")
                    for p in (component / folder).rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                },
                "binaries": {
                    str(p.relative_to(component)): digest(p)
                    for p in (
                        component / "build/python-release/vane-fs-mount",
                        component / "build/python-release/libvane_fs.a",
                        component / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so",
                    )
                },
            }
        before, after = (report["variants"][n]["sources"] for n in ("lookup", "coalesced"))
        assert before.keys() == after.keys()
        assert {p for p in before if before[p] != after[p]} == {"src/checkpoint.hpp"}
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
