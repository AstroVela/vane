# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build selective pre-OPEN backfill in an isolated non-editable installation."""

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
    out, parent = args.output.resolve(), args.parent.resolve()
    here = Path(__file__).resolve().parent
    repo, source = here.parents[3], parent / "coalesced"
    expected = json.loads((parent / "candidates.json").read_text())["variants"]["coalesced"]["sources"]
    assert all(digest(source / path) == sha for path, sha in expected.items())
    out.mkdir()
    component = out / "updated"
    sdk, fuse = repo / "vane_fs/vcpkg_installed/x64-linux-release", repo / ".cache/fuse-sdk/root/usr"
    report = dict(
        status="RUNNING",
        parent=str(parent),
        commands=[],
        frozen_policy=dict(
            external_clean_prefix_above=256 * 1024,
            passive_attempts=1,
            prepare_before_sql_transaction=True,
            skip_open=True,
            clean_fixed_ms=5,
            clean_uncopied_below=16 * 1024 * 1024,
            open_uncopied_bytes=16 * 1024 * 1024,
            admission_bytes=64 * 1024 * 1024,
        ),
        inputs={str(path): digest(path) for path in (Path(__file__), here / "prepare.patch", sdk / "lib/libsqlite3.a")},
    )

    def run(label, argv, cwd=None, env=None):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            done = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            dict(label=label, argv=argv, cwd=str(cwd), start=start, end=time.time(), returncode=done.returncode)
        )
        done.check_returncode()
        print(label, "PASS", flush=True)

    try:
        shutil.copytree(
            source,
            component,
            ignore=shutil.ignore_patterns("build", "venv", "sanitize", "allocation", "__pycache__", "*.orig"),
        )
        run("prepare-patch", ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(here / "prepare.patch")], component)
        changed = {path for path, sha in expected.items() if digest(component / path) != sha}
        assert changed == {"src/checkpoint.hpp", "src/workspace.cpp"}
        proof = out / "reverse-proof"
        for path in changed:
            (proof / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(component / path, proof / path)
        run(
            "reverse-patch",
            ["patch", "--batch", "--reverse", "--fuzz=0", "-p1", "-i", str(here / "prepare.patch")],
            proof,
        )
        assert all((proof / path).read_bytes() == (source / path).read_bytes() for path in changed)
        header = (component / "src/checkpoint.hpp").read_text()
        original_header = (source / "src/checkpoint.hpp").read_text()
        assert header[header.index("\tvoid Run() {") :] == original_header[original_header.index("\tvoid Run() {") :]
        python = component / "venv/bin/python"
        run("venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(component / "venv")])
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
            component,
            env,
        )
        report.update(
            status="PASS",
            changed_sources=sorted(changed),
            reverse_patch_exact=True,
            background_worker_identical=True,
            sources={path: digest(component / path) for path in expected},
            binaries={
                str(path): digest(path)
                for path in (
                    component / "build/python-release/vane-fs-mount",
                    component / "build/python-release/libvane_fs.a",
                    component / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so",
                )
            },
        )
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (out / "candidate.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
