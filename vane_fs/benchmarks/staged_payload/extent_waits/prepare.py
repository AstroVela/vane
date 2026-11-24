# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build reversible wait diagnostics for the frozen extent FUSE comparison."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACE = HERE.parent / "fuse_waits"
spec = importlib.util.spec_from_file_location("wait_helpers", TRACE / "prepare.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
digest, save, validate = helpers.digest, helpers.save, helpers.validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--recorder", type=Path, required=True)
    args = parser.parse_args()
    out, comparison, recorder = args.artifacts.resolve(), args.comparison.resolve(), args.recorder.resolve()
    prior = json.loads((comparison / "build.json").read_text())
    component = json.loads((Path(prior["component"]) / "build.json").read_text())
    source = component["source_manifest"]
    original = dict(baseline=Path(source["original"]), updated=Path(source["candidate"]))
    old = json.loads((recorder / "build.json").read_text())
    assert prior["status"] == component["status"] == old["status"] == "PASS"
    inputs = dict(prior["inputs"], **old["inputs"], **old["sources"], **old["binaries"])
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    out.mkdir()
    sdk, repo = Path(source["sdk"]), HERE.parents[3]
    report = dict(
        status="RUNNING",
        started_epoch=time.time(),
        parent=str(comparison),
        template=str(comparison / "full"),
        sdk=str(sdk),
        commands=[],
        cleanup=[],
        inputs=inputs,
        sources={},
        binaries={},
        reverse_proof={},
    )

    def run(label, argv, cwd=None, env=None):
        log, begin = out / (label + ".log"), time.time()
        with log.open("w") as stream:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=stream, timeout=600)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                cwd=str(cwd),
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

    try:
        for name, core in original.items():
            target = out / "diagnostic" / name
            target.parent.mkdir(exist_ok=True)
            if name == "baseline":
                target.symlink_to(recorder / "diagnostic/updated", target_is_directory=True)
            else:
                target.mkdir()
                for item in (
                    "src",
                    "include",
                    "python",
                    "tests",
                    "LICENSES",
                    "triplets",
                    "CMakeLists.txt",
                    "pyproject.toml",
                    "README.md",
                    "LICENSE",
                    "vcpkg.json",
                ):
                    path = core / item
                    if path.is_dir():
                        shutil.copytree(path, target / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    elif path.is_file():
                        shutil.copy2(path, target / item)
            for relative in (
                "CMakeLists.txt",
                "src/workspace.cpp",
                "src/checkpoint.hpp",
                "src/staging.hpp",
                "src/fuse.cpp",
            ):
                plain = (core / relative).read_text()
                instrumented, reverse, changes = helpers.instrument(plain, relative)
                assert reverse == plain
                if name == "baseline":
                    assert (target / relative).read_text() == instrumented
                else:
                    (target / relative).write_text(instrumented)
                report["reverse_proof"][str(target / relative)] = dict(
                    original_path=str(core / relative),
                    original_sha256=digest(core / relative),
                    instrumented_sha256=digest(target / relative),
                    exact_reverse=True,
                    changes=changes,
                )
            for file in ("wait_trace.hpp", "wait_trace.cpp"):
                if name == "updated":
                    shutil.copy2(TRACE / file, target / "src" / file)
                assert digest(target / "src" / file) == digest(TRACE / file)
            if name == "updated":
                python = target / "venv/bin/python"
                run("extent-venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(target / "venv")])
                (target / "venv/lib/python3.12/site-packages/support.pth").write_text(
                    str(core / "venv/lib/python3.12/site-packages")
                    + "\n"
                    + str(repo / ".venv/lib/python3.12/site-packages")
                    + "\n"
                )
                fuse = repo / ".cache/fuse-sdk/root/usr"
                env = dict(
                    os.environ,
                    SKBUILD_BUILD_DIR=str(target / "build/python-release"),
                    SKBUILD_CMAKE_BUILD_TYPE="Release",
                    CMAKE_BUILD_PARALLEL_LEVEL="2",
                    PKG_CONFIG_PATH=str(fuse / "lib/x86_64-linux-gnu/pkgconfig"),
                    PKG_CONFIG_SYSROOT_DIR=str(fuse.parent),
                )
                run(
                    "extent-install",
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
                        "-Ccmake.define.VANE_FS_BUILD_TESTS=OFF",
                    ],
                    target,
                    env,
                )
            for path in target.rglob("*"):
                if path.is_file() and not any(
                    p in ("build", "venv", "__pycache__") for p in path.relative_to(target).parts
                ):
                    report["sources"][str(path)] = digest(path)
            for path in [
                *(target / "build/python-release").glob("vane-fs*"),
                target / "build/python-release/libvane_fs.a",
                *target.glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"),
            ]:
                if path.is_file():
                    report["binaries"][str(path)] = digest(path)
            alias = out / name
            alias.mkdir()
            (alias / "venv").symlink_to(target / "venv", target_is_directory=True)
        for mode in ("release", "asan"):
            binary = out / ("trace-test-" + mode)
            flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if mode == "asan" else []
            run(
                mode + "-compile",
                [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-I" + str(sdk / "include"),
                    "-I" + str(TRACE),
                    str(TRACE / "test_trace.cpp"),
                    str(TRACE / "wait_trace.cpp"),
                    str(sdk / "lib/libsqlite3.a"),
                    "-ldl",
                    "-pthread",
                    *["-Wl,--wrap=" + n for n in helpers.WRAPS],
                    *flags,
                    "-o",
                    str(binary),
                ],
            )
            owned = Path(tempfile.mkdtemp(prefix="owned-extent-wait-", dir="/dev/shm"))
            try:
                run(
                    mode + "-trace",
                    [str(binary), str(owned / "test")],
                    env=dict(
                        os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"
                    ),
                )
                trace = json.loads((owned / "test.waits.json").read_text())
                assert validate(trace)["roots"] == 2
                root = next(r for t in trace["threads"] for r in t["roots"] if r[0] == "test")
                assert root[10][trace["stages"].index("checkpoint_wait")][1] >= 2500000 and root[9] > 0
                errors = [e for t in trace["threads"] for e in t["events"] if e[0] == "fsync"]
                assert len(errors) == 1 and errors[0][6:8] == [-1, 9]
                save(out / (mode + "-trace.json"), trace)
            finally:
                files = {
                    str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
                    for p in owned.rglob("*")
                    if p.is_file()
                }
                shutil.rmtree(owned)
                report["cleanup"].append(
                    dict(path=str(owned), files=files, removed=not owned.exists(), processes_stopped=True)
                )
            report["binaries"][str(binary)] = digest(binary)
        report["inputs"].update(
            {
                str(p): digest(p)
                for p in [
                    Path(__file__),
                    HERE / "run.py",
                    TRACE / "prepare.py",
                    TRACE / "wait_trace.hpp",
                    TRACE / "wait_trace.cpp",
                    TRACE / "test_trace.cpp",
                    comparison / "build.json",
                    recorder / "build.json",
                    sdk / "lib/libsqlite3.a",
                ]
            }
        )
        assert all(digest(Path(p)) == sha for p, sha in inputs.items())
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "build.json", report)


if __name__ == "__main__":
    main()
