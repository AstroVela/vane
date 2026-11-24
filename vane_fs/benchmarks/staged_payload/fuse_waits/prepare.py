# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build reversible FUSE wait diagnostics from both frozen buffer implementations."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

WRAPS = [
    "fsync",
    "fdatasync",
    "flock",
    "pread",
    "pread64",
    "pwrite",
    "pwrite64",
    "sqlite3_step",
    "sqlite3_exec",
    "sqlite3_wal_checkpoint_v2",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate(trace):
    assert trace["clock"] == "CLOCK_MONOTONIC" and trace["slow_event_threshold_ns"] == 1000000
    roots = events = 0
    for thread in trace["threads"]:
        for i, row in enumerate(thread["roots"]):
            name, ident, begin, end, cpu, offset, size, inode, caller, steps, counters = row
            assert ident == i and name and end >= begin and cpu >= 0 and steps >= 0
            assert len(counters) == len(trace["stages"])
            assert all(v >= 0 for c in counters for v in c)
            assert counters[0][0] == 1 and counters[0][1] == end - begin and counters[0][3] == cpu
            assert sum(c[2] for c in counters) == end - begin
            assert sum(c[4] for c in counters) == cpu
            roots += 1
        for event in thread["events"]:
            assert event[4] >= event[3] and event[5] >= 0
            if event[2] >= 0:
                root = thread["roots"][event[2]]
                assert root[2] <= event[3] <= event[4] <= root[3]
            events += 1
    return dict(roots=roots, events=events, exclusive_accounting="PASS")


def instrument(source, name):
    changes = []

    def replace(old, new, count=1):
        nonlocal source
        assert source.count(old) == count, (name, old, source.count(old), count)
        source = source.replace(old, new)
        changes.append((old, new, count))

    def span(signature, stage):
        replace(signature, signature + f"\n\tfuse_waits::Span wait_span(fuse_waits::{stage});")

    if name == "src/workspace.cpp":
        replace('#include "staging.hpp"', '#include "staging.hpp"\n#include "wait_trace.hpp"')
        replace("\t\tlock.lock();", "\t\t{ fuse_waits::Span wait_span(fuse_waits::DatabaseWait); lock.lock(); }")
        needle = "payload_guard = std::make_unique<PayloadFile::Guard>(*database.payload_file, exclusive_payload);"
        replace(needle, "{ fuse_waits::Span wait_span(fuse_waits::PayloadGuard); " + needle + " }")
        for statement, stage in [('write ? "BEGIN IMMEDIATE" : "BEGIN"', "Begin"), ('"COMMIT"', "Commit")]:
            needle = f"Control(database.db, {statement});"
            replace(needle, "{ fuse_waits::Span wait_span(fuse_waits::" + stage + "); " + needle + " }")
        span(
            "void WriteBytes(sqlite3 *db, Record &node, const std::string &data, int64_t offset, const View &view) {",
            "WriteBytes",
        )
    elif name == "src/checkpoint.hpp":
        replace('#include "staging.hpp"', '#include "staging.hpp"\n#include "wait_trace.hpp"')
        span("void BeforeWrite(sqlite3 *writer, bool external_payload = false) {", "Admission")
        span("void PrepareExternal() {", "PrepareExternal")
        replace(
            "std::lock_guard<std::mutex> checkpoint_lock(checkpoint_mutex);",
            "auto checkpoint_lock = fuse_waits::Lock(fuse_waits::CheckpointWait, checkpoint_mutex);",
            3,
        )
        replace("\t\t\tlock.unlock();", '\t\t\tlock.unlock();\n\t\t\tfuse_waits::Root wait_root("checkpoint_worker");')
    elif name == "src/staging.hpp":
        replace('#include "payload_file.hpp"', '#include "payload_file.hpp"\n#include "wait_trace.hpp"')
        replace("thread(stage.mutex)", "thread(fuse_waits::Lock(fuse_waits::PublicationWait, stage.mutex))")
        replace(
            "thread(stage.checkpoint_mutex)", "thread(fuse_waits::Lock(fuse_waits::LeaseWait, stage.checkpoint_mutex))"
        )
        for signature, stage in [
            ("void BeginBatch(sqlite3 *writer) {", "BeginBatch"),
            ("void BeforeFullCommit() {", "BeforeFull"),
            ("void AfterFullCommit() {", "AfterFull"),
            ("void Publish(sqlite3 *db) {", "Publish"),
        ]:
            span(signature, stage)
    elif name == "src/fuse.cpp":
        replace('#include "vane_fs/workspace.hpp"', '#include "vane_fs/workspace.hpp"\n#include "wait_trace.hpp"')
        matches = list(re.finditer(r"ops\.(\w+) = \[\]\([\s\S]*?\) \{", source))
        assert len(matches) == 21
        for match in matches:
            replace(match[0], match[0] + '\n\t\tfuse_waits::Root wait_root("' + match[1] + '");')
        for operation in ("read", "write"):
            replace(
                f'fuse_waits::Root wait_root("{operation}");',
                f'fuse_waits::Root wait_root("{operation}", offset, int64_t(size), int64_t(info->fh), fuse_req_ctx(req)->pid);',
            )
        replace(
            'fuse_waits::Root wait_root("fsync");',
            'fuse_waits::Root wait_root("fsync", -1, -1, int64_t(inode), fuse_req_ctx(req)->pid);',
        )
        span("void InvalidateAttributes(Mount &mount, int64_t inode) {", "Notify")
        replace(
            "int main(int argc, char **argv) {",
            "int main(int argc, char **argv) {\n\tif (argc > 1) fuse_waits::Install(argv[1]);",
        )
        replace(
            "\t\tmount.workspace.Close();",
            '\t\t{ fuse_waits::Root wait_root("workspace_close"); mount.workspace.Close(); }',
        )
    elif name == "CMakeLists.txt":
        replace(
            "add_library(vane_fs STATIC src/workspace.cpp)",
            "add_library(vane_fs STATIC src/workspace.cpp src/wait_trace.cpp)\ntarget_link_options(vane_fs PUBLIC "
            + " ".join('"LINKER:--wrap=' + name + '"' for name in WRAPS)
            + ")",
        )
    original = source
    for old, new, count in reversed(changes):
        assert original.count(new) == count
        original = original.replace(new, old)
    return source, original, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    out, parent, here = args.output.resolve(), args.parent.resolve(), Path(__file__).resolve().parent
    repo = here.parents[3]
    base = json.loads((parent / "build.json").read_text())
    borrow = json.loads((parent / "borrow-build.json").read_text())
    assert base["status"] == borrow["status"] == "PASS"
    sdk = Path(base["component"]["sdk"])
    originals = {"baseline": Path(base["component"]["path"]), "updated": parent / "borrowed"}
    frozen = dict(
        base["component"]["sources"],
        **base["component"]["binaries"],
        **borrow["sources"],
        **borrow["binaries"],
        **base["default"],
    )
    assert all(digest(Path(p)) == sha for p, sha in frozen.items())
    out.mkdir()
    report = dict(
        status="RUNNING",
        started_epoch=time.time(),
        parent=str(parent),
        sdk=str(sdk),
        commands=[],
        cleanup=[],
        inputs=frozen,
        sources={},
        binaries={},
        reverse_proof={},
    )

    def run(label, argv, cwd=None, env=None):
        begin = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                cwd=str(cwd),
                started_epoch=begin,
                ended_epoch=time.time(),
                returncode=result.returncode,
            )
        )
        save(out / "build.json", report)
        assert result.returncode == 0, label
        print(label, "PASS", flush=True)

    try:
        for name, original in originals.items():
            target = out / "diagnostic" / name
            target.mkdir(parents=True)
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
                path = original / item
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
                source = (target / relative).read_text()
                updated, reverse, changes = instrument(source, relative)
                assert reverse == source
                (target / relative).write_text(updated)
                report["reverse_proof"][str(target / relative)] = dict(
                    original_path=str(original / relative),
                    original_sha256=digest(original / relative),
                    instrumented_sha256=digest(target / relative),
                    exact_reverse=True,
                    changes=changes,
                )
            for filename in ("wait_trace.hpp", "wait_trace.cpp"):
                shutil.copy2(here / filename, target / "src" / filename)
            python = target / "venv/bin/python"
            run(name + "-venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(target / "venv")])
            (target / "venv/lib/python3.12/site-packages/support.pth").write_text(
                str(repo / ".venv/lib/python3.12/site-packages") + "\n"
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
                    "-Ccmake.define.VANE_FS_BUILD_TESTS=OFF",
                ],
                target,
                env,
            )
            report["sources"].update(
                {
                    str(p): digest(p)
                    for p in target.rglob("*")
                    if p.is_file()
                    and "build" not in p.relative_to(target).parts
                    and "venv" not in p.relative_to(target).parts
                }
            )
            for p in [
                target / "build/python-release/vane-fs-mount",
                target / "build/python-release/libvane_fs.a",
                *target.glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"),
            ]:
                report["binaries"][str(p)] = digest(p)
            alias = out / name
            alias.mkdir()
            (alias / "venv").symlink_to(target / "venv", target_is_directory=True)
        flags = ["-Wl,--wrap=" + name for name in WRAPS]
        for mode in ("release", "asan"):
            binary = out / ("trace-test-" + mode)
            argv = [
                "c++",
                "-std=c++17",
                "-O2",
                "-I" + str(sdk / "include"),
                "-I" + str(here),
                str(here / "test_trace.cpp"),
                str(here / "wait_trace.cpp"),
                str(sdk / "lib/libsqlite3.a"),
                "-ldl",
                "-pthread",
                *flags,
                "-o",
                str(binary),
            ]
            if mode == "asan":
                argv += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            run(mode + "-compile", argv)
            owned = Path(tempfile.mkdtemp(prefix="owned-wait-trace-", dir="/dev/shm"))
            try:
                run(
                    mode + "-trace",
                    [str(binary), str(owned / "test")],
                    env=dict(
                        os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"
                    ),
                )
                trace = json.loads((owned / "test.waits.json").read_text())
                stats = validate(trace)
                assert stats["roots"] == 2
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
        report["inputs"].update(
            {
                str(p): digest(p)
                for p in [
                    Path(__file__),
                    here / "wait_trace.hpp",
                    here / "wait_trace.cpp",
                    here / "test_trace.cpp",
                    sdk / "lib/libsqlite3.a",
                    parent / "build.json",
                    parent / "borrow-build.json",
                ]
            }
        )
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "build.json", report)


if __name__ == "__main__":
    main()
