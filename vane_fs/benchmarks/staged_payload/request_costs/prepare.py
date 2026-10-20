# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build short-request diagnostics from the unchanged, budgeted comparison."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def span(stage):
    return f"\n\t\trequest_trace::Span trace(request_trace::Stage::{stage});"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    args = parser.parse_args()
    out, parent = args.output.resolve(), args.parent.resolve()
    out.mkdir()
    here = Path(__file__).resolve().parent
    identity = json.loads((parent / "build.json").read_text())
    assert identity["status"] == "PASS"
    assert json.loads((parent / "writeback-tests.json").read_text())["status"] == "PASS"
    sdk, fuse = args.sqlite_prefix.resolve(), args.fuse_prefix.resolve()
    sqlite, fuselib = sdk / "lib/libsqlite3.a", fuse / "lib/x86_64-linux-gnu/libfuse3.so"
    report = dict(
        status="RUNNING",
        parent=str(parent),
        commands=[],
        inputs={
            str(p): digest(p) for p in (here / "prepare.py", here / "trace.cpp", here / "trace.hpp", sqlite, fuselib)
        },
        original_sources={},
        source_inputs={},
        binaries={},
    )

    def save():
        (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, argv):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            dict(label=label, argv=argv, returncode=result.returncode, started_epoch=start, ended_epoch=time.time())
        )
        save()
        result.check_returncode()
        print(label, "PASS", flush=True)

    try:
        for name in ("baseline", "updated"):
            component = parent / "updated" if name == "updated" else Path(identity["parent"]) / "updated"
            expected = identity["source_inputs"][name]
            assert all(digest(component / p) == h for p, h in expected.items())
            report["original_sources"][name] = dict(path=str(component), sha256=expected)
            target = out / "diagnostic" / name
            shutil.copytree(
                component,
                target,
                ignore=shutil.ignore_patterns("build", "venv", "sanitize", "allocation", "*.orig", "__pycache__"),
            )
            shutil.copy2(here / "trace.hpp", target / "src/request_trace.hpp")
            source = (target / "src/workspace.cpp").read_text()
            source = replace_once(
                source, '#include "staging.hpp"', '#include "staging.hpp"\n#include "request_trace.hpp"'
            )
            for signature, label in [
                ("void Exec(sqlite3 *db, const std::string &sql) {", "Exec"),
                ("Statement(sqlite3 *db, const std::string &sql) : db(db) {", "StatementAcquire"),
                ("~Statement() {", "StatementRelease"),
                ("bool Step() {", "StatementStep"),
            ]:
                source = replace_once(source, signature, signature + span(label))
            source = replace_once(
                source,
                "int code = sqlite3_step(statement);",
                "int code = sqlite3_step(statement);\n\t\trequest_trace::Steps(sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 1));",
            )
            source = replace_once(
                source,
                "\t\tlock.lock();",
                "\t\t{\n\t\t\trequest_trace::Span trace(request_trace::Stage::DatabaseWait);\n\t\t\tlock.lock();\n\t\t}",
            )
            source = replace_once(
                source,
                "payload_guard = std::make_unique<PayloadFile::Guard>(*database.payload_file, exclusive_payload);",
                "{\n\t\t\trequest_trace::Span trace(request_trace::Stage::PayloadGuard);\n\t\t\t"
                "payload_guard = std::make_unique<PayloadFile::Guard>(*database.payload_file, exclusive_payload);\n\t\t}",
            )
            for control, label in [('write ? "BEGIN IMMEDIATE" : "BEGIN"', "Begin"), ('"COMMIT"', "Commit")]:
                needle = f"Control(database.db, {control});"
                source = replace_once(source, needle, "{" + span(label) + "\n\t\t\t" + needle + "\n\t\t}")
            needle = "std::lock_guard<std::mutex> guard(registry.mutex);"
            assert source.count(needle) == 2
            source = source.replace(
                needle, "auto guard = request_trace::Lock(request_trace::Stage::PinsWait, registry.mutex);"
            )
            (target / "src/workspace.cpp").write_text(source)

            stage = (target / "src/staging.hpp").read_text()
            stage = replace_once(
                stage, '#include "payload_file.hpp"', '#include "payload_file.hpp"\n#include "request_trace.hpp"'
            )
            stage = replace_once(
                stage,
                "thread(stage.mutex)",
                "thread(request_trace::Lock(request_trace::Stage::PublicationWait, stage.mutex))",
            )
            for signature, label in [
                ("static void Exec(sqlite3 *db, const char *sql) {", "Exec"),
                ("void BeforeFullCommit() {", "Publication"),
            ]:
                stage = replace_once(stage, signature, signature + span(label))
            if name == "updated":
                stage = replace_once(
                    stage, "void CheckWritebackError() {", "void CheckWritebackError() {" + span("WritebackError")
                )
                stage = replace_once(stage, "if (this->stage) {", "if (this->stage) {" + span("PauseEnter"))
                stage = replace_once(stage, "if (stage) {", "if (stage) {" + span("PauseExit"))
            (target / "src/staging.hpp").write_text(stage)

            checkpoint = (target / "src/checkpoint.hpp").read_text()
            for signature, label in [
                ("void BeforeWrite(sqlite3 *writer) {", "Admission"),
                ("static int OnCommit(void *context, sqlite3 *, const char *, int frames) {", "WalHook"),
            ]:
                checkpoint = replace_once(checkpoint, signature, signature + span(label))
            checkpoint = replace_once(
                checkpoint,
                "auto &self = *static_cast<Checkpointer *>(context);",
                "request_trace::CommitFrames(frames);\n\t\tauto &self = *static_cast<Checkpointer *>(context);",
            )
            checkpoint = replace_once(
                checkpoint,
                "\t\t\tlock.unlock();",
                '\t\t\tlock.unlock();\n\t\t\trequest_trace::Scope request_scope("checkpoint_worker");',
            )
            (target / "src/checkpoint.hpp").write_text(checkpoint)
            fuse_source = (target / "src/fuse.cpp").read_text()
            fuse_source = '#include "request_trace.hpp"\n' + fuse_source
            pattern = r"(ops\.(\w+) = \[\]\([\s\S]*?\) \{)"
            fuse_source, count = re.subn(
                pattern,
                lambda m: m[1] + '\n\t\trequest_trace::Scope request_scope("' + m[2] + '");',
                fuse_source,
            )
            assert count == 21, count
            for operation in ("read", "write"):
                fuse_source = replace_once(
                    fuse_source,
                    f'request_trace::Scope request_scope("{operation}");',
                    f'request_trace::Scope request_scope("{operation}", offset, int64_t(size));',
                )
            fuse_source = replace_once(
                fuse_source,
                "void InvalidateAttributes(Mount &mount, int64_t inode) {",
                "void InvalidateAttributes(Mount &mount, int64_t inode) {" + span("Notify"),
            )
            fuse_source = replace_once(
                fuse_source,
                "int main(int argc, char **argv) {",
                "int main(int argc, char **argv) {\n\tif (argc > 1) request_trace::Install(argv[1]);",
            )
            (target / "src/fuse.cpp").write_text(fuse_source)
            report["source_inputs"][name] = {
                str(p.relative_to(target)): digest(p)
                for directory in ("src", "include")
                for p in (target / directory).rglob("*")
                if p.is_file()
            }
            build = target / "build"
            run(
                name + "-configure",
                [
                    "cmake",
                    "-S",
                    str(target),
                    "-B",
                    str(build),
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_PREFIX_PATH=" + str(sdk),
                    "-DVANE_FS_BUILD_PYTHON=OFF",
                    "-DVANE_FS_BUILD_FUSE=OFF",
                    "-DVANE_FS_BUILD_TESTS=OFF",
                    "-DVANE_FS_BUILD_TOOLS=OFF",
                ],
            )
            run(name + "-library", ["cmake", "--build", str(build), "-j", "2"])
            argv = [
                "c++",
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "-I" + str(target / "include"),
                "-I" + str(target / "src"),
                "-I" + str(sdk / "include"),
                "-I" + str(fuse / "include/fuse3"),
                str(target / "src/fuse.cpp"),
                str(here / "trace.cpp"),
                str(build / "libvane_fs.a"),
                str(sqlite),
                str(fuselib),
                "-Wl,-rpath," + str(fuselib.parent),
                "-ldl",
                "-pthread",
            ]
            argv += [
                "-Wl,--wrap=" + s
                for s in (
                    "sqlite3_wal_checkpoint_v2",
                    "fsync",
                    "fdatasync",
                    "pread",
                    "pwrite",
                    "pread64",
                    "pwrite64",
                    "sync_file_range",
                )
            ]
            argv += ["-o", str(build / "vane-fs-mount")]
            run(name + "-mount", argv)
            report["binaries"][str((build / "vane-fs-mount").relative_to(out))] = digest(build / "vane-fs-mount")
            cli = component / "build/python-release/vane-fs"
            (build / "vane-fs").symlink_to(cli)
            report["inputs"][str(cli)] = digest(cli)
            alias = out / name
            alias.mkdir()
            (alias / "venv").symlink_to(parent / "updated/venv", target_is_directory=True)
        shutil.copy2(parent / "tests.json", out / "tests.json")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
