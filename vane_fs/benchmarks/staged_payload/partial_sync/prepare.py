# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build publication/sync diagnostics from the frozen partial-edge comparison."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    parser.add_argument(
        "--writeback", action="store_true", help="Trace the isolated writeback comparison, including each hint"
    )
    args = parser.parse_args()
    out, parent = args.output.resolve(), args.parent.resolve()
    out.mkdir()
    here = Path(__file__).resolve().parent
    identity = json.loads((parent / "build.json").read_text())
    assert identity["status"] == "PASS"
    assert json.loads((parent / "tests.json").read_text())["status"] == "PASS"
    sdk, fuse = args.sqlite_prefix.resolve(), args.fuse_prefix.resolve()
    report = dict(status="RUNNING", parent=str(parent), commands=[], inputs={}, source_inputs={}, binaries={})

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
        timeline = here.parent / "block_sync/timeline.cpp"
        trace = timeline.read_text().replace("threads[8]", "threads[64]").replace("thread >= 8", "thread >= 64")
        trace += """
namespace payload_trace {
long long Now() { return timeline::Now(); }
void Record(const char *name, long long begin) { timeline::Record(name, begin, 0); }
}
"""
        if args.writeback:
            trace = replace_once(
                trace,
                "DiskStats disk_before, disk_after;",
                "DiskStats disk_before, disk_after;\n    long long range_bytes;\n    unsigned flags;",
            )
            trace = replace_once(
                trace,
                "const DiskStats *before = nullptr)",
                "const DiskStats *before = nullptr, long long range_bytes = -1, unsigned flags = 0)",
            )
            trace = replace_once(
                trace,
                "before ? *before : DiskStats {}, after});",
                "before ? *before : DiskStats {}, after, range_bytes, flags});",
            )
            trace = replace_once(
                trace,
                "f << '}';",
                'f << ",\\"range_bytes\\":" << r.range_bytes << ",\\"flags\\":" << r.flags << \'}\';',
            )
            trace += (here / "hint_trace.cpp").read_text()
        (out / "timeline.cpp").write_text(trace)
        sqlite = sdk / "lib/libsqlite3.a"
        fuselib = fuse / "lib/x86_64-linux-gnu/libfuse3.so"
        report["inputs"] = {str(p): digest(p) for p in (timeline, here / "trace.hpp", sqlite, fuselib)}
        if args.writeback:
            report["inputs"][str(here / "hint_trace.cpp")] = digest(here / "hint_trace.cpp")
        for name in ("baseline", "updated"):
            component = parent / name
            if args.writeback and name == "baseline":
                component = Path(identity["parent"]) / "updated"
            expected = identity["source_inputs"][name]
            assert all(digest(component / p) == h for p, h in expected.items() if "__pycache__" not in Path(p).parts)
            target = out / "diagnostic" / name
            shutil.copytree(
                component,
                target,
                ignore=shutil.ignore_patterns("build", "venv", "sanitize", "allocation", "*.orig", "__pycache__"),
            )
            # These headers add intervals and mutex-wait observations only.
            # Every durable syscall still executes the original implementation.
            shutil.copy2(here / "trace.hpp", target / "src/payload_trace.hpp")
            stage = (target / "src/staging.hpp").read_text()
            stage = replace_once(
                stage, '#include "payload_file.hpp"', '#include "payload_file.hpp"\n#include "payload_trace.hpp"'
            )
            stage = replace_once(
                stage, "thread(stage.mutex)", 'thread(payload_trace::Lock("publication_mutex_wait", stage.mutex))'
            )
            stage = replace_once(
                stage,
                "thread(stage.checkpoint_mutex)",
                'thread(payload_trace::Lock("stage_checkpoint_mutex_wait", stage.checkpoint_mutex))',
            )
            for signature, label in [
                ("void BeginBatch(sqlite3 *writer) {", "begin_batch"),
                ("void BeforeFullCommit() {", "payload_publication"),
                ("void Publish(sqlite3 *db) {", "publish"),
            ]:
                stage = replace_once(
                    stage, signature, signature + '\n        payload_trace::Span trace("' + label + '");'
                )
            if args.writeback and name == "updated":
                stage = replace_once(
                    stage,
                    "void Writeback(int64_t offset, int64_t bytes) {",
                    'void Writeback(int64_t offset, int64_t bytes) {\n        payload_trace::Span trace("payload_writeback_hint");',
                )
            stage = replace_once(
                stage,
                "void Save(bool pending) {",
                'void Save(bool pending) {\n        payload_trace::Span trace(pending ? "fence_open" : "fence_clean");',
            )
            (target / "src/staging.hpp").write_text(stage)
            checkpoint = (target / "src/checkpoint.hpp").read_text()
            checkpoint = replace_once(
                checkpoint,
                "void BeforeWrite(sqlite3 *writer) {",
                'void BeforeWrite(sqlite3 *writer) {\n        payload_trace::Span trace("admission");',
            )
            needle = "std::lock_guard<std::mutex> checkpoint_lock(checkpoint_mutex);"
            assert checkpoint.count(needle) == 2
            checkpoint = checkpoint.replace(
                needle,
                'auto checkpoint_lock = payload_trace::Lock("admission_checkpoint_mutex_wait", checkpoint_mutex);',
                1,
            )
            checkpoint = checkpoint.replace(
                needle,
                'auto checkpoint_lock = payload_trace::Lock("worker_checkpoint_mutex_wait", checkpoint_mutex);',
                1,
            )
            (target / "src/checkpoint.hpp").write_text(checkpoint)
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
            source = (target / "src/fuse.cpp").read_text()
            entry = "int main(int argc, char **argv) {"
            source = "void timeline_install(const char *);\n" + replace_once(
                source, entry, entry + "\n    if (argc > 1) timeline_install(argv[1]);"
            )
            (target / "fuse.cpp").write_text(source)
            argv = [
                "c++",
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "-DVANE_FS_SPLIT_WAL_SYNC=0",
                "-I" + str(target / "include"),
                "-I" + str(sdk / "include"),
                "-I" + str(fuse / "include/fuse3"),
                str(target / "fuse.cpp"),
                str(out / "timeline.cpp"),
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
                    "sqlite3_step",
                    "sqlite3_wal_checkpoint_v2",
                    "fsync",
                    "fdatasync",
                    "pread",
                    "pwrite",
                    "pread64",
                    "pwrite64",
                    "ftruncate",
                    "ftruncate64",
                )
            ]
            if args.writeback:
                argv.append("-Wl,--wrap=sync_file_range")
            argv += ["-o", str(build / "vane-fs-mount")]
            run(name + "-mount", argv)
            report["binaries"][str((build / "vane-fs-mount").relative_to(out))] = digest(build / "vane-fs-mount")
            # The unchanged CLI initializes the unmounted fixture before the
            # traced mount starts; it performs no measured or traced I/O.
            cli = component / "build/python-release/vane-fs"
            (build / "vane-fs").symlink_to(cli)
            report["inputs"][str(cli)] = digest(cli)
            alias = out / name
            alias.mkdir()
            (alias / "venv").symlink_to(component / "venv", target_is_directory=True)
        shutil.copy2(parent / "tests.json", out / "tests.json")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
