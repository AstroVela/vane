# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build plain native probes and separate diagnostic mounts from prepared components."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = args.output.resolve()
    if (out / "probe-build.json").exists() or (out / "diagnostic").exists():
        parser.error("Output already contains a probe build; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "started_epoch": time.time(), "commands": [], "inputs": {}, "outputs": {}}

    def save():
        (out / "probe-build.json").write_text(json.dumps(report, indent=2) + "\n")

    def build(label, command):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=log, check=False)
        report["commands"].append(
            {"argv": command, "returncode": result.returncode, "started_epoch": start, "ended_epoch": time.time()}
        )
        save()
        result.check_returncode()

    try:
        sdk = args.sqlite_prefix.resolve()
        fuse = args.fuse_prefix.resolve()
        timeline = here.parents[1] / "staged_payload/block_sync/timeline.cpp"
        source = timeline.read_text().replace("threads[8]", "threads[64]").replace("thread >= 8", "thread >= 64")
        # This measures the mutex wait *before* SQLITE_CHECKPOINT_RESTART, which
        # the SQLite wrapper alone cannot observe. Only diagnostic copies use it.
        source += """
void partial_trace(const char *name, std::chrono::steady_clock::time_point start) {
    auto begin = std::chrono::duration_cast<std::chrono::nanoseconds>(start - timeline::origin).count();
    timeline::Record(name, begin, 0, -1, -1, -1, -1, true);
}
void partial_write(size_t size, long long offset, std::chrono::steady_clock::time_point start) {
    auto begin = std::chrono::duration_cast<std::chrono::nanoseconds>(start - timeline::origin).count();
    auto name = "fuse_write_" + std::to_string(size);
    timeline::Record(name.c_str(), begin, 0, -1, -1, -1, offset, true);
}
"""
        trace = out / "timeline.cpp"
        trace.write_text(source)
        for name, component in (("baseline", args.baseline.resolve()), ("updated", args.candidate.resolve())):
            alias = out / name
            if not alias.exists():
                alias.symlink_to(component, target_is_directory=True)
            assert alias.resolve() == component
            library = component / "build/python-release/libvane_fs.a"
            sqlite = sdk / "lib/libsqlite3.a"
            fuselib = fuse / "lib/x86_64-linux-gnu/libfuse3.so"
            common = [
                "g++",
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "-I" + str(component / "include"),
                "-I" + str(sdk / "include"),
            ]
            for path in (library, sqlite, fuselib, timeline, trace, here / "native_probe.cpp"):
                report["inputs"][str(path)] = digest(path)
            plain = out / (name + "-native-probe")
            build(
                name + "-plain",
                common
                + [str(here / "native_probe.cpp"), str(library), str(sqlite), "-ldl", "-pthread", "-o", str(plain)],
            )
            alignment = out / (name + "-alignment-probe")
            counters = here.parent / "alignment_probe.cpp"
            report["inputs"][str(counters)] = digest(counters)
            symbols = ["fsync", "fdatasync", "sqlite3_prepare_v2", "sqlite3_step", "sqlite3_finalize", "sqlite3_exec"]
            build(
                name + "-alignment",
                common
                + [str(counters), str(library), str(sqlite), "-ldl", "-pthread"]
                + ["-Wl,--wrap=" + symbol for symbol in symbols]
                + ["-o", str(alignment)],
            )
            target = out / "diagnostic" / name
            shutil.copytree(component / "src", target / "src")
            (target / "build").mkdir()
            (target / "build/vane-fs").symlink_to(component / "build/python-release/vane-fs")
            header = target / "src/checkpoint.hpp"
            source = header.read_text()
            source = source.replace(
                "namespace vane_fs {",
                "void partial_trace(const char *, std::chrono::steady_clock::time_point);\n\nnamespace vane_fs {",
                1,
            )
            needle = "\n\t\tstd::lock_guard<std::mutex> checkpoint_lock(checkpoint_mutex);"
            assert source.count(needle) == 1
            source = source.replace(
                needle,
                """
        struct AdmissionScope {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            ~AdmissionScope() { partial_trace("admission", start); }
        } admission;
        auto wait_begin = std::chrono::steady_clock::now();
"""
                + needle
                + '\n\t\tpartial_trace("admission_mutex_wait", wait_begin);',
            )
            header.write_text(source)
            mount = target / "src/fuse.cpp"
            source = mount.read_text()
            needle = "int main(int argc, char **argv) {"
            assert source.count(needle) == 1
            source = (
                "#include <chrono>\nvoid timeline_install(const char *);\nvoid partial_write(size_t, long long, std::chrono::steady_clock::time_point);\n"
                + source.replace(needle, needle + "\n if(argc>1) timeline_install(argv[1]);")
            )
            needle = "ops.write = [](fuse_req_t req, fuse_ino_t, const char *buffer, size_t size, off_t offset, fuse_file_info *info) {"
            assert source.count(needle) == 1
            source = source.replace(
                needle,
                needle
                + """
        struct WriteScope {
            size_t size;
            long long offset;
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            ~WriteScope() { partial_write(size, offset, start); }
        } request {size, offset};
""",
            )
            mount.write_text(source)
            obj = target / "workspace.o"
            build(name + "-workspace", common + ["-c", str(target / "src/workspace.cpp"), "-o", str(obj)])
            wrapped = [
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
            ]
            binary = target / "build/vane-fs-mount"
            command = common + [
                "-DVANE_FS_SPLIT_WAL_SYNC=0",
                "-I" + str(fuse / "include/fuse3"),
                str(mount),
                str(trace),
                str(obj),
                str(library),
                str(sqlite),
                str(fuselib),
                "-Wl,-rpath," + str(fuselib.parent),
                "-ldl",
                "-pthread",
            ]
            build(name + "-mount", command + ["-Wl,--wrap=" + symbol for symbol in wrapped] + ["-o", str(binary)])
            for path in (plain, alignment, binary, obj, header, mount):
                report["outputs"][str(path)] = digest(path)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save()


if __name__ == "__main__":
    main()
