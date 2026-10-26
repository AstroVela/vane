# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build reversible native CPU instrumentation around the unchanged staged core."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

WRAPS = [
    "sqlite3_prepare_v2",
    "sqlite3_step",
    "sqlite3_exec",
    "sqlite3_wal_checkpoint_v2",
    "pwrite",
    "pwrite64",
    "pread",
    "pread64",
    "fsync",
    "fdatasync",
]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def clean(owned, report):
    assert str(owned) not in Path("/proc/mounts").read_text()
    files = {
        str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
        for p in owned.rglob("*")
        if p.is_file()
    }
    shutil.rmtree(owned)
    report["cleanup"].append(dict(path=str(owned), files=files, removed=not owned.exists(), processes_stopped=True))


def validate(row, profiled):
    assert row["status"] == "PASS" and row["content_and_integrity"]
    assert len(row["requests"]) == row["mib"]
    assert bool(row["events"]) == profiled
    for phase in row["phases"]:
        assert 0 <= phase["thread_cpu_ns"] and phase["begin_ns"] < phase["end_ns"]
        counters = phase["stages"]
        if profiled:
            assert sum(v[2] for v in counters.values()) == counters["root"][1]
            assert sum(v[4] for v in counters.values()) == counters["root"][3]
            assert all(v[2] >= 0 and v[4] >= 0 for v in counters.values())
            assert counters["root"][3] <= phase["thread_cpu_ns"]
        else:
            assert not counters
    if profiled:
        counters = next(p for p in row["phases"] if p["name"] == "writes")["stages"]
        groups = row["mib"] * 4
        assert counters["batch"][0] == groups
        assert counters["build_sql"][0] == counters["concatenate"][0] == counters["bind_batch"][0] == groups
        assert counters["step_payload"][0] == counters["step_version"][0] == groups
        assert counters["step_payload"][8] > 0 and counters["step_version"][8] > 0
        assert counters["payload_append"][7] == row["mib"] * 1024**2
        assert all(e[7] == 0 for e in row["events"] if e[0] != "checkpoint")
        assert all(e[7] in (0, 5, 6) for e in row["events"] if e[0] == "checkpoint")


def execute(binary, owned, mib, output):
    argv = [str(binary), str(owned / "db.sqlite"), str(mib), str(output)]
    start = time.time()
    process = None
    try:
        with output.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(
                argv,
                stdout=log,
                stderr=log,
                env=dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"),
            )
            code = process.wait(timeout=180)
        assert code == 0, output.with_suffix(".log").read_text()
        row = json.loads(output.read_text())
        row.update(command=argv, pid=process.pid, started_epoch=start, ended_epoch=time.time())
        return row
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=20)
        assert process is None or process.poll() is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    out, parent, here = args.output.resolve(), args.parent.resolve(), Path(__file__).resolve().parent
    repo = here.parents[3]
    old = json.loads((parent / "build.json").read_text())
    assert old["status"] == "PASS"
    component = old["components"]["fsync"]
    original, sdk = Path(component["path"]), Path(component["sdk"])
    assert all(digest(Path(p)) == sha for field in ("sources", "binaries") for p, sha in component[field].items())
    out.mkdir()
    report = dict(
        status="RUNNING",
        parent=str(parent),
        component=component,
        commands=[],
        tests=[],
        cleanup=[],
        started_epoch=time.time(),
        inputs={
            str(p): digest(p)
            for p in [
                Path(__file__),
                here / "costs.hpp",
                here / "costs.cpp",
                here / "probe.cpp",
                here / "test_costs.cpp",
                here.parent / "checkpoint_progress/fixture.hpp",
                parent / "build.json",
                sdk / "lib/libsqlite3.a",
                sdk / "include/sqlite3.h",
            ]
        },
        default={
            str(p): digest(p)
            for p in [
                repo / ".venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so",
                repo / "vane_fs/vcpkg_installed/x64-linux-release/lib/libsqlite3.a",
            ]
            + sorted((repo / "vane_fs/src").glob("*.[ch]pp"))
        },
        diagnostic_sources={},
        binaries={},
    )

    def run(label, argv, env=None):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=log, timeout=600)
        report["commands"].append(
            dict(label=label, argv=argv, started_epoch=start, ended_epoch=time.time(), returncode=result.returncode)
        )
        save(out / "build.json", report)
        assert result.returncode == 0, label
        print(label, "PASS", flush=True)

    try:
        target = out / "profile"
        target.mkdir()
        for name in ("src", "include"):
            shutil.copytree(original / name, target / name)
        shutil.copy2(original / "CMakeLists.txt", target / "CMakeLists.txt")
        shutil.copy2(here / "costs.hpp", target / "src/costs.hpp")
        edits = []

        def replace(source, old_text, new_text):
            assert source.count(old_text) == 1, old_text
            edits.append((old_text, new_text))
            return source.replace(old_text, new_text)

        source = (target / "src/workspace.cpp").read_text()
        source = replace(source, '#include "staging.hpp"', '#include "staging.hpp"\n#include "costs.hpp"')
        for signature, stage in [
            ("Statement(sqlite3 *db, const std::string &sql) : db(db) {", "AcquireStatement"),
            ("~Statement() {", "ReleaseStatement"),
            (
                "void WriteBytes(sqlite3 *db, Record &node, const std::string &data, int64_t offset, const View &view) {",
                "WriteBytes",
            ),
        ]:
            source = replace(source, signature, signature + f"\n\twrite_costs::Scope profile(write_costs::{stage});")
        begin, end = source.index("void InsertNewBlocks("), source.index("void RequireFile(")
        fragment = source[begin:end]
        fragment = replace(
            fragment,
            "\tstd::string payload_sql = ",
            "\twrite_costs::Scope profile(write_costs::Batch);\n\twrite_costs::Scope sql_profile(write_costs::BuildSql);\n\tstd::string payload_sql = ",
        )
        fragment = replace(fragment, "\tint64_t external = 0;", "\tsql_profile.Finish();\n\tint64_t external = 0;")
        fragment = replace(
            fragment,
            "\t\tstd::string bytes;",
            "\t\twrite_costs::Scope concatenate_profile(write_costs::Concatenate);\n\t\tstd::string bytes;",
        )
        fragment = replace(
            fragment,
            "\t\tStaging::Get(db).BeginBatch(db);",
            "\t\tconcatenate_profile.Finish();\n\t\tStaging::Get(db).BeginBatch(db);",
        )
        fragment = replace(
            fragment,
            "\tversions.Bind(1, inode);",
            "\twrite_costs::Scope bind_profile(write_costs::BindBatch);\n\tversions.Bind(1, inode);",
        )
        fragment = replace(fragment, "\tpayloads.Step();", "\tbind_profile.Finish();\n\tpayloads.Step();")
        source = source[:begin] + fragment + source[end:]
        restored = source
        for old_text, new_text in reversed(edits):
            assert restored.count(new_text) == 1
            restored = restored.replace(new_text, old_text)
        assert restored == (original / "src/workspace.cpp").read_text()
        (target / "src/workspace.cpp").write_text(source)
        payload = (target / "src/payload_file.hpp").read_text()
        old_text = "\tint64_t Append(const std::string &data) {"
        new_text = (
            old_text
            + "\n\t\twrite_costs::Scope profile(write_costs::PayloadAppend);\n\t\twrite_costs::AddBytes(write_costs::PayloadAppend, int64_t(data.size()));"
        )
        assert payload.count(old_text) == 1
        payload = payload.replace(old_text, new_text).replace(
            '#include "vane_fs/workspace.hpp"', '#include "vane_fs/workspace.hpp"\n#include "costs.hpp"'
        )
        restored_payload = payload.replace(new_text, old_text).replace(
            '#include "vane_fs/workspace.hpp"\n#include "costs.hpp"', '#include "vane_fs/workspace.hpp"'
        )
        assert restored_payload == (original / "src/payload_file.hpp").read_text()
        (target / "src/payload_file.hpp").write_text(payload)
        save(
            out / "source-proof.json",
            dict(status="PASS", workspace_reverse_exact=True, payload_reverse_exact=True, edits=edits),
        )
        report["diagnostic_sources"] = {str(p): digest(p) for p in target.rglob("*") if p.is_file()}
        for mode in ("profile", "asan"):
            build = out / (mode + "-build")
            configure = [
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
                "-DVANE_FS_BUILD_TESTS=OFF",
                "-DVANE_FS_BUILD_TOOLS=OFF",
                "-DVANE_FS_BUILD_FUSE=OFF",
            ]
            if mode == "asan":
                configure += ["-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer"]
            run(mode + "-configure", configure)
            run(mode + "-core", ["cmake", "--build", str(build), "-j", "2"])
        for mode in ("plain", "profile", "asan"):
            core = (
                original / "build/python-release/libvane_fs.a"
                if mode == "plain"
                else out / (mode + "-build/libvane_fs.a")
            )
            inc = original if mode == "plain" else target
            binary = out / ("probe-" + mode)
            argv = [
                "c++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I" + str(inc / "src"),
                "-I" + str(inc / "include"),
                "-I" + str(sdk / "include"),
                "-I" + str(here),
                str(here / "probe.cpp"),
            ]
            if mode != "plain":
                argv += [str(here / "costs.cpp"), "-DVANE_WRITE_PROFILE"]
            argv += [str(core), str(sdk / "lib/libsqlite3.a"), "-ldl", "-pthread", "-Wl,--wrap=sqlite3_wal_hook"]
            if mode != "plain":
                argv += ["-Wl,--wrap=" + name for name in WRAPS]
            if mode == "asan":
                argv += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            run(mode + "-probe-build", argv + ["-o", str(binary)])
            report["binaries"][str(binary)] = digest(binary)
            report["binaries"][str(core)] = digest(core)
            owned = Path(tempfile.mkdtemp(prefix="owned-native-write-validation-", dir="/dev/shm"))
            try:
                row = execute(binary, owned, 4, out / ("test-" + mode + ".json"))
                validate(row, mode != "plain")
                report["tests"].append(dict(mode=mode, result=row))
            finally:
                clean(owned, report)
        for mode in ("profile", "asan"):
            binary = out / ("test-counters-" + mode)
            argv = [
                "c++",
                "-std=c++17",
                "-O2",
                "-I" + str(sdk / "include"),
                str(here / "test_costs.cpp"),
                str(here / "costs.cpp"),
                str(sdk / "lib/libsqlite3.a"),
                "-ldl",
                "-pthread",
                "-o",
                str(binary),
            ]
            argv += ["-Wl,--wrap=" + name for name in WRAPS]
            if mode == "asan":
                argv += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            run(mode + "-counter-build", argv)
            run(mode + "-counter-validation", [str(binary)])
            report["binaries"][str(binary)] = digest(binary)
        report.update(status="PASS", ended_epoch=time.time())
    except BaseException as error:
        report.update(status="FAIL", ended_epoch=time.time(), error=repr(error))
        raise
    finally:
        save(out / "build.json", report)


if __name__ == "__main__":
    main()
