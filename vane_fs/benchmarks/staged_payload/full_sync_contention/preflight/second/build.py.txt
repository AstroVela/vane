# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Link the native diagnostic to frozen core/recorder sources and validate both arms."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse an already-configured failed validation directory")
    args = parser.parse_args()
    out, parent, here = args.output.resolve(), args.parent.resolve(), Path(__file__).resolve().parent
    repo = here.parents[3]
    if args.resume:
        assert json.loads((out / "build.json").read_text())["status"] == "FAIL"
    out.mkdir(exist_ok=args.resume)
    trace = parent / "trace"
    frozen = json.loads((trace / "build.json").read_text())
    assert frozen["status"] == "PASS" and json.loads((trace / "source-proof.json").read_text())["status"] == "PASS"
    core = trace / "diagnostic/updated"
    for path, sha in frozen["source_inputs"]["updated"].items():
        assert digest(core / path) == sha
    recorder = parent / "trace-generator"
    assert digest(recorder / "trace.hpp") == digest(core / "src/request_trace.hpp")
    for name in ("trace.hpp", "trace.cpp"):
        assert digest(recorder / name) == frozen["inputs"][str(recorder / name)]
    sdk = repo / "vane_fs/vcpkg_installed/x64-linux-release"
    release = core / "build/libvane_fs.a"
    report = dict(
        status="RUNNING",
        parent=str(parent),
        core=str(core),
        commands=[],
        tests=[],
        cleanup=[],
        inputs={
            str(path): digest(path)
            for path in (
                Path(__file__),
                here / "probe.cpp",
                here.parent / "checkpoint_progress/fixture.hpp",
                trace / "build.json",
                trace / "source-proof.json",
                parent / "tests.json",
                release,
                recorder / "trace.hpp",
                recorder / "trace.cpp",
                sdk / "lib/libsqlite3.a",
                sdk / "include/sqlite3.h",
            )
        },
        runtime_source_inputs={str(core / path): sha for path, sha in frozen["source_inputs"]["updated"].items()},
        unchanged_runtime_suites="Reuse 154 Python, 12 native, 12 ASAN/UBSAN and 56 checkpoint helpers from the exact parent implementation",
        binaries={},
    )

    def save():
        (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, argv, env=None):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            done = subprocess.run(argv, stdout=log, stderr=log, env=env, timeout=600)
        report["commands"].append(
            dict(label=label, argv=argv, start=start, end=time.time(), returncode=done.returncode)
        )
        save()
        done.check_returncode()
        print(label, "PASS", flush=True)

    try:
        sanitized = out / "sanitize"
        run(
            "asan-configure",
            [
                "cmake",
                "-S",
                str(core),
                "-B",
                str(sanitized),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                "-DCMAKE_PREFIX_PATH=" + str(sdk),
                "-DVANE_FS_BUILD_PYTHON=OFF",
                "-DVANE_FS_BUILD_TOOLS=OFF",
                "-DVANE_FS_BUILD_FUSE=OFF",
                "-DVANE_FS_BUILD_TESTS=OFF",
                "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer",
                "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
            ],
        )
        run("asan-library", ["cmake", "--build", str(sanitized), "-j", "2"])
        for name, library in (("release", release), ("asan", sanitized / "libvane_fs.a")):
            binary = out / ("probe-" + name)
            argv = [
                "c++",
                "-std=c++17",
                "-O2",
                "-DNDEBUG",
                "-I" + str(core / "src"),
                "-I" + str(core / "include"),
                "-I" + str(sdk / "include"),
                "-I" + str(recorder),
                str(here / "probe.cpp"),
                str(recorder / "trace.cpp"),
                str(library),
                str(sdk / "lib/libsqlite3.a"),
                "-ldl",
                "-pthread",
            ]
            if name == "asan":
                argv += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            argv += [
                "-Wl,--wrap=" + symbol
                for symbol in (
                    "sqlite3_wal_hook",
                    "sqlite3_wal_checkpoint_v2",
                    "fsync",
                    "fdatasync",
                    "pread",
                    "pwrite",
                    "pread64",
                    "pwrite64",
                    "sync_file_range",
                )
            ] + ["-o", str(binary)]
            run(name + "-probe", argv)
            report["binaries"][str(binary)] = digest(binary)
            for state in ("cold", "warm", "gc"):
                for variant in ("active", "quiesced"):
                    label = name + "-" + state + "-" + variant
                    owned = Path(tempfile.mkdtemp(prefix="owned-full-sync-test-", dir="/dev/shm"))
                    result = out / (label + ".json")
                    try:
                        run(
                            label,
                            [str(binary), str(owned / "db.sqlite"), state, variant, str(result)],
                            dict(
                                os.environ,
                                ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                                UBSAN_OPTIONS="halt_on_error=1",
                            ),
                        )
                        records = json.loads((owned / "db.sqlite.request-costs.json").read_text())
                        data = json.loads(result.read_text())
                        assert data["status"] == "PASS" and len(data["operation_ns"]) == 128
                        syncs, checkpoints = [], []
                        for thread in records["threads"]:
                            rows = [dict(zip(records["columns"], row, strict=True)) for row in thread["rows"]]
                            for value in thread["syncs"]:
                                sync = dict(zip(records["sync_columns"], value, strict=True))
                                sync["request"] = (
                                    rows[sync["request_index"]]["name"] if sync["request_index"] >= 0 else "unscoped"
                                )
                                syncs.append(sync)
                            checkpoints += [
                                v
                                for v in thread["checkpoints"]
                                if data["begin_ns"] <= v[5] < data["end_ns"] and v[1] != -1
                            ]
                        selected = [s for s in syncs if data["begin_ns"] <= s["begin_ns"] < data["end_ns"]]
                        wal = [s for s in selected if s["request"] == "fsync" and s["kind"] == "wal"]
                        assert len(wal) == 128 and all(s["result"] == s["errno"] == 0 for s in selected)
                        if variant == "quiesced" or state == "cold":
                            assert not checkpoints
                        else:
                            assert checkpoints
                        destination = out / (label + "-trace.json")
                        shutil.copy2(owned / "db.sqlite.request-costs.json", destination)
                        report["tests"].append(
                            dict(
                                label=label,
                                full_wal_syncs=len(wal),
                                maintenance_calls=len(checkpoints),
                                result=str(result),
                                trace=str(destination),
                            )
                        )
                    finally:
                        failed_trace = owned / "db.sqlite.request-costs.json"
                        if failed_trace.exists() and not (out / (label + "-trace.json")).exists():
                            shutil.copy2(failed_trace, out / (label + "-trace.json"))
                        shutil.rmtree(owned)
                        report["cleanup"].append(dict(path=str(owned), removed=not owned.exists()))
                        save()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
