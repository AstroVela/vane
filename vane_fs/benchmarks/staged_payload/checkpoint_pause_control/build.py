# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Link unchanged native cores with an identical activity-counter recorder and probe."""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from analyze import digest, save, validate_trace, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    parent = args.parent.resolve()
    here = Path(__file__).resolve().parent
    prior = json.loads((parent / "build.json").read_text())
    assert prior["status"] == "PASS"
    inputs = prior["inputs"] | prior["sources"] | prior["binaries"]
    for path in [
        parent / "build.json",
        parent / "ext4-diagnostic/results.json",
        here.parent / "checkpoint_progress/fixture.hpp",
        *here.glob("*.py"),
        *here.glob("*.cpp"),
        *here.glob("*.hpp"),
    ]:
        inputs[str(path)] = digest(path)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    out.mkdir()
    cores = {name: parent / "diagnostic" / name for name in ("baseline", "updated")}
    assert (cores["baseline"] / "src/wait_trace.hpp").read_bytes() == (
        cores["updated"] / "src/wait_trace.hpp"
    ).read_bytes()
    original = (cores["baseline"] / "src/wait_trace.cpp").read_text()
    assert original == (cores["updated"] / "src/wait_trace.cpp").read_text()
    changes = [
        (
            "static bool enabled = false;",
            "static bool enabled = false;\nstatic std::atomic<int> payload_roots {0};\nbool PayloadWorkerIdle() { return payload_roots.load() == 0; }",
        ),
        (
            "\trows.push_back({name, int64_t(rows.size()), offset, bytes, inode, caller});",
            '\tif (std::string_view(name) == "prewriteback_worker")\n\t\t++payload_roots;\n\trows.push_back({name, int64_t(rows.size()), offset, bytes, inode, caller});',
        ),
        (
            "\tspan.reset();\n\trow = nullptr;",
            '\tspan.reset();\n\tif (std::string_view(row->name) == "prewriteback_worker")\n\t\t--payload_roots;\n\trow = nullptr;',
        ),
    ]
    text = original
    for before, after in changes:
        assert text.count(before) == 1
        text = text.replace(before, after)
    reverse = text
    for before, after in reversed(changes):
        assert reverse.count(after) == 1
        reverse = reverse.replace(after, before)
    assert reverse == original
    (out / "recorder.cpp").write_text(text)
    offsets = [value * 4096 for value in random.Random(892).sample(range(16384), 512)]
    raw = json.loads((parent / "ext4-diagnostic/results.json").read_text())
    for state in ("fuse", "aged_fuse"):
        for row in raw[state]:
            phase = next(p for p in row["samples"] if p["workload"] == "random_write_4k")
            assert [p["offset"] for p in phase["diagnostic_intervals"] if p["kind"] == "write"] == offsets
    (out / "offsets.hpp").write_text(
        "// SPDX-FileCopyrightText: 2026 Vane contributors\n// SPDX-License-Identifier: Apache-2.0\n#include <array>\n#include <cstdint>\nstatic constexpr std::array<int64_t,512> OFFSETS = {"
        + ",".join(map(str, offsets))
        + "};\n"
    )
    sdk = Path(prior["sdk"])
    wraps = [
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
        "sync_file_range",
        "sqlite3_wal_hook",
    ]
    report = dict(
        status="RUNNING",
        started_epoch=time.time(),
        parent=str(parent),
        sdk=str(sdk),
        commands=[],
        tests=[],
        cleanup=[],
        inputs=inputs,
        binaries={},
        core_libraries={name: str(core / "build/python-release/libvane_fs.a") for name, core in cores.items()},
        runtime_sources_unchanged=True,
        recorder_changes=changes,
        recorder_exact_reverse=True,
        original_recorder_sha256=digest(cores["baseline"] / "src/wait_trace.cpp"),
        prior_algorithm_suites_reused_by_exact_identity=True,
    )
    if args.preflight:
        preflight = args.preflight.resolve()
        failed = json.loads((preflight / "build.json").read_text())
        assert failed["status"] == "FAIL" and failed["error"] == "AssertionError()"
        target = out / "preflight"
        target.mkdir()
        retained = [
            *preflight.glob("*.json"),
            *preflight.glob("*.log"),
            preflight / "recorder.cpp",
            preflight / "offsets.hpp",
            *preflight.glob("preflight-sources/*"),
        ]
        for path in retained:
            destination = target / path.relative_to(preflight)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        report["preflight"] = dict(
            path=str(preflight),
            reason="One FULL root issued two WAL fsync calls; the original exactly-one syscall assumption was invalid. Native operations passed. Validate every FULL root has successful WAL fsyncs and count all calls independently.",
            cleanup=failed["cleanup"],
            files={str(p): digest(p) for p in target.rglob("*") if p.is_file()},
        )
        for result in target.glob("baseline-release-probe-*.json"):
            if not result.name.endswith("-trace.json"):
                verify(
                    json.loads(result.read_text()),
                    json.loads(result.with_name(result.stem + "-trace.json").read_text()),
                )

    def run(label, argv, env=None):
        log = out / (label + ".log")
        begin = time.time()
        with log.open("w") as stream:
            process = subprocess.run(argv, env=env, stdout=stream, stderr=stream, timeout=300)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                started_epoch=begin,
                ended_epoch=time.time(),
                returncode=process.returncode,
                log=str(log),
                log_sha256=digest(log),
            )
        )
        save(out / "build.json", report)
        assert process.returncode == 0, log
        print(label, "PASS", flush=True)

    try:
        for mode in ("release", "asan"):
            flags = (
                ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
                if mode == "asan"
                else ["-O2", "-DNDEBUG"]
            )
            recorder = out / ("recorder-" + mode + ".o")
            run(
                mode + "-recorder",
                [
                    "c++",
                    "-std=c++17",
                    *flags,
                    "-I" + str(cores["baseline"] / "src"),
                    "-I" + str(sdk / "include"),
                    "-c",
                    str(out / "recorder.cpp"),
                    "-o",
                    str(recorder),
                ],
            )
            for component, core in cores.items():
                library = core / "build/python-release/libvane_fs.a"
                if mode == "asan":
                    library = out / (component + "-asan-core.o")
                    run(
                        component + "-asan-core",
                        [
                            "c++",
                            "-std=c++17",
                            *flags,
                            "-I" + str(core / "include"),
                            "-I" + str(core / "src"),
                            "-I" + str(sdk / "include"),
                            "-c",
                            str(core / "src/workspace.cpp"),
                            "-o",
                            str(library),
                        ],
                    )
                for program, source in [("probe", "probe.cpp"), ("faults", "test_pause.cpp")]:
                    binary = out / f"{component}-{mode}-{program}"
                    mapping = out / (binary.name + ".map")
                    run(
                        binary.name,
                        [
                            "c++",
                            "-std=c++17",
                            *flags,
                            "-I" + str(core / "src"),
                            "-I" + str(core / "include"),
                            "-I" + str(sdk / "include"),
                            "-I" + str(out),
                            str(here / source),
                            str(recorder),
                            str(library),
                            str(sdk / "lib/libsqlite3.a"),
                            "-pthread",
                            "-ldl",
                            *["-Wl,--wrap=" + symbol for symbol in wraps],
                            "-Wl,-Map=" + str(mapping),
                            "-o",
                            str(binary),
                        ],
                    )
                    assert not re.search(r"libvane_fs\.a\([^)]*wait_trace\.cpp\.o\)", mapping.read_text())
                    report["binaries"][str(binary)] = digest(binary)
                    cases = (
                        [("faults",)]
                        if program == "faults"
                        else [(state, arm) for state in ("warm", "gc") for arm in ("active", "paused")]
                    )
                    for case in cases:
                        label = binary.name + "-" + "-".join(case)
                        owned = Path(tempfile.mkdtemp(prefix="owned-checkpoint-pause-test-", dir="/dev/shm"))
                        result = out / (label + ".json")
                        db = owned / "db.sqlite"
                        try:
                            argv = (
                                [str(binary), str(db)]
                                if program == "faults"
                                else [str(binary), str(db), *case, str(result)]
                            )
                            run(
                                label,
                                argv,
                                dict(
                                    os.environ,
                                    ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                                    UBSAN_OPTIONS="halt_on_error=1",
                                ),
                            )
                            trace_path = Path(str(db) + ".waits.json")
                            trace = json.loads(trace_path.read_text())
                            value = (
                                validate_trace(trace)
                                if program == "faults"
                                else verify(json.loads(result.read_text()), trace)
                            )
                            target = out / (label + "-trace.json")
                            shutil.copy2(trace_path, target)
                            report["tests"].append(
                                dict(
                                    label=label,
                                    program=program,
                                    component=component,
                                    mode=mode,
                                    trace=str(target),
                                    validation=value,
                                )
                            )
                        finally:
                            assert str(owned) not in Path("/proc/mounts").read_text()
                            trace_path = Path(str(db) + ".waits.json")
                            if trace_path.exists() and not (out / (label + "-trace.json")).exists():
                                shutil.copy2(trace_path, out / (label + "-trace.json"))
                            files = {
                                str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
                                for p in owned.rglob("*")
                                if p.is_file()
                            }
                            shutil.rmtree(owned)
                            report["cleanup"].append(
                                dict(
                                    path=str(owned),
                                    removed=not owned.exists(),
                                    process_stopped=True,
                                    mount_absent=True,
                                    files=files,
                                )
                            )
                            save(out / "build.json", report)
        assert len(report["tests"]) == 20
        for path in (out / "recorder.cpp", out / "offsets.hpp"):
            report["inputs"][str(path)] = digest(path)
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
